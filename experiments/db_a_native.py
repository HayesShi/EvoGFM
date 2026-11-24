"""Isolated native event replay of the full released A predictor.

Examples (from EvoGFM):
  python -m experiments.db_a_native prepare --dataset bitcoin-otc --output ../runs/a_native/bo
  python -m experiments.db_a_native run --trace ../runs/a_native/bo/trace.json --output ../runs/a_native/bo_runs

``run`` makes a separate Full reference, then five independently spawned paired
repeats with randomized method order. Validation opens disk outputs only after
workers exit. It does not tune weights, refit a readout, or select a query seed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from experiments.db_a_native_trace import identity, prepare_trace, write_json


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    os.environ.get("VLDB_CHECKPOINT", WORKSPACE / "checkpoints/reference.pt")
).resolve()
DEFAULT_READOUT = Path(
    os.environ.get(
        "VLDB_READOUT", WORKSPACE / "runs/vldb/readout/selected_readout.joblib"
    )
).resolve()
METHODS = ("full", "strict_adapted", "lazy", "hybrid_closure")
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def peak_rss_bytes():
    """OS process lifetime peak; Windows reports PeakWorkingSetSize."""
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "quota_peak_paged",
                    "quota_paged",
                    "quota_peak_nonpaged",
                    "quota_nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.peak)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def replay_temporal_history(n_nodes):
    if n_nodes > 750_000:
        from evogfm.compact_temporal import CompactTemporalHistory

        return CompactTemporalHistory(n_nodes)
    from evogfm.temporal import TemporalHistory

    return TemporalHistory(n_nodes)


def ingest_history(history, events, origin):
    """Exactly ingest all timestamps without one Python call per event.

    This is TemporalHistory's directed count/last-occurrence contract, batched
    with each pair's actual last timestamp. No event or self interaction drops.
    """
    if not len(events):
        return
    compact_ingest = getattr(history, "ingest_native", None)
    if compact_ingest is not None:
        compact_ingest(events, origin)
        return
    source, destination = events[:, 0], events[:, 1]
    times = (events[:, 2] - origin) / 604800.0
    keys, inverse, counts = np.unique(
        source * history.n + destination, return_inverse=True, return_counts=True
    )
    last = np.full(len(keys), -np.inf, dtype=np.float64)
    np.maximum.at(last, inverse, times)
    for key, count, timestamp in zip(keys.tolist(), counts.tolist(), last.tolist()):
        history.counts[key] = history.counts.get(key, 0) + count
        history.last[key] = timestamp
    np.add.at(history.out_count, source, 1)
    np.add.at(history.in_count, destination, 1)
    # TemporalHistory.last_node is float32 by construction; cast each input
    # before max so the result equals chronological scalar assignment.
    np.maximum.at(history.last_node, source, times.astype(np.float32))
    np.maximum.at(history.last_node, destination, times.astype(np.float32))


def check_identity(entry):
    current = identity(entry["path"])
    if current != entry:
        raise ValueError(
            f"Locked artifact changed since trace preparation: {entry['path']}"
        )


def counters(scorer):
    return {
        key: float(value) if isinstance(value, (float, np.floating)) else int(value)
        for key, value in getattr(scorer, "counters", {}).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def difference(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def readout_arrays(value):
    """Expose sklearn's array members to the shared storage deduplicator."""
    found, seen = [], set()

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            found.append(item)
        elif isinstance(item, dict):
            for member in item.values():
                visit(member)
        elif isinstance(item, (tuple, list)):
            for member in item:
                visit(member)
        elif type(item).__module__.startswith("sklearn.") and hasattr(item, "__dict__"):
            visit(vars(item))

    visit(value)
    return found


def preload_method(method):
    """Resolve method code and native import dependencies before service time."""
    if method == "full":
        from evogfm.incremental import MaintainedLinkScorer

        return MaintainedLinkScorer
    if method == "strict_adapted":
        from evogfm.strict_dependency import (
            StrictDependencyScorer,
            load_author_recompute,
        )

        # Import the pinned author planner and DGL here. The constructor's
        # subsequent loader call resolves the already resident modules.
        load_author_recompute()
        return StrictDependencyScorer
    from evogfm.demand_maintenance import DemandLinkScorer

    return DemandLinkScorer


def scorer_for(method, implementation, model, head, raw, edges, encoded, fraction):
    if method == "full":
        return implementation(
            model, head, raw, raw, edges, method="full", encoded=encoded
        )
    if method == "strict_adapted":
        return implementation(model, head, raw, edges, encoded=encoded)
    return implementation(
        model,
        head,
        raw,
        edges,
        encoded=encoded,
        policy="lazy" if method == "lazy" else "closure",
        materialize_fraction=fraction,
    )


def worker(args):
    # Numerical libraries are imported only in this single-method child.
    for key in THREAD_ENV:
        os.environ[key] = str(args.threads)
    import_started = time.perf_counter()
    import joblib
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from evogfm.data import canonical_edges
    from evogfm.incremental import ingest_edges, numeric_payload_bytes
    from experiments.full_graph import load_checkpoint
    from experiments.idea_a import frozen_structural_features
    from experiments.db_a_quality import predict_readout

    implementation = preload_method(args.method)
    excluded_import_s = time.perf_counter() - import_started

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    trace = json.loads(args.trace.read_text(encoding="utf8"))
    if trace["seed"] != 42:
        raise ValueError("First A experiment fixes seed 42")
    for key in ("checkpoint", "readout"):
        check_identity(trace[key])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"Worker result already exists: {output}")
    # Read complete replay inputs and frozen artifacts before the service clock.
    # References are never opened by this process.
    events = np.array(np.load(trace["events_path"], mmap_mode="r"), copy=True)
    with np.load(trace["queries_path"]) as pack:
        pair_arrays = {name: torch.from_numpy(pack[name].copy()) for name in pack.files}
    model, head, _ = load_checkpoint(trace["checkpoint"]["path"])
    readout = joblib.load(trace["readout"]["path"])
    expected = dict(input_dim=100, hidden=128, rank=16, heads=4)
    for key, value in expected.items():
        if model.config.get(key) != value:
            raise ValueError(
                f"Released full model requires {key}={value}: {model.config}"
            )
    if len(model.layers) != 2 or not model.config.get("joint_task_backbone"):
        raise ValueError("Full released two-layer joint backbone is required")
    if (
        readout.get("selected_id") != "union24_hgb15"
        or readout["model"]["input"] != "union24"
    ):
        raise ValueError("Use the already selected BO union24_hgb15 readout")
    stages = dict(
        initial_topology_s=0.0,
        raw_inputs_s=0.0,
        encode_s=0.0,
        scorer_init_s=0.0,
        history_init_s=0.0,
        topology_ingest_s=0.0,
        history_ingest_s=0.0,
        update_s=0.0,
        graph_features_s=0.0,
        history_features_s=0.0,
        released_head_s=0.0,
        readout_s=0.0,
    )
    query_log, update_log = [], []
    by_snapshot = {}
    for request in trace["requests"]:
        by_snapshot.setdefault((request["batch_id"], request["side"]), []).append(
            request
        )
    origin, h, g, n = (
        trace["native_timestamp_origin"],
        trace["initial_stop"],
        0,
        trace["n_nodes"],
    )

    def timed(name, callback):
        started = time.perf_counter()
        value = callback()
        duration = time.perf_counter() - started
        stages[name] += duration
        return value, duration

    with torch.no_grad(), threadpool_limits(limits=args.threads):

        def initialize_topology():
            edges = canonical_edges(torch.from_numpy(events[:h, :2].T.copy()), n)
            return edges, (edges[0] * n + edges[1]).numpy()

        (visible, known), _ = timed("initial_topology_s", initialize_topology)
        raw, _ = timed(
            "raw_inputs_s", lambda: frozen_structural_features(visible, n, 100)
        )
        encoded, _ = timed("encode_s", lambda: model.encode_features(raw))
        scorer, _ = timed(
            "scorer_init_s",
            lambda: scorer_for(
                args.method,
                implementation,
                model,
                head,
                raw,
                visible,
                encoded,
                args.materialize_fraction,
            ),
        )

        def initialize_history():
            value = replay_temporal_history(n)
            ingest_history(value, events[:h], origin)
            return value

        history, _ = timed("history_init_s", initialize_history)
        init_peak = peak_rss_bytes()

        def query(request):
            if (g, h) != (request["g"], request["h"]):
                raise AssertionError(
                    "Replay reached an inconsistent graph/history snapshot"
                )
            pairs = pair_arrays[request["array_key"]]
            before = counters(scorer)
            feature, graph_s = timed("graph_features_s", lambda: scorer.features(pairs))
            statistics, history_s = timed(
                "history_features_s",
                lambda: history.statistics(
                    pairs.numpy(), request["history_time_weeks"]
                ),
            )
            released, head_s = timed("released_head_s", lambda: head(feature).flatten())
            probability, readout_s = timed(
                "readout_s",
                lambda: predict_readout(
                    readout, feature.numpy(), statistics.numpy(), protocol="undergs"
                ),
            )
            # Copies, serialization and telemetry inspection are outside service
            # timers identically for Full, reference and every measured method.
            row = dict(
                request,
                graph_features_s=graph_s,
                history_features_s=history_s,
                released_head_s=head_s,
                readout_s=readout_s,
                service_s=graph_s + history_s + head_s + readout_s,
                work=difference(counters(scorer), before),
                prediction_file=f"q{request['request_id']:05d}.npz",
            )
            np.savez(
                output / row["prediction_file"],
                base17=feature.numpy(),
                statistics7=statistics.numpy(),
                released_head=released.numpy(),
                probability=probability,
            )
            query_log.append(row)

        for batch in trace["batches"]:
            for request in by_snapshot.get((batch["batch_id"], "before"), []):
                query(request)
            if h != batch["start"]:
                raise AssertionError("Noncontiguous event replay")
            ingested, ingest_s = timed(
                "topology_ingest_s",
                lambda: ingest_edges(
                    visible,
                    torch.from_numpy(
                        events[batch["start"] : batch["stop"], :2].T.copy()
                    ),
                    n,
                    known,
                ),
            )
            visible, added, known = ingested
            _, history_s = timed(
                "history_ingest_s",
                lambda: ingest_history(
                    history, events[batch["start"] : batch["stop"]], origin
                ),
            )
            if added.shape[1] != batch["new_simple_edges"]:
                raise AssertionError("Prepared trace and native topology disagree")
            metadata, update_s = timed(
                "update_s", lambda: scorer.update(visible, added)
            )
            h = batch["stop"]
            g += int(added.shape[1] > 0)
            if g != batch["after_g"]:
                raise AssertionError("Incorrect topology version")
            update_log.append(
                dict(
                    batch,
                    topology_ingest_s=ingest_s,
                    history_ingest_s=history_s,
                    update_s=update_s,
                    method_timings=metadata,
                )
            )
            for request in by_snapshot.get((batch["batch_id"], "after"), []):
                query(request)
        measured_peak = peak_rss_bytes()
        # Numeric payload accounting and full history object traversal are
        # excluded from both the service clock and the reported service peak.
        numeric_state = scorer.cache_bytes()
        history_bytes = history.state_bytes()
        readout_bytes = numeric_payload_bytes(readout_arrays(readout))
        pools = threadpool_info()
    latencies = [row["service_s"] for row in query_log]
    result = dict(
        schema_version=1,
        dataset=trace["dataset"],
        seed=42,
        method=args.method,
        role=args.role,
        repeat=args.repeat,
        pid=os.getpid(),
        threads=args.threads,
        n_nodes=n,
        initial_edges=trace["initial_edges"],
        final_edges=visible.shape[1],
        universe_events=len(events),
        processed_events=h,
        topology_versions=g,
        excluded_import_s=excluded_import_s,
        excluded_import_scope="Worker numerical/experiment modules and selected method implementation, including strict author planner/DGL imports, are resolved before service stages. Process startup and runner top-level imports precede this diagnostic clock. Import memory remains in the process lifetime peak.",
        history_only_batches=trace["history_only_batches"],
        requests=len(query_log),
        scored_pairs=sum(row["pairs"] for row in query_log),
        stages=stages,
        total_s=sum(stages.values()),
        query_p50_ms=float(np.quantile(latencies, 0.5)) * 1000,
        query_p95_ms=float(np.quantile(latencies, 0.95)) * 1000,
        query_p99_ms=float(np.quantile(latencies, 0.99)) * 1000,
        initial_peak_working_set_bytes=init_peak,
        peak_working_set_bytes=measured_peak,
        peak_scope="Isolated worker OS lifetime peak up through service replay; includes imports, loaded inputs/artifacts and result output buffers. No reference arrays are loaded.",
        numeric_state_bytes=numeric_state,
        temporal_history_bytes=history_bytes,
        readout_numeric_bytes=readout_bytes,
        counters=counters(scorer),
        model_configuration=model.config,
        model_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint=trace["checkpoint"],
        readout=trace["readout"],
        adapter=(
            dict(
                label="Source-derived strict dependency adaptation on the common predictor/backend; not official native NeutronRT timing",
                provenance=getattr(scorer, "provenance", None),
            )
            if args.method == "strict_adapted"
            else None
        ),
        timing_scope="Sum of nonoverlapping service stages: initialization topology, prefix raw features, encoding, scorer and history initialization, every event ingestion/index/update, graph features/recovery/scheduler, history features, released head and locked HGBT readout. File IO, result serialization, payload accounting and validation excluded equally. Method substage timings/counters overlap their containing service stage.",
        readout_semantics=trace["readout_semantics"],
        software=dict(
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            torch=torch.__version__,
            pools=pools,
        ),
    )
    write_json(output / "queries.json", query_log)
    write_json(output / "updates.json", update_log)
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("method", "repeat", "total_s", "peak_working_set_bytes")
            }
        )
    )
    return result


def error_summary(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        return dict(
            allclose=False, max_abs=None, max_relative=None, shape_mismatch=True
        )
    same_infinity = np.isinf(actual) & (actual == expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    if not np.all(finite | same_infinity):
        return dict(
            allclose=False, max_abs=None, max_relative=None, invalid_nonfinite=True
        )
    delta = np.abs(
        actual[finite].astype(np.float64) - expected[finite].astype(np.float64)
    )
    denominator = np.maximum(np.abs(expected[finite]), max(atol, 1e-12))
    return dict(
        allclose=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        max_abs=float(delta.max(initial=0)),
        max_relative=float((delta / denominator).max(initial=0)),
    )


def verify(args):
    rows = json.loads((args.actual / "queries.json").read_text(encoding="utf8"))
    expected_rows = json.loads(
        (args.reference / "queries.json").read_text(encoding="utf8")
    )
    if len(rows) != len(expected_rows):
        raise AssertionError("Different request counts")
    reports = []
    for row, expected_row in zip(rows, expected_rows):
        for key in ("request_id", "g", "h", "t", "pairs", "array_key"):
            if row[key] != expected_row[key]:
                raise AssertionError(f"Request mismatch in {key}")
        with (
            np.load(args.actual / row["prediction_file"]) as actual,
            np.load(args.reference / expected_row["prediction_file"]) as expected,
        ):
            errors = {
                key: error_summary(actual[key], expected[key], *tolerance)
                for key, tolerance in dict(
                    base17=(2e-4, 2e-4),
                    statistics7=(0.0, 0.0),
                    released_head=(2e-4, 2e-4),
                    probability=(1e-6, 1e-6),
                ).items()
            }
        reports.append(
            dict(
                request_id=row["request_id"],
                g=row["g"],
                h=row["h"],
                t=row["t"],
                work=row["work"],
                errors=errors,
                valid=all(value["allclose"] for value in errors.values()),
            )
        )
    result = dict(
        valid=all(row["valid"] for row in reports),
        requests=len(reports),
        reference=str(args.reference.resolve()),
        actual=str(args.actual.resolve()),
        per_query=reports,
        max_abs={
            key: max((row["errors"][key]["max_abs"] or 0.0) for row in reports)
            for key in ("base17", "statistics7", "released_head", "probability")
        },
    )
    write_json(args.actual / "verification.json", result)
    if not result["valid"]:
        raise AssertionError(
            f"Numerical parity failed; inspect {args.actual / 'verification.json'}"
        )
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "schedule.json").exists():
        raise FileExistsError(
            "Use a new run output directory to preserve paired measurements"
        )
    rng = np.random.default_rng(42)
    schedule = [
        dict(repeat=repeat, methods=rng.permutation(args.methods).tolist())
        for repeat in range(args.repeats)
    ]
    write_json(
        output / "schedule.json",
        dict(
            seed=42,
            paired_repeats=args.repeats,
            methods=args.methods,
            schedule=schedule,
            trace=str(args.trace.resolve()),
            formal_first_batch=args.repeats == 5 and set(args.methods) == set(METHODS),
            strict_semantics="Source-derived strict dependency adaptation on the common predictor/backend; never labeled an official native NeutronRT run",
        ),
    )
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in THREAD_ENV})
    reference = output / "reference_full"

    def launch(method, destination, repeat, role):
        command = [
            sys.executable,
            "-m",
            "experiments.db_a_native",
            "worker",
            "--trace",
            str(args.trace.resolve()),
            "--output",
            str(destination),
            "--method",
            method,
            "--repeat",
            str(repeat),
            "--role",
            role,
            "--threads",
            str(args.threads),
            "--materialize-fraction",
            str(args.materialize_fraction),
        ]
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "command.json", command)
        with (destination / "worker.log").open("w", encoding="utf8") as log:
            subprocess.run(
                command,
                cwd=WORKSPACE / "EvoGFM",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    launch("full", reference, -1, "reference")
    results = []
    for block in schedule:
        for method in block["methods"]:
            destination = output / f"r{block['repeat']:02d}_{method}"
            launch(method, destination, block["repeat"], "timed")
            verification = verify(
                argparse.Namespace(actual=destination, reference=reference)
            )
            result = json.loads(
                (destination / "result.json").read_text(encoding="utf8")
            )
            result.update(valid=verification["valid"], max_abs=verification["max_abs"])
            results.append(result)
            write_json(
                output / "progress.json", dict(results=results, schedule=schedule)
            )
            print(
                json.dumps(
                    dict(
                        method=method,
                        repeat=block["repeat"],
                        total_s=result["total_s"],
                        peak_working_set_bytes=result["peak_working_set_bytes"],
                        valid=True,
                    )
                ),
                flush=True,
            )
    for result in results:
        full = next(
            (
                other
                for other in results
                if other["method"] == "full" and other["repeat"] == result["repeat"]
            ),
            None,
        )
        result["paired_speedup_vs_full"] = (
            full["total_s"] / result["total_s"] if full else None
        )
    write_json(output / "summary.json", dict(results=results, schedule=schedule))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Prepare native chronology and fixed workload only"
    )
    prepare.add_argument("--root", type=Path, default=WORKSPACE)
    prepare.add_argument(
        "--dataset", choices=("bitcoin-otc", "superuser"), required=True
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--raw-bitcoin", type=Path)
    prepare.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    prepare.add_argument("--readout", type=Path, default=DEFAULT_READOUT)
    prepare.add_argument("--seed", type=int, choices=(42,), default=42)
    prepare.add_argument("--prefix-fraction", type=float, default=1.0)
    prepare.add_argument("--initial-fraction", type=float, default=0.5)
    prepare.add_argument("--versions", type=int, default=20)
    prepare.add_argument("--queries-per-snapshot", type=int, default=3)
    prepare.add_argument("--query-pairs", type=int, default=32)
    prepare.add_argument("--dense-pairs", type=int, default=256)
    run_parser = sub.add_parser(
        "run",
        help="Reference plus independent paired workers; launches real computation",
    )
    worker_parser = sub.add_parser(
        "worker", help="One isolated method; never opens Full references"
    )
    for child in (run_parser, worker_parser):
        child.add_argument("--trace", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--threads", type=int, default=1)
        child.add_argument("--materialize-fraction", type=float, default=0.35)
    run_parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    run_parser.add_argument("--repeats", type=int, default=5)
    worker_parser.add_argument("--method", choices=METHODS, required=True)
    worker_parser.add_argument("--repeat", type=int, default=0)
    worker_parser.add_argument(
        "--role", choices=("reference", "timed", "pilot"), default="pilot"
    )
    verify_parser = sub.add_parser(
        "verify", help="Offline per-query parity, after worker exit"
    )
    verify_parser.add_argument("--reference", type=Path, required=True)
    verify_parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if not (0 < args.prefix_fraction <= 1 and 0 < args.initial_fraction < 1):
            parser.error("prefix-fraction must be in (0,1], initial-fraction in (0,1)")
        if (
            min(
                args.versions,
                args.queries_per_snapshot,
                args.query_pairs,
                args.dense_pairs,
            )
            < 1
        ):
            parser.error("Versions and query counts must be positive")
        trace = prepare_trace(args)
        print(
            json.dumps(
                {
                    key: trace[key]
                    for key in (
                        "dataset",
                        "n_nodes",
                        "universe_events",
                        "topology_versions",
                        "history_only_batches",
                    )
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "worker":
        worker(args)
    elif args.command == "run":
        if args.repeats < 1 or len(set(args.methods)) != len(args.methods):
            parser.error("Use positive repeats and unique method names")
        run(args)
    else:
        result = verify(args)
        print(
            json.dumps(
                dict(
                    valid=result["valid"],
                    requests=result["requests"],
                    max_abs=result["max_abs"],
                )
            )
        )


if __name__ == "__main__":
    main()
