"""Paired full-predictor replay: strong Head-fused baseline vs shared execution.

All service stages and the frozen readout follow db_a_native. Workers never
load the reference. Validation is offline and uses the manuscript tolerance.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from experiments.db_a_native import (
    WORKSPACE,
    THREAD_ENV,
    peak_rss_bytes,
    ingest_history,
    check_identity,
    counters,
    difference,
    readout_arrays,
    error_summary,
)
from experiments.db_a_native_trace import identity, write_json

VERSION = "shared-attention-v3-indexed"
METHODS = ("head_fused", "shared")


def source_record():
    return [
        identity(WORKSPACE / path)
        for path in (
            "evogfm/shared_attention.py",
            "evogfm/demand_maintenance.py",
            "evogfm/model.py",
            "experiments/db_a_native_trace.py",
            "experiments/db_a_shared.py",
        )
    ]


def preload_method(method):
    from evogfm.shared_attention import SharedAttentionLinkScorer

    return SharedAttentionLinkScorer


def scorer_for(
    method, implementation, model, head, raw, edges, encoded, budget_mib, tile_edges
):
    return implementation(
        model,
        head,
        raw,
        edges,
        encoded=encoded,
        policy="lazy",
        shared_budget_bytes=0 if method == "head_fused" else int(budget_mib * 1024**2),
        tile_edges=tile_edges,
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
    from evogfm.temporal import TemporalHistory
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
    expected = dict(input_dim=100, d=100, hidden=128, rank=16, heads=4)
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

    def progress(phase, **extra):
        write_json(
            output / "status.json",
            dict(
                pid=os.getpid(),
                method=args.method,
                phase=phase,
                updated_unix=time.time(),
                **extra,
            ),
        )

    progress("initializing")
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
                args.budget_mib,
                args.tile_edges,
            ),
        )

        def initialize_history():
            value = TemporalHistory(n)
            ingest_history(value, events[:h], origin)
            return value

        history, _ = timed("history_init_s", initialize_history)
        init_peak = peak_rss_bytes()
        progress("serving", completed_requests=0, total_requests=len(trace["requests"]))

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
            if len(query_log) % 16 == 0 or len(query_log) == len(trace["requests"]):
                progress(
                    "serving",
                    completed_requests=len(query_log),
                    total_requests=len(trace["requests"]),
                    service_s=sum(stages.values()),
                    graph_features_s=stages["graph_features_s"],
                    shared_cache=scorer.cache_stats(),
                    work=counters(scorer),
                )

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
        shared_cache = scorer.cache_stats()
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
        implementation_version=VERSION,
        source_record=source_record(),
        shared_cache=shared_cache,
        tile_edges=args.tile_edges,
        budget_mib=args.budget_mib,
        protocol_scope="Complete fixed E0 native replay, 50% initialization prefix; not the new 70/10/20 quality protocol",
        backend="CPU FP32 torch/scipy full-neighborhood",
        peak_gpu_allocated_bytes=0,
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
    progress("complete", completed_requests=len(query_log), total_s=result["total_s"])
    return result


def verify(actual, reference):
    rows = json.loads((actual / "queries.json").read_text(encoding="utf8"))
    refs = json.loads((reference / "queries.json").read_text(encoding="utf8"))
    if len(rows) != len(refs):
        raise AssertionError("Different request counts")
    reports = []
    tolerances = dict(
        base17=(1e-4, 1e-5),
        statistics7=(0.0, 0.0),
        released_head=(1e-4, 1e-5),
        probability=(1e-6, 1e-6),
    )
    for row, ref in zip(rows, refs):
        for key in ("request_id", "g", "h", "t", "pairs", "array_key"):
            if row[key] != ref[key]:
                raise AssertionError(f"Different request {key}")
        with (
            np.load(actual / row["prediction_file"]) as a,
            np.load(reference / ref["prediction_file"]) as b,
        ):
            errors = {
                key: error_summary(a[key], b[key], *tol)
                for key, tol in tolerances.items()
            }
        reports.append(
            dict(
                request_id=row["request_id"],
                errors=errors,
                valid=all(e["allclose"] for e in errors.values()),
            )
        )
    result = dict(
        valid=all(row["valid"] for row in reports),
        requests=len(reports),
        tolerances=tolerances,
        reference=str(reference),
        actual=str(actual),
        per_query=reports,
        max_abs={
            key: max(row["errors"][key]["max_abs"] or 0.0 for row in reports)
            for key in tolerances
        },
    )
    write_json(actual / "verification.json", result)
    if not result["valid"]:
        raise AssertionError(f"Numerical mismatch: {actual / 'verification.json'}")
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    schedule = [
        dict(repeat=r, methods=rng.permutation(METHODS).tolist())
        for r in range(args.repeats)
    ]
    settings = dict(
        seed=42,
        schedule=schedule,
        trace=identity(args.trace),
        reference=str(args.reference.resolve()),
        threads=args.threads,
        budget_mib=args.budget_mib,
        tile_edges=args.tile_edges,
        source_record=source_record(),
        scope="Full E0 native-event systems replay; identical weights, encodings, requests and readout",
    )
    schedule_path = output / "schedule.json"
    if schedule_path.exists():
        if json.loads(schedule_path.read_text(encoding="utf8")) != settings:
            raise ValueError(
                "Run configuration/source changed: use a new output directory"
            )
    else:
        write_json(schedule_path, settings)
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in THREAD_ENV})
    results = []
    for block in schedule:
        for method in block["methods"]:
            dest = output / f"r{block['repeat']:02d}_{method}"
            dest.mkdir(exist_ok=True)
            if not (dest / "result.json").exists():
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.db_a_shared",
                    "worker",
                    "--trace",
                    str(args.trace.resolve()),
                    "--output",
                    str(dest),
                    "--method",
                    method,
                    "--repeat",
                    str(block["repeat"]),
                    "--role",
                    "timed",
                    "--threads",
                    str(args.threads),
                    "--budget-mib",
                    str(args.budget_mib),
                    "--tile-edges",
                    str(args.tile_edges),
                ]
                write_json(dest / "command.json", command)
                with (dest / "worker.log").open("a", encoding="utf8") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=WORKSPACE / "EvoGFM",
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    write_json(
                        output / "active.json",
                        dict(
                            pid=os.getpid(),
                            worker_pid=child.pid,
                            phase="running",
                            method=method,
                            repeat=block["repeat"],
                            directory=str(dest),
                            updated_unix=time.time(),
                        ),
                    )
                    code = child.wait()
                if code:
                    write_json(
                        output / "active.json",
                        dict(
                            pid=os.getpid(),
                            phase="failed",
                            method=method,
                            repeat=block["repeat"],
                            directory=str(dest),
                            exit_code=code,
                        ),
                    )
                    raise RuntimeError(f"Worker failed ({code}): {dest / 'worker.log'}")
            try:
                comparison = verify(dest, args.reference.resolve())
            except Exception as error:
                write_json(
                    output / "active.json",
                    dict(
                        pid=os.getpid(),
                        phase="validation_failed",
                        method=method,
                        repeat=block["repeat"],
                        directory=str(dest),
                        error=str(error),
                        updated_unix=time.time(),
                    ),
                )
                raise
            result = json.loads((dest / "result.json").read_text(encoding="utf8"))
            if result["source_record"] != settings["source_record"]:
                raise ValueError("Completed worker source mismatch")
            result.update(valid=comparison["valid"], max_abs=comparison["max_abs"])
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
                        valid=True,
                        max_abs=comparison["max_abs"],
                        shared_cache=result["shared_cache"],
                    )
                ),
                flush=True,
            )
    for result in results:
        base = next(
            r
            for r in results
            if r["repeat"] == result["repeat"] and r["method"] == "head_fused"
        )
        result["paired_speedup_vs_head_fused"] = base["total_s"] / result["total_s"]
    write_json(output / "summary.json", dict(results=results, schedule=schedule))
    write_json(
        output / "active.json",
        dict(pid=os.getpid(), phase="complete", updated_unix=time.time()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        child = sub.add_parser(name)
        child.add_argument("--trace", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--threads", type=int, default=1)
        child.add_argument("--budget-mib", type=float, default=64.0)
        child.add_argument("--tile-edges", type=int, default=1024)
        if name == "run":
            child.add_argument("--reference", type=Path, required=True)
            child.add_argument("--repeats", type=int, default=5)
        else:
            child.add_argument("--method", choices=METHODS, required=True)
            child.add_argument("--repeat", type=int, default=0)
            child.add_argument("--role", choices=("timed", "pilot"), default="pilot")
    args = parser.parse_args()
    if args.threads < 1 or args.tile_edges < 1 or args.budget_mib < 0:
        parser.error("threads/tile must be positive; budget nonnegative")
    if args.command == "worker":
        worker(args)
    else:
        if args.repeats < 1:
            parser.error("positive repeats required")
        run(args)


if __name__ == "__main__":
    main()
