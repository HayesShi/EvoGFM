"""Paired full-predictor replay: strong Head-fused baseline vs adaptive dense sharing.

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
    replay_temporal_history,
    check_identity,
    counters,
    difference,
    readout_arrays,
    error_summary,
)
from experiments.db_a_native_trace import identity, write_json

VERSION = "shared-attention-v4-dense"
METHODS = ("head_fused", "shared")


def source_record():
    return [
        identity(WORKSPACE / path)
        for path in (
            "evogfm/shared_attention.py",
            "evogfm/shared_attention_fast.py",
            "evogfm/demand_maintenance.py",
            "evogfm/model.py",
            "evogfm/temporal.py",
            "evogfm/compact_temporal.py",
            "experiments/db_a_native.py",
            "experiments/db_a_native_trace.py",
            "experiments/db_a_shared_fast.py",
        )
    ]


def preload_method(method):
    from evogfm.shared_attention_fast import SharedAttentionFastLinkScorer

    return SharedAttentionFastLinkScorer


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


def apply_edits(keys, added, removed, nodes):
    if len(removed):
        indices = np.searchsorted(keys, removed)
        if np.any(indices >= len(keys)) or not np.array_equal(keys[indices], removed):
            raise ValueError("Deletion requires an active edge")
        mask = np.ones(len(keys), dtype=bool)
        mask[indices] = False
        keys = keys[mask]
    if len(added):
        added = np.sort(added)
        indices = np.searchsorted(keys, added)
        inside = indices < len(keys)
        if np.any(keys[indices[inside]] == added[inside]):
            raise ValueError("Insertion requires an inactive edge")
        positions = indices + np.arange(len(added))
        mask = np.ones(len(keys) + len(added), dtype=bool)
        mask[positions] = False
        combined = np.empty(len(mask), dtype=np.int64)
        combined[positions], combined[mask] = added, keys
        keys = combined

    def decode(values):
        return torch.from_numpy(np.stack((values // nodes, values % nodes)))

    import torch

    return decode(keys), decode(added), decode(removed), keys


def ingest_sorted_native(previous, incoming, nodes, keys):
    """Insert only novel batch keys into the currently sorted simple topology."""
    from evogfm.data import canonical_edges

    candidates = canonical_edges(incoming, nodes)
    batch_keys = (candidates[0] * nodes + candidates[1]).numpy()
    positions = np.searchsorted(keys, batch_keys)
    inside = positions < len(keys)
    known = np.zeros(len(batch_keys), dtype=bool)
    known[inside] = keys[positions[inside]] == batch_keys[inside]
    added = batch_keys[~known]
    if not len(added):
        return previous, candidates[:, :0], keys
    edges, appended, _, updated = apply_edits(
        keys, added, np.empty(0, dtype=np.int64), nodes
    )
    return edges, appended, updated


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
    native_ingest = (
        ingest_sorted_native if getattr(args, "sorted_ingest", False) else ingest_edges
    )
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
    events = np.load(trace["events_path"], mmap_mode="r")
    compact = getattr(args, "compact", False)
    if compact:
        pair_arrays = np.load(trace["queries_path"])
        score_offsets = np.r_[0, np.cumsum([row["pairs"] for row in trace["requests"]])]
        scores = np.lib.format.open_memmap(
            output / "probabilities.npy",
            mode="w+",
            dtype=np.float32,
            shape=(int(score_offsets[-1]),),
        )
    else:
        with np.load(trace["queries_path"]) as pack:
            pair_arrays = {
                name: torch.from_numpy(pack[name].copy()) for name in pack.files
            }
    model, head, _ = load_checkpoint(trace["checkpoint"]["path"])
    readout = joblib.load(trace["readout"]["path"])
    expected = dict(input_dim=100, d=100, hidden=128, rank=16, heads=4)
    for key, value in expected.items():
        if model.config.get(key) != value:
            raise ValueError(
                f"Released full model requires {key}={value}: {model.config}"
            )
    if len(model.layers) != trace.get("model_depth", 2) or not model.config.get(
        "joint_task_backbone"
    ):
        raise ValueError(
            "The full frozen backbone must match the trace's declared depth"
        )
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
    changes = np.load(trace["changes_path"]) if "changes_path" in trace else None
    request_reuse = getattr(args, "request_reuse_counters", False)
    capture_requests = set(getattr(args, "capture_request", ()))
    arrival_reference = getattr(args, "arrival_reference", None)
    rate = None
    arrival_offset = 0.0
    if arrival_reference:
        calibration = json.loads(arrival_reference.read_text(encoding="utf-8"))
        if (
            calibration["dataset"] != trace["dataset"]
            or calibration["method"] != "neutronrt_adapt"
        ):
            raise ValueError(
                "Arrival calibration requires the matched NeutronRT validation replay"
            )
        service = calibration["total_s"] - sum(
            calibration["stages"][stage]
            for stage in (
                "initial_topology_s",
                "raw_inputs_s",
                "encode_s",
                "scorer_init_s",
                "history_init_s",
            )
        )
        rate = args.arrival_factor * calibration["scored_pairs"] / service

    def timed(name, callback):
        allocator = getattr(args, "allocator_profile", None)
        before = allocator.begin() if allocator else None
        started = time.perf_counter()
        value = callback()
        duration = time.perf_counter() - started
        if allocator:
            allocator.finish(name, before)
        stages[name] += duration
        return value, duration

    with torch.no_grad(), threadpool_limits(limits=args.threads):

        def initialize_history():
            value = replay_temporal_history(n)
            ingest_history(value, events[:h], origin)
            return value

        large_history = n > 750_000
        if large_history:
            history, _ = timed("history_init_s", initialize_history)

        def initialize_topology():
            edges = canonical_edges(
                torch.from_numpy(
                    events[trace.get("initial_start", 0) : h, :2].T.copy()
                ),
                n,
            )
            return edges, (edges[0] * n + edges[1]).numpy()

        (visible, known), _ = timed("initial_topology_s", initialize_topology)

        def initialize_raw():
            encoding_stop = trace.get("encoding_stop", h)
            encoding_edges = (
                visible
                if encoding_stop == h and not trace.get("initial_start")
                else canonical_edges(
                    torch.from_numpy(events[:encoding_stop, :2].T.copy()), n
                )
            )
            return frozen_structural_features(encoding_edges, n, 100)

        raw, _ = timed("raw_inputs_s", initialize_raw)
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
        if not large_history:
            history, _ = timed("history_init_s", initialize_history)
        init_peak = peak_rss_bytes()
        progress("serving", completed_requests=0, total_requests=len(trace["requests"]))
        replay_start = time.perf_counter()
        next_progress_s = 60.0

        def wait_until_release():
            if rate is None:
                return None
            remaining = replay_start + arrival_offset - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            return arrival_offset

        def query(request):
            nonlocal arrival_offset, next_progress_s
            if (g, h) != (request["g"], request["h"]):
                raise AssertionError(
                    "Replay reached an inconsistent graph/history snapshot"
                )
            release = wait_until_release()
            if rate is not None:
                arrival_offset += request["pairs"] / rate
            pairs = pair_arrays[request["array_key"]]
            if compact:
                pairs = torch.from_numpy(pairs)
            if request_reuse:
                endpoints = np.unique(pairs.numpy())
                hidden_hits = int(
                    np.count_nonzero(scorer.stamps["h"][endpoints] == scorer.version)
                )
            before = counters(scorer)
            feature, graph_s = timed("graph_features_s", lambda: scorer.features(pairs))
            statistics, history_s = timed(
                "history_features_s",
                lambda: history.statistics(
                    pairs.numpy(), request["history_time_weeks"]
                ),
            )
            released, head_s = timed("released_head_s", lambda: head(feature).flatten())

            def readout_query():
                ages = (
                    request["history_time_weeks"] - history.last_node[pairs.numpy()]
                    if "node_activity" in readout
                    else None
                )
                return predict_readout(
                    readout,
                    feature.numpy(),
                    statistics.numpy(),
                    protocol="undergs",
                    node_ages=ages,
                )

            probability, readout_s = timed("readout_s", readout_query)
            completed = time.perf_counter() - replay_start
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
            if compact:
                row["work"] = {
                    key: value for key, value in row["work"].items() if value
                }
            if request_reuse:
                row["work"].update(
                    hidden_requested_endpoints=len(endpoints),
                    hidden_hit_endpoints=hidden_hits,
                )
            if release is not None:
                row.update(
                    arrival_s=release,
                    completed_s=completed,
                    response_s=completed - release,
                )
            if compact:
                first, last = score_offsets[
                    request["request_id"] : request["request_id"] + 2
                ]
                scores[first:last] = np.asarray(probability).reshape(-1)
                row.pop("prediction_file")
                row["prediction_offset"] = int(first)
                if request["request_id"] in capture_requests:
                    np.savez(
                        output / f"capture_{request['request_id']:05d}.npz",
                        base17=feature.numpy(),
                        statistics7=statistics.numpy(),
                        released_head=released.numpy(),
                        probability=probability,
                    )
            else:
                np.savez(
                    output / row["prediction_file"],
                    base17=feature.numpy(),
                    statistics7=statistics.numpy(),
                    released_head=released.numpy(),
                    probability=probability,
                )
            query_log.append(row)
            if (
                len(query_log) % 16 == 0
                or len(query_log) == len(trace["requests"])
                or (rate is None and completed >= next_progress_s)
            ):
                next_progress_s = completed + 60.0
                progress(
                    "serving",
                    completed_requests=len(query_log),
                    total_requests=len(trace["requests"]),
                    service_s=sum(stages.values()),
                    graph_features_s=stages["graph_features_s"],
                    peak_gpu_allocated_bytes=(
                        args.allocator_profile.peak
                        if getattr(args, "allocator_profile", None)
                        else (
                            torch.cuda.max_memory_allocated()
                            if getattr(args, "defer_result", False)
                            else 0
                        )
                    ),
                    shared_cache=scorer.cache_stats(),
                    work=counters(scorer),
                )

        for batch in trace["batches"]:
            for request in by_snapshot.get((batch["batch_id"], "before"), []):
                query(request)
            if h != batch["start"]:
                raise AssertionError("Noncontiguous event replay")
            release = wait_until_release()
            if changes is None:
                ingested, ingest_s = timed(
                    "topology_ingest_s",
                    lambda: native_ingest(
                        visible,
                        torch.from_numpy(
                            events[batch["start"] : batch["stop"], :2].T.copy()
                        ),
                        n,
                        known,
                    ),
                )
                visible, added, known = ingested
                removed = None
            else:
                bid = batch["batch_id"]
                new, expired = changes[f"a{bid}"], changes[f"r{bid}"]
                ingested, ingest_s = timed(
                    "topology_ingest_s", lambda: apply_edits(known, new, expired, n)
                )
                visible, added, removed, known = ingested
            _, history_s = timed(
                "history_ingest_s",
                lambda: ingest_history(
                    history, events[batch["start"] : batch["stop"]], origin
                ),
            )
            if added.shape[1] != batch["new_simple_edges"]:
                raise AssertionError("Prepared trace and native topology disagree")
            metadata, update_s = timed(
                "update_s", lambda: scorer.update(visible, added, removed)
            )
            completed = time.perf_counter() - replay_start
            h = batch["stop"]
            g += int(
                added.shape[1] > 0 or (removed is not None and removed.shape[1] > 0)
            )
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
            if release is not None:
                update_log[-1].update(
                    arrival_s=release,
                    completed_s=completed,
                    response_s=completed - release,
                )
            for request in by_snapshot.get((batch["batch_id"], "after"), []):
                query(request)
        measured_peak = peak_rss_bytes()
        if compact:
            scores.flush()
            pair_arrays.close()
        if changes is not None:
            changes.close()
        # Numeric payload accounting and full history object traversal are
        # excluded from both the service clock and the reported service peak.
        numeric_state = scorer.cache_bytes()
        shared_cache = scorer.cache_stats()
        history_bytes = history.state_bytes()
        readout_bytes = numeric_payload_bytes(readout_arrays(readout))
        pools = threadpool_info()
    latencies = [row.get("response_s", row["service_s"]) for row in query_log]
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
        temporal_history_backend=type(history).__name__,
        readout_numeric_bytes=readout_bytes,
        counters=counters(scorer),
        implementation_version=VERSION,
        source_record=source_record(),
        shared_cache=shared_cache,
        tile_edges=args.tile_edges,
        budget_mib=args.budget_mib,
        protocol_scope=trace.get(
            "protocol_scope",
            "Complete fixed E0 native replay, 50% initialization prefix; not the new 70/10/20 quality protocol",
        ),
        backend="CPU FP32 torch/scipy full-neighborhood",
        peak_gpu_allocated_bytes=0,
        model_configuration=model.config,
        model_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint=trace["checkpoint"],
        readout=trace["readout"],
        score_storage="probabilities.npy" if compact else "per-request arrays",
        probability_dtype="float32",
        native_topology_ingestion=(
            "sorted simple-edge insertion"
            if getattr(args, "sorted_ingest", False)
            else "canonical sort and deduplication"
        ),
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
    if rate is not None:
        arrival = np.array([row["arrival_s"] for row in query_log])
        finished = np.array([row["completed_s"] for row in query_log])
        elapsed = max(finished[-1], update_log[-1]["completed_s"])
        queued = np.searchsorted(arrival, finished, side="right") - np.arange(
            1, len(finished) + 1
        )
        result["arrival"] = dict(
            candidate_pairs_per_second=rate,
            factor=args.arrival_factor,
            elapsed_s=float(elapsed),
            candidate_throughput=result["scored_pairs"] / elapsed,
            update_p99_ms=float(
                np.quantile([row["response_s"] for row in update_log], 0.99)
            )
            * 1000,
            max_waiting_requests=int(queued.max(initial=0)),
            waiting_requests_at_last_arrival=int(
                len(arrival) - np.searchsorted(finished, arrival[-1], side="right")
            ),
            update_release="Eligible at the next version's first scheduled query; FIFO service processes update first",
        )
    if request_reuse:
        result["request_reuse"] = {
            key: sum(row["work"].get(key, 0) for row in query_log)
            for key in (
                "hidden_requested_endpoints",
                "hidden_hit_endpoints",
                "shared_score_hits",
                "shared_score_evaluations",
            )
        }
    write_json(output / "queries.json", query_log)
    write_json(output / "updates.json", update_log)
    if not getattr(args, "defer_result", False):
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
                    "experiments.db_a_shared_fast",
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
