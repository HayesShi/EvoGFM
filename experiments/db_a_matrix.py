"""Prepare the controlled A workload traces and verify matched probability vectors."""

import argparse
import json
from pathlib import Path
import random

import numpy as np

from experiments.db_a_execution_suite import write_json
from experiments.db_a_native import DEFAULT_CHECKPOINT, DEFAULT_READOUT, WORKSPACE
from experiments.db_a_native_trace import (
    array_path,
    first_occurrences,
    identity,
    topology_novelty,
)


SYSTEM_METHODS = (
    "full_refresh",
    "head_fused",
    "neutronrt_adapt",
    "ours",
    "demand_only",
    "shared_lru",
)
CONTROL_METHODS = ("neutronrt_adapt", "head_fused", "shared_lru", "ours")
CUDA_FLAGS = [
    "--resident-scores",
    "--direct-hops",
    "--async-indices",
    "--gpu-propagation",
    "--compact",
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scores(trace_path, actual, reference):
    trace = load(trace_path)

    def scores(directory):
        packed = directory / "probabilities.npy"
        if packed.is_file():
            return np.load(packed, mmap_mode="r")
        values = []
        for request in trace["requests"]:
            filename = f"q{request['request_id']:05d}.npz"
            with np.load(directory / filename) as pack:
                values.append(pack["probability"].reshape(-1))
        return np.concatenate(values)

    a, b = scores(actual), scores(reference)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Incompatible probability files")
    orders, mismatch, violations, max_abs, max_rel = 0, 0, 0, 0.0, 0.0
    first = 0
    with np.load(trace["queries_path"]) as pairs:
        for request in trace["requests"]:
            last = first + request["pairs"]
            av, bv = a[first:last], b[first:last]
            endpoints = pairs[request["array_key"]]
            error = np.abs(av - bv)
            max_abs = max(max_abs, float(error.max(initial=0.0)))
            max_rel = max(
                max_rel, float((error / np.maximum(np.abs(bv), 1e-12)).max(initial=0.0))
            )
            violations += int((~np.isclose(av, bv, rtol=1e-6, atol=1e-6)).sum())
            # Duplicate pairs retain their trace order after both endpoint IDs.
            order_a = np.lexsort((endpoints[0], endpoints[1], -av))
            order_b = np.lexsort((endpoints[0], endpoints[1], -bv))
            mismatch += int((order_a != order_b).sum())
            orders += len(av)
            first = last
    if first != len(a):
        raise ValueError("Probability vectors do not cover the trace")
    result = dict(
        valid=violations == 0,
        reference=str(reference),
        maximum_probability_abs_error=max_abs,
        maximum_probability_relative_error=max_rel,
        probability_violation_count=violations,
        request_rank_position_disagreement=mismatch / max(1, orders),
        pairs=orders,
        tolerance={"atol": 1e-6, "rtol": 1e-6},
    )
    if trace.get("groups_per_update"):
        from sklearn.metrics import roc_auc_score

        targets = np.zeros(len(a), dtype=bool)
        targets[::101] = True
        with np.load(trace["queries_path"]) as pairs:
            destination = np.concatenate(
                [pairs[q["array_key"]][1] for q in trace["requests"]]
            ).reshape(-1, 101)

        def metrics(values):
            records = values.reshape(-1, 101)
            pos, neg = records[:, :1], records[:, 1:]
            ranks = 1 + (
                (neg > pos) | ((neg == pos) & (destination[:, 1:] < destination[:, :1]))
            ).sum(1)
            return {
                "auc": float(roc_auc_score(targets, values)),
                "mrr": float(np.mean(1 / ranks)),
                "positives": len(records),
            }

        ar = np.lexsort((destination, -a.reshape(-1, 101)), axis=1)
        br = np.lexsort((destination, -b.reshape(-1, 101)), axis=1)
        result.update(
            actual_quality=metrics(a),
            reference_quality=metrics(b),
            rank_position_disagreement=float((ar != br).mean()),
        )
    write_json(actual / "verification.json", result)
    return result


def matrix_job(trace, output, method, group, repeat, budget=64, extra=()):
    output = output.resolve()
    args = [
        "-m",
        "experiments.db_a_cuda",
        "worker",
        "--trace",
        str(trace.resolve()),
        "--output",
        str(output),
        "--method",
        method,
        "--budget-mib",
        str(budget),
        "--repeat",
        str(repeat),
        *CUDA_FLAGS,
        *extra,
    ]
    details = load(trace)
    dataset = details["dataset"]
    if details["n_nodes"] > 750_000:
        args.append("--host-buffers")
        if details.get("groups_per_update"):
            args.append("--sorted-ingest")
    return dict(
        id=f"{dataset}/{output.parent.parent.name}/{output.parent.name}/{output.name}",
        group=group,
        dataset=dataset,
        method=method,
        output=str(output),
        arguments=args,
        attempts=2,
    )


def replay_jobs(trace, output, methods, group, repeats=5, budget=64, extra=()):
    rng = random.Random(42)
    return [
        matrix_job(
            trace,
            output / f"r{repeat:02d}" / method,
            method,
            group,
            repeat,
            budget,
            extra,
        )
        for repeat in range(repeats)
        for method in rng.sample(methods, len(methods))
    ]


def source(dataset, root):
    path = WORKSPACE / "data/a_temporal_scale" / dataset / "manifest.json"
    if not path.is_file():
        path = root / "data" / dataset / "manifest.json"
    meta = load(path)
    events_path = array_path(meta, "events", path).resolve()
    return (
        path,
        events_path,
        np.load(events_path, mmap_mode="r"),
        meta["statistics"]["vertices"],
    )


def cutoff(events, fraction):
    location = min(len(events) - 1, round(fraction * len(events)))
    t = events[location, 2]
    start, stop = np.searchsorted(events[:, 2], t, side="left"), np.searchsorted(
        events[:, 2], t, side="right"
    )
    if round(fraction * len(events)) - start <= stop - round(fraction * len(events)):
        return int(start)
    return int(stop)


def pair_grid(pool, rng, count=256):
    left = rng.integers(len(pool), size=count)
    right = (left + rng.integers(1, len(pool), size=count)) % len(pool)
    return np.stack((pool[left], pool[right]))


def coverage_pairs(pool, rng, fraction):
    nodes = rng.choice(pool, size=max(2, round(fraction * len(pool))), replace=False)
    if len(nodes) % 2:
        nodes = np.r_[nodes, nodes[0]]
    return nodes.reshape(-1, 2).T.copy()


def edge_keys(events, n):
    u, v = events[:, 0], events[:, 1]
    return np.minimum(u, v) * n + np.maximum(u, v)


def mutation_batches(events, n, initial, batch_size, n_versions, sliding, novelty):
    edges = []
    batches = []
    keys = edge_keys(events, n) if sliding else None
    window = initial - initial // 2 if sliding else None
    initial_start = initial - window if sliding else 0
    if sliding:
        ids, counts = np.unique(keys[initial_start:initial], return_counts=True)
        multiplicity = dict(zip(ids.tolist(), counts.tolist()))
    start = initial
    for offset in range(initial, len(events)):
        if sliding:
            add, remove = int(keys[offset]), int(keys[offset - window])
            if add != remove:
                for key, change in ((remove, -1), (add, 1)):
                    old = multiplicity.get(key, 0)
                    new = old + change
                    if new:
                        multiplicity[key] = new
                    else:
                        del multiplicity[key]
                    if (old == 0 or new == 0) and key // n != key % n:
                        edges.append((key, change))
        elif novelty[offset]:
            edges.append((int(edge_keys(events[offset : offset + 1], n)[0]), 1))
        while len(edges) >= batch_size:
            batch, edges = edges[:batch_size], edges[batch_size:]
            net = {}
            for key, change in batch:
                net[key] = net.get(key, 0) + change
            up = np.array(
                sorted(key for key, balance in net.items() if balance > 0),
                dtype=np.int64,
            )
            down = np.array(
                sorted(key for key, balance in net.items() if balance < 0),
                dtype=np.int64,
            )
            if len(up) + len(down) != batch_size:
                # Take later, real events until a commit contains exactly the
                # target number of net effective edits.
                edges = [
                    (key, balance) for key, balance in net.items() if balance
                ] + edges
                break
            stop = offset + 1
            batches.append((start, stop, up, down))
            start = stop
            if len(batches) == n_versions:
                return batches, initial_start
    raise ValueError(
        "The release does not contain enough effective edge edits for this workload"
    )


def prepare_control(
    dataset,
    root,
    initial,
    encoding_stop,
    batches,
    initial_start,
    label,
    output,
    first,
    novelty,
    events_path,
    source_path,
    n,
    events,
    coverage=None,
    pattern="uniform",
    request_count=1,
    checkpoint=DEFAULT_CHECKPOINT,
    readout=DEFAULT_READOUT,
    depth=2,
    *,
    requests_by_version=None,
    candidates_per_request=256,
):
    output.mkdir(parents=True, exist_ok=True)
    changelog, queries, requests, batch_rows = {}, {}, [], []
    rng = np.random.default_rng(42)
    initial_active = np.flatnonzero(first < initial)
    fixed_hotspot = rng.choice(
        initial_active, min(512, len(initial_active)), replace=False
    )
    origin = int(events[0, 2])
    for version, (start, stop, added, removed) in enumerate(batches):
        active = np.flatnonzero(first < stop)
        pool = active
        if pattern == "fixed-hotspot":
            pool = fixed_hotspot
        elif pattern == "rotating-hotspot":
            pool = rng.choice(active, min(512, len(active)), replace=False)
        repeated = (
            pair_grid(pool, rng, candidates_per_request)
            if pattern == "identical"
            else None
        )
        count = (
            request_count
            if requests_by_version is None
            else requests_by_version[version]
        )
        for _ in range(count):
            if coverage is not None:
                query = coverage_pairs(pool, rng, coverage)
            elif repeated is not None:
                query = repeated
            else:
                query = pair_grid(pool, rng, candidates_per_request)
            index = len(requests)
            key = f"q{index:05d}"
            queries[key] = query
            timestamp = int(events[max(0, stop - 1), 2])
            requests.append(
                dict(
                    request_id=index,
                    array_key=key,
                    batch_id=version,
                    side="after",
                    g=version + 1,
                    h=stop,
                    t=timestamp,
                    history_time_weeks=(timestamp - origin) / 604800.0,
                    visible_nodes=len(active),
                    pairs=query.shape[1],
                    distinct_endpoints=int(len(np.unique(query))),
                    workload_phase=label,
                )
            )
        changelog[f"a{version}"] = added
        changelog[f"r{version}"] = removed
        batch_rows.append(
            dict(
                batch_id=version,
                start=start,
                stop=stop,
                raw_events=stop - start,
                new_simple_edges=len(added),
                removed_simple_edges=len(removed),
                topology_changed=True,
                before_g=version,
                after_g=version + 1,
                effective_updates=len(added) + len(removed),
            )
        )
    np.savez(output / "queries.npz", **queries)
    np.savez(output / "changes.npz", **changelog)
    visible = edge_keys(events[initial_start:initial], n)
    initial_edges = int(len(np.unique(visible[visible // n != visible % n])))
    trace = dict(
        schema_version=2,
        dataset=dataset,
        seed=42,
        n_nodes=n,
        initial_stop=initial,
        initial_start=initial_start,
        encoding_stop=encoding_stop,
        universe_events=len(events),
        events_path=str(events_path),
        event_source_manifest=str(source_path),
        queries_path=str(output / "queries.npz"),
        changes_path=str(output / "changes.npz"),
        initial_edges=initial_edges,
        topology_versions=len(batches),
        history_only_batches=0,
        native_timestamp_origin=origin,
        checkpoint=identity(checkpoint),
        readout=identity(readout),
        model_depth=depth,
        readout_semantics="Identical frozen linked checkpoint and union24 readout for all methods",
        protocol_scope=label,
        requests=requests,
        batches=batch_rows,
    )
    write_json(output / "trace.json", trace)
    return output / "trace.json"


def controlled(dataset, root, panels):
    source_path, events_path, events, n = source(dataset, root)
    first = first_occurrences(events, n)
    novelty = topology_novelty(events, n)
    train, test = cutoff(events, 0.7), cutoff(events, 0.8)
    outputs = root / "traces" / dataset
    jobs = []

    def prep(tag, prefix, encoded, changes, base=0, **kwargs):
        return prepare_control(
            dataset,
            root,
            prefix,
            encoded,
            changes,
            base,
            tag,
            outputs / tag,
            first,
            novelty,
            events_path,
            source_path,
            n,
            events,
            **kwargs,
        )

    def add(tag, trace, methods, group, repeats=5, budget=64, extra=()):
        jobs.extend(
            replay_jobs(
                trace, root / dataset / tag, methods, group, repeats, budget, extra
            )
        )

    for group in panels:
        if group in (4, 5, 7, 8):
            base, _ = mutation_batches(events, n, test, 256, 16, False, novelty)
        if group == 4:
            for fraction in (0.001, 0.01, 0.05, 0.2, 1.0):
                tag = f"e4_coverage_{fraction:g}"
                add(
                    tag,
                    prep(tag, test, train, base, coverage=fraction),
                    CONTROL_METHODS,
                    "4 coverage",
                    repeats=1,
                )
            for count in (1, 4, 16, 64):
                for pattern in (
                    "uniform",
                    "fixed-hotspot",
                    "rotating-hotspot",
                    "identical",
                ):
                    tag = f"e4_{pattern}_{count}"
                    add(
                        tag,
                        prep(
                            tag, test, train, base, pattern=pattern, request_count=count
                        ),
                        CONTROL_METHODS,
                        "4 overlap",
                        repeats=1,
                        extra=("--request-reuse-counters",),
                    )
        elif group == 5:
            valid, _ = mutation_batches(
                events[:test], n, train, 256, 16, False, novelty[:test]
            )
            tag = "e5_calibration"
            valid_trace = prep(
                tag, train, train, valid, request_count=4, pattern="fixed-hotspot"
            )
            add(
                tag,
                valid_trace,
                ("neutronrt_adapt",),
                "5 validation arrival calibration",
                repeats=1,
            )
            validation = (
                root / dataset / tag / "r00" / "neutronrt_adapt" / "result.json"
            )
            tag = "e5_test"
            trace = prep(
                tag, test, train, base, request_count=4, pattern="fixed-hotspot"
            )
            for factor in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
                add(
                    f"{tag}_{factor:g}",
                    trace,
                    CONTROL_METHODS,
                    "5 paced tail latency",
                    repeats=1,
                    extra=(
                        "--arrival-reference",
                        str(validation),
                        "--arrival-factor",
                        str(factor),
                    ),
                )
        elif group == 6:
            for sliding in (False, True):
                for count in (1, 16, 256, 4096):
                    native, start = mutation_batches(
                        events, n, test, count, 8, sliding, novelty
                    )
                    tag = f"e6_{'window' if sliding else 'insert'}_{count}"
                    trace = prep(tag, test, train, native, start)
                    add(
                        tag,
                        trace,
                        ("ours", "ours_rebuild_moments"),
                        "6 effective update batch sizes",
                        repeats=1,
                    )
        elif group == 7:
            tag = "e7_work"
            trace = prep(
                tag, test, train, base, pattern="fixed-hotspot", request_count=16
            )
            add(
                tag,
                trace,
                (
                    "demand_only",
                    "head_fused",
                    "shared_lru",
                    "ours",
                    "ours_no_scores",
                    "ours_no_projections",
                    "ours_rebuild_moments",
                    "ours_feature_layer_one",
                ),
                "7 exact work and ablation",
                repeats=1,
                extra=("--work-profile",),
            )
            for scale in (0.10, 0.25, 0.50, 1.0):
                prefix = len(events) if scale == 1.0 else cutoff(events, scale)
                keys = np.unique(edge_keys(events[:prefix], n))
                keys = keys[keys // n != keys % n]
                selected = np.random.default_rng(42).choice(
                    keys, size=2048, replace=False
                )
                empty = np.empty(0, dtype=np.int64)
                mixed = [
                    (
                        prefix,
                        prefix,
                        empty if i % 2 == 0 else block,
                        block if i % 2 == 0 else empty,
                    )
                    for part in selected.reshape(8, 256)
                    for i, block in enumerate((part, part))
                ]
                tag = f"e7_scale_{scale:g}"
                add(
                    tag,
                    prep(tag, prefix, min(train, prefix), mixed, request_count=4),
                    ("neutronrt_adapt", "head_fused", "ours"),
                    "7 scale and stage profile",
                    repeats=1,
                    extra=("--stage-profile",),
                )
        elif group == 8:
            tag = "e8_budget"
            trace = prep(
                tag, test, train, base, pattern="rotating-hotspot", request_count=16
            )
            add(
                f"{tag}_head_fused",
                trace,
                ("head_fused",),
                "8 memory and budget",
                repeats=1,
                extra=("--resource-profile",),
            )
            for budget in (0, 4, 16, 64, 256):
                add(
                    f"{tag}_{budget}",
                    trace,
                    ("shared_lru", "ours"),
                    "8 memory and budget",
                    repeats=1,
                    budget=budget,
                    extra=("--work-profile", "--resource-profile"),
                )
    write_json(
        root / "queue" / f"020_{dataset}_groups{''.join(map(str, panels))}.json", jobs
    )


def depth_replay(dataset, root, checkpoint_root, priority=18):
    source_path, events_path, events, n = source(dataset, root)
    first = first_occurrences(events, n)
    novelty = topology_novelty(events, n)
    train, test = cutoff(events, 0.7), cutoff(events, 0.8)
    batches, initial_start = mutation_batches(events, n, test, 256, 16, False, novelty)
    jobs = []
    for depth in (2, 3):
        tag = f"e9_depth_{depth}"
        ckpt = checkpoint_root / f"depth{depth}" / "best.pt"
        trace = prepare_control(
            dataset,
            root,
            test,
            train,
            batches,
            initial_start,
            tag,
            root / "traces" / dataset / tag,
            first,
            novelty,
            events_path,
            source_path,
            n,
            events,
            pattern="fixed-hotspot",
            request_count=16,
            checkpoint=ckpt,
            depth=depth,
        )
        jobs.extend(
            replay_jobs(
                trace,
                root / dataset / tag,
                ("head_fused", "ours"),
                "9 trained depth comparison",
            )
        )
    write_json(root / "queue" / f"{priority:03d}_{dataset}_depth.json", jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    control = sub.add_parser("control")
    control.add_argument("--dataset", required=True)
    control.add_argument("--root", type=lambda x: Path(x).resolve(), required=True)
    control.add_argument(
        "--panels", nargs="+", type=int, choices=(4, 5, 6, 7, 8), required=True
    )
    depth = sub.add_parser("depth")
    depth.add_argument("--dataset", required=True)
    depth.add_argument("--root", type=lambda x: Path(x).resolve(), required=True)
    depth.add_argument("--checkpoints", type=lambda x: Path(x).resolve(), required=True)
    depth.add_argument("--queue-priority", type=int, default=18)
    verify = sub.add_parser("verify")
    verify.add_argument("--trace", type=Path, required=True)
    verify.add_argument("--actual", type=Path, required=True)
    verify.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "control":
        controlled(args.dataset, args.root, args.panels)
    elif args.action == "depth":
        depth_replay(args.dataset, args.root, args.checkpoints, args.queue_priority)
    else:
        print(json.dumps(compare_scores(args.trace, args.actual, args.reference)))


if __name__ == "__main__":
    main()
