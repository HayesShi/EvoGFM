"""Prepare seed-42 causal test replay with fixed candidates and timestamp groups."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.db_a_native import DEFAULT_CHECKPOINT, DEFAULT_READOUT, WORKSPACE
from experiments.db_a_native_trace import (
    array_path,
    first_occurrences,
    identity,
    topology_novelty,
    write_json,
)


def prepare(
    dataset,
    output,
    groups_per_update,
    pairs_per_request,
    readout,
    source=None,
    checkpoint=DEFAULT_CHECKPOINT,
    split="test",
    typed_destinations=None,
):
    legacy_source = source is None
    source = source or WORKSPACE / "data/a_temporal_scale" / dataset / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    events_path = array_path(manifest, "events", source).resolve()
    events = np.load(events_path, mmap_mode="r")
    n = manifest["statistics"]["vertices"]
    groups = np.r_[0, np.flatnonzero(events[1:, 2] != events[:-1, 2]) + 1, len(events)]
    train_group = int(np.argmin(np.abs(groups - 0.7 * len(events))))
    test_group = int(np.argmin(np.abs(groups - 0.8 * len(events))))
    train_stop, test_start = map(int, groups[[train_group, test_group]])
    rng = np.random.default_rng(42)
    first = first_occurrences(events, n)
    destinations = None
    if typed_destinations:
        destinations = np.zeros(n, dtype=bool)
        destinations[np.load(typed_destinations)] = True
    novel = topology_novelty(events, n)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trace.json").exists():
        raise FileExistsError(output / "trace.json")
    requests, batches, arrays, version = [], [], {}, 0
    origin = int(events[0, 2])
    start_group, stop_group = (
        (test_group, len(groups) - 1) if split == "test" else (train_group, test_group)
    )
    for batch_id, group in enumerate(range(start_group, stop_group, groups_per_update)):
        start, stop = int(groups[group]), int(
            groups[min(group + groups_per_update, stop_group)]
        )
        at_time = events[start : groups[group + 1]]
        positives = at_time[at_time[:, 0] != at_time[:, 1], :2]
        eligible = first < start
        active = np.flatnonzero(eligible)
        pool = (
            active if destinations is None else np.flatnonzero(eligible & destinations)
        )
        candidates = np.empty((len(positives), 101, 2), dtype=np.int64)
        for u in np.unique(positives[:, 0]):
            rows = np.flatnonzero(positives[:, 0] == u)
            blocked = np.r_[at_time[at_time[:, 0] == u, 1], u]
            if len(pool) - len(np.unique(blocked)) < 100:
                allowed = np.setdiff1d(pool, blocked)
            else:
                allowed = None
            if allowed is not None and len(allowed) < 100:
                raise ValueError("Fewer than 100 eligible negatives for a query")
            for row in rows:
                candidates[row, :, 0] = u
                candidates[row, 0, 1] = positives[row, 1]
                if legacy_source or len(pool) < 512:
                    allowed = (
                        np.setdiff1d(pool, blocked) if allowed is None else allowed
                    )
                    candidates[row, 1:, 1] = rng.choice(
                        allowed, size=100, replace=False
                    )
                else:
                    accepted, seen = [], set(blocked.tolist())
                    while len(accepted) < 100:
                        draw = rng.choice(pool, size=100 - len(accepted), replace=False)
                        for value in draw.tolist():
                            if value not in seen:
                                accepted.append(value)
                                seen.add(value)
                    candidates[row, 1:, 1] = accepted
        pairs = candidates.reshape(-1, 2).T.copy()
        for offset in range(0, pairs.shape[1], pairs_per_request):
            block = pairs[:, offset : offset + pairs_per_request].copy()
            request_id = len(requests)
            key = f"q{request_id:05d}"
            arrays[key] = block
            requests.append(
                dict(
                    request_id=request_id,
                    array_key=key,
                    batch_id=batch_id,
                    side="before",
                    workload_phase=f"causal_{split}",
                    g=version,
                    h=start,
                    t=int(at_time[0, 2]),
                    history_time_weeks=(int(at_time[0, 2]) - origin) / 604800.0,
                    visible_nodes=len(active),
                    pairs=block.shape[1],
                    distinct_endpoints=int(len(np.unique(block))),
                )
            )
        additions = int(novel[start:stop].sum())
        batches.append(
            dict(
                batch_id=batch_id,
                start=start,
                stop=stop,
                raw_events=stop - start,
                new_simple_edges=additions,
                topology_changed=bool(additions),
                before_g=version,
                after_g=version + int(bool(additions)),
                first_native_time=int(events[start, 2]),
                last_native_time=int(events[stop - 1, 2]),
                positives=len(positives),
            )
        )
        version += int(bool(additions))
    queries = output / "queries.npz"
    np.savez(queries, **arrays)
    trace = dict(
        schema_version=1,
        dataset=dataset,
        seed=42,
        n_nodes=n,
        protocol_scope=f"70/10/20 causal {split} replay with 100 negatives per positive at each update block's first timestamp",
        events_path=str(events_path),
        event_source_manifest=str(source.resolve()),
        queries_path=str(queries.resolve()),
        universe_events=len(events),
        initial_stop=int(groups[start_group]),
        encoding_stop=train_stop,
        initial_edges=int(novel[: groups[start_group]].sum()),
        topology_versions=version,
        history_only_batches=sum(not b["topology_changed"] for b in batches),
        native_timestamp_origin=origin,
        checkpoint=identity(checkpoint),
        readout=identity(readout),
        readout_semantics="Frozen readout from prior validation; held fixed for all execution methods",
        groups_per_update=groups_per_update,
        pairs_per_request=pairs_per_request,
        batches=batches,
        requests=requests,
    )
    write_json(output / "trace.json", trace)
    print(
        json.dumps(
            dict(
                dataset=dataset,
                train_events=train_stop,
                test_events=len(events) - test_start,
                update_blocks=len(batches),
                requests=len(requests),
                positive_queries=sum(b["positives"] for b in batches),
                candidates=sum(q["pairs"] for q in requests),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-update", type=int, default=256)
    parser.add_argument("--pairs-per-request", type=int, default=256)
    parser.add_argument("--readout", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--typed-destinations", type=Path)
    args = parser.parse_args()
    if min(args.groups_per_update, args.pairs_per_request) < 1:
        parser.error("block and request sizes must be positive")
    prepare(
        args.dataset,
        args.output.resolve(),
        args.groups_per_update,
        args.pairs_per_request,
        args.readout,
        args.source,
        args.checkpoint,
        args.split,
        args.typed_destinations,
    )


if __name__ == "__main__":
    main()
