"""Native chronology and past-only query traces for ``db_a_native``.

Preparation is deliberately independent of model loading and timed replay.
Every raw event survives; simple topology and directed event history differ.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def array_path(manifest, name, manifest_path):
    entry = manifest["arrays"][name]
    relative = entry.get("path_from_dataset_dir")
    if relative:
        return Path(manifest_path).parent / relative
    return Path(entry["absolute_path"])


def prepare_bitcoin(root, raw_path=None):
    """Adapt the released gzip, retaining its stable exact native chronology."""
    root = Path(root)
    directory = root / "data/a_temporal_scale/bitcoin-otc"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    raw_path = (
        Path(raw_path)
        if raw_path
        else root / "data/temporal/soc-sign-bitcoinotc.csv.gz"
    )
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Existing native Bitcoin OTC gzip required: {raw_path}"
        )
    with gzip.open(raw_path, "rt") as source:
        raw = np.loadtxt(source, delimiter=",")
    node_ids = np.unique(raw[:, :2].astype(np.int64))
    endpoints = np.searchsorted(node_ids, raw[:, :2].astype(np.int64))
    order = np.argsort(raw[:, 3], kind="stable")
    events = np.column_stack((endpoints[order], raw[order, 3])).astype(np.int64)
    first = np.full(len(node_ids), len(events), dtype=np.int64)
    positions = np.arange(len(events), dtype=np.int64)
    np.minimum.at(first, events[:, 0], positions)
    np.minimum.at(first, events[:, 1], positions)
    values = dict(
        events=events,
        source_event_index=order.astype(np.int64),
        node_ids=node_ids,
        node_first_seen_event_index=first,
        node_first_seen_timestamp=events[first, 2],
    )
    processed = directory / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, value in values.items():
        path = processed / f"{name}.npy"
        if path.exists():
            raise FileExistsError(f"Incomplete native preparation exists: {path}")
        np.save(path, value, allow_pickle=False)
        entries[name] = dict(
            path_from_dataset_dir=f"processed/{name}.npy",
            absolute_path=str(path.resolve()),
            shape=list(value.shape),
            dtype=str(value.dtype),
        )
    low = np.minimum(events[:, 0], events[:, 1])
    high = np.maximum(events[:, 0], events[:, 1])
    nonself = low != high
    manifest = dict(
        schema_version=1,
        dataset_key="bitcoin-otc",
        artifact_kind="native_snap_temporal_preprocessing",
        arrays=entries,
        source=dict(
            local_path=str(raw_path.resolve()),
            download_url="https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
            chronology="Ascending exact Unix timestamp, stable original source ordinal at ties",
        ),
        statistics=dict(
            raw_events=len(events),
            vertices=len(node_ids),
            self_events=int((~nonself).sum()),
            unique_canonical_undirected_edges=int(
                len(np.unique(low[nonself] * len(node_ids) + high[nonself]))
            ),
            timestamp_min=int(events[0, 2]),
            timestamp_max=int(events[-1, 2]),
        ),
        event_and_topology_contract=dict(
            event_history="Every native directed event, including repeats and self events",
            topology="Canonical simple undirected edges; no self edges",
            rating="Original event rating is preserved in the source gzip; the locked predictor does not use it",
            node_dictionary="Complete ID inventory is for addressing/allocation only",
        ),
    )
    write_json(manifest_path, manifest)
    return manifest_path


def first_occurrences(events, n):
    first = np.full(n, len(events), dtype=np.int64)
    positions = np.arange(len(events), dtype=np.int64)
    np.minimum.at(first, events[:, 0], positions)
    np.minimum.at(first, events[:, 1], positions)
    return first


def topology_novelty(events, n):
    low = np.minimum(events[:, 0], events[:, 1])
    high = np.maximum(events[:, 0], events[:, 1])
    nonself = np.flatnonzero(low != high)
    _, first = np.unique(low[nonself] * n + high[nonself], return_index=True)
    novel = np.zeros(len(events), dtype=bool)
    novel[nonself[first]] = True
    return novel


def boundaries(events, novel, start, versions):
    """Make ~versions real topology batches and expose natural no-op history.

    Each equally spaced event interval is split immediately before a final
    existing-edge/self event, when available. Nothing is reordered or omitted.
    Internal history-only events remain in the topology-bearing batch history.
    """
    targets = np.unique(np.linspace(start, len(events), versions + 1, dtype=np.int64))
    cuts = [start]
    for target in targets[1:-1]:
        candidates = np.flatnonzero(~novel[cuts[-1] : target])
        cut = int(cuts[-1] + candidates[-1] + 1) if len(candidates) else int(target)
        if cut > cuts[-1]:
            cuts.append(cut)
    cuts.append(len(events))
    output = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        # A real no-op suffix (often one event) makes history advancement and
        # graph-state reuse observable without fabricating any update.
        candidates = np.flatnonzero(~novel[left:right])
        if len(candidates) and novel[left:right].any():
            suffix_start = int(left + candidates[-1])
            while suffix_start > left and not novel[suffix_start - 1]:
                suffix_start -= 1
            if suffix_start > left:
                output.append((int(left), suffix_start))
                left = suffix_start
            # If the no-op run is internal, keep its following new edges in a
            # separate batch; this can add a topology version, declared below.
            suffix_stop = int(left + 1)
            while suffix_stop < right and not novel[suffix_stop]:
                suffix_stop += 1
            output.append((int(left), suffix_stop))
            left = suffix_stop
        if left < right:
            output.append((int(left), int(right)))
    return output


def query_pairs(events, h, active, count, kind, rng, used):
    """Access proxy from already observed events; no future endpoint leakage."""
    recent = np.asarray(events[max(0, h - 8192) : h, :2])
    activity, frequency = np.unique(recent.reshape(-1), return_counts=True)
    hotspots = activity[np.argsort(-frequency, kind="stable")[: min(16, len(activity))]]
    pairs = []
    attempts = 0
    while len(pairs) < count:
        attempts += 1
        if attempts > max(10000, count * 200):
            raise ValueError(
                "Too few visible ordered pairs for a nonrepeating query trace"
            )
        if kind == "observed_event_proxy" and rng.random() < 0.65:
            u, v = recent[int(rng.integers(len(recent)))]
        else:
            u = rng.choice(hotspots if kind == "hotspot" else active)
            v = rng.choice(active)
        pair = int(u), int(v)
        if pair[0] == pair[1] or pair in used:
            continue
        used.add(pair)
        pairs.append(pair)
    return np.asarray(pairs, dtype=np.int64).T


def prepare_trace(args):
    root, output = args.root.resolve(), args.output.resolve()
    if (output / "trace.json").exists():
        raise FileExistsError(
            f"Use a new output directory; trace already exists: {output}"
        )
    supplied_manifest = getattr(args, "manifest", None)
    manifest_path = (
        Path(supplied_manifest).resolve()
        if supplied_manifest
        else (
            prepare_bitcoin(root, args.raw_bitcoin)
            if args.dataset == "bitcoin-otc"
            else root / "data/a_temporal_scale/superuser/manifest.json"
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    source_events = np.load(
        array_path(manifest, "events", manifest_path), mmap_mode="r"
    )
    stop = max(4, int(len(source_events) * args.prefix_fraction))
    if stop > len(source_events):
        raise ValueError("Prefix exceeds the native event release")
    output.mkdir(parents=True, exist_ok=True)
    events_path = array_path(manifest, "events", manifest_path).resolve()
    n = int(manifest["statistics"]["vertices"])
    events = source_events
    if stop != len(source_events):
        # Compact only the already selected chronological prefix for capacity
        # pilots. Full release remains its original complete ID dictionary.
        inventory = np.unique(source_events[:stop, :2])
        events = np.array(source_events[:stop], copy=True)
        events[:, :2] = np.searchsorted(inventory, events[:, :2])
        n = len(inventory)
        events_path = output / "prefix_events.npy"
        np.save(events_path, events, allow_pickle=False)
        np.save(output / "prefix_to_native_dense_id.npy", inventory, allow_pickle=False)
    initial_stop = max(2, int(len(events) * args.initial_fraction))
    if initial_stop >= len(events):
        raise ValueError("Initialization must leave at least one event")
    first = first_occurrences(events, n)
    novel = topology_novelty(events, n)
    if not novel[:initial_stop].any():
        raise ValueError("Initialization prefix has no simple nonself edge")
    rng = np.random.default_rng(args.seed)
    batches, requests, arrays, version = [], [], {}, 0
    for batch_id, (left, right) in enumerate(
        boundaries(events, novel, initial_stop, args.versions)
    ):
        before_version = version
        topology_changed = bool(novel[left:right].any())
        version += int(topology_changed)
        batch = dict(
            batch_id=batch_id,
            start=left,
            stop=right,
            raw_events=right - left,
            new_simple_edges=int(novel[left:right].sum()),
            topology_changed=topology_changed,
            before_g=before_version,
            after_g=version,
            first_native_time=int(events[left, 2]),
            last_native_time=int(events[right - 1, 2]),
        )
        batches.append(batch)
        for side, h, g, native_t in (
            ("before", left, before_version, events[left, 2]),
            ("after", right, version, events[right - 1, 2]),
        ):
            active = np.flatnonzero(first < h)
            if len(active) < 2:
                raise ValueError("Query snapshot has fewer than two observed nodes")
            # Pair uniqueness is across all requests at this snapshot, while
            # recurring endpoints intentionally allow lower-layer reuse.
            used = set()
            phase = ("observed_event_proxy", "hotspot", "dense")[(batch_id // 2) % 3]
            for ordinal in range(args.queries_per_snapshot):
                count = args.dense_pairs if phase == "dense" else args.query_pairs
                pairs = query_pairs(events, h, active, count, phase, rng, used)
                request_id = len(requests)
                key = f"q{request_id:05d}"
                arrays[key] = pairs
                requests.append(
                    dict(
                        request_id=request_id,
                        array_key=key,
                        batch_id=batch_id,
                        side=side,
                        workload_phase=phase,
                        g=g,
                        h=h,
                        t=int(native_t),
                        history_time_weeks=float((native_t - events[0, 2]) / 604800.0),
                        visible_nodes=len(active),
                        pairs=pairs.shape[1],
                        distinct_endpoints=int(len(np.unique(pairs))),
                    )
                )
    np.savez(output / "queries.npz", **arrays)
    trace = dict(
        schema_version=1,
        dataset=args.dataset,
        seed=args.seed,
        events_path=str(events_path),
        event_source_manifest=str(manifest_path.resolve()),
        queries_path=str(output / "queries.npz"),
        n_nodes=n,
        universe_events=len(events),
        prefix_fraction=args.prefix_fraction,
        initial_fraction=args.initial_fraction,
        initial_stop=initial_stop,
        initial_edges=int(novel[:initial_stop].sum()),
        topology_versions=version,
        history_only_batches=sum(not b["topology_changed"] for b in batches),
        native_timestamp_origin=int(events[0, 2]),
        history_time_unit="elapsed native weeks (604800 seconds)",
        checkpoint=identity(args.checkpoint),
        readout=identity(args.readout),
        readout_semantics="Frozen BO UnderGS validation-selected union24_hgb15. Superuser is transfer of this readout, without fitting or a new quality claim.",
        query_semantics="Generated access proxy, not production logs. Observed pairs/hotspots use only events before h; all endpoints first_seen<h. Pairs differ within each snapshot. Dense requests sample visible IDs.",
        snapshot_semantics="(g,h,t): topology version, number of directed raw events ingested, native query time. Before requests run before batch ingestion; after requests run after it. Every native event including repeats/self events reaches history.",
        encoding="100-dimensional frozen_structural_features from initialization prefix only, including zero-degree bias for unobserved allocated IDs; no future topology inputs.",
        batches=batches,
        requests=requests,
    )
    write_json(output / "trace.json", trace)
    return trace
