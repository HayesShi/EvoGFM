"""Dynamic-source training and independent causal deployment for direction A.

This entry point never reads the historical static reference checkpoint.
Training uses bounded query-neighborhood minibatches of complete source streams;
deployment consumes every update and evaluates a fixed sampled query workload.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

from evogfm.data import canonical_edges, seed_all
from evogfm.features import adjacency, contexts, pair_channels, task_features
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.readout import fit_residual, new_head, predict_residual
from evogfm.temporal import replace_temporal
from evogfm.source_target_state import CompactTemporalHistory as TemporalHistory
from experiments.full_graph import FullGraphGFM, load_checkpoint


TITLE = "Incremental Predictive Analytics over Evolving Graphs"
VERSION = "dynamic-source-target-a-v3"
SOURCES = ("uci", "bitcoin-alpha", "bitcoin-otc", "wiki-talk")
TARGETS = (
    "wikipedia",
    "reddit-body",
    "reddit-title",
    "stack-overflow",
    "mathoverflow",
    "askubuntu",
    "superuser",
    "email-eu",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def dataset_seed(name, seed, offset=0):
    return seed + int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) + offset


def load_events(args, name):
    from evogfm.source_target_data import load_events as loader

    return loader(name, args.data_root, snapshots=args.snapshots)


def offsets(events):
    supplied = events.metadata.get("snapshot_offsets")
    if supplied is not None:
        return np.asarray(supplied, dtype=np.int64)
    _, counts = np.unique(events.time, return_counts=True)
    return np.r_[0, counts.cumsum()]


def phase_name(events, snapshot):
    value = events.phase[offsets(events)[snapshot]]
    if isinstance(value, (str, np.str_)):
        return str(value)
    return ("train", "validation", "test")[int(value)]


def snapshot_pairs(events, snapshot, boundaries=None):
    boundaries = offsets(events) if boundaries is None else boundaries
    low, high = boundaries[snapshot : snapshot + 2]
    return np.stack((events.source[low:high], events.destination[low:high]))


def available_destinations(events, snapshot):
    from evogfm.source_target_data import eligible_destinations

    pool = eligible_destinations(events, snapshot)
    if not len(pool):
        raise ValueError("No causally available destinations")
    return pool


def queryable_events(events, snapshot, positive_all, indices):
    """Drop queries only when their source has no legal negative destination."""
    pool = available_destinations(events, snapshot)
    forbidden = np.unique(positive_all[0] * events.n_nodes + positive_all[1])
    allowed = np.zeros(events.n_nodes, dtype=bool)
    allowed[pool] = True
    impossible = []
    for source in np.unique(positive_all[0, indices]):
        low, high = np.searchsorted(
            forbidden, [source * events.n_nodes, (source + 1) * events.n_nodes]
        )
        current_targets = forbidden[low:high] % events.n_nodes
        blocked = int(allowed[current_targets].sum()) + int(
            allowed[source] and source not in current_targets
        )
        if blocked >= len(pool):
            impossible.append(source)
    return indices[~np.isin(positive_all[0, indices], impossible)]


def sample_negatives(events, snapshot, positive, all_positive, count, rng):
    """Same source, typed/currently available destinations, no future filtering.

    Current directed positive pairs and self links are rejected. Historical
    interactions remain eligible because this is repeated-event prediction.
    Sampling is with replacement, and ranking ties are handled explicitly.
    """
    pool = available_destinations(events, snapshot)
    source = np.repeat(positive[0], count)
    target = pool[rng.integers(len(pool), size=len(source))]
    forbidden = np.unique(all_positive[0] * events.n_nodes + all_positive[1])
    invalid = (source == target) | np.isin(source * events.n_nodes + target, forbidden)
    for _ in range(100):
        if not invalid.any():
            break
        target[invalid] = pool[rng.integers(len(pool), size=int(invalid.sum()))]
        invalid = (source == target) | np.isin(
            source * events.n_nodes + target, forbidden
        )
    if invalid.any():
        for node in np.unique(source[invalid]):
            selected = np.flatnonzero((source == node) & invalid)
            legal = pool[
                (pool != node) & ~np.isin(node * events.n_nodes + pool, forbidden)
            ]
            if not len(legal):
                raise ValueError(
                    f"No legal negative destination for source {node} at {snapshot}"
                )
            target[selected] = legal[rng.integers(len(legal), size=len(selected))]
    return np.stack((source, target))


def protocol_config(args):
    return dict(
        version=VERSION,
        run_kind="engineering_smoke" if args.smoke else "formal",
        seed=args.seed,
        snapshots=args.snapshots,
        warmup_snapshots=args.warmup_snapshots,
        max_query_positives=args.query_positives,
        ranking_negatives=args.ranking_negatives,
        input_dimension=args.dimension,
        feature_frame=args.feature_frame,
        feature_frame_configuration=frame_settings(args),
        observation="predict-before-ingest",
        split="chronological nonempty snapshots 70/10/20 from loader",
        source_selection="last20% of source training prefix, excluding original validation/test",
        node_universe="external transductive node ID/type registry; fixed warmup-only structure plus optional deterministic static random attributes",
        negatives="same-source typed destinations first_seen<t; reject current positives/self; replacement; no future filtering; omit sources with no legal negative",
        aggregation="pooled balanced-workload AUC/AP; event-mean MRR with average tied rank",
        graph="undirected simple topology for spectral/backbone; directed event recurrence channels",
        query_scope="fixed sampled positives per snapshot; all update events ingested",
    )


def frame_settings(args, name=None):
    structure_dims = min(args.dimension - 5, 24) + 5
    hash_dims = (
        min(args.hash_dims, 32, args.dimension - structure_dims)
        if args.feature_frame == "causal_hash_structure"
        else 0
    )
    return dict(
        version=VERSION,
        feature_frame=args.feature_frame,
        seed=args.seed,
        dimension=args.dimension,
        warmup_snapshots=args.warmup_snapshots,
        structure_dims=structure_dims,
        hash_dims_requested=args.hash_dims,
        hash_dims_effective=hash_dims,
        dataset_hash_seed=(
            dataset_seed(name, args.seed)
            if name is not None
            else "seed42+sha256(dataset) prefix"
        ),
        hash_algorithm="row/channel SplitMix64 Rademacher; independent of node count, chunk size, edges and labels",
        block_normalization="independent row L2 normalization; no cross-node fitted normalization",
        structure_weight=float(1 / np.sqrt(2)) if hash_dims else 1.0,
        hash_weight=float(1 / np.sqrt(2)) if hash_dims else 0.0,
        cold_node_input="zero structure block; static hash block is nonzero when enabled",
        hash_trainable_parameters=0,
        future_topology_or_labels_used=False,
    )


def frame_path(args, name):
    setting = frame_settings(args, name)
    filename = f"frame_{args.feature_frame}_h{setting['hash_dims_effective']}_s{args.seed}_{identity(setting)}.npy"
    return args.output / "datasets" / name / filename


def hash_rows(start, stop, dimensions, seed):
    """Deterministic normalized random attributes, computed without fitted state.

    A row depends only on registered node ID, channel and dataset seed. Extending
    the registry or changing generation chunk sizes cannot change existing rows.
    """
    row = np.arange(start, stop, dtype=np.uint64)[:, None] + np.uint64(1)
    channel = np.arange(dimensions, dtype=np.uint64)[None, :] + np.uint64(1)
    with np.errstate(over="ignore"):
        value = (
            row * np.uint64(0x9E3779B97F4A7C15)
            ^ channel * np.uint64(0xD1B54A32D192ED03)
            ^ np.uint64(seed)
        )
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    signs = ((value & np.uint64(1)).astype(np.float32) * 2 - 1) / np.sqrt(dimensions)
    return signs.astype(np.float32, copy=False)


def compact_metadata(events):
    return {
        key: value
        for key, value in events.metadata.items()
        if key not in ("destination_pool", "first_seen", "snapshot_offsets")
    }


def ensure_protocol(args, name, events):
    directory = args.output / "datasets" / name / "protocol"
    directory.mkdir(parents=True, exist_ok=True)
    config = protocol_config(args)
    protocol_id = identity(config)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["protocol_id"] != protocol_id:
            raise ValueError(
                f"Protocol changed at {directory}; choose a new output directory"
            )
        if (directory / "index.json").exists():
            return json.loads((directory / "index.json").read_text()), directory
    boundaries = offsets(events)
    index = []
    for snapshot in range(len(boundaries) - 1):
        positive_all = snapshot_pairs(events, snapshot, boundaries)
        rng = np.random.default_rng(dataset_seed(name, args.seed, snapshot))
        selected = np.sort(
            rng.choice(
                positive_all.shape[1],
                min(args.query_positives, positive_all.shape[1]),
                replace=False,
            )
        )
        if snapshot < args.warmup_snapshots:
            selected = selected[:0]
        else:
            selected = queryable_events(events, snapshot, positive_all, selected)
        positive = positive_all[:, selected]
        negative = (
            sample_negatives(events, snapshot, positive, positive_all, 1, rng)
            if len(selected)
            else positive.copy()
        )
        ranking = (
            sample_negatives(
                events, snapshot, positive, positive_all, args.ranking_negatives, rng
            )
            if len(selected)
            else positive.copy()
        )
        filename = f"snapshot_{snapshot:04d}.npz"
        np.savez_compressed(
            directory / filename,
            positive=positive,
            auc_negative=negative,
            ranking_negative=ranking,
            sampled_event_index=events.event_index[boundaries[snapshot] + selected],
        )
        index.append(
            dict(
                snapshot=snapshot,
                time=snapshot,
                phase=phase_name(events, snapshot),
                file=filename,
                positive_queries=len(selected),
                update_events=positive_all.shape[1],
                ranking_negatives=args.ranking_negatives,
            )
        )
    write_json(
        metadata_path,
        dict(
            protocol_id=protocol_id,
            configuration=config,
            dataset=compact_metadata(events),
            nodes=events.n_nodes,
            events=len(events.source),
            snapshots=len(index),
        ),
    )
    write_json(directory / "index.json", index)
    return index, directory


def warmup_frame(args, name, events):
    """Fixed causal structure and optional parameter-free registered-ID hashes."""
    setting = frame_settings(args, name)
    path = frame_path(args, name)
    metadata_path = path.with_suffix(".json")
    boundaries = offsets(events)
    warmup = min(args.warmup_snapshots, len(boundaries) - 2)
    stop = int(boundaries[warmup])
    graph = canonical_edges(
        torch.from_numpy(np.stack((events.source[:stop], events.destination[:stop]))),
        events.n_nodes,
    )
    if not graph.shape[1]:
        raise ValueError("Warmup graph is empty after removing self links")
    if not (path.exists() and metadata_path.exists()):
        matrix = adjacency(graph, events.n_nodes).astype(np.float32).tocsr()
        degree = np.asarray(matrix.sum(1)).ravel()
        bins = min(args.dimension - 5, 24)
        boundary = np.geomspace(1, max(2.0, float(degree.max())), bins)
        assigned = np.minimum(bins - 1, np.searchsorted(boundary, degree))
        observed = np.flatnonzero(degree > 0)
        onehot = sp.csr_matrix(
            (np.ones(len(observed), np.float32), (observed, assigned[observed])),
            shape=(events.n_nodes, bins),
        )
        histogram = (matrix @ onehot).tocsr()
        neighbor_degree = matrix @ degree
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        result = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(events.n_nodes, args.dimension),
        )
        blocksize = 32768
        for start in range(0, events.n_nodes, blocksize):
            end = min(events.n_nodes, start + blocksize)
            block = result[start:end]
            block[:] = 0
            structure = block[:, : setting["structure_dims"]]
            structure[:, :bins] = np.log1p(histogram[start:end].toarray())
            structure[:, bins] = np.log1p(degree[start:end])
            structure[:, bins + 1] = np.log1p(neighbor_degree[start:end]) / np.maximum(
                1, degree[start:end]
            )
            structure[:, bins + 2] = degree[start:end] > 0
            # diag(A^2) = degree for this symmetric simple warmup graph.
            structure[:, bins + 3] = np.log1p(degree[start:end])
            structure[:, bins + 4] = degree[start:end] > 0
            norm = np.sqrt(np.square(structure).sum(1, keepdims=True))
            structure[:] = (
                structure / np.maximum(norm, 1e-12) * setting["structure_weight"]
            )
            if setting["hash_dims_effective"]:
                hash_start = setting["structure_dims"]
                hash_end = hash_start + setting["hash_dims_effective"]
                block[:, hash_start:hash_end] = (
                    hash_rows(
                        start,
                        end,
                        setting["hash_dims_effective"],
                        setting["dataset_hash_seed"],
                    )
                    * setting["hash_weight"]
                )
        result.flush()
        del block, structure, result, histogram, onehot, matrix
        temporary.replace(path)
        write_json(
            metadata_path,
            dict(
                configuration=setting,
                configuration_id=identity(setting),
                fitted_through_snapshot=warmup - 1,
                fixed_during_stream=True,
                dimension=args.dimension,
                hash_generation_block_rows=blocksize,
                node_count=events.n_nodes,
                cold_node_input=setting["cold_node_input"],
                cold_nodes=int((degree == 0).sum()),
                fitting_events=stop,
            ),
        )
    else:
        saved = json.loads(metadata_path.read_text())
        if (
            saved["configuration_id"] != identity(setting)
            or saved["node_count"] != events.n_nodes
        ):
            raise ValueError(
                "Feature frame cache differs from its complete versioned configuration"
            )
    # Copy-on-write mmap permits torch's read-only use without mutating the saved frame.
    return torch.from_numpy(np.load(path, mmap_mode="c")), graph, warmup


def prepare(args):
    for name in args.datasets:
        started = time.perf_counter()
        events = load_events(args, name)
        index, directory = ensure_protocol(args, name, events)
        raw, graph, warmup = warmup_frame(args, name, events)
        result = dict(
            stage="prepare",
            dataset=name,
            events=len(events.source),
            nodes=events.n_nodes,
            snapshots=len(index),
            warmup_snapshots=warmup,
            protocol=str(directory),
            elapsed_s=time.perf_counter() - started,
        )
        write_json(args.output / "datasets" / name / "prepared.json", result)
        print(json.dumps(result), flush=True)
        del events, raw, graph
        gc.collect()


def model_config(args):
    return dict(
        input_dim=args.dimension,
        d=args.dimension,
        hidden=128,
        layers=2,
        heads=4,
        rank=16,
        k=32,
        dropout=0.1,
        inverse_basis="conditional_hat",
        frame_aligned=True,
        input_encoder="identity",
        whiten_floor=0.05,
        attention_normalization="frobenius",
        relation_gate_init=0.0,
        joint_task_backbone=True,
    )


def sampled_neighborhood(graph, n_nodes, pairs, raw, args, rng, matrix=None):
    """Bounded two-hop neighbor sampling on the complete past-only graph."""
    matrix = adjacency(graph, n_nodes).tocsr() if matrix is None else matrix
    seeds = np.unique(pairs)
    if len(seeds) > args.train_max_nodes:
        raise ValueError("Query endpoint count exceeds --train-max-nodes")
    if n_nodes <= args.train_max_nodes and graph.shape[1] <= args.train_max_edges:
        nodes = np.arange(n_nodes)
        edges = graph
    else:
        selected = set(seeds.tolist())
        frontier = seeds
        sampled = []
        for _ in range(2):
            next_nodes = []
            for node in frontier:
                neighbors = matrix.indices[
                    matrix.indptr[node] : matrix.indptr[node + 1]
                ]
                if len(neighbors) > args.train_fanout:
                    neighbors = rng.choice(neighbors, args.train_fanout, replace=False)
                for other in neighbors:
                    if (
                        int(other) not in selected
                        and len(selected) >= args.train_max_nodes
                    ):
                        continue
                    if len(sampled) >= args.train_max_edges:
                        break
                    sampled.append((int(node), int(other)))
                    if int(other) not in selected:
                        selected.add(int(other))
                        next_nodes.append(int(other))
            frontier = np.asarray(next_nodes, np.int64)
        if not sampled:
            # Cold endpoint queries still share an analytical state from past edges.
            sampled = [tuple(graph[:, 0].tolist())]
            selected.update(sampled[0])
        nodes = np.asarray(sorted(selected), dtype=np.int64)
        edges = canonical_edges(
            torch.from_numpy(np.searchsorted(nodes, np.asarray(sampled).T)), len(nodes)
        )
    local_pairs = torch.from_numpy(np.searchsorted(nodes, pairs)).long()
    return raw[torch.from_numpy(nodes)].clone(), edges, local_pairs


def train_batch(
    model, head, raw, graph, history, pairs, snapshot, args, rng, matrix=None
):
    local_raw, local_edges, local_pairs = sampled_neighborhood(
        graph, len(raw), pairs, raw, args, rng, matrix
    )
    ctx = contexts(local_edges, len(local_raw), local_pairs)
    base = pair_channels(local_raw, local_edges, local_pairs)
    device = torch.device(args.device)
    local_raw, local_edges, local_pairs, base = (
        value.to(device) for value in (local_raw, local_edges, local_pairs, base)
    )
    ctx = {key: value.to(device) for key, value in ctx.items()}
    state = model.state(local_raw, local_edges)
    features, _ = task_features(
        model, state, local_raw, local_edges, local_pairs, ctx, base
    )
    statistics = history.statistics(pairs, snapshot).to(device)
    features = replace_temporal(
        features, statistics, "recency_frequency", args.half_life
    )
    return head(features).flatten(), dict(
        nodes=len(local_raw), edges=local_edges.shape[1]
    )


def source_pass(
    args, name, model, head, optimizer, epoch, validation_only=False, heartbeat=None
):
    started = time.perf_counter()
    events = load_events(args, name)
    ensure_protocol(args, name, events)
    raw, graph, warmup = warmup_frame(args, name, events)
    boundaries = offsets(events)
    train_end = sum(
        phase_name(events, i) == "train" for i in range(len(boundaries) - 1)
    )
    selection_start = max(warmup + 1, int(train_end * 0.8))
    history = TemporalHistory(events.n_nodes)
    for snapshot in range(warmup):
        history.ingest(snapshot_pairs(events, snapshot, boundaries), snapshot)
    known = (graph[0] * events.n_nodes + graph[1]).numpy()
    losses, validation_labels, validation_scores, workload = [], [], [], []
    log_path = args.output / "source" / "events.jsonl"
    for snapshot in range(warmup, train_end):
        positive_all = snapshot_pairs(events, snapshot, boundaries)
        validation = snapshot >= selection_start
        if validation != validation_only:
            history.ingest(positive_all, snapshot)
            graph, _, known = ingest_edges(
                graph, torch.from_numpy(positive_all), events.n_nodes, known
            )
            if heartbeat:
                heartbeat(name, snapshot, validation_only)
            continue
        draw_seed = dataset_seed(
            name, args.seed, snapshot + (0 if validation else epoch * 10000)
        )
        rng = np.random.default_rng(draw_seed)
        count = min(
            args.source_validation_positives if validation else args.train_positives,
            positive_all.shape[1],
        )
        selected = np.sort(rng.choice(positive_all.shape[1], count, replace=False))
        selected = queryable_events(events, snapshot, positive_all, selected)
        count = len(selected)
        if count == 0:
            append_json(
                log_path,
                dict(
                    stage="no_queryable_source_events",
                    dataset=name,
                    epoch=epoch,
                    snapshot=snapshot,
                    update_events=positive_all.shape[1],
                ),
            )
            history.ingest(positive_all, snapshot)
            graph, _, known = ingest_edges(
                graph, torch.from_numpy(positive_all), events.n_nodes, known
            )
            if heartbeat:
                heartbeat(name, snapshot, validation_only)
            continue
        positive = positive_all[:, selected]
        negative = sample_negatives(events, snapshot, positive, positive_all, 1, rng)
        batch_losses = []
        matrix = adjacency(graph, events.n_nodes).tocsr()
        for start in range(0, count, args.train_positives):
            stop = min(count, start + args.train_positives)
            pairs = np.concatenate(
                (positive[:, start:stop], negative[:, start:stop]), axis=1
            )
            labels = torch.cat(
                (torch.ones(stop - start), torch.zeros(stop - start))
            ).to(args.device)
            model.train(not validation)
            head.train(not validation)
            with torch.set_grad_enabled(not validation):
                logits, size = train_batch(
                    model, head, raw, graph, history, pairs, snapshot, args, rng, matrix
                )
                bce = F.binary_cross_entropy_with_logits(logits, labels)
                pos, neg = logits[: stop - start], logits[stop - start :]
                hard = neg.topk(max(1, len(neg) // 10)).values
                loss = bce + 0.2 * F.softplus(hard[None, :] - pos[:, None]).mean()
                if not validation:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()), 2.0
                    )
                    optimizer.step()
                    model.inverse.constrain()
                else:
                    validation_labels.append(labels.detach().cpu().numpy())
                    validation_scores.append(logits.detach().cpu().numpy())
            batch_losses.append(float(bce.detach()))
            workload.append(size)
        if not validation:
            losses.extend(batch_losses)
        row = dict(
            stage="source_selection" if validation else "source_train",
            dataset=name,
            epoch=epoch,
            snapshot=snapshot,
            positive_queries=count,
            visible_edges_before=graph.shape[1],
            update_events=positive_all.shape[1],
            bce=float(np.mean(batch_losses)),
            elapsed_s=time.perf_counter() - started,
        )
        append_json(log_path, row)
        history.ingest(positive_all, snapshot)
        graph, _, known = ingest_edges(
            graph, torch.from_numpy(positive_all), events.n_nodes, known
        )
        del matrix
        if heartbeat:
            heartbeat(name, snapshot, validation_only)
    if not workload:
        raise ValueError(
            f"{name}: no queryable source {'selection' if validation_only else 'training'} events"
        )
    result = dict(
        dataset=name,
        epoch=epoch,
        train_bce=float(np.mean(losses)) if losses else None,
        selection_snapshots=[selection_start, train_end - 1],
        elapsed_s=time.perf_counter() - started,
        max_sampled_nodes=max(row["nodes"] for row in workload),
        max_sampled_edges=max(row["edges"] for row in workload),
    )
    if validation_only:
        truth, score = np.concatenate(validation_labels), np.concatenate(
            validation_scores
        )
        result.update(
            selection_bce=float(
                F.binary_cross_entropy_with_logits(
                    torch.from_numpy(score), torch.from_numpy(truth)
                )
            ),
            selection_auc=float(roc_auc_score(truth, score)),
            selection_examples=len(truth),
        )
    print(json.dumps(dict(stage="source_pass", **result)), flush=True)
    del raw, graph, history, events
    gc.collect()
    return result


def training_settings(args):
    return dict(
        version=VERSION,
        protocol=protocol_config(args),
        source_graphs=args.sources,
        model=model_config(args),
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
        weight_decay=1e-4,
        train_positives_per_snapshot=args.train_positives,
        source_validation_positives=args.source_validation_positives,
        train_neighborhood_hops=2,
        fanout=args.train_fanout,
        max_nodes=args.train_max_nodes,
        max_edges=args.train_max_edges,
        optimizer="AdamW",
        gradient_clip=2.0,
        objective="BCE +0.2 top10%-negative softplus ranking",
        source_selection="equal-source mean BCE on final20% inside training prefix",
        source_original_validation_test_used=False,
        temporal_transform="recency_frequency",
        half_life=args.half_life,
        device=args.device,
        deployment_device="cpu",
    )


def train(args):
    invocation_started = time.perf_counter()
    directory = args.output / "source"
    directory.mkdir(parents=True, exist_ok=True)
    settings = training_settings(args)
    training_id = identity(settings)
    config_path = directory / "configuration.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous["training_id"] != training_id:
            raise ValueError(
                "Source training configuration changed; choose a new output root"
            )
    write_json(config_path, dict(training_id=training_id, configuration=settings))
    progress_path = directory / "progress.json"
    previous_progress = (
        json.loads(progress_path.read_text()) if progress_path.exists() else {}
    )
    if (
        previous_progress.get("status") == "complete"
        and (directory / "inference.pt").exists()
    ):
        print(
            json.dumps(dict(stage="source_already_complete", **previous_progress)),
            flush=True,
        )
        return
    seed_all(args.seed)
    model = FullGraphGFM(**model_config(args)).to(args.device)
    head = new_head(True).to(args.device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    state = dict(
        epoch=0,
        source_index=0,
        current_rows=[],
        history=[],
        best=float("inf"),
        stale=0,
        total_training_wall_s=0.0,
    )
    latest = directory / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=args.device, weights_only=False)
        if checkpoint["training_id"] != training_id:
            raise ValueError("Resume checkpoint has another training configuration")
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["progress"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if args.device.startswith("cuda") and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in checkpoint["cuda_rng"]]
            )
    state["total_training_wall_s"] = max(
        state.get("total_training_wall_s", 0.0),
        previous_progress.get("total_training_wall_s", 0.0),
    )
    began = time.perf_counter()
    last_tick = invocation_started

    def capture_elapsed():
        nonlocal last_tick
        now = time.perf_counter()
        state["total_training_wall_s"] += now - last_tick
        last_tick = now

    def heartbeat(name, snapshot, validation_only):
        capture_elapsed()
        write_json(
            progress_path,
            dict(
                status="running",
                epoch=state["epoch"],
                source=name,
                source_snapshot=snapshot,
                selection=validation_only,
                total_training_wall_s=state["total_training_wall_s"],
                training_id=training_id,
            ),
        )

    def pack():
        capture_elapsed()
        elapsed_pass_s = sum(
            sum(row["elapsed_s"] for row in epoch.get("source_results", []))
            + sum(row["elapsed_s"] for row in epoch.get("source_validation", []))
            for epoch in state["history"]
        )
        result = dict(
            model=model.state_dict(),
            head=head.state_dict(),
            model_config=model.config,
            optimizer=optimizer.state_dict(),
            progress=deepcopy(state),
            training_id=training_id,
            settings=settings,
            source_graphs=list(args.sources),
            source_protocol="dynamic-training-prefix-only",
            version=VERSION,
            pretraining_elapsed_pass_s=elapsed_pass_s,
            total_training_wall_s=state["total_training_wall_s"],
            epoch=state["epoch"],
            torch_rng=torch.get_rng_state(),
            validation_loss=state["best"],
        )
        if args.device.startswith("cuda"):
            result["cuda_rng"] = torch.cuda.get_rng_state_all()
        return result

    for epoch in range(state["epoch"], args.epochs):
        for source_index in range(state["source_index"], len(args.sources)):
            name = args.sources[source_index]
            row = source_pass(
                args, name, model, head, optimizer, epoch, heartbeat=heartbeat
            )
            state["current_rows"].append(row)
            state["source_index"] = source_index + 1
            save_torch(latest, pack())
        # Evaluate every source under the same final epoch parameters. Scores
        # collected before training later sources cannot select this checkpoint.
        validation_rows = [
            source_pass(
                args,
                name,
                model,
                head,
                optimizer,
                epoch,
                validation_only=True,
                heartbeat=heartbeat,
            )
            for name in args.sources
        ]
        value = float(np.mean([row["selection_bce"] for row in validation_rows]))
        record = dict(
            epoch=epoch + 1,
            selection_bce=value,
            source_results=deepcopy(state["current_rows"]),
            source_validation=validation_rows,
        )
        state["history"].append(record)
        improved = value < state["best"]
        state["best"] = min(value, state["best"])
        state["stale"] = 0 if improved else state["stale"] + 1
        state.update(epoch=epoch + 1, source_index=0, current_rows=[])
        if improved:
            save_torch(directory / "best.pt", pack())
        save_torch(latest, pack())
        write_json(directory / "history.json", state["history"])
        write_json(
            directory / "progress.json",
            dict(
                status="running",
                epoch=state["epoch"],
                best_selection_bce=state["best"],
                elapsed_this_invocation_s=time.perf_counter() - began,
                total_training_wall_s=state["total_training_wall_s"],
                training_id=training_id,
            ),
        )
        print(json.dumps(dict(stage="source_epoch", **record)), flush=True)
        if state["stale"] >= args.patience:
            break
    # The inference artifact contains only serving weights/provenance. Optimizer
    # and RNG snapshots remain training audit/recovery artifacts.
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    inference_keys = (
        "model",
        "head",
        "model_config",
        "training_id",
        "settings",
        "source_graphs",
        "source_protocol",
        "version",
        "epoch",
        "validation_loss",
        "pretraining_elapsed_pass_s",
    )
    inference = {key: selected[key] for key in inference_keys}
    capture_elapsed()
    inference["total_training_wall_s"] = state["total_training_wall_s"]
    save_torch(directory / "inference.pt", inference)
    capture_elapsed()
    write_json(
        directory / "progress.json",
        dict(
            status="complete",
            epoch=state["epoch"],
            best_selection_bce=state["best"],
            elapsed_this_invocation_s=time.perf_counter() - began,
            total_training_wall_s=state["total_training_wall_s"],
            cost_scope="resumable measured wall clock including loading, training, source validation and checkpoint writes; idle time between invocations excluded",
            checkpoint=str(directory / "inference.pt"),
            training_id=training_id,
        ),
    )


def features_for_pairs(scorer, history, pairs, snapshot, args):
    chunks = []
    for start in range(0, pairs.shape[1], args.query_batch_size):
        block = pairs[:, start : start + args.query_batch_size]
        base = scorer.features(torch.from_numpy(block))
        chunks.append(
            replace_temporal(
                base,
                history.statistics(block, snapshot),
                "recency_frequency",
                args.half_life,
            )
        )
    return torch.cat(chunks)


def load_dynamic_checkpoint(path):
    model, head, checkpoint = load_checkpoint(path)
    if checkpoint.get("source_protocol") != "dynamic-training-prefix-only":
        raise ValueError(
            "This experiment requires its new dynamic-source checkpoint, not reference.pt"
        )
    return model, head, checkpoint


def adaptation(args, name):
    directory = args.output / "datasets" / name / "adaptation"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "source" / "inference.pt"
    model, head, checkpoint = load_dynamic_checkpoint(checkpoint_path)
    config = dict(
        training_id=checkpoint["training_id"],
        checkpoint_epoch=checkpoint["epoch"],
        protocol=protocol_config(args),
        ridges=args.ridges,
        half_life=args.half_life,
    )
    run_id = identity(config)
    if (directory / "summary.json").exists():
        previous = json.loads((directory / "summary.json").read_text())
        if previous["adaptation_id"] == run_id:
            return previous
        raise ValueError("Adaptation inputs changed; use a new output root")
    started = time.perf_counter()
    events = load_events(args, name)
    index, protocol = ensure_protocol(args, name, events)
    raw, visible, warmup = warmup_frame(args, name, events)
    boundaries = offsets(events)
    history = TemporalHistory(events.n_nodes)
    for snapshot in range(warmup):
        history.ingest(snapshot_pairs(events, snapshot, boundaries), snapshot)
    known = (visible[0] * events.n_nodes + visible[1]).numpy()
    scorer = MaintainedLinkScorer(model, head, raw, raw, visible, method="full")
    training_features, training_offsets, training_labels = [], [], []
    validation_features, validation_offsets, validation_labels = [], [], []
    rows = []
    with torch.no_grad():
        for entry in index[warmup:]:
            if entry["phase"] == "test":
                break
            snapshot = entry["snapshot"]
            with np.load(protocol / entry["file"]) as query:
                pairs = np.concatenate(
                    (query["positive"], query["auc_negative"]), axis=1
                )
            if not pairs.shape[1]:
                incoming = snapshot_pairs(events, snapshot, boundaries)
                history.ingest(incoming, snapshot)
                visible, added, known = ingest_edges(
                    visible, torch.from_numpy(incoming), events.n_nodes, known
                )
                scorer.update(visible, added)
                continue
            feature_start = time.perf_counter()
            features = features_for_pairs(scorer, history, pairs, snapshot, args)
            offset = head(features).flatten()
            labels = torch.cat(
                (torch.ones(pairs.shape[1] // 2), torch.zeros(pairs.shape[1] // 2))
            )
            if entry["phase"] == "train":
                training_features.append(features)
                training_offsets.append(offset)
                training_labels.append(labels)
            else:
                validation_features.append(features)
                validation_offsets.append(offset)
                validation_labels.append(labels)
            feature_s = time.perf_counter() - feature_start
            ingest_start = time.perf_counter()
            incoming = snapshot_pairs(events, snapshot, boundaries)
            history.ingest(incoming, snapshot)
            visible, added, known = ingest_edges(
                visible, torch.from_numpy(incoming), events.n_nodes, known
            )
            update = scorer.update(visible, added)
            rows.append(
                dict(
                    snapshot=snapshot,
                    phase=entry["phase"],
                    positive_queries=entry["positive_queries"],
                    update_events=incoming.shape[1],
                    feature_s=feature_s,
                    ingest_and_update_s=time.perf_counter() - ingest_start,
                    **update,
                )
            )
    fit_started = time.perf_counter()
    if not training_features or not validation_features:
        raise ValueError(
            f"{name}: adaptation requires nonempty training and validation query workloads"
        )
    features, offset, labels = map(
        torch.cat, (training_features, training_offsets, training_labels)
    )
    vf, vo, vy = map(
        torch.cat, (validation_features, validation_offsets, validation_labels)
    )
    trials = []
    best = None
    for ridge in args.ridges:
        fitted = fit_residual(features, offset, labels, ridge=ridge, steps=12)
        prediction = predict_residual(vf, vo, fitted)
        loss = float(F.binary_cross_entropy_with_logits(prediction, vy.double()))
        trial = dict(
            ridge=ridge,
            validation_bce=loss,
            validation_auc=float(roc_auc_score(vy.numpy(), prediction.numpy())),
        )
        trials.append(trial)
        if best is None or loss < best[0]:
            best = (loss, fitted, ridge)
    fit_s = time.perf_counter() - fit_started
    save_torch(
        directory / "readout.pt",
        dict(
            fitted=best[1],
            ridge=best[2],
            adaptation_id=run_id,
            source_training_id=checkpoint["training_id"],
        ),
    )
    result = dict(
        dataset=name,
        run_kind="engineering_smoke" if args.smoke else "formal",
        adaptation_id=run_id,
        configuration=config,
        source_checkpoint=str(checkpoint_path.resolve()),
        source_training_id=checkpoint["training_id"],
        source_checkpoint_epoch=checkpoint["epoch"],
        source_graphs=checkpoint["source_graphs"],
        independent_target_initialization=True,
        training_examples=len(labels),
        validation_examples=len(vy),
        ridge=best[2],
        trials=trials,
        adaptation_s=time.perf_counter() - started,
        readout_fit_s=fit_s,
        readout_coefficients=18,
        rows=rows,
        test_labels_used=False,
        status="complete",
    )
    write_json(directory / "summary.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("rows", "configuration", "trials")
            }
        ),
        flush=True,
    )
    return result


def rss_bytes():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


def artifact_storage(args, name, events, protocol, deployment_directory):
    """Actual file lengths for required inputs, shared state, and audit output.

    CSR indices, sufficient statistics and representation caches are constructed
    in memory. They are not silently counted as persistent files that do not
    exist; their numeric payload is reported separately by the scorer/history.
    """
    prepared = Path(events.source.filename).parent
    categories = dict(
        prepared_event_graph_and_time_index=[
            prepared / f"{key}.npy"
            for key in ("source", "destination", "time", "event_index", "phase")
        ],
        node_type_and_first_seen_index=[
            prepared / f"{key}.npy" for key in ("first_seen", "destination_pool")
        ],
        prepared_manifest=[prepared / "manifest.json"],
        fixed_warmup_input=[frame_path(args, name)],
        shared_model_checkpoint=[args.output / "source" / "inference.pt"],
        independent_target_head=[
            args.output / "datasets" / name / "adaptation" / "readout.pt"
        ],
        fixed_query_workload=list(protocol.glob("*.npz"))
        + [protocol / "metadata.json", protocol / "index.json"],
    )
    lengths = {
        key: sum(path.stat().st_size for path in paths if path.is_file())
        for key, paths in categories.items()
    }
    all_paths = {
        path.resolve()
        for paths in categories.values()
        for path in paths
        if path.is_file()
    }
    audit_predictions = list(deployment_directory.glob("snapshot_*.npz"))
    training_audit = [
        path
        for path in (args.output / "source").glob("*")
        if path.is_file() and path.name != "inference.pt"
    ]
    return dict(
        required_disk_bytes=sum(path.stat().st_size for path in all_paths),
        required_artifact_bytes=lengths,
        immutable_stream_and_query_inputs_shared_across_methods=True,
        resident_graph_index_persisted_bytes=0,
        resident_history_persisted_bytes=0,
        resident_representation_cache_persisted_bytes=0,
        cache_storage_policy="reconstructed from event graph at cold start; resident numeric payload reported separately",
        optional_audit_prediction_bytes=sum(
            path.stat().st_size for path in audit_predictions
        ),
        optional_training_recovery_and_audit_bytes=sum(
            path.stat().st_size for path in training_audit
        ),
        audit_outputs_required_for_serving=False,
    )


def scores_and_metrics(scorer, head, fitted, history, query, snapshot, args):
    npos = query["positive"].shape[1]
    pairs = np.concatenate(
        (query["positive"], query["auc_negative"], query["ranking_negative"]), axis=1
    )
    probabilities = []
    for start in range(0, pairs.shape[1], args.query_batch_size):
        features = features_for_pairs(
            scorer,
            history,
            pairs[:, start : start + args.query_batch_size],
            snapshot,
            args,
        )
        prediction = predict_residual(
            features, head(features).flatten(), fitted
        ).sigmoid()
        probabilities.append(prediction.numpy())
    prediction = np.concatenate(probabilities)
    positive, negative = prediction[:npos], prediction[npos : 2 * npos]
    ranking = prediction[2 * npos :].reshape(npos, args.ranking_negatives)
    greater = (ranking > positive[:, None]).sum(1)
    tied = (ranking == positive[:, None]).sum(1)
    reciprocal = 1 / (1 + greater + tied / 2)
    labels = np.r_[np.ones(npos), np.zeros(npos)]
    metrics = dict(
        auc=float(roc_auc_score(labels, np.r_[positive, negative])),
        ap=float(average_precision_score(labels, np.r_[positive, negative])),
        mrr=float(reciprocal.mean()),
        pessimistic_mrr=float(np.mean(1 / (1 + greater + tied))),
        tied_queries=int((tied > 0).sum()),
        positive_queries=npos,
    )
    return metrics, prediction, reciprocal


def deploy_method(args, name, method, common):
    directory = args.output / "datasets" / name / "deployment" / method
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        adaptation_id=common["adaptation_id"],
        method=method,
        threshold=args.drift_threshold,
        period=args.refresh_period,
        query_batch_size=args.query_batch_size,
        threads=args.threads,
    )
    deployment_id = identity(config)
    if (directory / "summary.json").exists():
        previous = json.loads((directory / "summary.json").read_text())
        if previous["deployment_id"] == deployment_id:
            return previous
        raise ValueError("Deployment configuration changed; use a new output root")
    started = time.perf_counter()
    peak_rss = rss_bytes()
    model, head, checkpoint = load_dynamic_checkpoint(
        args.output / "source" / "inference.pt"
    )
    fitted = torch.load(
        args.output / "datasets" / name / "adaptation" / "readout.pt",
        weights_only=False,
    )["fitted"]
    events = load_events(args, name)
    index, protocol = ensure_protocol(args, name, events)
    raw, _, warmup = warmup_frame(args, name, events)
    boundaries = offsets(events)
    first_test = next(entry["snapshot"] for entry in index if entry["phase"] == "test")
    stop = int(boundaries[first_test])
    visible = canonical_edges(
        torch.from_numpy(np.stack((events.source[:stop], events.destination[:stop]))),
        events.n_nodes,
    )
    known = (visible[0] * events.n_nodes + visible[1]).numpy()
    history = TemporalHistory(events.n_nodes)
    for snapshot in range(first_test):
        history.ingest(snapshot_pairs(events, snapshot, boundaries), snapshot)
    load_and_graph_s = time.perf_counter() - started
    initialize_start = time.perf_counter()
    scorer = MaintainedLinkScorer(
        model,
        head,
        raw,
        raw,
        visible,
        method=method,
        threshold=args.drift_threshold,
        period=args.refresh_period,
    )
    initialize_s = time.perf_counter() - initialize_start
    peak_rss = max(peak_rss, rss_bytes())
    rows, pooled_labels, pooled_scores, reciprocal_all = [], [], [], []
    reference_directory = directory.parent / "full"
    with torch.no_grad():
        for entry in index[first_test:]:
            snapshot = entry["snapshot"]
            stage_start = time.perf_counter()
            candidate_start = time.perf_counter()
            with np.load(protocol / entry["file"]) as archive:
                query = {key: archive[key] for key in archive.files}
            candidate_load_s = time.perf_counter() - candidate_start
            query_start = time.perf_counter()
            if query["positive"].shape[1]:
                metrics, prediction, reciprocal = scores_and_metrics(
                    scorer, head, fitted, history, query, snapshot, args
                )
            else:
                metrics = dict(
                    auc=None,
                    ap=None,
                    mrr=None,
                    pessimistic_mrr=None,
                    tied_queries=0,
                    positive_queries=0,
                )
                prediction, reciprocal = np.empty(0), np.empty(0)
            query_s = time.perf_counter() - query_start
            peak_rss = max(peak_rss, rss_bytes())
            npos = metrics["positive_queries"]
            if npos:
                pooled_labels.append(np.r_[np.ones(npos), np.zeros(npos)])
                pooled_scores.append(prediction[: 2 * npos])
                reciprocal_all.append(reciprocal)
            difference = {}
            if method != "full":
                with np.load(reference_directory / entry["file"]) as archive:
                    reference = archive["prediction"]
                difference = dict(
                    prediction_mae_vs_full=(
                        float(np.abs(prediction - reference).mean())
                        if len(prediction)
                        else 0.0
                    ),
                    prediction_max_error_vs_full=(
                        float(np.abs(prediction - reference).max())
                        if len(prediction)
                        else 0.0
                    ),
                )
            serialize_start = time.perf_counter()
            np.savez_compressed(
                directory / entry["file"],
                prediction=prediction,
                reciprocal_rank=reciprocal,
                sampled_event_index=query["sampled_event_index"],
            )
            serialization_s = time.perf_counter() - serialize_start
            update_start = time.perf_counter()
            incoming = snapshot_pairs(events, snapshot, boundaries)
            history.ingest(incoming, snapshot)
            visible, added, known = ingest_edges(
                visible, torch.from_numpy(incoming), events.n_nodes, known
            )
            ingestion_s = time.perf_counter() - update_start
            update = scorer.update(visible, added)
            transition_s = time.perf_counter() - update_start
            peak_rss = max(peak_rss, rss_bytes())
            row = dict(
                snapshot=snapshot,
                update_events=incoming.shape[1],
                new_simple_edges=added.shape[1],
                candidate_load_s=candidate_load_s,
                query_s=query_s,
                ingestion_s=ingestion_s,
                transition_s=transition_s,
                prediction_serialization_s=serialization_s,
                wall_s=time.perf_counter() - stage_start,
                **metrics,
                **update,
                **difference,
            )
            append_json(directory / "events.jsonl", row)
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        stage="deployment_snapshot",
                        dataset=name,
                        method=method,
                        snapshot=snapshot,
                        auc=metrics["auc"],
                        mrr=metrics["mrr"],
                        transition_s=transition_s,
                        query_s=query_s,
                    )
                ),
                flush=True,
            )
    before_measurement_s = time.perf_counter() - started
    truth = np.concatenate(pooled_labels) if pooled_labels else None
    probability = np.concatenate(pooled_scores) if pooled_scores else None
    cache_bytes = scorer.cache_bytes()
    history_bytes = history.state_bytes()
    weight_bytes = sum(
        value.numel() * value.element_size()
        for value in list(model.state_dict().values())
        + list(head.state_dict().values())
    )
    result = dict(
        dataset=name,
        role="Source" if name in args.sources else "Target",
        method=method,
        run_kind="engineering_smoke" if args.smoke else "formal",
        deployment_id=deployment_id,
        configuration=config,
        source_training_id=checkpoint["training_id"],
        auc=float(roc_auc_score(truth, probability)) if truth is not None else None,
        ap=(
            float(average_precision_score(truth, probability))
            if truth is not None
            else None
        ),
        mrr=float(np.concatenate(reciprocal_all).mean()) if reciprocal_all else None,
        snapshot_mean_auc=(
            float(np.mean([row["auc"] for row in rows if row["auc"] is not None]))
            if pooled_labels
            else None
        ),
        positive_queries=sum(row["positive_queries"] for row in rows),
        update_events=sum(row["update_events"] for row in rows),
        test_snapshots=len(rows),
        source_checkpoint_epoch=checkpoint["epoch"],
        source_pretraining_pass_s=checkpoint.get("pretraining_elapsed_pass_s"),
        source_pretraining_wall_s=json.loads(
            (args.output / "source" / "progress.json").read_text()
        )["total_training_wall_s"],
        model_parameters=sum(value.numel() for value in model.parameters())
        + sum(value.numel() for value in head.parameters()),
        model_weight_bytes=weight_bytes,
        readout_coefficient_count=18,
        scorer_numeric_cache_bytes=cache_bytes,
        history_numeric_state_bytes=history_bytes,
        history_storage="exact sorted int64 keys/uint32 counts/int32 timestamps; float32 node activity",
        sampled_process_peak_rss_bytes=peak_rss,
        storage=artifact_storage(args, name, events, protocol, directory),
        timing=dict(
            load_and_graph_s=load_and_graph_s,
            cache_initialization_s=initialize_s,
            candidate_load_s=sum(row["candidate_load_s"] for row in rows),
            query_s=sum(row["query_s"] for row in rows),
            transition_s=sum(row["transition_s"] for row in rows),
            transition_mean_s=float(np.mean([row["transition_s"] for row in rows])),
            transition_p95_s=float(
                np.percentile([row["transition_s"] for row in rows], 95)
            ),
            prediction_serialization_s=sum(
                row["prediction_serialization_s"] for row in rows
            ),
            deployment_wall_before_resource_accounting_s=before_measurement_s,
            common_adaptation_s=common["adaptation_s"],
        ),
        status="complete",
        protocol=protocol_config(args),
        approximation_trigger="attention-program Frobenius drift, not a prediction-loss bound",
        observation="all current queries scored before that snapshot is ingested",
        rows=rows,
    )
    result["timing"]["updates_per_second"] = result["update_events"] / max(
        1e-12, result["timing"]["transition_s"]
    )
    result["timing"][
        "latency_unit"
    ] = "one complete snapshot update batch; not individual-event latency"
    result["timing"][
        "throughput_definition"
    ] = "all ingested directed interaction events divided by ingest+maintenance transition wall time"
    result["timing"]["online_service_s"] = (
        result["timing"]["query_s"] + result["timing"]["transition_s"]
    )
    result["timing"]["deployment_service_s"] = (
        load_and_graph_s
        + initialize_s
        + result["timing"]["candidate_load_s"]
        + result["timing"]["online_service_s"]
    )
    result["timing"]["cold_start_adapt_and_service_s"] = (
        common["adaptation_s"] + result["timing"]["deployment_service_s"]
    )
    if method != "full":
        result["prediction_mae_vs_full"] = float(
            np.mean([row["prediction_mae_vs_full"] for row in rows])
        )
        result["prediction_max_error_vs_full"] = max(
            row["prediction_max_error_vs_full"] for row in rows
        )
        if (
            method == "exact_incremental"
            and result["prediction_max_error_vs_full"] > 1e-4
        ):
            result["exact_incremental_parity"] = False
            write_json(directory / "summary.json", result)
            raise RuntimeError(
                "Exact incremental predictions differ from full by more than 1e-4"
            )
        if method == "exact_incremental":
            result["exact_incremental_parity"] = True
    write_json(directory / "summary.json", result)
    return result


def evaluate(args):
    for name in args.datasets:
        common = adaptation(args, name)
        gc.collect()
        methods = ["full"] + [method for method in args.methods if method != "full"]
        for method in methods:
            result = deploy_method(args, name, method, common)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("rows", "configuration", "protocol", "timing")
                    }
                ),
                flush=True,
            )
            gc.collect()
    collect(args)


def collect(args):
    rows = []
    for path in sorted((args.output / "datasets").glob("*/deployment/*/summary.json")):
        result = json.loads(path.read_text())
        rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in ("rows", "protocol", "configuration")
            }
        )
    write_json(
        args.output / "results.json",
        dict(
            title=TITLE,
            version=VERSION,
            run_kind="engineering_smoke" if args.smoke else "formal",
            results=rows,
        ),
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--stage",
        choices=("prepare", "train", "adapt", "evaluate", "collect", "all"),
        required=True,
    )
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Label isolated engineering checks; never use in the paper result table",
    )
    result.add_argument("--data-root", type=Path, default=Path("data/source_target_a"))
    result.add_argument("--output", type=Path, default=Path("runs/vldb/source_target"))
    result.add_argument("--sources", nargs="+", default=list(SOURCES))
    result.add_argument("--datasets", nargs="+", default=list(SOURCES + TARGETS))
    result.add_argument("--snapshots", type=int, default=100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--dimension", type=int, default=100)
    result.add_argument(
        "--feature-frame",
        choices=("causal_hash_structure", "warmup_structure"),
        default="causal_hash_structure",
    )
    result.add_argument("--hash-dims", type=int, default=32)
    result.add_argument("--warmup-snapshots", type=int, default=5)
    result.add_argument("--query-positives", type=int, default=2048)
    result.add_argument("--ranking-negatives", type=int, default=100)
    result.add_argument("--query-batch-size", type=int, default=512)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--learning-rate", type=float, default=0.003)
    result.add_argument("--train-positives", type=int, default=512)
    result.add_argument("--source-validation-positives", type=int, default=2048)
    result.add_argument("--train-fanout", type=int, default=10)
    result.add_argument("--train-max-nodes", type=int, default=30000)
    result.add_argument("--train-max-edges", type=int, default=100000)
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--half-life", type=float, default=4.0)
    result.add_argument("--ridges", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    result.add_argument(
        "--methods",
        nargs="+",
        choices=MaintainedLinkScorer.METHODS,
        default=["full", "exact_incremental", "error_aware"],
    )
    result.add_argument("--drift-threshold", type=float, default=0.02)
    result.add_argument("--refresh-period", type=int, default=5)
    return result


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed != 42:
        raise ValueError("The agreed experiment fixes seed42")
    if (
        min(
            args.query_positives,
            args.ranking_negatives,
            args.train_positives,
            args.source_validation_positives,
            args.epochs,
        )
        < 1
    ):
        raise ValueError("Query/training/epoch counts must be positive")
    if args.dimension < 8 or args.warmup_snapshots < 1:
        raise ValueError("Need dimension>=8 and a nonempty warmup")
    if args.hash_dims < 1 or args.hash_dims > 32:
        raise ValueError("Static hash dimension must be between1 and32")
    if (
        args.feature_frame == "causal_hash_structure"
        and frame_settings(args)["hash_dims_effective"] < 1
    ):
        raise ValueError(
            "The input dimension must leave room beyond the warmup structure block for static hashes"
        )
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("train", "all"):
        train(args)
    if args.stage == "adapt":
        for name in args.datasets:
            adaptation(args, name)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    if args.stage == "collect":
        collect(args)


if __name__ == "__main__":
    main()
