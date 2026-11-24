"""Shared causal snapshot probes and locked predictor fits on official releases."""

import argparse
import json
from pathlib import Path
import random
import time

import joblib
from numba import njit
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from evogfm.data import canonical_edges
from evogfm.incremental import ingest_edges
from evogfm.shared_attention_cuda import CudaLinkScorer
from evogfm.three_hop_csr import three_hop_walk_counts
from evogfm.undergs_protocol import CausalNeighborhood
from experiments.db_a_execution_suite import write_json
from experiments.db_a_native import (
    DEFAULT_CHECKPOINT,
    WORKSPACE,
    ingest_history,
    replay_temporal_history,
)
from experiments.db_a_matrix import source
from experiments.db_a_quality import TREE_GRID, features, predict_readout
from experiments.db_a_shared_fast import ingest_sorted_native
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features
from experiments.temporal_official import original_undergs_modules


FREQUENCY = {
    "wikipedia": 3600,
    "reddit-body": 604800,
    "reddit-title": 604800,
    "wiki-talk": 604800,
    "stack-overflow": 30 * 86400,
}


@njit
def unpack_containers(containers, width):
    keys = np.full((len(containers), width, 3), -1, dtype=np.int64)
    weights = np.zeros((len(containers), width), dtype=np.float32)
    for i in range(len(containers)):
        for j, (key, value) in enumerate(containers[i].items()):
            keys[i, j, 0], keys[i, j, 1], keys[i, j, 2] = key
            weights[i, j] = value
    return keys, weights


class ScopedCausalNeighborhood(CausalNeighborhood):
    """Run the author's Container on touched nodes and retain its slot order."""

    def score(self, sources, destinations, negatives, times, edge_idxs, train=False):
        nodes = np.unique(np.concatenate((sources, destinations, negatives)))
        model = self.model
        outer = model.node_features[nodes].clone()
        inner = model.embedding_module.node_features[nodes].clone()
        result = model.compute_edge_probabilities(
            sources, destinations, negatives, times, edge_idxs, train=train
        )
        model.node_features[nodes] = outer
        model.embedding_module.node_features[nodes] = inner
        return result

    def update(
        self,
        source_nodes,
        update_node_list,
        current_timestamp,
        edge_idxs,
        n_samples,
        unique_edge_indices,
    ):
        if not self.observing:
            return super().update(
                source_nodes,
                update_node_list,
                current_timestamp,
                edge_idxs,
                n_samples,
                unique_edge_indices,
            )
        active = update_node_list
        size = len(active)
        store = self.store
        old_keys = store.tcns_keys[active].cpu().numpy().copy()
        weights = store.tcns_weights[active].cpu().numpy()
        mapping = np.arange(store.num_nodes, dtype=np.int64) + size
        mapping[active] = np.arange(size)
        key_neighbours = old_keys[:, :, 1]
        present = old_keys[:, :, 0] >= 0
        key_neighbours[present] = mapping[key_neighbours[present]]
        inverse = np.r_[active, np.arange(store.num_nodes)]
        container = self.util.Container(size, store.k, store.lambda_1, store.lambda_2)
        container.load_tcns(old_keys, weights)
        neighbours, edges, delta, output_weights = container.maintain(
            mapping[source_nodes].astype(np.int32),
            np.arange(size, dtype=np.int64),
            current_timestamp,
            edge_idxs,
            n_samples,
            unique_edge_indices,
        )
        used_edges = edges != 0
        neighbours[used_edges] = inverse[neighbours[used_edges]]
        new_keys, new_weights = unpack_containers(container.container_list, store.k)
        present = new_keys[:, :, 0] >= 0
        new_keys[:, :, 1][present] = inverse[new_keys[:, :, 1][present]]
        store.tcns_keys[active] = torch.from_numpy(new_keys).to(store.device)
        store.tcns_weights[active] = torch.from_numpy(new_weights).to(store.device)
        return tuple(
            torch.from_numpy(value)
            .to(store.device)
            .to(torch.long if i < 2 else torch.float32)
            for i, value in enumerate((neighbours, edges, delta, output_weights))
        )


def sample_snapshot(at_time, pool, ratio, phase, rng):
    kept = np.zeros(at_time.shape[1], dtype=bool)
    negatives = np.empty((at_time.shape[1], ratio), dtype=np.int64)
    chronological = np.argsort(at_time[0], kind="stable")
    cuts = np.r_[
        0, np.flatnonzero(np.diff(at_time[0, chronological])) + 1, len(chronological)
    ]
    for first, last in zip(cuts[:-1], cuts[1:]):
        rows = chronological[first:last]
        node = at_time[0, rows[0]]
        if not len(pool):
            break
        blocked = set(at_time[1, rows].tolist()) | {int(node)}
        allowed = (
            np.setdiff1d(pool, np.fromiter(blocked, dtype=np.int64))
            if len(pool) < 512 + len(blocked)
            else None
        )
        if allowed is not None and len(allowed) < ratio:
            if phase != "train":
                raise ValueError(
                    "A held-out positive does not have enough causal negatives"
                )
            continue
        for row in rows:
            kept[row] = True
            if allowed is not None:
                negatives[row] = rng.choice(allowed, ratio, replace=False)
                continue
            accepted, skip = [], set()
            while len(accepted) < ratio:
                for candidate in rng.choice(
                    pool, ratio - len(accepted), replace=False
                ).tolist():
                    if candidate not in blocked and candidate not in skip:
                        accepted.append(candidate)
                        skip.add(candidate)
            negatives[row] = accepted
    indices = np.flatnonzero(kept)
    return indices, negatives if kept.all() else negatives[indices]


def prepare(dataset, root, output):
    manifest, event_path, events, n = source(dataset, root)
    t0 = int(events[0, 2])
    if dataset == "stack-overflow":
        first_month = np.datetime64(t0, "s").astype("datetime64[M]")
        last_month = np.datetime64(int(events[-1, 2]), "s").astype("datetime64[M]")
        months = np.arange(first_month, last_month + 1, dtype="datetime64[M]")
        boundaries = months.astype("datetime64[s]").astype(np.int64)
        offsets = np.unique(
            np.r_[np.searchsorted(events[:, 2], boundaries), len(events)]
        )
        group_time = np.arange(len(offsets) - 1)
    else:
        time = (events[:, 2] - t0) // FREQUENCY[dataset]
        offsets = np.r_[0, np.flatnonzero(time[:-1] != time[1:]) + 1, len(time)]
        group_time = time[offsets[:-1]]
    train_end = int(np.argmin(np.abs(offsets - 0.7 * len(events))))
    test_start = int(np.argmin(np.abs(offsets - 0.8 * len(events))))
    typed = manifest.with_name("destination_nodes.npy")
    valid_kind = np.ones(n, dtype=bool)
    if typed.is_file():
        valid_kind[:] = False
        valid_kind[np.load(typed)] = True
    seen = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(42)
    stream = n > 750_000
    first_seen_path = output / "first_seen.npy"
    if stream:
        first_seen = np.full(n, len(events), dtype=np.int64)
        for start in range(0, len(events), 500_000):
            stop = min(len(events), start + 500_000)
            places = np.arange(start, stop)
            np.minimum.at(first_seen, events[start:stop, 0], places)
            np.minimum.at(first_seen, events[start:stop, 1], places)
        np.save(first_seen_path, first_seen)
    entries = []
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        at_time = np.asarray(events[start:stop, :2]).T
        interval = (
            "train" if i < train_end else "validation" if i < test_start else "test"
        )
        ratio = 100 if interval == "test" else 1
        filename = None
        positives = None
        if not stream:
            indices, negative = sample_snapshot(
                at_time, np.flatnonzero(seen & valid_kind), ratio, interval, rng
            )
            filename = f"snapshot_{i:04d}.npz"
            positives = len(indices)
            np.savez_compressed(output / filename, indices=indices, negative=negative)
        entries.append(
            dict(
                id=i,
                file=filename,
                start=int(start),
                stop=int(stop),
                time=int(group_time[i]),
                positives=positives,
                phase=interval,
                negatives_per_positive=ratio,
            )
        )
        seen[at_time] = True
        if (i + 1) % 20 == 0 or i + 1 == len(offsets) - 1:
            write_json(
                output / "status.json",
                dict(
                    phase="generating fixed causal candidates",
                    completed_snapshots=i + 1,
                    total_snapshots=len(offsets) - 1,
                ),
            )
    write_json(
        output / "result.json",
        dict(
            dataset=dataset,
            source_manifest=str(manifest),
            event_path=str(event_path),
            seed=42,
            n=n,
            snapshots=entries,
            train_stop=int(offsets[train_end]),
            test_start=int(offsets[test_start]),
            chronological_semantics="Complete native time bins, boundaries nearest cumulative 70/80 event percentiles, past-observed negative targets",
            time_bin_seconds=(
                None if dataset == "stack-overflow" else FREQUENCY[dataset]
            ),
            binning=(
                "calendar month UTC"
                if dataset == "stack-overflow"
                else "fixed period from the first native event"
            ),
            event_origin=t0,
            streamed_candidates=stream,
            candidate_seed=(
                "SeedSequence([42, snapshot ordinal])"
                if stream
                else "Single chronological seed-42 stream"
            ),
        ),
    )


def query(directory, events, row):
    observed = events[row["start"] : row["stop"]]
    if row["file"] is None:
        first = np.load(directory / "first_seen.npy", mmap_mode="r")
        pool = np.flatnonzero(first < row["start"])
        rng = np.random.default_rng(np.random.SeedSequence([42, row["id"]]))
        indices, negative = sample_snapshot(
            observed[:, :2].T, pool, row["negatives_per_positive"], row["phase"], rng
        )
    else:
        with np.load(directory / row["file"]) as pack:
            indices, negative = pack["indices"], pack["negative"]
    return observed, indices, negative


def auc_from_counts(positive, negative):
    """Count wins and half-credit ties without a candidate-length sort index."""
    values, positive_count = np.unique(positive, return_counts=True)
    negative_values, negative_count = np.unique(negative, return_counts=True)
    prefix = np.r_[0, np.cumsum(negative_count, dtype=np.int64)]
    below = np.searchsorted(negative_values, values, side="left")
    through = np.searchsorted(negative_values, values, side="right")
    twice_wins = np.sum(
        positive_count * (prefix[below] + prefix[through]), dtype=np.int64
    )
    return twice_wins / (2 * len(positive) * negative.size)


def metrics(positive, negative, destinations, negative_targets):
    ranks = np.empty(len(positive), dtype=np.int64)
    for start in range(0, len(positive), 8192):
        index = slice(start, start + 8192)
        pos = positive[index, None]
        neg = negative[index]
        ranks[index] = 1 + (
            (neg > pos)
            | ((neg == pos) & (negative_targets[index] < destinations[index, None]))
        ).sum(axis=1)
    if negative.size >= 10_000_000:
        auc = auc_from_counts(positive, negative)
    else:
        labels = np.arange(len(positive) + negative.size) < len(positive)
        auc = roc_auc_score(labels, np.r_[positive, negative.ravel()])
    return {
        "auc": float(auc),
        "mrr": float(np.mean(1 / ranks)),
        "positives": len(positive),
    }


def aggregate(rows):
    return {
        "auc": float(np.mean([row["auc"] for row in rows])),
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "snapshots": len(rows),
        "positives": sum(row["positives"] for row in rows),
    }


def candidate_batches(source, destination, negatives, batch_size=2048):
    width = negatives.shape[1] + 1
    total = len(source) * width
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        rows, columns = np.divmod(np.arange(start, stop), width)
        targets = negatives[rows, np.maximum(columns - 1, 0)]
        positive = columns == 0
        targets[positive] = destination[rows[positive]]
        yield slice(start, stop), np.stack((source[rows], targets))


def ours(directory, metadata, events, output):
    from experiments.db_a_large_readout import FeatureArchive, fit_batched

    n = metadata["n"]
    ingest = ingest_sorted_native if n > 750_000 else ingest_edges
    training = events[: metadata["train_stop"]]
    model, head, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    topology = canonical_edges(torch.from_numpy(training[:, :2].T.copy()), n)
    raw = frozen_structural_features(topology, n, 100)
    for first in metadata["snapshots"]:
        visible = canonical_edges(
            torch.from_numpy(events[: first["stop"], :2].T.copy()), n
        )
        if visible.shape[1]:
            break
    if not visible.shape[1] or first["phase"] != "train":
        raise ValueError("No nonempty training graph exists before the held-out period")
    known = (visible[0] * n + visible[1]).numpy()
    history = replay_temporal_history(n)
    ingest_history(history, events[: first["stop"]], metadata["event_origin"])
    scorer = CudaLinkScorer(
        model,
        head,
        raw,
        visible,
        resident_scores=True,
        gpu_propagation=True,
        path_counts=three_hop_walk_counts,
        tile_edges=32768,
        async_indices=True,
        host_buffers=n > 750_000,
    )
    fitted, validation = {}, {}
    train_features, train_labels = [], []
    dev_features, dev_labels = [], []
    calibration_bytes = 2 * metadata["test_start"] * 24 * np.dtype(np.float32).itemsize
    archive = (
        FeatureArchive(output / "calibration_features")
        if calibration_bytes >= 4 * 1024**3
        else None
    )
    result = []
    start = time.perf_counter()
    last_progress = start
    lock = output / "selected_readout.joblib"
    selected = None
    for row in metadata["snapshots"][first["id"] + 1 :]:
        if row["phase"] == "test" and selected is None:
            if archive is None:
                train_x, train_y = np.concatenate(train_features), np.concatenate(
                    train_labels
                )
                validation_x, validation_y = np.concatenate(
                    dev_features
                ), np.concatenate(dev_labels)
                train_features.clear()
                dev_features.clear()
            for settings in TREE_GRID:
                for half_life in (1.0, 4.0, 16.0):
                    name = f"union24_{settings['name']}_{half_life:g}"
                    config = dict(settings)
                    config.pop("name")
                    classifier = HistGradientBoostingClassifier(
                        **config, early_stopping=False, random_state=42
                    )
                    if archive is None:
                        train_input = features(
                            train_x[:, :17], train_x[:, 17:], "union24", half_life
                        )
                        test_input = features(
                            validation_x[:, :17],
                            validation_x[:, 17:],
                            "union24",
                            half_life,
                        )
                        classifier.fit(train_input, train_y)
                        auc = roc_auc_score(
                            validation_y, classifier.predict_proba(test_input)[:, 1]
                        )
                    else:
                        write_json(
                            output / "status.json",
                            dict(
                                phase="fitting frozen histogram readout",
                                candidate=name,
                                training_rows=archive.rows["train"],
                                validation_rows=archive.rows["validation"],
                                completed_candidates=len(validation),
                                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            ),
                        )
                        fit_batched(
                            classifier,
                            archive.rows["train"],
                            lambda: archive.blocks("train", half_life),
                        )
                        auc = archive.validation_auc(classifier, half_life)
                    fitted[name] = dict(
                        model=dict(family="hgb", input="union24", fitted=classifier),
                        half_life=half_life,
                        selected_id=f"union24_{settings['name']}",
                        seed=42,
                    )
                    validation[name] = float(auc)
            name = max(validation, key=validation.get)
            selected = fitted[name]
            if archive is None:
                del (
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    train_input,
                    test_input,
                )
            joblib.dump(selected, lock, compress=3)
            write_json(
                output / "validation.json", dict(selected=name, candidates=validation)
            )
        observed, indices, negatives = query(directory, events, row)
        source, destination = observed[indices, :2].T
        timestamp = (observed[0, 2] - metadata["event_origin"]) / 604800.0
        if len(indices):
            candidate_count = len(indices) * (negatives.shape[1] + 1)
            if selected is None:
                values = np.empty((candidate_count, 24), dtype=np.float32)
            else:
                probabilities = np.empty(candidate_count, dtype=np.float32)
            for location, pairs in candidate_batches(source, destination, negatives):
                base = scorer.features(torch.from_numpy(pairs)).numpy()
                statistics = history.statistics(pairs, timestamp).numpy()
                if selected is None:
                    values[location, :17] = base
                    values[location, 17:] = statistics
                else:
                    probabilities[location] = predict_readout(
                        selected, base, statistics
                    )
                if n > 750_000 and time.perf_counter() - last_progress >= 60:
                    last_progress = time.perf_counter()
                    write_json(
                        output / "status.json",
                        dict(
                            phase=row["phase"],
                            completed_snapshots=row["id"],
                            total_snapshots=len(metadata["snapshots"]),
                            scored_pairs_in_snapshot=location.stop,
                            candidates_in_snapshot=candidate_count,
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        ),
                    )
            if archive is not None and selected is None:
                archive.append(row["phase"], values)
            elif row["phase"] == "train":
                train_features.append(values)
                train_labels.append(np.arange(candidate_count) % 2 == 0)
            elif row["phase"] == "validation":
                dev_features.append(values)
                dev_labels.append(np.arange(candidate_count) % 2 == 0)
            else:
                scores = probabilities.reshape(len(indices), -1)
                result.append(
                    metrics(scores[:, 0], scores[:, 1:], destination, negatives)
                )
        ingest_history(history, observed, metadata["event_origin"])
        visible, added, known = ingest(
            visible, torch.from_numpy(observed[:, :2].T.copy()), n, known
        )
        scorer.update(visible, added)
        if (row["id"] + 1) % 10 == 0:
            write_json(
                output / "status.json",
                {
                    "phase": row["phase"],
                    "completed_snapshots": row["id"] + 1,
                    "total_snapshots": len(metadata["snapshots"]),
                    "scored_test_snapshots": len(result),
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
            )
    return dict(
        **aggregate(result),
        rows=result,
        elapsed_s=time.perf_counter() - start,
        validation=validation,
        readout=str(lock),
        temporal_history_backend=type(history).__name__,
        calibration_features=str(archive.directory) if archive else None,
    )


def undergs(directory, metadata, events, output):
    module, util, _ = original_undergs_modules(WORKSPACE / "baselines")
    n = metadata["n"]
    large_release = metadata["dataset"] != "wikipedia"
    config = argparse.Namespace(
        n_nodes=n + 1,
        n_edges=len(events) + 1,
        topk=16,
        lambda_1=0.1,
        lambda_2=0.05,
        filter_ratio=0.4 if large_release else 0.3,
        device=torch.device("cuda"),
    )
    past_attributes = Path(metadata["source_manifest"]).with_name("edge_features.npy")
    attributes = (
        np.array(np.load(past_attributes, mmap_mode="r"), copy=True)
        if past_attributes.is_file()
        else np.zeros((len(events) + 1, 1), dtype=np.float32)
    )
    if past_attributes.is_file():
        selection = attributes[1 : metadata["train_stop"] + 1]
        center, scale = selection.mean(0), selection.std(0).clip(min=1e-6)
        attributes[1:] = (attributes[1:] - center) / scale
    model = module.UnderGS(
        None, attributes, config.device, 0.1, 128, 8, "gcn", config
    ).cuda()
    adapter = ScopedCausalNeighborhood(model, util)
    learning_rate = 0.0005 if large_release else 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    criterion = torch.nn.BCELoss()
    stopper = util.EarlyStopMonitor(max_round=5)
    started = time.perf_counter()
    last_progress = started
    runtime_tensors = {
        "outer_features": model.node_features,
        "inner_features": model.embedding_module.node_features,
        "tcns_keys": model.embedding_module.tcns.tcns_keys,
        "tcns_weights": model.embedding_module.tcns.tcns_weights,
    }

    def observe(row, observed):
        index = np.arange(row["start"], row["stop"]) + 1
        adapter.observe(
            observed[:, 0] + 1,
            observed[:, 1] + 1,
            np.full(len(index), row["time"]),
            index,
        )
        model.embedding_module.tcns.snapshot_decay(config.lambda_2)

    def score(row, observed, indices, negatives, fit=False):
        nonlocal last_progress
        positive, negative = [], []
        for start in range(0, len(indices), 256):
            current = indices[start : start + 256]
            destinations = negatives[start : start + 256]
            s, d = observed[current, :2].T + 1
            times = np.full(len(current), row["time"])
            pos, neg = adapter.score(
                s,
                d,
                (destinations.T + 1).ravel(),
                times,
                row["start"] + current + 1,
                train=fit,
            )
            if fit:
                loss = criterion(pos, torch.ones_like(pos)) + criterion(
                    neg, torch.zeros_like(neg)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                positive.append(pos.cpu().numpy().reshape(-1))
                negative.append(neg.cpu().numpy().reshape(negatives.shape[1], -1).T)
            if n > 750_000 and time.perf_counter() - last_progress >= 60:
                last_progress = time.perf_counter()
                write_json(
                    output / "status.json",
                    dict(
                        phase=row["phase"],
                        completed_epochs=(
                            len(curve) if row["phase"] == "test" else epoch
                        ),
                        completed_snapshots=row["id"],
                        total_snapshots=len(metadata["snapshots"]),
                        scored_positives_in_snapshot=start + len(current),
                        positives_in_snapshot=len(indices),
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    ),
                )
        if not fit:
            return metrics(
                np.concatenate(positive),
                np.concatenate(negative),
                observed[indices, 1],
                negatives,
            )

    best, curve = None, []
    checkpoint = output / "best.pt"
    for epoch in range(50):
        model.embedding_module.tcns.reset_tcns()
        model.node_features.zero_()
        model.embedding_module.node_features.zero_()
        validation = []
        for row in metadata["snapshots"]:
            if row["phase"] == "test":
                break
            observed, indices, negatives = query(directory, events, row)
            fitting = row["phase"] == "train"
            model.train(fitting)
            if len(indices):
                with torch.set_grad_enabled(fitting):
                    measured = score(row, observed, indices, negatives, fit=fitting)
                if not fitting:
                    validation.append(measured)
            observe(row, observed)
        scheduler.step()
        auc = aggregate(validation)["auc"]
        curve.append({"epoch": epoch + 1, "validation_auc": auc})
        stop = stopper.early_stop_check(auc)
        if epoch == stopper.best_epoch:
            best = auc
            saved_runtime = {
                name: value.detach().to("cpu", copy=True)
                for name, value in runtime_tensors.items()
            }
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), runtime=saved_runtime), temporary)
            temporary.replace(checkpoint)
            del saved_runtime
        write_json(
            output / "status.json",
            dict(
                phase="training",
                completed_epochs=epoch + 1,
                validation_auc=auc,
                best_validation_auc=best,
                elapsed_s=time.perf_counter() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
        if stop:
            break
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    for name, value in runtime_tensors.items():
        value.copy_(selected["runtime"][name])
    del selected
    model.eval()
    train_s = time.perf_counter() - started
    test = []
    held_out = [row for row in metadata["snapshots"] if row["phase"] == "test"]
    with torch.no_grad():
        for index, row in enumerate(held_out, 1):
            observed, indices, negatives = query(directory, events, row)
            test.append(score(row, observed, indices, negatives))
            observe(row, observed)
            if index % 10 == 0 or index == len(held_out):
                write_json(
                    output / "status.json",
                    {
                        "phase": "test",
                        "completed_snapshots": index,
                        "total_snapshots": len(held_out),
                        "scored_candidate_pairs": 101
                        * sum(e["positives"] for e in test),
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    },
                )
    return dict(
        **aggregate(test),
        rows=test,
        training_s=train_s,
        validation_curve=curve,
        best_validation_auc=best,
        hyperparameters={
            "learning_rate": learning_rate,
            "filter_ratio": config.filter_ratio,
        },
        attributes=(
            "87 historic SNAP edge attributes, scaled on train; no accrued full-release node vectors"
            if past_attributes.is_file()
            else "zero-valued edge channel and released learned node state"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--dataset", choices=FREQUENCY, required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--candidates", type=Path, required=True)
    fit.add_argument("--method", choices=("ours", "undergs"), required=True)
    for option in (pre, fit):
        option.add_argument(
            "--output", type=lambda value: Path(value).resolve(), required=True
        )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.dataset, args.root, args.output)
        return

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((args.candidates / "result.json").read_text(encoding="utf-8"))
    events = np.load(metadata["event_path"], mmap_mode="r")
    if args.method == "ours":
        with torch.no_grad():
            report = ours(args.candidates, metadata, events, args.output)
    else:
        report = undergs(args.candidates, metadata, events, args.output)
    write_json(
        args.output / "result.json",
        dict(
            dataset=metadata["dataset"],
            seed=42,
            method=args.method,
            candidates=str(args.candidates),
            gpu=torch.cuda.get_device_name(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            **report,
        ),
    )


if __name__ == "__main__":
    main()
