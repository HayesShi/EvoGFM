"""Temporal datasets and candidate/metric parity with two public DGNN codes."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch


URLS = {
    "uci": "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
    "bitcoin-alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "bitcoin-otc": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
}
REVISIONS = {
    "undergs": "b87c16f5a5364ec2cec48bb18eca3228edc311c3",
    "decoupled": "3b96b9c509169fc1a7a258a4a52c33d9fdefdba2",
}


@dataclass
class TemporalEvents:
    source: np.ndarray
    destination: np.ndarray
    time: np.ndarray
    event_index: np.ndarray
    phase: np.ndarray
    n_nodes: int
    edge_features: np.ndarray
    metadata: dict

    @property
    def times(self):
        return np.unique(self.time)

    def events_at(self, time):
        return np.flatnonzero(self.time == time)


def prepare_dataset(dataset, protocol, data="data/temporal", baselines="baselines"):
    directory = Path(data)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / URLS[dataset].rsplit("/", 1)[-1]
    if not path.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URLS[dataset], timeout=120) as response:
            path.write_bytes(response.read())
    with gzip.open(path, "rt") as stream:
        raw = np.loadtxt(stream, delimiter=None if dataset == "uci" else ",")
    ids = np.unique(raw[:, :2].astype(np.int64))
    endpoints = np.searchsorted(ids, raw[:, :2].astype(np.int64))
    timestamps = raw[:, -1]
    # Public Bitcoin preprocessing uses np.argsort's default sorting kind.
    order = np.argsort(timestamps)
    source, destination = endpoints[order].T
    features = np.zeros((len(raw) + 1, 1), dtype=np.float32)
    if dataset != "uci":
        rating = raw[:, 2]
        features[1:, 0] = (rating - rating.min()) / max(
            1.0, rating.max() - rating.min()
        )
    provenance = "public-script reconstruction"
    if protocol == "undergs":
        if dataset == "bitcoin-otc":
            supplied = Path(baselines) / "UnderGS/data/bitcoinotc/ml_bitcoinotc.csv"
            frame = pd.read_csv(supplied)
            source, destination = frame.u.to_numpy() - 1, frame.i.to_numpy() - 1
            time = frame.ts.to_numpy().astype(np.int64)
            event_index = frame.idx.to_numpy().astype(np.int64)
            features = np.load(supplied.with_suffix(".npy")).astype(np.float32)
            provenance = "official repository supplied ml_bitcoinotc.csv and .npy"
        else:
            shifted = (
                timestamps[order]
                if dataset == "uci"
                else timestamps[order] - timestamps.min()
            )
            frequency = "D" if dataset == "uci" else "W"
            periods = pd.to_datetime(shifted, unit="s").to_period(frequency).astype(str)
            time = pd.factorize(periods)[0].astype(np.int64)
            event_index = order + 1
            if dataset == "uci":
                provenance = "paper Table1 calendar-daily 193 snapshots; missing official UCI preprocessing script; reconstructed by calendar-day/factorize"
        train_end, val_end = int(time.max() * 0.7), int(time.max() * 0.8)
    elif protocol == "decoupled":
        interval = 190080 if dataset == "uci" else 1200000
        time = (timestamps[order] // interval).astype(np.int64)
        event_index = order + 1
        distinct = np.unique(time)
        train_index = int(len(distinct) * (0.71 if dataset == "uci" else 0.7))
        val_index = int(len(distinct) * 0.8) + (2 if dataset == "uci" else 0)
        train_end, val_end = int(distinct[train_index]), int(distinct[val_index])
    else:
        raise ValueError(protocol)
    phase = np.where(
        time <= train_end, "train", np.where(time <= val_end, "validation", "test")
    )
    metadata = dict(
        dataset=dataset,
        protocol=protocol,
        source_url=URLS[dataset],
        source_file=str(path.resolve()),
        official_revision=REVISIONS[protocol],
        preprocessing_provenance=provenance,
        nodes=len(ids),
        interactions=len(source),
        snapshots=len(np.unique(time)),
        train_end=int(train_end),
        validation_end=int(val_end),
        node_index="zero-based contiguous IDs; UnderGS baseline adds official one-based offset",
        positive_semantics="directed raw interactions, including repeated events and self-links",
        snapshot_counts={
            name: int(len(np.unique(time[phase == name])))
            for name in ("train", "validation", "test")
        },
    )
    if protocol == "decoupled" and dataset == "uci":
        snapshot_times, snapshot_sizes = np.unique(time, return_counts=True)
        metadata["public_preprocessing_discrepancy"] = dict(
            csv_snapshots=len(snapshot_times),
            time_edge_map_snapshots=int(np.count_nonzero(snapshot_sizes >= 2)),
            singleton_snapshot_times=snapshot_times[snapshot_sizes < 2].tolist(),
            policy="retain public CSV snapshots and CSV-based split; original build_time_edge_map drops singleton bins, yielding 87 rather than 89 snapshots and missing keys in training",
            paper_reproduction_complete=False,
        )
    return TemporalEvents(
        np.asarray(source, np.int64),
        np.asarray(destination, np.int64),
        np.asarray(time, np.int64),
        np.asarray(event_index, np.int64),
        phase,
        len(ids),
        features,
        metadata,
    )


def undergs_destinations(source_pool, destination_pool, size, seed):
    rng = np.random.RandomState(seed)
    # This discarded source draw changes the destination RNG position.
    rng.randint(0, len(np.unique(source_pool)), size)
    return np.unique(destination_pool)[
        rng.randint(0, len(np.unique(destination_pool)), size)
    ]


def decoupled_candidates(source, destination, n_nodes, count, rng):
    unique = np.unique(source)
    repeated = np.repeat(unique, int(1.5 * count))
    target = rng.choice(n_nodes, len(repeated), replace=True)
    keys = repeated * n_nodes + target
    positive_keys = source * n_nodes + destination
    keep = ~np.isin(keys, positive_keys)
    repeated, target = repeated[keep], target[keep]
    selected = []
    for node in unique:
        indices = np.flatnonzero(repeated == node)
        if len(indices) < count:
            raise ValueError(
                "official 1.5x oversampling produced fewer than requested negatives"
            )
        selected.extend(indices[:count])
    return np.stack((repeated[selected], target[selected])), unique


def export_protocol(events, protocol, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(
        events.metadata,
        seed=int(seed),
        ranking_negatives=100 if protocol == "undergs" else 1000,
        negative_duplicates_allowed=True,
        negative_self_links_allowed=True,
        undergs_sampler="phase seed 0(validation)/2(test), reset each snapshot; positives/history not filtered",
        decoupled_sampler="1500 draws per unique source, replacement, exclude current directed positives, retain first 1000; independent seeded export RNG instead of training-dependent RNG state",
        visibility="queries exported before ingest; adapters choose and record observation policy",
        aggregation="unweighted nonempty-snapshot mean",
    )
    (directory / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        directory / "events.npz",
        source=events.source,
        destination=events.destination,
        time=events.time,
        event_index=events.event_index,
        phase=events.phase,
    )
    training = events.phase == "train"
    rng = np.random.RandomState(seed)
    index = []
    for time in events.times:
        selected = events.events_at(time)
        phase = str(events.phase[selected[0]])
        positive = np.stack((events.source[selected], events.destination[selected]))
        if protocol == "undergs":
            pool = (
                training if phase != "test" else np.ones(len(events.source), dtype=bool)
            )
            sample_seed = (
                0
                if phase == "validation"
                else 2 if phase == "test" else seed + int(time)
            )
            auc_dst = undergs_destinations(
                events.source[pool],
                events.destination[pool],
                len(selected),
                sample_seed,
            )
            count = 100 if phase != "train" else 1
            rank_dst = undergs_destinations(
                events.source[pool],
                events.destination[pool],
                len(selected) * count,
                sample_seed,
            )
            auc_negative = np.stack((positive[0], auc_dst))
            ranking_negative = np.stack((np.tile(positive[0], count), rank_dst))
            rank_sources = positive[0]
        else:
            auc_negative = np.stack(
                (positive[0], rng.randint(0, events.n_nodes, len(selected)))
            )
            if phase == "train":
                ranking_negative, rank_sources = positive[:, :0], np.unique(positive[0])
            else:
                ranking_negative, rank_sources = decoupled_candidates(
                    positive[0],
                    positive[1],
                    events.n_nodes,
                    1000,
                    rng,
                )
        file = f"snapshot_{int(time)}.npz"
        np.savez_compressed(
            directory / file,
            positive=positive,
            auc_negative=auc_negative,
            ranking_negative=ranking_negative,
            rank_sources=rank_sources,
            event_indices=selected,
        )
        index.append(
            dict(
                time=int(time),
                phase=phase,
                file=file,
                positives=len(selected),
                ranking_candidates=ranking_negative.shape[1],
            )
        )
    (directory / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def ranking_mrr(
    protocol, positive_scores, negative_scores, positive_sources, rank_sources
):
    positive_scores = np.asarray(positive_scores).reshape(-1)
    negative_scores = np.asarray(negative_scores).reshape(-1)
    if protocol == "undergs":
        ranks = []
        for index, score in enumerate(positive_scores):
            combined = np.r_[score, negative_scores[index :: len(positive_scores)]]
            order = np.argsort(-combined)
            ranks.append(int(np.flatnonzero(order == 0)[0]) + 1)
    else:
        best = np.asarray(
            [
                positive_scores[np.asarray(positive_sources) == node].max()
                for node in rank_sources
            ]
        )
        negative = negative_scores.reshape(len(rank_sources), -1)
        ranks = 1 + (negative >= best[:, None]).sum(1)
    return float(np.mean(1 / np.asarray(ranks, dtype=np.float64)))


def ranking_tie_diagnostics(
    protocol, positive_scores, negative_scores, positive_sources, rank_sources
):
    """Keep official ranks separate from deterministic and random tie controls."""
    positive_scores, negative_scores = np.asarray(positive_scores).reshape(
        -1
    ), np.asarray(negative_scores).reshape(-1)
    if protocol == "undergs":
        positive = positive_scores
        negative = negative_scores.reshape(-1, len(positive)).T
    else:
        positive = np.asarray(
            [
                positive_scores[np.asarray(positive_sources) == node].max()
                for node in rank_sources
            ]
        )
        negative = negative_scores.reshape(len(positive), -1)
    greater = (negative > positive[:, None]).sum(1)
    ties = (negative == positive[:, None]).sum(1)
    harmonic = np.r_[0.0, np.cumsum(1 / np.arange(1, negative.shape[1] + 2))]
    expected = (harmonic[greater + ties + 1] - harmonic[greater]) / (ties + 1)
    return dict(
        random_tie_expected_mrr=float(expected.mean()),
        pessimistic_mrr=float(np.mean(1 / (greater + ties + 1))),
        mean_tied_rank_mrr=float(np.mean(1 / (greater + 1 + ties / 2))),
        query_with_ties_fraction=float(np.mean(ties > 0)),
        negative_tie_fraction=float(ties.sum() / negative.size),
        zero_positive_fraction=float(np.mean(positive == 0)),
    )


class TemporalHistory:
    """Past-only directed counts, last occurrence and source activity statistics."""

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.counts, self.last = {}, {}
        self.out_count = np.zeros(n_nodes, dtype=np.float32)
        self.in_count = np.zeros(n_nodes, dtype=np.float32)
        self.last_node = np.full(n_nodes, -np.inf, dtype=np.float32)

    def ingest(self, pairs, time):
        source, destination = np.asarray(pairs, dtype=np.int64)
        keys, counts = np.unique(source * self.n + destination, return_counts=True)
        for key, count in zip(keys.tolist(), counts.tolist()):
            self.counts[key] = self.counts.get(key, 0) + count
            self.last[key] = time
        np.add.at(self.out_count, source, 1)
        np.add.at(self.in_count, destination, 1)
        self.last_node[np.unique(np.r_[source, destination])] = time

    def lookup(self, keys):
        last = np.fromiter(
            (self.last.get(int(key), -np.inf) for key in keys),
            dtype=np.float32,
            count=len(keys),
        )
        frequency = np.fromiter(
            (self.counts.get(int(key), 0) for key in keys),
            dtype=np.float32,
            count=len(keys),
        )
        return last, frequency

    def statistics(self, pairs, time):
        source, destination = np.asarray(pairs, dtype=np.int64)
        last, frequency = self.lookup(source * self.n + destination)
        reverse_last, reverse = self.lookup(destination * self.n + source)
        age = time - last
        reverse_age = time - reverse_last
        normalized = frequency / np.sqrt(
            np.maximum(1, self.out_count[source] * self.in_count[destination])
        )
        return torch.from_numpy(
            np.stack(
                (
                    age,
                    np.log1p(frequency),
                    reverse_age,
                    np.log1p(reverse),
                    np.log1p(self.out_count[source]),
                    np.log1p(self.in_count[destination]),
                    normalized,
                ),
                axis=1,
            )
        ).float()

    def channels(self, pairs, time, half_life=4.0):
        statistics = self.statistics(pairs, time)
        statistics[:, [0, 2]] = torch.exp2(-statistics[:, [0, 2]] / half_life)
        return statistics

    def replace(
        self, features, pairs, time, variant="recency_frequency", half_life=4.0
    ):
        return replace_temporal(
            features, self.statistics(pairs, time), variant, half_life
        )

    def state_bytes(self):
        """Recursive Python object sizes, including owning NumPy payloads."""
        seen = set()

        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, dict):
                total += sum(size(key) + size(item) for key, item in value.items())
            elif isinstance(value, (tuple, list)):
                total += sum(size(item) for item in value)
            return total

        return sys.getsizeof(self) + size(vars(self))


def replace_temporal(features, statistics, variant="recency_frequency", half_life=4.0):
    output = features.clone()
    temporal = statistics.clone()
    temporal[:, [0, 2]] = torch.exp2(-temporal[:, [0, 2]] / half_life)
    if variant == "recency_frequency":
        output[:, [0, 1, 2, 3, 10, 11, 12]] = temporal
    elif variant == "recency":
        output[:, [0, 1, 2]] = temporal[:, [0, 2, 6]]
    elif variant == "pair_frequency":
        output[:, 16] = temporal[:, 1]
    elif variant == "pair_frequency_recency":
        output[:, [10, 11]] = temporal[:, [1, 0]]
    elif variant != "original":
        raise ValueError(variant)
    return output
