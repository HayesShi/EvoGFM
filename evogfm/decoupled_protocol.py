"""Released-code data and evaluation conventions for DecoupledDGNN."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from evogfm.temporal import prepare_dataset


def prepare_released_events(dataset, data="data/temporal", baselines="baselines"):
    """Use the released map's >=2-edge rule and its CSV-based split endpoints."""
    events = prepare_dataset(dataset, "decoupled", data=data, baselines=baselines)
    times, counts = np.unique(events.time, return_counts=True)
    retained = np.isin(events.time, times[counts >= 2])
    order = np.flatnonzero(retained)
    order = order[np.argsort(events.event_index[order], kind="stable")]
    metadata = dict(events.metadata)
    metadata.update(
        raw_interactions=len(events.source),
        raw_csv_snapshots=len(times),
        dropped_singleton_times=times[counts < 2].tolist(),
        dropped_event_indices=events.event_index[~retained].tolist(),
        interactions=len(order),
        snapshots=len(np.unique(events.time[order])),
        snapshot_counts={
            phase: len(np.unique(events.time[order][events.phase[order] == phase]))
            for phase in ("train", "validation", "test")
        },
        preprocessing_policy="released build_time_edge_map >=2 edges; original raw-row order within each snapshot",
        split_policy="released get_data_transductive endpoints computed on unfiltered CSV, then restrict execution to existing time-map keys",
        singleton_csv_map_repair=bool(np.any(~retained)),
        author_processed_artifact_available=False,
    )
    return replace(
        events,
        source=events.source[order],
        destination=events.destination[order],
        time=events.time[order],
        event_index=events.event_index[order],
        phase=events.phase[order],
        metadata=metadata,
    )


def first_source_negatives(sources, count=1000):
    unique, counts = np.unique(sources, return_counts=True)
    if np.any(counts < count):
        raise ValueError("Released 1.5x oversampling has insufficient negatives")
    starts = np.cumsum(counts) - counts
    selected = (starts[:, None] + np.arange(count)[None, :]).reshape(-1)
    if not np.array_equal(sources[selected], np.repeat(unique, count)):
        raise ValueError("Released candidates must retain source-major order")
    return selected, unique


def released_auc_indices(n_events, batch_size=1024):
    """Retain Execute.test's 2*batch_size count and batch_size slicing."""
    n_batches = int(np.ceil(n_events / (2 * batch_size)))
    return np.concatenate(
        [
            np.arange(i * batch_size, min(n_events, (i + 1) * batch_size))
            for i in range(n_batches)
        ]
    )
