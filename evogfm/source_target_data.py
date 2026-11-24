"""Pinned temporal releases for the Source/Target deployment experiment.

All updates are retained. Empty equal-duration time bins are omitted before a
70/10/20 snapshot split. IDs are an external node registry; candidate eligibility
is restricted separately by first observation and destination type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .temporal import TemporalEvents


RELEASES = {
    "uci": ("https://snap.stanford.edu/data/CollegeMsg.txt.gz", "space"),
    "bitcoin-alpha": (
        "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
        "bitcoin",
    ),
    "bitcoin-otc": (
        "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        "bitcoin",
    ),
    "wiki-talk": ("https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz", "space"),
    "wikipedia": ("https://snap.stanford.edu/jodie/wikipedia.csv", "jodie"),
    "reddit-body": (
        "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
        "reddit",
    ),
    "reddit-title": (
        "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
        "reddit",
    ),
    "stack-overflow": (
        "https://snap.stanford.edu/data/sx-stackoverflow.txt.gz",
        "space",
    ),
    "mathoverflow": ("https://snap.stanford.edu/data/sx-mathoverflow.txt.gz", "space"),
    "askubuntu": ("https://snap.stanford.edu/data/sx-askubuntu.txt.gz", "space"),
    "superuser": ("https://snap.stanford.edu/data/sx-superuser.txt.gz", "space"),
    "email-eu": (
        "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
        "space",
    ),
}
ALIASES = {
    "uci_message": "uci",
    "bitcoin_alpha": "bitcoin-alpha",
    "bitcoin_otc": "bitcoin-otc",
    "wiki_talk": "wiki-talk",
    "wikipedia_jodie": "wikipedia",
    "stack_overflow": "stack-overflow",
    "reddit_body": "reddit-body",
    "reddit_title": "reddit-title",
    "email_eu": "email-eu",
}
PHASE_NAMES = np.array(["train", "validation", "test"])


class PhaseCodes(np.ndarray):
    """One byte per event while preserving the existing string phase API."""

    def __new__(cls, values):
        return np.asanyarray(values, dtype=np.uint8).view(cls)

    def __eq__(self, other):
        if isinstance(other, str):
            match = np.flatnonzero(PHASE_NAMES == other)
            return np.asarray(self) == (int(match[0]) if len(match) else -1)
        return np.asarray(self) == other

    def __ne__(self, other):
        return ~self.__eq__(other)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return str(PHASE_NAMES[int(value)]) if np.isscalar(value) else value


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_release(name, data_root):
    name = ALIASES.get(name, name)
    url, _ = RELEASES[name]
    raw = Path(data_root).resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / url.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size:
        return path
    # Reuse the three small releases already downloaded by the public runner.
    local = Path(__file__).resolve().parents[1] / "data/temporal" / path.name
    if local.exists():
        shutil.copyfile(local, path)
        return path
    from .release_download import download_ranges

    error = None
    for attempt in range(3):
        try:
            return download_ranges(url, path)
        except OSError as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise OSError(f"Cannot download {name} from {url}: {error}")


def _chunks(path, kind):
    common = dict(chunksize=500_000, compression="infer")
    if kind == "reddit":
        reader = pd.read_csv(path, sep="\t", usecols=[0, 1, 3], **common)
        for frame in reader:
            yield frame.iloc[:, 0].astype(str).to_numpy(), frame.iloc[:, 1].astype(
                str
            ).to_numpy(), (
                pd.to_datetime(frame.iloc[:, 2], utc=True).astype("int64").to_numpy()
                / 1e9
            )
    else:
        reader = pd.read_csv(
            path,
            sep="," if kind in ("bitcoin", "jodie") else r"\s+",
            header=0 if kind == "jodie" else None,
            comment="#",
            usecols=[0, 1, 2 if kind == "jodie" else (3 if kind == "bitcoin" else 2)],
            **common,
        )
        for frame in reader:
            yield frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy(), frame.iloc[
                :, 2
            ].to_numpy(dtype=np.float64)


def _map_ids(values, registry, namespace):
    unique, inverse = np.unique(values, return_inverse=True)
    mapped = np.empty(len(unique), dtype=np.int64)
    for i, value in enumerate(unique.tolist()):
        key = (namespace, value)
        if key not in registry:
            registry[key] = len(registry)
        mapped[i] = registry[key]
    return mapped[inverse]


def prepare_release(name, data_root, snapshots=100):
    name = ALIASES.get(name, name)
    if snapshots < 10:
        raise ValueError("at least ten bins are required for 70/10/20")
    root = Path(data_root).resolve()
    output = root / "prepared" / f"{name}_bins{snapshots}_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        return output
    begin = time.perf_counter()
    path = fetch_release(name, root)
    kind = RELEASES[name][1]
    stage = output / "staging"
    stage.mkdir(exist_ok=True)
    registry, items, counts = {}, set(), []
    print(json.dumps(dict(stage="parse", dataset=name, file=str(path))), flush=True)
    for i, (u, v, timestamp) in enumerate(_chunks(path, kind)):
        source = _map_ids(u, registry, "user" if kind == "jodie" else "node")
        destination = _map_ids(v, registry, "item" if kind == "jodie" else "node")
        if kind == "jodie":
            items.update(np.unique(destination).tolist())
        matrix = np.empty((len(source), 3), dtype=np.float64)
        matrix[:, 0], matrix[:, 1], matrix[:, 2] = source, destination, timestamp
        np.save(stage / f"part{i:04d}.npy", matrix)
        counts.append(len(matrix))
        print(
            json.dumps(
                dict(
                    stage="parsed_chunk",
                    dataset=name,
                    events=sum(counts),
                    nodes=len(registry),
                )
            ),
            flush=True,
        )
    size, n = sum(counts), len(registry)
    if size == 0:
        raise ValueError(f"empty release {name}")
    # Keep large arrays on disk; no dense all-event Unicode phase array.
    source = np.lib.format.open_memmap(
        output / "source.npy", mode="w+", dtype=np.int64, shape=(size,)
    )
    destination = np.lib.format.open_memmap(
        output / "destination.npy", mode="w+", dtype=np.int64, shape=(size,)
    )
    timestamp = np.lib.format.open_memmap(
        output / "timestamp.npy", mode="w+", dtype=np.float64, shape=(size,)
    )
    offset = 0
    for i, count in enumerate(counts):
        part = np.load(stage / f"part{i:04d}.npy", mmap_mode="r")
        source[offset : offset + count] = part[:, 0]
        destination[offset : offset + count] = part[:, 1]
        timestamp[offset : offset + count] = part[:, 2]
        offset += count
        del part
        (stage / f"part{i:04d}.npy").unlink()
    stage.rmdir()
    del registry
    monotone = bool(np.all(timestamp[1:] >= timestamp[:-1]))
    order = None if monotone else np.argsort(timestamp, kind="stable")
    if order is not None:
        for array in (source, destination, timestamp):
            array[:] = np.asarray(array)[order]
    # Stable sort leaves raw within-timestamp order unchanged.
    indices = np.arange(size, dtype=np.int64) if order is None else order
    np.save(output / "event_index.npy", indices)
    del indices, order
    low, high = float(timestamp[0]), float(timestamp[-1])
    bin_id = np.minimum(
        ((np.asarray(timestamp) - low) / max(high - low, 1e-12) * snapshots).astype(
            np.int32
        ),
        snapshots - 1,
    )
    distinct, time_id = np.unique(bin_id, return_inverse=True)
    time_id = time_id.astype(np.int16 if snapshots < 32768 else np.int32)
    times = len(distinct)
    if times < 10:
        raise ValueError(f"{name} has only {times} nonempty time bins")
    train_count, val_stop = int(times * 0.7), int(times * 0.8)
    phases = np.where(
        time_id < train_count, 0, np.where(time_id < val_stop, 1, 2)
    ).astype(np.uint8)
    np.save(output / "time.npy", time_id)
    np.save(output / "phase.npy", phases)
    first_seen = np.full(n, times, dtype=np.int32)
    np.minimum.at(first_seen, source, time_id)
    np.minimum.at(first_seen, destination, time_id)
    np.save(output / "first_seen.npy", first_seen)
    destination_pool = (
        np.array(sorted(items), dtype=np.int64)
        if kind == "jodie"
        else np.arange(n, dtype=np.int64)
    )
    np.save(output / "destination_pool.npy", destination_pool)
    offsets = np.r_[0, np.cumsum(np.bincount(time_id, minlength=times))].tolist()
    source.flush(), destination.flush(), timestamp.flush()
    metadata = dict(
        schema_version=1,
        dataset=name,
        source_url=RELEASES[name][0],
        source_file=str(path),
        source_bytes=path.stat().st_size,
        source_mtime_ns=path.stat().st_mtime_ns,
        nodes=n,
        interactions=size,
        snapshots=times,
        requested_time_bins=snapshots,
        time_binning="equal-duration, nonempty bins only; stable within-timestamp order",
        snapshot_counts=dict(
            train=train_count, validation=val_stop - train_count, test=times - val_stop
        ),
        train_end=train_count - 1,
        validation_end=val_stop - 1,
        snapshot_offsets=offsets,
        positive_semantics="directed repeated raw interactions; signed ratings are observed edges",
        state_semantics="prediction task is directed; undirected simple graph for spectral/propagation state",
        node_index="first-encounter registry; separate user/item namespaces for JODIE",
        candidate_policy="past-observed nodes of valid destination type, excluding current positives; future edges not consulted",
        event_features="not used; current ratings, sentiment, JODIE state labels and text excluded",
        preprocessing_seconds=time.perf_counter() - begin,
        raw_timestamp_range=[low, high],
        raw_was_monotone=monotone,
        storage_bytes=sum(p.stat().st_size for p in output.glob("*.npy")),
    )
    _json(manifest, metadata)
    return output


def load_events(name, data_root, snapshots=100):
    output = prepare_release(name, data_root, snapshots)
    metadata = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metadata["destination_pool"] = np.load(output / "destination_pool.npy").tolist()
    metadata["first_seen"] = np.load(output / "first_seen.npy").tolist()
    arrays = {
        key: np.load(output / f"{key}.npy", mmap_mode="r")
        for key in ("source", "destination", "time", "event_index", "phase")
    }
    return TemporalEvents(
        arrays["source"],
        arrays["destination"],
        arrays["time"],
        arrays["event_index"],
        PhaseCodes(arrays["phase"]),
        metadata["nodes"],
        np.empty((0, 0), dtype=np.float32),
        metadata,
    )


def eligible_destinations(events, time_index):
    pool = np.asarray(events.metadata["destination_pool"], dtype=np.int64)
    first = np.asarray(events.metadata["first_seen"], dtype=np.int32)
    return pool[first[pool] < time_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(RELEASES))
    parser.add_argument("--data", default="data/vldb")
    parser.add_argument("--snapshots", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    status = []
    for name in args.datasets:
        try:
            output = prepare_release(name, root, args.snapshots)
            row = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            row["status"] = "prepared"
            status.append(row)
            print(json.dumps(row), flush=True)
        except Exception as error:
            row = dict(dataset=name, status="failed", error=repr(error))
            status.append(row)
            print(json.dumps(row), flush=True)
        _json(root / "preparation_status.json", status)
    if any(row["status"] == "failed" for row in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
