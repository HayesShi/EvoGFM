"""Read the official temporal graph releases without time binning or edge filtering."""

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import struct
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd

from experiments.db_a_native_trace import identity, write_json

SOURCES = {
    "wikipedia": (157474, 9227, "https://doi.org/10.5281/zenodo.7008205", "CC-BY-4.0"),
    "reddit-body": (
        286561,
        27863,
        "https://snap.stanford.edu/data/soc-RedditHyperlinks.html",
        None,
    ),
    "reddit-title": (
        571927,
        43695,
        "https://snap.stanford.edu/data/soc-RedditHyperlinks.html",
        None,
    ),
    "wiki-talk": (
        7833140,
        1140149,
        "https://snap.stanford.edu/data/wiki-talk-temporal.html",
        None,
    ),
    "stack-overflow": (
        63497050,
        2601977,
        "https://snap.stanford.edu/data/sx-stackoverflow.html",
        None,
    ),
}
ROLAND_MD5 = {
    "reddit-body": "4ed2bb05718498b66450a10216e29b86",
    "reddit-title": "e1babc01eb264be47cbab4fe134546de",
}


def chunks(dataset, source):
    if dataset == "wikipedia":
        if source.suffix == ".range":
            data = source.read_bytes()
            header = struct.unpack("<4s5H3I2H", data[:30])
            length, extra = header[-2:]
            if (
                data[30 : 30 + length]
                != b"TG_network_datasets/wikipedia/ml_wikipedia.csv"
            ):
                raise ValueError("Unexpected Zenodo archive member")
            payload = zlib.decompress(data[30 + length + extra :], wbits=-15)
            if len(payload) != 5_623_675 or zlib.crc32(payload) != 0x00EB2865:
                raise ValueError(
                    "Zenodo member failed its published archive CRC/length check"
                )
            stream = io.BytesIO(payload)
            for frame in pd.read_csv(
                stream, usecols=["u", "i", "ts"], chunksize=200_000
            ):
                yield frame[["u", "i"]].to_numpy(), frame["ts"].to_numpy()
        else:
            with ZipFile(source) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith("/wikipedia.csv")
                ]
                if len(members) != 1:
                    raise ValueError(
                        f"Expected one Wikipedia source CSV, found {members}"
                    )
                with archive.open(members[0]) as stream:
                    for frame in pd.read_csv(
                        stream, usecols=[0, 1, 2], chunksize=200_000
                    ):
                        yield frame.iloc[:, :2].to_numpy(), frame.iloc[:, 2].to_numpy()
    elif dataset.startswith("reddit-"):
        # SNAP stores POST_ID before TIMESTAMP.
        for frame in pd.read_csv(
            source, sep="\t", usecols=[0, 1, 3], chunksize=200_000
        ):
            times = pd.to_datetime(frame.iloc[:, 2], utc=True).astype("int64") // 10**9
            yield frame.iloc[:, :2].to_numpy(), times.to_numpy()
    else:
        with gzip.open(source, "rb") as stream:
            for frame in pd.read_csv(
                stream,
                sep=r"\s+",
                comment="#",
                header=None,
                dtype=np.int64,
                usecols=[0, 1, 2],
                chunksize=500_000,
            ):
                yield frame.iloc[:, :2].to_numpy(), frame.iloc[:, 2].to_numpy()


def prepare(dataset, source, output):
    count, nodes, url, license_name = SOURCES[dataset]
    digest = None
    if dataset in ROLAND_MD5:
        md5 = hashlib.md5()
        with source.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                md5.update(block)
        digest = md5.hexdigest()
        if digest != ROLAND_MD5[dataset]:
            raise ValueError(
                "Official SNAP download did not match the ROLAND release checksum"
            )
    raw = np.lib.format.open_memmap(
        output / "raw_events.npy", mode="w+", dtype=np.int64, shape=(count, 3)
    )
    registry = {}
    source_nodes = set()
    # SNAP integer IDs can be indexed directly, while subreddits are names.
    encode = dataset == "wikipedia" or dataset.startswith("reddit-")
    stop = 0
    for endpoints, times in chunks(dataset, source):
        start, stop = stop, stop + len(times)
        if stop > count:
            raise ValueError(f"Release exceeds the expected {count} events")
        if encode:
            for row, (u, v) in enumerate(endpoints, start):
                source_key, destination_key = (
                    (("user", u), ("item", v)) if dataset == "wikipedia" else (u, v)
                )
                raw[row, 0] = registry.setdefault(source_key, len(registry))
                raw[row, 1] = registry.setdefault(destination_key, len(registry))
                if dataset.startswith("reddit-"):
                    source_nodes.add(source_key)
        else:
            raw[start:stop, :2] = endpoints
        if not np.all(np.asarray(times, dtype=np.int64) == times):
            raise ValueError(
                "Subsecond timestamps require integer storage at finer precision"
            )
        raw[start:stop, 2] = times
        write_json(
            output / "status.json",
            {
                "phase": "reading release",
                "total_events": count,
                "completed_events": stop,
            },
        )
    if stop != count:
        raise ValueError(f"Release contains {stop} events, expected {count}")
    ids = np.arange(len(registry)) if encode else np.union1d(raw[:, 0], raw[:, 1])
    if len(ids) != nodes and len(source_nodes) != nodes:
        raise ValueError(f"Release contains {len(ids)} nodes, expected {nodes}")
    order = np.argsort(raw[:, 2], kind="stable")
    events_path = output / "events.npy"
    events = np.lib.format.open_memmap(
        events_path, mode="w+", dtype=np.int64, shape=raw.shape
    )
    for start in range(0, count, 500_000):
        stop = min(count, start + 500_000)
        events[start:stop] = raw[order[start:stop]]
        if not encode:
            events[start:stop, :2] = np.searchsorted(ids, events[start:stop, :2])
    events.flush()
    if dataset == "wikipedia":
        item_nodes = np.array(
            [value for key, value in registry.items() if key[0] == "item"]
        )
        np.save(output / "destination_nodes.npy", item_nodes)
    if dataset.startswith("reddit-"):
        auxiliary = np.lib.format.open_memmap(
            output / "raw_edge_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 87),
        )
        offset = 0
        for frame in pd.read_csv(
            source,
            sep="\t",
            usecols=["LINK_SENTIMENT", "PROPERTIES"],
            chunksize=100_000,
        ):
            props = np.stack(
                [
                    np.fromstring(text, dtype=np.float32, sep=",")
                    for text in frame["PROPERTIES"]
                ]
            )
            stop = offset + len(frame)
            auxiliary[offset:stop, 0] = frame["LINK_SENTIMENT"]
            auxiliary[offset:stop, 1:] = props
            offset = stop
        attrs = np.lib.format.open_memmap(
            output / "edge_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count + 1, 87),
        )
        for first in range(0, count, 500_000):
            last = min(count, first + 500_000)
            attrs[first + 1 : last + 1] = auxiliary[order[first:last]]
        attrs.flush()
        del auxiliary
        (output / "raw_edge_features.npy").unlink()
    write_json(
        output / "manifest.json",
        dict(
            dataset_key=dataset,
            source={
                **identity(source),
                "url": url,
                "license": license_name,
                "member": (
                    "wikipedia/ml_wikipedia.csv" if source.suffix == ".range" else None
                ),
                "roland_release_md5": digest,
            },
            statistics=dict(
                raw_events=count,
                vertices=len(ids),
                reported_vertices=nodes,
                unique_source_vertices=len(source_nodes) if source_nodes else None,
                timestamp_min=int(events[0, 2]),
                timestamp_max=int(events[-1, 2]),
            ),
            chronology="Stable sort by native timestamp. Directed history retains every released event.",
            arrays={"events": {"path_from_dataset_dir": events_path.name}},
        ),
    )
    del raw
    (output / "raw_events.npy").unlink()
    write_json(
        output / "status.json",
        {"phase": "complete", "total_events": count, "completed_events": count},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=lambda value: Path(value).resolve(), required=True
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.dataset, args.source, args.output)


if __name__ == "__main__":
    main()
