"""Read official OGB archives directly, without duplicating the extracted data."""

from pathlib import Path
from contextlib import contextmanager
import gzip
import json
import zipfile
import tarfile

import numpy as np
import scipy.sparse as sp
import torch

from .data import canonical_edges, download


def location(data_root, name):
    """Optional data/locations.json maps names to already downloaded raw folders."""
    path = Path(data_root) / "locations.json"
    locations = (
        json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    )
    return Path(locations[name]) if name in locations else None


@contextmanager
def official_files(name, data_root):
    alias = {
        "ogbl-ppa": "ppassoc",
        "ogbl-citation2": "citation-v2",
        "ogbl-collab": "collab",
        "ogbl-ddi": "ddi",
        "ogbn-products": "products",
    }[name]
    existing = location(data_root, name)
    archive = None
    if existing is None:
        family = "nodeproppred" if name.startswith("ogbn") else "linkproppred"
        path = download(
            f"https://snap.stanford.edu/ogb/data/{family}/{alias}.zip",
            Path(data_root) / "raw/ogb" / f"{alias}.zip",
        )
        archive = zipfile.ZipFile(path)

    def open_file(relative):
        return (
            (existing / relative).open("rb")
            if archive is None
            else archive.open(alias + "/" + relative)
        )

    try:
        yield open_file
    finally:
        if archive is not None:
            archive.close()


def csv_array(open_file, relative, dtype):
    with open_file(relative) as source, gzip.GzipFile(fileobj=source) as stream:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None:
            return pd.read_csv(stream, header=None, dtype=dtype).values.squeeze()
        return np.loadtxt(stream, delimiter=",", dtype=dtype)


def load_ogb_links(name, data_root="data"):
    """Original OGB graph and train/valid/test pairs, including citation2's targets."""
    split_name = (
        "target"
        if name == "ogbl-ddi"
        else ("throughput" if name == "ogbl-ppa" else "time")
    )
    with official_files(name, data_root) as open_file:
        splits = {}
        for part in ("train", "valid", "test"):
            with open_file(f"split/{split_name}/{part}.pt") as source:
                splits[part] = {
                    key: torch.as_tensor(value)
                    for key, value in torch.load(
                        source, map_location="cpu", weights_only=False
                    ).items()
                }
        if name == "ogbl-ddi":
            n = int(csv_array(open_file, "raw/num-node-list.csv.gz", np.int64))
            edges = canonical_edges(splits["train"]["edge"].T, n)
            degree = torch.bincount(edges.flatten(), minlength=n)
            x = (degree.clamp_max(63)[:, None] == torch.arange(64)).float()
        else:
            x = torch.from_numpy(
                csv_array(open_file, "raw/node-feat.csv.gz", np.float32)
            )
            train = splits["train"]
            e = (
                torch.stack((train["source_node"], train["target_node"]))
                if "source_node" in train
                else train["edge"].T
            )
            edges = canonical_edges(e, len(x))
    return dict(x=x, edges=edges), splits


def load_large_nodes(name, data_root):
    if name == "reddit":
        folder = location(data_root, name)
        if folder is None:
            raise FileNotFoundError(
                "Set data/locations.json['reddit'] to the original PyG raw folder"
            )
        with np.load(folder / "reddit_data.npz") as source:
            graph = dict(
                x=torch.from_numpy(source["feature"]).float(),
                y=torch.from_numpy(source["label"]).long(),
            )
            for label, code in (("train", 1), ("val", 2), ("test", 3)):
                graph[label] = torch.from_numpy(
                    np.flatnonzero(source["node_types"] == code)
                )
        matrix = sp.load_npz(folder / "reddit_graph.npz").tocoo()
        graph["edges"] = canonical_edges(
            np.stack((matrix.row, matrix.col)), len(graph["x"])
        )
        return graph
    with official_files(name, data_root) as open_file:
        graph = dict(
            x=torch.from_numpy(
                csv_array(open_file, "raw/node-feat.csv.gz", np.float32)
            ),
            y=torch.from_numpy(csv_array(open_file, "raw/node-label.csv.gz", np.int64)),
        )
        edges = torch.from_numpy(csv_array(open_file, "raw/edge.csv.gz", np.int64)).T
        graph["edges"] = canonical_edges(edges, len(graph["x"]))
        for part in ("train", "valid", "test"):
            graph["val" if part == "valid" else part] = torch.from_numpy(
                csv_array(open_file, f"split/sales_ranking/{part}.csv.gz", np.int64)
            )
    return graph


class ChunkReader:
    """Sequential stream of already verified one-MiB archive range files."""

    def __init__(self, folder):
        self.paths = iter(sorted(Path(folder).glob("*.part")))
        self.buffer = b""

    def read(self, count):
        while len(self.buffer) < count:
            path = next(self.paths, None)
            if path is None:
                break
            self.buffer += path.read_bytes()
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result


def heart_candidates(path, name, split, data_root="data"):
    """Load released candidates and the released PPA positive indices, in order."""
    path = Path(path)
    reader = ChunkReader(path) if path.is_dir() else path.open("rb")
    candidates, indices = None, None
    target = f"dataset/{name}/heart_{split}_samples.npy"
    index_name = f"dataset/{name}/{split}_samples_index.pt"
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if member.name.lstrip("./") == target:
                    with archive.extractfile(member) as source:
                        # The compressed tar member has no OS file descriptor.
                        class ArrayStream:
                            def read(self, count=-1):
                                return source.read(count)

                        candidates = np.lib.format.read_array(
                            ArrayStream(), allow_pickle=False
                        )
                elif member.name.lstrip("./") == index_name:
                    import io

                    with archive.extractfile(member) as source:
                        indices = torch.load(
                            io.BytesIO(source.read()), weights_only=False
                        )
                if candidates is not None and (
                    name != "ogbl-ppa" or indices is not None
                ):
                    break
    finally:
        if not path.is_dir():
            reader.close()
    if candidates is None or (name == "ogbl-ppa" and indices is None):
        raise FileNotFoundError(
            f"Official {name} {split} candidates/indices missing from archive"
        )
    return candidates, indices
