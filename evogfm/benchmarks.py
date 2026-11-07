"""Remaining GraphAny datasets, using its published variants and masks."""

from pathlib import Path
import zipfile

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

from .data import canonical_edges, download


def graphany_masks(graph):
    """GraphAny: 20*C total stratified labels, remainder split equally (seed 42)."""
    labels = graph["y"].numpy()
    ids = np.arange(len(labels))
    train, rest = train_test_split(
        ids,
        test_size=(len(ids) - 20 * len(np.unique(labels))) / len(ids),
        random_state=42,
        stratify=labels,
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=42, stratify=labels[rest]
    )
    graph.update(
        {
            key: torch.from_numpy(value)
            for key, value in zip(("train", "val", "test"), (train, val, test))
        }
    )
    return graph


def social_archive(name, folder):
    """Recover the original 128D PyG inputs from public research caches."""
    filename = "deezer_europe.npz" if name == "deezer" else "lastfm_asia.npz"
    path = folder / filename
    if path.exists():
        return path
    if name == "lastfm-asia":
        return download(
            "https://raw.githubusercontent.com/wuyucheng2002/PPGSL/"
            "84b67e2ea709f4740a0cb7b4c11012664bc64f7b/"
            "data/LastFMAsia/raw/lastfm_asia.npz",
            path,
        )
    try:
        return download("https://graphmining.ai/datasets/ptg/" + filename, path)
    except OSError:
        pass
    try:
        import dgl
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Deezer's original host is unavailable. Install a DGL build compatible "
            "with your PyTorch to read its public graph cache, or place the original "
            f"128D deezer_europe.npz at {path}."
        ) from error
    cached = download(
        "https://raw.githubusercontent.com/Qy-ds/G-CORMoL/"
        "d298647acb960bf09483fc0fd3cdf9ba061c807f/datas/DeezerEurope.bin",
        folder / "DeezerEurope.bin",
    )
    graphs, metadata = dgl.load_graphs(str(cached))
    graph = graphs[0]
    source, target = graph.edges()
    features = graph.ndata["feat"].cpu().numpy()
    if features.shape != (28281, 128) or graph.num_edges() != 185504:
        raise ValueError("Unexpected DeezerEurope mirror dimensions")
    np.savez_compressed(
        path,
        features=features,
        target=metadata["labels"].cpu().numpy().reshape(-1),
        edges=torch.stack((source, target), dim=1).cpu().numpy(),
    )
    return path


def load_node_benchmark(name, raw):
    raw = Path(raw)
    folder = raw / name
    if name in ("reddit", "ogbn-products"):
        from .ogb_data import load_large_nodes

        return load_large_nodes(name, raw.parent)
    if name in ("air-brazil", "air-eu", "air-us"):
        country = {"air-brazil": "brazil", "air-eu": "europe", "air-us": "usa"}[name]
        base = "https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/"
        label_file = download(
            base + f"labels-{country}-airports.txt", folder / "labels.txt"
        )
        edge_file = download(
            base + f"{country}-airports.edgelist", folder / "edges.txt"
        )
        records = np.loadtxt(label_file, skiprows=1, dtype=np.int64)
        lookup = {int(node): i for i, node in enumerate(records[:, 0])}
        edges = torch.tensor(
            [
                [lookup[int(a)], lookup[int(b)]]
                for a, b in np.loadtxt(edge_file, dtype=np.int64)
            ]
        ).T
        graph = dict(
            x=torch.eye(len(records)),
            y=torch.from_numpy(records[:, 1]),
            edges=canonical_edges(edges, len(records)),
        )
    elif name == "texas4":
        path = download(
            "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/texas_4_classes.npz",
            folder / "texas_4_classes.npz",
        )
        with np.load(path) as source:
            graph = dict(
                x=torch.from_numpy(source["node_features"]).float(),
                y=torch.from_numpy(source["node_labels"]).long(),
            )
            graph["edges"] = canonical_edges(source["edges"].T, len(graph["x"]))
            for part in ("train", "val", "test"):
                masks = source[part + "_masks"]
                graph[part] = torch.from_numpy(
                    np.flatnonzero(
                        masks[0] if masks.shape[0] != len(graph["x"]) else masks[:, 0]
                    )
                )
        return graph
    elif name in ("coauthor-physics", "dblp", "full-cora"):
        filename = {
            "coauthor-physics": "ms_academic_phy.npz",
            "dblp": "dblp.npz",
            "full-cora": "cora.npz",
        }[name]
        base = (
            "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/"
            if name == "coauthor-physics"
            else "https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/"
        )
        path = download(base + filename, folder / filename)
        from torch_geometric.io import read_npz

        data = read_npz(str(path))
        graph = dict(
            x=data.x, y=data.y, edges=canonical_edges(data.edge_index, len(data.x))
        )
    elif name in ("deezer", "lastfm-asia"):
        folder.mkdir(parents=True, exist_ok=True)
        path = social_archive(name, folder)
        with np.load(path, allow_pickle=True) as source:
            graph = dict(
                x=torch.from_numpy(source["features"]).float(),
                y=torch.from_numpy(source["target"]).long(),
            )
            graph["edges"] = canonical_edges(source["edges"].T, len(graph["x"]))
    elif name in ("wiki", "blogcatalog"):
        filenames = ("attrs.npz", "edgelist.txt", "labels.txt")
        if not all((folder / filename).exists() for filename in filenames):
            identifier = {
                "wiki": "1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV",
                "blogcatalog": "178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS",
            }[name]
            path = download(
                f"https://drive.usercontent.google.com/download?id={identifier}&export=download&confirm=t",
                folder / "data.zip",
            )
            with zipfile.ZipFile(path) as archive:
                for filename in filenames:
                    matches = [
                        member
                        for member in archive.namelist()
                        if Path(member).name == filename
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Cannot identify {filename} in official archive"
                        )
                    (folder / filename).write_bytes(archive.read(matches[0]))
        features = torch.from_numpy(sp.load_npz(folder / "attrs.npz").toarray()).float()
        labels = [
            row.split()[1:]
            for row in (folder / "labels.txt").read_text().splitlines()
            if row.strip()
        ]
        if any(len(row) != 1 for row in labels):
            raise ValueError("GraphAny protocol requires single-label classification")
        edges = np.loadtxt(folder / "edgelist.txt", dtype=np.int64).T
        graph = dict(
            x=features,
            y=torch.tensor([int(row[0]) - 1 for row in labels]),
            edges=canonical_edges(edges, len(features)),
        )
    else:
        raise ValueError(f"Unsupported node benchmark: {name}")
    return graphany_masks(graph)
