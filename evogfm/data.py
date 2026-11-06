"""Graph loading, fixed evaluation protocols, and native masked-edge splits."""

from pathlib import Path
import concurrent.futures
import json
import pickle
import urllib.request
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def seed_all(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_edges(edges, n, directed=False):
    e = torch.as_tensor(edges, dtype=torch.long).cpu()
    if e.ndim != 2 or e.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    e = e[:, e[0] != e[1]]
    if not directed:
        e = torch.stack((e.min(0).values, e.max(0).values))
    return torch.stack(
        (torch.unique(e[0] * n + e[1]) // n, torch.unique(e[0] * n + e[1]) % n)
    )


def negatives(n, positive_edges, count, rng, directed=False):
    """Sampling without replacement, excluding every known positive (split creation only)."""
    e = canonical_edges(positive_edges, n, directed).numpy()
    forbidden = set((e[0] * n + e[1]).tolist())
    population = n * (n - 1) if directed else n * (n - 1) // 2
    count = min(count, population - len(forbidden))
    chosen = set()
    while len(chosen) < count:
        a = rng.integers(n, size=max(1024, 2 * (count - len(chosen))))
        b = rng.integers(n, size=len(a))
        if not directed:
            a, b = np.minimum(a, b), np.maximum(a, b)
        for u, v in zip(a.tolist(), b.tolist()):
            k = u * n + v
            if u != v and k not in forbidden:
                chosen.add(k)
                if len(chosen) == count:
                    break
    keys = np.array(sorted(chosen), dtype=np.int64)
    # Sorting alone would allocate low-node-ID negatives to calibration and
    # high-node-ID negatives to test when the caller slices this array.
    keys = keys[rng.permutation(len(keys))]
    return torch.from_numpy(np.stack((keys // n, keys % n)))


def pair_set(pos, neg, density):
    e = torch.cat((pos, neg), dim=1)
    y = torch.cat((torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])))
    # Inverse-stratum weights, normalized to mean 1.
    w = torch.cat(
        (
            torch.full((pos.shape[1],), density / max(1, pos.shape[1])),
            torch.full((neg.shape[1],), (1 - density) / max(1, neg.shape[1])),
        )
    )
    w = w / w.mean().clamp_min(1e-12)
    return {"pairs": e, "y": y, "w": w}


def make_pack(
    name,
    x,
    edge_index,
    y=None,
    seed=42,
    directed=False,
    edge_ratios=(0.60, 0.20, 0.10, 0.10),
    shot=1,
    query_cap=20000,
):
    """Native masked-edge protocol; never identified as an official OFA split."""
    x = torch.as_tensor(x).float().cpu()
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite node features")
    n = len(x)
    rng = np.random.default_rng(seed)
    e = canonical_edges(edge_index, n, directed)
    e = e[:, torch.from_numpy(rng.permutation(e.shape[1]))]
    cut = np.cumsum([int(e.shape[1] * r) for r in edge_ratios[:3]])
    s, c, v, q = torch.tensor_split(e, cut.tolist(), dim=1)
    if min(s.shape[1], c.shape[1], v.shape[1], q.shape[1]) == 0:
        raise ValueError("need at least one edge per structural split")
    neg = negatives(
        n, e, sum(min(z.shape[1], query_cap) for z in [c, v, q]), rng, directed
    )
    pop = n * (n - 1) if directed else n * (n - 1) // 2
    # Density estimated from S and its known observation rate, not query labels.
    rho = float(s.shape[1] / (edge_ratios[0] * pop))
    parts, off = [], 0
    for z in [c, v, q]:
        k = min(z.shape[1], query_cap)
        parts.append(pair_set(z[:, :k], neg[:, off : off + k], rho))
        off += k
    pack = dict(
        schema=2,
        name=name,
        x=x,
        directed=directed,
        full_edges=e,
        state_edges=s,
        state_observation_rate=edge_ratios[0],
        density=rho,
        calibration=parts[0],
        lp_val=parts[1],
        lp_test=parts[2],
        protocol="native_masked_60_20_10_10_seed42",
        seed=seed,
    )
    if y is not None:
        y = torch.as_tensor(y).long().view(-1).cpu()
        if len(y) != n:
            raise ValueError("node label count mismatch")
        shots, val, test = [], [], []
        for cls in y.unique().tolist():
            if cls < 0:
                continue
            ids = torch.where(y == cls)[0].numpy()
            ids = rng.permutation(ids)
            k = min(shot, len(ids))
            nv = max(1, int(0.1 * (len(ids) - k)))
            shots.extend(ids[:k])
            val.extend(ids[k : k + nv])
            test.extend(ids[k + nv :])
        pack.update(
            node_y=y,
            nc_shots=torch.tensor(shots, dtype=torch.long),
            nc_val=torch.tensor(val, dtype=torch.long),
            nc_test=torch.tensor(test, dtype=torch.long),
        )
    return pack


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def planetoid(name, raw_root):
    directory = Path(raw_root) / name
    directory.mkdir(parents=True, exist_ok=True)
    suffixes = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]

    def fetch(suffix):
        filename = "ind." + name + "." + suffix
        dest = directory / filename
        if not dest.exists():
            url = (
                "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/"
                + filename
            )
            with urllib.request.urlopen(url, timeout=50) as response:
                data = response.read()
            dest.with_suffix(dest.suffix + ".download").write_bytes(data)
            dest.with_suffix(dest.suffix + ".download").replace(dest)
        return suffix

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in tqdm(
            concurrent.futures.as_completed([pool.submit(fetch, s) for s in suffixes]),
            total=8,
            desc="raw " + name,
        ):
            future.result()

    def read(suffix):
        with (directory / ("ind." + name + "." + suffix)).open("rb") as stream:
            return pickle.load(stream, encoding="latin1")

    x, tx, allx = [read(s) for s in ["x", "tx", "allx"]]
    y, ty, ally = [read(s) for s in ["y", "ty", "ally"]]
    test_reorder = np.loadtxt(
        directory / ("ind." + name + ".test.index"), dtype=np.int64
    )
    test_sorted = np.sort(test_reorder)
    if name == "citeseer":
        full_range = np.arange(test_sorted.min(), test_sorted.max() + 1)
        expanded = sp.lil_matrix((len(full_range), x.shape[1]), dtype=tx.dtype)
        expanded[test_sorted - full_range.min()] = tx
        tx = expanded
        expanded_y = np.zeros((len(full_range), y.shape[1]), dtype=ty.dtype)
        expanded_y[test_sorted - full_range.min()] = ty
        ty = expanded_y
    features = sp.vstack([allx, tx]).tolil()
    features[test_reorder] = features[test_sorted]
    labels = np.vstack([ally, ty])
    labels[test_reorder] = labels[test_sorted]
    graph = read("graph")
    edges = torch.tensor([(u, v) for u, vs in graph.items() for v in vs]).T
    return dict(
        x=torch.from_numpy(features.toarray()).float(),
        y=torch.from_numpy(labels.argmax(1)),
        edges=canonical_edges(edges, features.shape[0]),
    )


def download(url, destination):
    """Cache a public data file and show its download progress."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url, timeout=90) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        with (
            temporary.open("wb") as output,
            tqdm(
                total=total, unit="B", unit_scale=True, desc=destination.name
            ) as progress,
        ):
            while chunk := response.read(256 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)
    return destination


def load_graph(name, data_root="data"):
    """Load original numerical features; cache graphs separately from results."""
    data_root = Path(data_root)
    cache = data_root / "graphs" / f"{name}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    raw = data_root / "raw"
    if name in ["cora", "citeseer", "pubmed"]:
        graph = planetoid(name, raw)
        classes = len(graph["y"].unique())
        train_count = 20 * classes
        graph.update(
            train=torch.arange(train_count),
            val=torch.arange(train_count, train_count + 500),
            test=torch.from_numpy(
                np.loadtxt(raw / name / f"ind.{name}.test.index", dtype=np.int64)
            ),
        )
    elif name in ["texas", "wisconsin"]:
        base = (
            "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/"
            + name
        )
        nodes_file = download(
            base + "/out1_node_feature_label.txt",
            raw / name / "out1_node_feature_label.txt",
        )
        edges_file = download(
            base + "/out1_graph_edges.txt", raw / name / "out1_graph_edges.txt"
        )
        records = [line.split("\t") for line in nodes_file.read_text().splitlines()[1:]]
        nodes = {
            int(row[0]): (list(map(float, row[1].split(","))), int(row[2]))
            for row in records
        }
        ids = sorted(nodes)
        index = {node: i for i, node in enumerate(ids)}
        links = [line.split("\t") for line in edges_file.read_text().splitlines()[1:]]
        edges = torch.tensor([[index[int(a)], index[int(b)]] for a, b in links]).T
        graph = dict(
            x=torch.tensor([nodes[i][0] for i in ids]),
            y=torch.tensor([nodes[i][1] for i in ids]),
            edges=canonical_edges(edges, len(ids)),
        )
    elif name == "photo":
        filename = "amazon_electronics_photo.npz"
        path = download(
            "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/"
            + filename,
            raw / name / filename,
        )
        with np.load(path) as source:
            x = sp.csr_matrix(
                (source["attr_data"], source["attr_indices"], source["attr_indptr"]),
                shape=source["attr_shape"],
            )
            a = sp.csr_matrix(
                (source["adj_data"], source["adj_indices"], source["adj_indptr"]),
                shape=source["adj_shape"],
            )
            graph = dict(
                x=torch.from_numpy(x.toarray()).float(),
                y=torch.from_numpy(source["labels"]).long(),
                edges=canonical_edges(
                    torch.from_numpy(np.stack(a.nonzero())), x.shape[0]
                ),
            )
    elif name == "roman-empire":
        path = download(
            "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/roman_empire.npz",
            raw / name / "roman_empire.npz",
        )
        with np.load(path) as source:
            graph = dict(
                x=torch.from_numpy(source["node_features"]).float(),
                y=torch.from_numpy(source["node_labels"]).long(),
                edges=canonical_edges(source["edges"].T, len(source["node_labels"])),
            )
            # Column zero is the saved seed-42 GraphAny mask used by this pilot.
            for part in ["train", "val", "test"]:
                masks = torch.from_numpy(source[part + "_masks"])
                mask = masks[:, 0] if masks.shape[0] == len(graph["x"]) else masks[0]
                graph[part] = torch.where(mask)[0]
    else:
        raise ValueError(f"Unsupported graph: {name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, cache)
    return graph


def ncn_split(graph):
    """NCN's 70/10/20 public protocol, using its original PyG split operations.

    Source: https://github.com/GraphPKU/NeuralCommonNeighbor/blob/main/ogbdataset.py
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected, train_test_split_edges

    def unique_edges(edges):
        edges = to_undirected(edges)
        return edges[:, edges[0] < edges[1]].T

    seed_all(42)
    edges = torch.cat((graph["edges"], graph["edges"].flip(0)), dim=1)
    # PyG's Planetoid reader coalesces by destination. The split RNG acts on
    # this order, so row-sorting the same graph would change every positive set.
    edges = edges[:, (edges[1] * len(graph["x"]) + edges[0]).argsort()]
    data = Data(x=graph["x"], edge_index=edges, num_nodes=len(graph["x"]))
    data = train_test_split_edges(data, 0.2, 0.2)
    count = int(data.val_pos_edge_index.shape[1] * 0.1 / 0.2)
    order = torch.randperm(data.val_pos_edge_index.shape[1])
    val = data.val_pos_edge_index[:, order]
    return dict(
        train=dict(
            edge=unique_edges(
                torch.cat((data.train_pos_edge_index, val[:, :-count]), dim=1)
            )
        ),
        valid=dict(
            edge=unique_edges(val[:, -count:]),
            edge_neg=unique_edges(data.val_neg_edge_index),
        ),
        test=dict(
            edge=unique_edges(data.test_pos_edge_index),
            edge_neg=unique_edges(data.test_neg_edge_index),
        ),
    )


def load_protocol(name, task, data_root="data"):
    path = Path(data_root) / "protocols" / f"{name}_{task}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    graph = load_graph(name, data_root)
    if task == "nc":
        if "train" not in graph:
            raise ValueError(f"{name} has no public NC masks in this pilot")
        protocol = {key: graph[key] for key in ["train", "val", "test", "edges"]}
        protocol["seed"] = 42
    elif task == "lp":
        protocol = ncn_split(graph)
    else:
        raise ValueError("task must be nc or lp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(protocol, path)
    return protocol
