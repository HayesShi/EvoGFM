"""Download and cache graphs and the public evaluation splits."""

import argparse
from evogfm.data import load_graph, load_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pubmed", "photo", "texas", "cora", "citeseer", "roman-empire"],
    )
    args = parser.parse_args()
    for name in args.datasets:
        graph = load_graph(name, args.data_root)
        if "train" in graph:
            load_protocol(name, "nc", args.data_root, graph=graph)
        if name in ["cora", "citeseer", "pubmed"]:
            load_protocol(name, "lp", args.data_root, graph=graph)
        print(
            f"{name}: {len(graph['x'])} nodes, {graph['edges'].shape[1]} undirected edges",
            flush=True,
        )
