"""Export original NC features and exact episodes for external environments."""

import argparse
from pathlib import Path

import numpy as np
import torch

from evogfm.data import load_graph, load_protocol
from evogfm.readout import episode
from . import settings
from .baseline_feature_ridge import TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=settings.DATA)
    parser.add_argument("--output", type=Path, default=settings.ROOT / "nc_inputs")
    parser.add_argument("--graphs", nargs="+", default=TARGETS)
    args = parser.parse_args()
    inputs, episodes = args.output / "inputs", args.output / "episodes"
    inputs.mkdir(parents=True, exist_ok=True)
    episodes.mkdir(exist_ok=True)
    for name in args.graphs:
        graph = load_graph(name, args.data_root)
        protocol = load_protocol(name, "nc", args.data_root)
        np.savez_compressed(
            inputs / f"{name}_identity.npz",
            **{key: graph[key].numpy() for key in ["x", "y", "edges"]},
            **{key: protocol[key].numpy() for key in ["train", "val", "test"]},
        )
        for seed in [42, 43, 44]:
            for shots in [1, 5, 20]:
                support = episode(
                    graph["y"],
                    protocol["train"],
                    torch.Generator().manual_seed(seed),
                    shots=shots,
                    queries=0,
                )[0]
                np.savez_compressed(
                    episodes / f"{name}_s{seed}_k{shots}.npz",
                    support=support.numpy(),
                    test=protocol["test"].numpy(),
                )
        print(f"Exported {name}: original inputs and nine episodes", flush=True)


if __name__ == "__main__":
    main()
