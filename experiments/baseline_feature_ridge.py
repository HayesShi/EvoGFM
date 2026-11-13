"""Feature-only ridge control using the standard NC support and test IDs.

Uses original row-normalized inputs, no pretraining or graph propagation, and
fixed ridge 0.1. It is a simple control, not a graph foundation model.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evogfm.data import load_graph, load_protocol
from evogfm.readout import episode, fit_ridge
from . import settings


TARGETS = [
    "air-brazil",
    "air-eu",
    "air-us",
    "wikics",
    "blogcatalog",
    "deezer",
    "lastfm-asia",
    "ogbn-arxiv",
    "wiki",
    "coauthor-physics",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=settings.DATA)
    parser.add_argument("--output", type=Path, default=settings.ROOT / "feature_ridge")
    parser.add_argument("--graphs", nargs="+", default=TARGETS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    if (args.output / "scores.csv").exists():
        with (args.output / "scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["graph"], int(row["seed"]), int(row["shots"])) for row in rows}
    for name in args.graphs:
        if all(
            (name, seed, shot) in done for seed in [42, 43, 44] for shot in [1, 5, 20]
        ):
            continue
        graph = load_graph(name, args.data_root)
        protocol = load_protocol(name, "nc", args.data_root)
        features = F.normalize(graph["x"], dim=1)
        labels = graph["y"]
        for seed in [42, 43, 44]:
            for shot in [1, 5, 20]:
                if (name, seed, shot) in done:
                    continue
                support = episode(
                    labels,
                    protocol["train"],
                    torch.Generator().manual_seed(seed),
                    shots=shot,
                    queries=0,
                )[0]
                # A single fixed alpha; no test or validation tuning.
                fitted = fit_ridge(
                    features, labels, support, protocol["val"], alphas=(0.1,)
                )
                classes = labels[support].unique(sorted=True)
                predictions = torch.cat(
                    [
                        classes[
                            (
                                (features[ids].double() - fitted.mean) @ fitted.weight
                                + fitted.bias
                            ).argmax(1)
                        ]
                        for ids in protocol["test"].split(1024)
                    ]
                )
                row = dict(
                    graph=name,
                    seed=seed,
                    shots=shot,
                    accuracy_percent=100
                    * float(predictions.eq(labels[protocol["test"]]).double().mean()),
                    support_count=len(support),
                    test_count=len(protocol["test"]),
                )
                np.savez_compressed(
                    args.output / f"{name}_s{seed}_k{shot}.npz",
                    predictions=predictions.numpy(),
                    support=support.numpy(),
                    test=protocol["test"].numpy(),
                )
                rows.append(row)
                with (args.output / "scores.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)
        del graph, features, fitted
        gc.collect()


if __name__ == "__main__":
    main()
