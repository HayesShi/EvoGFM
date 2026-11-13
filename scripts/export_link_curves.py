"""Export the measured ROC/PR curves from the original prediction artifacts."""

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

GRAPHS = ["chameleon", "cora", "computers", "squirrel"]
SEEDS = [42, 43, 44]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def query_scores(graph, seed):
    u = BASE / "unilp"
    random = load(u / "table_a1_fixed_solver" / f"predictions_{graph}_s{seed}.pt")
    mixed_protocol = load(
        u / "context_mixed_diagnostic" / f"protocol_{graph}_s{seed}_k200.pt"
    )
    assert torch.equal(random["query_pairs"], mixed_protocol["query_pairs"])
    assert random["positive_count"] == mixed_protocol["positive_count"]
    mixed_evo = load(u / "solver_diagnostic" / f"predictions_{graph}_mixed_s{seed}.pt")
    mixed_unilp = load(
        u / "context_mixed_diagnostic" / f"predictions_{graph}_s{seed}_k200.pt"
    )
    scores = {
        "EvoGFM, random support": random["evogfm_scores"],
        "EvoGFM, mixed support": mixed_evo["converged_fixed_ridge"],
        "UniLP, random support": random["unilp_scores"],
        "UniLP, mixed support": mixed_unilp["scores"],
    }
    n = int(random["positive_count"])
    assert all(len(s) == 3 * n and torch.isfinite(s).all() for s in scores.values())
    return n, scores


def interpolate_curve(kind, labels, scores, grid):
    if kind == "ROC":
        x, y, _ = roc_curve(labels, scores, drop_intermediate=False)
    else:
        precision, recall, _ = precision_recall_curve(labels, scores)
        x, y = recall[::-1], precision[::-1]
    # Retain the upper endpoint at duplicate x coordinates before interpolation.
    unique, first = np.unique(x, return_index=True)
    y = np.maximum.reduceat(y, first)
    return np.interp(grid, unique, y)


def reproduction_scores(folder, graph, seed, unilp=None):
    scores = {}
    reference = None
    for model, source in [("EvoGFM", folder), ("UniLP", unilp)]:
        if source is None:
            continue
        for mode, label in [("uniform", "random"), ("mixed", "mixed")]:
            saved = load(source / f"predictions_{graph}_{mode}_s{seed}.pt")
            if reference is None:
                reference = saved
            assert torch.equal(reference["query_pairs"], saved["query_pairs"])
            assert reference["positive_count"] == saved["positive_count"]
            scores[f"{model}, {label} support"] = saved["scores"]
    return reference["positive_count"], scores


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--source", type=Path, help="Original baseline result directory"
    )
    inputs.add_argument(
        "--predictions", type=Path, help="Output of experiments.link_prediction"
    )
    parser.add_argument(
        "--unilp", type=Path, help="UniLP predictions in the same file format"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/data/link_curves.npz")
    )
    args = parser.parse_args()
    BASE = args.source
    grid = np.linspace(0, 1, 501)
    methods = [
        "EvoGFM, random support",
        "EvoGFM, mixed support",
        "UniLP, random support",
        "UniLP, mixed support",
    ]
    if args.predictions and not args.unilp:
        methods = methods[:2]
    values = np.empty((2, 4, 2, len(methods), 3, len(grid)))
    for gi, graph in enumerate(GRAPHS):
        for si, seed in enumerate(SEEDS):
            n, scores = (
                reproduction_scores(args.predictions, graph, seed, args.unilp)
                if args.predictions
                else query_scores(graph, seed)
            )
            labels = np.r_[np.ones(n), np.zeros(n)]
            for di, start in enumerate([n, 2 * n]):
                for mi, method in enumerate(methods):
                    score = torch.cat(
                        [scores[method][:n], scores[method][start : start + n]]
                    ).numpy()
                    for ki, kind in enumerate(["ROC", "PR"]):
                        values[ki, gi, di, mi, si] = interpolate_curve(
                            kind, labels, score, grid
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        grid=grid,
        curves=values,
        graphs=GRAPHS,
        methods=methods,
        seeds=SEEDS,
        kinds=["ROC", "PR"],
        difficulties=["Random", "Hard"],
    )
    print(args.output)


if __name__ == "__main__":
    main()
