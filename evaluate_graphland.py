"""GraphLand AP future probes and scalar regression mechanism controls."""

import argparse
from pathlib import Path
import time
import torch
from evogfm.data import save_json, seed_all
from evogfm.graphland import load_graphland
from evogfm.features import (
    global_svd_features,
    node_representations,
    ordinary_representations,
)
from evogfm.linear_readout import fit_linear, predict_linear, metric_score
from evogfm.training import load_checkpoint


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    model, _, checkpoint = load_checkpoint(args.checkpoint)
    model.requires_grad_(False)
    initial = model.inverse.theta().detach().clone()
    snapshots = (
        torch.load(Path(args.states) / "nc_states.pt", weights_only=False)
        if args.states
        else None
    )
    output = Path(args.output)
    rows = []
    for name in args.datasets:
        graph = load_graphland(name, args.data_root)
        raw = graph["x"]
        x = global_svd_features(raw)
        state = model.state(x, graph["edges"])
        metric = "r2" if graph["task"] == "regression" else "ap"
        groups = (
            snapshots
            if snapshots is not None
            else {kind: [None] for kind in ["full", "zero-inverse", "propagation"]}
        )
        measured = {}
        for kind, states in groups.items():
            for prefix, snapshot in enumerate(states):
                begin = time.perf_counter()
                theta = (
                    torch.as_tensor(snapshot["theta"]).float()
                    if snapshot is not None
                    else initial
                )
                key = (theta.numpy().tobytes(), kind if snapshots is None else "future")
                if key in measured:
                    original = measured[key]
                    row = dict(
                        original,
                        control=kind,
                        prefix=prefix,
                        seconds=0.0,
                        reused_from=[original["control"], original["prefix"]],
                    )
                    rows.append(row)
                    save_json(
                        output / "results.json",
                        dict(
                            seed=42,
                            checkpoint=args.checkpoint,
                            states=args.states,
                            source_epoch=checkpoint["epoch"],
                            split="RL",
                            shared_updates=0,
                            rows=rows,
                        ),
                    )
                    print("REUSED", name, kind, prefix, flush=True)
                    continue
                model.inverse.set_theta(
                    torch.zeros_like(initial) if kind == "zero-inverse" else theta
                )
                with torch.no_grad():
                    hops = 2 if snapshots is not None else 4
                    if kind == "propagation":
                        parts, values = ordinary_representations(
                            raw, graph["edges"], hops=hops
                        )
                    else:
                        parts, values = node_representations(
                            model, state, raw, graph["edges"], hops=hops
                        )
                    features = torch.cat(parts + [values], dim=1)
                    fitted, candidates = fit_linear(
                        features, graph["y"], graph["train"], graph["val"], metric
                    )
                    prediction = predict_linear(features, fitted)
                row = dict(
                    dataset=name,
                    control=kind,
                    prefix=prefix,
                    metric=metric,
                    validation_percent=100
                    * metric_score(prediction, graph["y"], graph["val"], metric),
                    test_percent=100
                    * metric_score(prediction, graph["y"], graph["test"], metric),
                    alpha=fitted["alpha"],
                    candidates=candidates,
                    seconds=time.perf_counter() - begin,
                    nodes=len(raw),
                    edges=graph["edges"].shape[1],
                    dimensions=features.shape[1],
                    training_labels=len(graph["train"]),
                    preprocessing=graph["preprocessing"],
                )
                rows.append(row)
                measured[key] = row
                save_json(
                    output / "results.json",
                    dict(
                        seed=42,
                        checkpoint=args.checkpoint,
                        states=args.states,
                        source_epoch=checkpoint["epoch"],
                        split="RL",
                        shared_updates=0,
                        rows=rows,
                    ),
                )
                print("RESULT", row, flush=True)
        del graph, state, features, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--states", help="Saved NC stream; absent means static mechanism ablations"
    )
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=3)
    evaluate(parser.parse_args())
