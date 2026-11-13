"""Label-free inputs for featureless graphs and matched airport diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evogfm.data import load_graph, load_protocol
from evogfm.features import preprocess, node_representations
from evogfm.readout import episode, fit_ridge
from .full_graph import load_checkpoint
from . import settings


def structural_input(edges, n, policy):
    """Encode degrees with a fixed 64-bin rule, independent of all labels."""
    degree = torch.bincount(edges.flatten(), minlength=n)
    if policy == "degree64":
        bins = degree.clamp_max(63)
    elif policy == "degree_quantile64":
        values, inverse, counts = degree.unique(
            sorted=True, return_inverse=True, return_counts=True
        )
        del values
        # Equal degrees share a bin. Mid-ranks retain high-degree distinctions
        # that a fixed cap of 63 removes; graph labels never enter this rule.
        midpoint = counts.cumsum(0).double() - counts.double() / 2
        bins = (64 * midpoint[inverse] / n).long().clamp_max(63)
    else:
        raise ValueError(policy)
    return F.one_hot(bins, num_classes=64).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=settings.DATA)
    parser.add_argument("--checkpoint", type=Path, default=settings.CHECKPOINT)
    parser.add_argument("--output", type=Path, default=settings.ROOT / "airport_inputs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.output / "inputs"
    inputs.mkdir(exist_ok=True)
    design = dict(
        graphs=["air-brazil", "air-eu", "air-us"],
        seeds=[42, 43, 44],
        shots=[1, 5, 20],
        policies=["identity", "degree64", "degree_quantile64"],
        ridge=0.1,
        selection="Fixed protocols; no test-based policy or regularization selection",
        checkpoint=str(args.checkpoint),
        representation="Full EvoGFM two-hop and Transformer",
        control="Raw input ridge with the same supports and fixed regularization",
    )
    (args.output / "design.json").write_text(json.dumps(design, indent=2))
    torch.set_num_threads(2)
    model, _, _ = load_checkpoint(args.checkpoint)
    rows, audits = [], []
    for name in design["graphs"]:
        graph = load_graph(name, args.data_root)
        protocol = load_protocol(name, "nc", args.data_root)
        n = len(graph["y"])
        for policy in design["policies"]:
            x = (
                graph["x"]
                if policy == "identity"
                else structural_input(graph["edges"], n, policy)
            )
            raw, projected = preprocess(dict(x=x))
            np.savez_compressed(
                inputs / f"{name}_{policy}.npz",
                x=x.numpy(),
                y=graph["y"].numpy(),
                edges=graph["edges"].numpy(),
                **{k: protocol[k].numpy() for k in ["train", "val", "test"]},
            )
            with torch.no_grad():
                state = model.state(projected, graph["edges"])
                parts, hidden = node_representations(
                    model,
                    state,
                    raw,
                    graph["edges"],
                    hops=2,
                    inference_device=args.device,
                )
                representation = torch.cat([*parts, hidden], dim=1)
            degree = torch.bincount(graph["edges"].flatten(), minlength=n)
            audits.append(
                dict(
                    graph=name,
                    policy=policy,
                    nodes=n,
                    raw_dimensions=x.shape[1],
                    projected_zero_rows=int((projected.norm(dim=1) == 0).sum()),
                    nodes_above_degree63=int((degree > 63).sum()),
                )
            )
            for seed in design["seeds"]:
                for shot in design["shots"]:
                    support = episode(
                        graph["y"],
                        protocol["train"],
                        torch.Generator().manual_seed(seed),
                        shots=shot,
                        queries=0,
                    )[0]
                    classes = graph["y"][support].unique(sorted=True)
                    for method, features in [
                        ("EvoGFM", representation),
                        ("InputRidge", F.normalize(raw, dim=1)),
                    ]:
                        fit = fit_ridge(
                            features,
                            graph["y"],
                            support,
                            protocol["val"],
                            alphas=(0.1,),
                        )
                        logits = (
                            features[protocol["test"]].double() - fit.mean
                        ) @ fit.weight + fit.bias
                        predictions = classes[logits.argmax(1)]
                        accuracy = 100 * float(
                            predictions.eq(graph["y"][protocol["test"]]).double().mean()
                        )
                        row = dict(
                            graph=name,
                            policy=policy,
                            method=method,
                            seed=seed,
                            shots=shot,
                            accuracy_percent=accuracy,
                            support_count=len(support),
                            test_count=len(protocol["test"]),
                        )
                        rows.append(row)
                        np.savez_compressed(
                            args.output
                            / f"{name}_{policy}_{method}_s{seed}_k{shot}.npz",
                            predictions=predictions.numpy(),
                            support=support.numpy(),
                            test=protocol["test"].numpy(),
                        )
                        print(json.dumps(row), flush=True)
            with (args.output / "scores.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (args.output / "input_audit.json").write_text(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
