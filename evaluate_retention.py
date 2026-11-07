"""Revisit old domains with their first-acquisition NC head held fixed."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from evogfm.data import load_graph, load_protocol, save_json
from evogfm.features import (
    preprocess,
    contexts,
    pair_channels,
    task_features,
    node_representations,
)
from evogfm.linear_readout import fit_linear, predict_linear, metric_score
from evogfm.training import load_checkpoint
from evogfm.online import serving_theta


def evaluate(args):
    torch.set_num_threads(args.threads)
    model, _, _ = load_checkpoint(args.checkpoint)
    model.requires_grad_(False)
    prior = model.inverse.theta().detach().clone()
    directory = Path(args.states)
    config = json.loads((directory / "nc_configuration.json").read_text())
    stream = config["stream"]
    states = torch.load(directory / "nc_states.pt", weights_only=False)
    matrices = {
        kind: [[None] * len(stream) for _ in snapshots]
        for kind, snapshots in states.items()
    }
    heads = {}
    for column, name in enumerate(tqdm(stream, desc="fixed-head retention")):
        graph = load_graph(name, args.data_root)
        raw, x = preprocess(graph)
        edges = graph["edges"]
        protocol = load_protocol(name, "nc", args.data_root)
        state = model.state(x, edges)
        dummy = edges[:, :1]
        ctx, base = contexts(edges, len(raw), dummy), pair_channels(raw, edges, dummy)
        metric = "auc" if len(graph["y"].unique()) == 2 else "accuracy"
        for kind, snapshots in states.items():
            fitted = None
            for prefix in range(column + 1, len(snapshots)):
                model.inverse.set_theta(
                    serving_theta(snapshots[prefix], prior, args.inverse_strength)
                )
                with torch.no_grad():
                    if args.hops == 2:
                        _, parts = task_features(
                            model, state, raw, edges, dummy, ctx, base
                        )
                    else:
                        parts, values = node_representations(
                            model, state, raw, edges, hops=args.hops
                        )
                        parts = parts + [values]
                    features = torch.cat(parts, dim=1)
                    if fitted is None:
                        fitted, _ = fit_linear(
                            features,
                            graph["y"],
                            protocol["train"],
                            metric=metric,
                            alphas=(0.1,),
                        )
                        heads[f"{kind}/{name}"] = fitted
                    prediction = predict_linear(features, fitted)
                matrices[kind][prefix][column] = metric_score(
                    prediction, graph["y"], protocol["val"], metric, fitted["classes"]
                )
        save_json(Path(args.output) / "fixed_nc_matrices.json", matrices)
    summary = {}
    for kind, matrix in matrices.items():
        final = np.asarray(matrix[-1])
        diagonal = np.asarray([matrix[j + 1][j] for j in range(len(stream))])
        forgetting = [
            max(row[j] for row in matrix[j + 1 :]) - final[j]
            for j in range(len(stream) - 1)
        ]
        summary[kind] = dict(
            final_average=float(final.mean()),
            old_domains_final_average=float(final[:-1].mean()),
            forgetting=float(np.mean(forgetting)),
            backward_transfer=float((final[:-1] - diagonal[:-1]).mean()),
            new_domain_scores=diagonal.tolist(),
            matrix=matrix,
        )
    save_json(
        Path(args.output) / "retention.json",
        dict(
            seed=42,
            stream=stream,
            split="validation",
            inverse_strength=args.inverse_strength,
            hops=args.hops,
            nc_head="first-acquisition fixed alpha0.1 ridge",
            unseen_cells=None,
            summary=summary,
            evaluator_head_bytes=sum(
                value.numel() * value.element_size()
                for head in heads.values()
                for value in head.values()
                if isinstance(value, torch.Tensor)
            ),
            shared_state_bytes=sum(
                torch.as_tensor(states["cumulative"][-1][key]).numel() * 8
                for key in ("theta", "metric")
            ),
            lp_note="Native LP already uses the frozen source head; its recorded matrix is fixed-head retention.",
        ),
    )
    torch.save(heads, Path(args.output) / "fixed_nc_heads.pt")
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--inverse-strength", type=float, default=1.0)
    parser.add_argument("--hops", type=int, choices=range(1, 5), default=2)
    evaluate(parser.parse_args())
