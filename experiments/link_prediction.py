"""Matched random/mixed-support LP with the converged residual solver."""

import argparse
import csv
import json
import types

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import cross_task as cs, settings
from .full_graph import load_checkpoint
from .io import atomic_json
from .protocols import build_protocol
from .residual_solver import fit_residual_converged
from .study import Study, save_pt
from evogfm.data import make_pack
from evogfm.link_inference import LinkScorer
from evogfm.readout import predict_residual


def prepare_graph(name, study):
    """Use the original full graph, native edge masks and disjoint hard pools."""
    path = cs.ROOT / "cache" / f"{name}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    old = study.pack(name)
    native = make_pack(name, old["raw"], old["edges"], seed=42, query_cap=10**9)
    edges = native["state_edges"]
    state = cs.remask_state(study.model, old["state"], edges)
    hard = cs.safe_hard_pool(edges, old["edges"], len(old["y"]))
    permutation = torch.randperm(
        hard.shape[1], generator=torch.Generator().manual_seed(31415)
    )
    split = max(1, len(permutation) // 2)
    pack = dict(
        name=name,
        raw=old["raw"],
        y=old["y"],
        protocol=old["protocol"],
        edges=edges,
        state=state,
        native=native,
        hard_train=hard[:, permutation[:split]],
        hard_val=hard[:, permutation[split:]],
    )
    save_pt(path, pack)
    return pack


@torch.no_grad()
def features(scorer, pairs):
    return torch.cat([scorer.features(batch) for batch in pairs.split(512, dim=1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs", nargs="+", default=["chameleon", "cora", "computers", "squirrel"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--modes", nargs="+", choices=["uniform", "mixed"], default=["uniform", "mixed"]
    )
    args = parser.parse_args()
    cs.setup()
    output = settings.ROOT / "link_prediction"
    output.mkdir(parents=True, exist_ok=True)
    study = Study(
        argparse.Namespace(
            output=str(settings.NC_RUN),
            data=str(settings.DATA),
            threads=4,
            eval_device=settings.DEVICE,
        )
    )
    atomic_json(
        output / "configuration.json",
        dict(
            checkpoint=str(settings.CHECKPOINT),
            graphs=args.graphs,
            seeds=args.seeds,
            modes=args.modes,
            ridge=0.1,
            support_positive=200,
            support_negative=200,
            solver="float64 Armijo Newton; gradient infinity norm <= 1e-8",
        ),
    )
    rows = []
    for graph in args.graphs:
        pack = prepare_graph(graph, study)
        model, head, _ = load_checkpoint(settings.CHECKPOINT)
        model.encode_features = types.MethodType(
            lambda self, x, z=pack["state"].z: z, model
        )
        scorer = LinkScorer(
            model, head, pack["raw"], pack["raw"], pack["edges"], observation_rate=0.6
        )
        for seed in args.seeds:
            query = None
            query_features = None
            for mode in args.modes:
                tag = f"{graph}_{mode}_s{seed}"
                protocol = build_protocol(pack, seed, 200, mode)
                save_pt(output / f"protocol_{tag}.pt", protocol)
                if query is None:
                    query = protocol["query_pairs"]
                    query_features = features(scorer, query)
                assert torch.equal(query, protocol["query_pairs"])
                fitted_features = features(scorer, protocol["support_pairs"])
                with torch.no_grad():
                    offset = head(fitted_features).view(-1)
                    query_offset = head(query_features).view(-1)
                fitted = fit_residual_converged(
                    fitted_features, offset, protocol["context_labels"], ridge=0.1
                )
                score = predict_residual(query_features, query_offset, fitted)
                n = protocol["positive_count"]
                save_pt(
                    output / f"predictions_{tag}.pt",
                    dict(query_pairs=query, positive_count=n, scores=score, fit=fitted),
                )
                for difficulty, start in [("Random", n), ("Hard", 2 * n)]:
                    y = np.r_[np.ones(n), np.zeros(n)]
                    values = torch.cat([score[:n], score[start : start + n]]).numpy()
                    row = dict(
                        graph=graph,
                        seed=seed,
                        support=mode,
                        negative=difficulty,
                        auc_percent=100 * roc_auc_score(y, values),
                        ap_percent=100 * average_precision_score(y, values),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        del scorer, model, head, pack
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
