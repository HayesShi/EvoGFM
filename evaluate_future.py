"""Measure future domains at saved continual prefixes, without further updates."""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from evogfm.data import load_graph, load_protocol, make_pack, save_json, seed_all
from evogfm.features import preprocess, contexts, pair_channels, task_features
from evogfm.readout import ridge
from evogfm.training import load_checkpoint


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    model, head, checkpoint = load_checkpoint(args.checkpoint)
    model.requires_grad_(False)
    head.requires_grad_(False)
    prepared = {}
    for name in args.datasets:
        graph = load_graph(name, args.data_root)
        raw, x = preprocess(graph)
        edges = graph["edges"]
        protocol = load_protocol(name, "nc", args.data_root)
        pack = make_pack(name, x, edges, seed=42, query_cap=2048)
        pairs = pack["lp_val"]["pairs"]
        prepared[name] = dict(
            raw=raw,
            y=graph["y"],
            protocol=protocol,
            edges=edges,
            nc_state=model.state(x, edges),
            nc_pairs=edges[:, :1],
            nc_ctx=contexts(edges, len(x), edges[:, :1]),
            nc_base=pair_channels(raw, edges, edges[:, :1]),
            lp_edges=pack["state_edges"],
            lp_state=model.state(x, pack["state_edges"], 0.6),
            lp_pairs=pairs,
            lp_labels=pack["lp_val"]["y"],
            lp_ctx=contexts(pack["state_edges"], len(x), pairs),
            lp_base=pair_channels(raw, pack["state_edges"], pairs),
        )
        print("Prepared future domain", name, flush=True)
    tasks = ("nc", "lp") if args.task == "both" else (args.task,)
    for task in tasks:
        snapshots = torch.load(
            Path(args.states) / f"{task}_states.pt", weights_only=False
        )
        trajectories, summary = {}, {}
        for control, states in snapshots.items():
            records = []
            for prefix, state in enumerate(
                tqdm(states, desc=f"future {task} {control}")
            ):
                model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                scores = {}
                with torch.no_grad():
                    for name, p in prepared.items():
                        f, parts = task_features(
                            model,
                            p[f"{task}_state"],
                            p["raw"],
                            p["edges" if task == "nc" else "lp_edges"],
                            p[f"{task}_pairs"],
                            p[f"{task}_ctx"],
                            p[f"{task}_base"],
                        )
                        if task == "nc":
                            metric = (
                                "auc"
                                if len(p["y"].unique()) == 2
                                and args.nc_metric == "auto"
                                else "accuracy"
                            )
                            _, info = ridge(
                                torch.cat(parts, dim=1),
                                p["y"],
                                p["protocol"]["train"],
                                p["protocol"]["val"],
                                alphas=(0.1,),
                                metric=metric,
                            )
                            key = (
                                "validation_auc_percent"
                                if metric == "auc"
                                else "validation_percent"
                            )
                            scores[name] = dict(score=info[key] / 100, metric=metric)
                        else:
                            prediction = head(f).squeeze(1).numpy()
                            scores[name] = dict(
                                score=float(roc_auc_score(p["lp_labels"], prediction)),
                                metric="auc",
                            )
                records.append(dict(prefix=prefix, scores=scores))
            trajectories[control] = records
            summary[control] = dict(
                future_curves={
                    name: [r["scores"][name]["score"] for r in records]
                    for name in prepared
                },
                future_gain={
                    name: records[-1]["scores"][name]["score"]
                    - records[0]["scores"][name]["score"]
                    for name in prepared
                },
            )
            save_json(
                Path(args.output) / f"{task}_validation.json",
                dict(
                    seed=42,
                    source_epoch=checkpoint["epoch"],
                    states=args.states,
                    datasets=args.datasets,
                    task=task,
                    split="validation",
                    updates=0,
                    trajectories=trajectories,
                    summary=summary,
                ),
            )
            print(task, control, summary[control]["future_gain"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--states", default="runs/continual")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/future")
    parser.add_argument("--task", choices=["both", "nc", "lp"], default="both")
    parser.add_argument("--nc-metric", choices=["auto", "accuracy"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    evaluate(parser.parse_args())
