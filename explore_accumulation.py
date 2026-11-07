"""Evaluate future graphs across fixed label budgets and saved stream states."""

import argparse
import json
from pathlib import Path
import time
import torch
from sklearn.metrics import average_precision_score
from evogfm.data import load_graph, load_protocol, save_json, seed_all
from evogfm.graphland import load_graphland
from evogfm.features import preprocess, global_svd_features, node_representations
from evogfm.linear_readout import fit_linear, predict_linear, metric_score
from evogfm.training import load_checkpoint
from evogfm.online import serving_theta


def supports(labels, pool, shots, seed=42):
    if not shots:
        return pool
    generator = torch.Generator().manual_seed(seed)
    return torch.cat(
        [
            ids[torch.randperm(len(ids), generator=generator)[:shots]]
            for value in labels[pool].unique(sorted=True)
            for ids in [pool[labels[pool] == value]]
        ]
    )


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    model, _, checkpoint = load_checkpoint(args.checkpoint)
    model.requires_grad_(False)
    prior = model.inverse.theta().detach().clone()
    output = Path(args.output)
    experiments = {}
    for path in args.states:
        directory = Path(path)
        if directory.name in experiments:
            raise ValueError("Stream directories must have distinct names")
        config = json.loads(
            (directory / "nc_configuration.json").read_text(encoding="utf8")
        )
        if config["sources"] != checkpoint["source_graphs"]:
            raise ValueError("Stream source metadata does not match the backbone")
        snapshots = torch.load(directory / "nc_states.pt", weights_only=False)
        initial = torch.as_tensor(next(iter(snapshots.values()))[0]["theta"]).float()
        if not torch.equal(initial, model.inverse.theta().detach()):
            raise ValueError("Stream was initialized from another model")
        experiments[directory.name] = dict(
            config=config, snapshots=snapshots, path=str(directory)
        )
    configuration = dict(
        seed=42,
        checkpoint=args.checkpoint,
        states=args.states,
        datasets=args.datasets,
        budgets=args.budgets,
        alpha=0.1,
        inverse_strength=args.inverse_strength,
        hops=args.hops,
        split=args.split,
        shared_updates=0,
        final_only=args.final_only,
        head_selection="fixed alpha0.1; no future validation labels for head selection",
    )
    previous = output / "configuration.json"
    if args.resume and previous.exists():
        saved = json.loads(previous.read_text(encoding="utf8"))
        saved.setdefault("final_only", False)
        saved.setdefault("inverse_strength", 1.0)
        saved.setdefault("hops", 2)
        if saved != configuration:
            raise ValueError("Resume requires the same evaluation configuration")
    save_json(previous, configuration)
    records, profiles = [], {}
    for name in args.datasets:
        finished = output / f"{name}.json"
        if args.resume and finished.exists():
            saved = json.loads(finished.read_text(encoding="utf8"))
            records.extend(saved["rows"])
            profiles[name] = saved["profile"]
            continue
        allowed = {}
        for experiment, data in experiments.items():
            learned = set(data["config"]["sources"]) | set(data["config"]["stream"])
            if name in learned or (
                name in ("tolokers", "tolokers-2")
                and learned & {"tolokers", "tolokers-2"}
            ):
                continue
            allowed[experiment] = data
        if not allowed:
            continue
        begin = time.perf_counter()
        industrial = name in ("artnet-exp", "city-reviews", "tolokers-2")
        graph = (
            load_graphland(name, args.data_root)
            if industrial
            else load_graph(name, args.data_root)
        )
        raw, x = (
            (graph["x"], global_svd_features(graph["x"]))
            if industrial
            else preprocess(graph)
        )
        protocol = (
            graph
            if industrial
            else load_protocol(name, "nc", args.data_root, graph=graph)
        )
        labels = torch.nan_to_num(graph["y"].float(), nan=-1).long()
        edges = graph["edges"]
        choices = {
            shots: supports(labels, protocol["train"], shots) for shots in args.budgets
        }
        state = model.state(x, edges)
        train_mask = torch.zeros(len(raw), dtype=torch.bool)
        train_mask[protocol["train"]] = True
        known = train_mask[edges[0]] & train_mask[edges[1]]
        profiles[name] = dict(
            nodes=len(raw),
            edges=edges.shape[1],
            dimensions=raw.shape[1],
            validation_nodes=len(protocol["val"]),
            test_nodes=len(protocol["test"]),
            classes=len(labels[protocol["train"]].unique()),
            training_edge_homophily=(
                float(
                    (labels[edges[0, known]] == labels[edges[1, known]]).float().mean()
                )
                if known.any()
                else None
            ),
            support_ids={str(k): v.tolist() for k, v in choices.items()},
        )
        rows, measured = [], {}
        for experiment, data in allowed.items():
            for control, snapshots in data["snapshots"].items():
                for prefix, snapshot in enumerate(snapshots):
                    if args.final_only and prefix not in (0, len(snapshots) - 1):
                        continue
                    theta = serving_theta(snapshot, prior, args.inverse_strength)
                    key = theta.numpy().tobytes()
                    if key not in measured:
                        model.inverse.set_theta(theta)
                        with torch.no_grad():
                            parts, values = node_representations(
                                model, state, raw, edges, hops=args.hops
                            )
                            features = torch.cat(parts + [values], dim=1)
                            fitted_rows = []
                            for shots, train in choices.items():
                                binary = len(labels[train].unique()) == 2
                                metric = "auc" if binary else "accuracy"
                                fitted, _ = fit_linear(
                                    features,
                                    labels,
                                    train,
                                    metric=metric,
                                    alphas=(0.1,),
                                )
                                ids = protocol[
                                    "val" if args.split == "validation" else "test"
                                ]
                                prediction = predict_linear(features[ids], fitted)
                                quality = metric_score(
                                    prediction,
                                    labels[ids],
                                    torch.arange(len(ids)),
                                    metric,
                                    fitted["classes"],
                                )
                                ap = (
                                    float(
                                        average_precision_score(
                                            labels[ids].numpy(),
                                            (
                                                prediction[:, 1] - prediction[:, 0]
                                            ).numpy(),
                                        )
                                    )
                                    if binary
                                    else None
                                )
                                fitted_rows.append(
                                    dict(
                                        shots=shots,
                                        train_labels=len(train),
                                        metric="ap" if industrial else metric,
                                        score_percent=100
                                        * (ap if industrial else quality),
                                        auc_percent=100 * quality if binary else None,
                                        ap_percent=100 * ap if binary else None,
                                    )
                                )
                            measured[key] = fitted_rows
                    for fitted_row in measured[key]:
                        rows.append(
                            dict(
                                dataset=name,
                                experiment=experiment,
                                control=control,
                                prefix=prefix,
                                **fitted_row,
                            )
                        )
        records.extend(rows)
        save_json(
            finished,
            dict(
                dataset=name,
                split=args.split,
                rows=rows,
                profile=profiles[name],
                seconds=time.perf_counter() - begin,
            ),
        )
        save_json(
            output / "results.json",
            dict(seed=42, split=args.split, rows=records, profiles=profiles),
        )
        print(
            "COMPLETE",
            name,
            "rows",
            len(rows),
            "seconds",
            round(time.perf_counter() - begin, 1),
            flush=True,
        )
        del graph, raw, x, state, features, parts, values
    save_json(
        output / "results.json",
        dict(seed=42, split=args.split, rows=records, profiles=profiles),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 20, 0])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--inverse-strength", type=float, default=1.0)
    parser.add_argument("--hops", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Evaluate initial and final states only",
    )
    evaluate(parser.parse_args())
