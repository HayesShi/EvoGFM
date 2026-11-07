"""Public NC/LP evaluation with a frozen shared backbone."""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .data import load_graph, load_protocol, seed_all, save_json
from .features import (
    preprocess,
    contexts,
    task_features,
    pair_channels,
    node_representations,
    hard_negative_pool,
)
from .readout import (
    ridge,
    hits100,
    sampled_negatives,
    fit_residual,
    predict_residual,
    multiscale_ridge,
    enhanced_ridge,
)
from .training import load_checkpoint


def masked_link_examples(
    model, head, raw, x, full, observation_rate=0.7, negative_mode="uniform"
):
    """Fit 18 coefficients using masked training edges, never public queries."""
    n = len(raw)
    generator = torch.Generator().manual_seed(42)
    train = []
    held = []
    train_ids = set()
    held_ids = set()
    # Positive and negative pair IDs are disjoint across train/holdout in
    # ALL views. A view-specific split alone can repeat held pairs in train.
    global_order = torch.randperm(full.shape[1], generator=generator)
    nheld = max(1, int(0.05 * len(global_order)))
    fixed_held = global_order[:nheld]
    eligible = global_order[nheld:]
    negative_pool = sampled_negatives(n, full, 2 * full.shape[1], generator)
    if negative_mode == "mixed":
        hard = hard_negative_pool(full, n)
        count = min(len(hard.T), len(negative_pool.T) // 2)
        selected = torch.randperm(
            len(hard.T), generator=torch.Generator().manual_seed(43)
        )[:count]
        uniform_selected = torch.randperm(
            len(negative_pool.T), generator=torch.Generator().manual_seed(44)
        )[: len(negative_pool.T) - count]
        negative_pool = torch.cat(
            (hard[:, selected], negative_pool[:, uniform_selected]), dim=1
        )
        keys = (negative_pool[0] * n + negative_pool[1]).unique(sorted=True)
        negative_pool = torch.stack((keys // n, keys % n))
    elif negative_mode != "uniform":
        raise ValueError("negative_mode must be uniform or mixed")
    negative_order = torch.randperm(negative_pool.shape[1], generator=generator)
    negative_held = negative_pool[:, negative_order[:nheld]]
    negative_train = negative_pool[:, negative_order[nheld:]]
    with torch.no_grad():
        for view in range(4):
            order = eligible[torch.randperm(len(eligible), generator=generator)]
            cut = int(0.2 * len(global_order)) - nheld
            pos = full[:, order[:cut]]
            support = full[:, order[cut:]]
            held_positive = full[:, fixed_held]
            neg = negative_train[
                :,
                torch.randperm(negative_train.shape[1], generator=generator)[
                    : pos.shape[1]
                ],
            ]
            train_pairs = torch.cat((pos, neg), dim=1)
            held_pairs = torch.cat((held_positive, negative_held), dim=1)
            pairs = torch.cat((train_pairs, held_pairs), dim=1)
            train_y = torch.cat((torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])))
            held_y = torch.cat((torch.ones(nheld), torch.zeros(nheld)))
            y = torch.cat((train_y, held_y))
            state = model.state(x, support, observation_rate * 0.8)
            ctx = contexts(support, n, pairs)
            base = pair_channels(raw, support, pairs)
            f, _ = task_features(model, state, raw, support, pairs, ctx, base)
            offset = head(f).squeeze(1)
            ix = torch.randperm(len(train_y), generator=generator)[:1024]
            train.append((f[ix], offset[ix], y[ix]))
            keys = pairs[0, ix] * n + pairs[1, ix]
            train_ids.update(keys.tolist())
            ix = len(train_y) + torch.randperm(len(held_y), generator=generator)[:512]
            held.append((f[ix], offset[ix], y[ix]))
            keys = pairs[0, ix] * n + pairs[1, ix]
            held_ids.update(keys.tolist())
        if train_ids & held_ids:
            raise RuntimeError("calibration pair IDs overlap across augmented views")
        tf, to, ty = [torch.cat([row[i] for row in train]) for i in range(3)]
        hf, ho, hy = [torch.cat([row[i] for row in held]) for i in range(3)]
    return (
        (tf, to, ty),
        (hf, ho, hy),
        dict(
            train_pairs=len(ty),
            unique_train_pairs=len(train_ids),
            unique_holdout_pairs=len(held_ids),
        ),
    )


def fit_link_readout(
    model, head, raw, x, full, observation_rate=0.7, negative_mode="uniform"
):
    """Fit 18 coefficients using masked training edges, never public queries."""
    (tf, to, ty), (hf, ho, hy), statistics = masked_link_examples(
        model, head, raw, x, full, observation_rate, negative_mode
    )
    candidates = []
    begin = time.perf_counter()
    for strength in [0.1, 1.0, 10.0, 100.0]:
        fitted = fit_residual(tf, to, ty, strength)
        loss = float(
            F.binary_cross_entropy_with_logits(
                predict_residual(hf, ho, fitted), hy.double()
            )
        )
        candidates.append((loss, strength, fitted))
    loss, strength, fitted = min(candidates, key=lambda row: row[0])
    return dict(
        fitted=fitted,
        selected_ridge=strength,
        calibration_holdout_bce=loss,
        optimizer_seconds=time.perf_counter() - begin,
        **statistics,
    )


def evaluate(
    checkpoint="runs/source/best.pt",
    data_root="data",
    output="runs/public",
    split="val",
    ablation="none",
    nc=("cora", "roman-empire"),
    lp=("cora", "citeseer"),
    threads=4,
    node_readout="current",
    nc_metric="accuracy",
    shots=0,
    edge_rate=1.0,
):
    if shots < 0 or not 0 < edge_rate <= 1:
        raise ValueError("shots must be nonnegative and edge_rate must be in (0, 1]")
    seed_all(42)
    torch.set_num_threads(threads)
    model, head, ckpt = load_checkpoint(checkpoint)
    if ablation == "zero-inverse":
        model.inverse.set_theta(torch.zeros_like(model.inverse.theta()))
    rows = []
    readouts = {}
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    for name in tqdm(nc, desc="node classification"):
        graph = load_graph(name, data_root)
        raw, x = preprocess(graph)
        edges = graph["edges"]
        protocol = load_protocol(name, "nc", data_root, graph=graph)
        train = protocol["train"]
        if shots:
            generator = torch.Generator().manual_seed(42)
            selected = []
            for label in graph["y"][train].unique():
                ids = train[graph["y"][train] == label]
                selected.append(
                    ids[torch.randperm(len(ids), generator=generator)[:shots]]
                )
            train = torch.cat(selected)
        if edge_rate < 1:
            order = torch.randperm(
                edges.shape[1], generator=torch.Generator().manual_seed(42)
            )
            edges = edges[:, order[: int(edge_rate * len(order))]]
        with torch.no_grad():
            state = model.state(x, edges, edge_rate)
            metric = (
                "auc"
                if nc_metric == "auto" and len(graph["y"].unique()) == 2
                else ("accuracy" if nc_metric == "auto" else nc_metric)
            )
            parts, values, transition = node_representations(
                model,
                state,
                raw,
                edges,
                hops=2 if node_readout == "current" else 4,
                return_transition=True,
            )
            if node_readout in ("enhanced", "advanced"):
                scores, info = enhanced_ridge(
                    parts,
                    values,
                    graph["y"],
                    train,
                    protocol["val"],
                    transition,
                    metric=metric,
                    advanced=node_readout == "advanced",
                )
            elif node_readout == "adaptive":
                scores, info = multiscale_ridge(
                    parts,
                    values,
                    graph["y"],
                    train,
                    protocol["val"],
                    metric=metric,
                )
            else:
                scores, info = ridge(
                    torch.cat(parts + [values], dim=1),
                    graph["y"],
                    train,
                    protocol["val"],
                    metric=metric,
                )
        classes = graph["y"][train].unique(sorted=True)
        ids = protocol[split]
        value = 100 * float(
            (classes[scores[ids].argmax(1)] == graph["y"][ids]).float().mean()
        )
        row = dict(dataset=name, task="nc", split=split, accuracy_percent=value, **info)
        if len(classes) == 2:
            row["auc_percent"] = 100 * float(
                roc_auc_score(graph["y"][ids], scores[ids, 1] - scores[ids, 0])
            )
        row["selection_metric"] = metric
        row.update(shots=shots, train_labels=len(train), edge_rate=edge_rate)
        rows.append(row)
        print(row, flush=True)
    for name in tqdm(lp, desc="link prediction"):
        graph = load_graph(name, data_root)
        raw, x = preprocess(graph)
        protocol = load_protocol(name, "lp", data_root, graph=graph)
        edges = protocol["train"]["edge"].T
        fitted = fit_link_readout(model, head, raw, x, edges)
        readouts[name] = fitted
        part = protocol["valid" if split == "val" else "test"]
        pairs = torch.cat((part["edge"], part["edge_neg"])).T
        count = len(part["edge"])
        with torch.no_grad():
            state = model.state(x, edges, 0.7)
            features, _ = task_features(
                model,
                state,
                raw,
                edges,
                pairs,
                contexts(edges, len(raw), pairs),
                pair_channels(raw, edges, pairs),
            )
            scores = predict_residual(
                features, head(features).squeeze(1), fitted["fitted"]
            )
        labels = torch.cat((torch.ones(count), torch.zeros(len(scores) - count)))
        row = dict(
            dataset=name,
            task="lp",
            split=split,
            hits100_percent=100 * hits100(scores[:count], scores[count:]),
            auc=float(roc_auc_score(labels, scores)),
            selected_ridge=fitted["selected_ridge"],
        )
        rows.append(row)
        print(row, flush=True)
    result = dict(
        seed=42,
        checkpoint=str(checkpoint),
        source_epoch=ckpt["epoch"],
        ablation=ablation,
        rows=rows,
    )
    save_json(directory / f"{split}_{ablation}.json", result)
    torch.save(readouts, directory / f"readouts_{ablation}.pt")
    return result
