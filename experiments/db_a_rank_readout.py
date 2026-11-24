"""Validate ranking-oriented training weights at the original readout capacity."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from scipy.special import expit
from threadpoolctl import threadpool_limits

from experiments.db_a_native import WORKSPACE
from experiments.db_a_quality import (
    auc_pack,
    baseline,
    cases,
    features,
    model_capacity,
    ranking_metrics,
    records,
    write_json,
)


def inputs(case, phase):
    yield from (
        (row, features(row["base"], row["statistics"], "union24", case["half_life"]))
        for row in records(case, phase, full=phase != "train")
    )


def measure(model, data):
    auc, mrr, pessimistic, tied = [], [], [], []
    for row, x in data:
        score = (
            model.predict_proba(x)[:, 1]
            if hasattr(model, "predict_proba")
            else expit(model.predict(x))
        ).astype(np.float32)
        rank = ranking_metrics(row, score, "undergs")
        auc.append(auc_pack(row, score)["auc"])
        mrr.append(rank["mrr"])
        pessimistic.append(rank["pessimistic_mrr"])
        tied.append(rank["query_with_ties_fraction"])
    return dict(
        auc=float(np.mean(auc)),
        mrr=float(np.mean(mrr)),
        pessimistic_mrr=float(np.mean(pessimistic)),
        tie_fraction=float(np.mean(tied)),
    )


def tree_nodes(model):
    if hasattr(model, "_predictors"):
        return sum(tree.nodes.size for trees in model._predictors for tree in trees)
    return int((2 * model.get_tree_leaf_counts() - 1).sum())


def pairwise_rankers(training):
    from catboost import CatBoostRanker, Pool

    pairs = []
    for row, values in training:
        count = row["npos"]
        if row["nneg"] != count:
            raise ValueError(
                "Training candidates must have one matched negative per positive"
            )
        pairs.append(np.stack((values[:count], values[count:]), axis=1))
    pairs = np.concatenate(pairs)
    ranked = Pool(
        pairs.reshape(-1, pairs.shape[-1]),
        label=np.tile([1, 0], len(pairs)),
        group_id=np.repeat(np.arange(len(pairs)), 2),
    )
    return {
        f"gpu_pairlogit_d{depth}": CatBoostRanker(
            loss_function="PairLogit",
            iterations=count,
            depth=depth,
            learning_rate=0.1,
            random_seed=42,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.3,
            border_count=64,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ).fit(ranked)
        for depth, count in ((3, 100), (4, 80))
    }


def lock_readout(case, existing, output, gpu_ranker):
    root = existing / case["dataset"]
    state = joblib.load(root / "selected_readout.joblib")
    old = state["model"]["fitted"]
    if state["model"]["input"] != "union24" or state["model"]["family"] != "hgb":
        raise ValueError("This sweep requires the locked full-feature tree readout")
    case["half_life"] = state["half_life"]
    training = list(inputs(case, "train"))
    x = np.concatenate([features for _, features in training])
    y = np.concatenate(
        [np.r_[np.ones(row["npos"]), np.zeros(row["nneg"])] for row, _ in training]
    )
    negative = y == 0
    teacher = old.predict_proba(x[negative])[:, 1]
    difficult = np.zeros(len(y), dtype=np.bool_)
    difficult[negative] = teacher >= np.quantile(teacher, 0.8)
    prior_edge = negative & (
        np.concatenate([row["statistics"][:, 1] for row, _ in training]) > 0
    )
    validation = list(inputs(case, "validation"))
    models = {"locked_reference": old}
    for name, mask, weight in (
        ("hard2", difficult, 2),
        ("hard4", difficult, 4),
        ("hard8", difficult, 8),
        ("prior4", prior_edge, 4),
        ("prior8", prior_edge, 8),
    ):
        if mask.any():
            sample_weight = np.where(mask, weight, 1.0).astype(np.float64)
            sample_weight /= sample_weight.mean()
            models[name] = clone(old).fit(x, y, sample_weight=sample_weight)
    if gpu_ranker:
        models.update(pairwise_rankers(training))
    metrics = {name: measure(model, validation) for name, model in models.items()}
    base = metrics["locked_reference"]
    # Prefer a ranking improvement with essentially the same validation AUC.
    eligible = [
        name
        for name, row in metrics.items()
        if row["auc"] >= base["auc"] - 0.0025
        and row["pessimistic_mrr"] >= base["pessimistic_mrr"]
        and tree_nodes(models[name]) <= model_capacity(state["model"])
    ]
    selected = max(
        eligible,
        key=lambda name: (
            metrics[name]["mrr"],
            metrics[name]["auc"],
            name == "locked_reference",
        ),
    )
    state["model"]["fitted"] = models[selected]
    state["ranking_variant"] = selected
    if selected.startswith("gpu_pairlogit"):
        state["model"]["family"] = "pair_rank"
    folder = output / case["dataset"]
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, folder / "selected_readout.joblib", compress=3)
    result = dict(
        dataset=case["dataset"],
        selected=selected,
        val=metrics,
        hard_negative_count=int(difficult.sum()),
        prior_edge_negative_count=int(prior_edge.sum()),
        tree_nodes=tree_nodes(models[selected]),
    )
    write_json(folder / "validation_lock.json", result)
    print(
        json.dumps(
            dict(
                dataset=case["dataset"], selected=selected, validation=metrics[selected]
            )
        ),
        flush=True,
    )
    return result


def run(output, gpu_ranker):
    existing = WORKSPACE / ("runs/vldb/" "results/a_quality/undergs")
    work = [case for case in cases() if case["protocol"] == "undergs"]
    locks = [lock_readout(case, existing, output, gpu_ranker) for case in work]
    write_json(output / "all_validation_locks.json", locks)
    results = []
    for case, lock in zip(work, locks):
        folder = output / case["dataset"]
        model = joblib.load(folder / "selected_readout.joblib")["model"]["fitted"]
        old = joblib.load(existing / case["dataset"] / "selected_readout.joblib")[
            "model"
        ]["fitted"]
        held_out = list(inputs(case, "test"))
        result = dict(
            **lock,
            seed=42,
            reference=baseline(case),
            test=measure(model, held_out),
            old_test=measure(old, held_out),
            scope="Existing causal UnderGS candidates and cached FP32 features; previously used development split",
        )
        results.append(result)
        write_json(folder / "result.json", result)
        print(
            json.dumps(
                dict(
                    dataset=case["dataset"],
                    test=result["test"],
                    reference=result["reference"],
                )
            ),
            flush=True,
        )
    write_json(output / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ranker", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output.resolve(), args.gpu_ranker)
