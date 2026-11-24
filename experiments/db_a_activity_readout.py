"""Use previously inactive feature slots for endpoint activity on first contacts."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from experiments.db_a_quality import (
    apply_node_activity,
    baseline,
    cases,
    records,
    features,
    write_json,
)
from experiments.db_a_rank_readout import measure, tree_nodes
from experiments.db_a_native import WORKSPACE


def inputs_with_ages(case, phase):
    folder = Path(case["candidates"])
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    shape = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    last = np.full(shape["nodes"], -np.inf, dtype=np.float32)
    data = iter(records(case, phase, full=phase != "train"))
    for entry in index:
        time = entry["time"]
        with np.load(folder / entry["file"]) as query:
            positive = query["positive"]
            if entry["phase"] == phase and entry is not index[0]:
                row = next(data)
                if time != row["time"]:
                    raise ValueError("Candidate and feature snapshots differ")
                pairs = [positive, query["auc_negative"]]
                if phase != "train":
                    pairs.append(query["ranking_negative"])
                endpoints = np.concatenate(pairs, axis=1)
                base = features(
                    row["base"], row["statistics"], "union24", case["half_life"]
                )
                yield row, base, time - last[endpoints]
        last[np.unique(positive)] = time


def apply(data, activity):
    for row, x, ages in data:
        yield row, (
            apply_node_activity(x.copy(), row["statistics"], ages, activity)
            if activity
            else x
        )


def select(case, previous, output):
    dataset = case["dataset"]
    origin = previous / dataset
    state = joblib.load(origin / "selected_readout.joblib")
    case["half_life"] = state["half_life"]
    reference = state["model"]["fitted"]
    training = list(inputs_with_ages(case, "train"))
    validation = list(inputs_with_ages(case, "validation"))
    truth = np.concatenate(
        [np.r_[np.ones(row["npos"]), np.zeros(row["nneg"])] for row, _, _ in training]
    )
    negative = truth == 0
    locked = json.loads((origin / "validation_lock.json").read_text(encoding="utf-8"))
    original = (
        WORKSPACE
        / ("runs/vldb/" "results/a_quality/undergs")
        / dataset
        / "selected_readout.joblib"
    )
    original_model = joblib.load(original)["model"]["fitted"]
    train_original = np.concatenate([x for _, x in apply(training, None)])
    base_score = original_model.predict_proba(train_original[negative])[:, 1]
    top = base_score >= np.quantile(base_score, 0.8)
    weight = np.ones(len(truth))
    if locked["selected"].startswith("hard"):
        weight[np.flatnonzero(negative)[top]] = int(locked["selected"][4:])
    weight /= weight.mean()
    grid = {"previous": None}
    models = {"previous": reference}
    val = {"previous": measure(reference, apply(validation, None))}
    for horizon in (1, 4, 16):
        for use_source in (False, True):
            name = f"activity_{horizon}_{'both' if use_source else 'target'}"
            config = {"half_life": horizon, "use_source": use_source}
            x = np.concatenate([row for _, row in apply(training, config)])
            models[name] = clone(reference).fit(x, truth, sample_weight=weight)
            val[name] = measure(models[name], apply(validation, config))
            grid[name] = config
    eligible = [
        name
        for name, row in val.items()
        if row["auc"] >= val["previous"]["auc"] - 0.0025
        and row["pessimistic_mrr"] >= val["previous"]["pessimistic_mrr"]
        and tree_nodes(models[name]) <= tree_nodes(reference)
    ]
    chosen = max(
        eligible,
        key=lambda name: (val[name]["mrr"], val[name]["auc"], name == "previous"),
    )
    seeded = {42: models[chosen]}
    seed_scores = {42: val[chosen]}
    selected_x = np.concatenate([row for _, row in apply(training, grid[chosen])])
    for seed in (41, 43):
        seeded[seed] = (
            clone(reference)
            .set_params(random_state=seed)
            .fit(selected_x, truth, sample_weight=weight)
        )
        seed_scores[seed] = measure(seeded[seed], apply(validation, grid[chosen]))
    spread = max(row["mrr"] for row in seed_scores.values()) - min(
        row["mrr"] for row in seed_scores.values()
    )
    selected_seed = 42
    if spread >= 0.01:
        eligible_seeds = [
            seed
            for seed, row in seed_scores.items()
            if row["auc"] >= val["previous"]["auc"] - 0.0025
            and row["pessimistic_mrr"] >= val["previous"]["pessimistic_mrr"]
            and tree_nodes(seeded[seed]) <= tree_nodes(reference)
        ]
        selected_seed = max(
            eligible_seeds,
            key=lambda seed: (
                seed_scores[seed]["mrr"],
                seed_scores[seed]["auc"],
                seed == 42,
            ),
        )
    state["model"]["fitted"] = seeded[selected_seed]
    state["training_seed"] = selected_seed
    if grid[chosen]:
        state["node_activity"] = grid[chosen]
    folder = output / dataset
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, folder / "selected_readout.joblib", compress=3)
    result = dict(
        dataset=dataset,
        selected=chosen,
        activity=grid[chosen],
        val=val,
        seed=selected_seed,
        objective="Rank validation MRR subject to AUC and pessimistic-MRR floors",
        validation_seeds=seed_scores,
        seed_mrr_range=spread,
        tree_nodes=tree_nodes(seeded[selected_seed]),
    )
    write_json(folder / "validation_lock.json", result)
    print(
        json.dumps(dict(dataset=dataset, selected=chosen, val=val[chosen])), flush=True
    )
    return result


def run(previous, output):
    work = [case for case in cases() if case["protocol"] == "undergs"]
    locks = [select(case, previous, output) for case in work]
    write_json(output / "all_validation_locks.json", locks)
    results = []
    for case, lock in zip(work, locks):
        folder = output / case["dataset"]
        state = joblib.load(folder / "selected_readout.joblib")
        result = dict(
            **lock,
            base=baseline(case),
            test=measure(
                state["model"]["fitted"],
                apply(inputs_with_ages(case, "test"), state.get("node_activity")),
            ),
        )
        results.append(result)
        write_json(folder / "result.json", result)
        print(
            json.dumps(
                dict(dataset=case["dataset"], test=result["test"], base=result["base"])
            ),
            flush=True,
        )
    write_json(output / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.previous.resolve(), args.output.resolve())
