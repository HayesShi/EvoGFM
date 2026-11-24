"""A0 quality repair on existing development caches, selected on validation only.

The dynamic graph feature extractor, event visibility, candidates and official
metrics are unchanged. This entry point compares a small predeclared readout
grid. It does not measure graph extraction or end-to-end maintenance cost.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import time

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

try:
    from experiments.diagnose_temporal_rootcause import fit as fit_logistic
except ModuleNotFoundError:
    from diagnose_temporal_rootcause import fit as fit_logistic


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("original17", "temporal7", "union24")
TREE_GRID = (
    dict(
        name="hgb7",
        max_leaf_nodes=7,
        max_iter=100,
        learning_rate=0.08,
        min_samples_leaf=100,
        l2_regularization=1.0,
    ),
    dict(
        name="hgb15",
        max_leaf_nodes=15,
        max_iter=100,
        learning_rate=0.08,
        min_samples_leaf=100,
        l2_regularization=10.0,
    ),
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cases():
    output = []
    for protocol in ("decoupled", "undergs"):
        for dataset in ("bitcoin-alpha", "bitcoin-otc", "uci"):
            base = ROOT / "runs/protocol_alignment" / protocol / dataset / "s42"
            cache = (
                base / "features_full"
                if protocol == "decoupled"
                else base / "causal_adaptation/frozen/features"
            )
            candidates = base / (
                "protocol" if protocol == "decoupled" else "candidates"
            )
            lock = (
                ROOT
                / "runs/optimization/temporal_calibration_s42/local_statistics"
                / protocol
                / dataset
                / "validation_lock.json"
            )
            old_config = json.loads(lock.read_text())["selected"]["configuration"]
            output.append(
                dict(
                    protocol=protocol,
                    dataset=dataset,
                    feature_cache=str(cache),
                    candidates=str(candidates),
                    half_life=float(old_config["half_life"]),
                    half_life_source=str(lock),
                    baseline_directory=str(base),
                )
            )
    return output


def predeclare(output):
    path = output / "predeclared_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    declaration = dict(
        created_utc=now(),
        seed=42,
        threads=1,
        project="Incremental Predictive Analytics over Evolving Graphs",
        study="A0 lightweight readout quality repair on existing development benchmarks",
        evaluation_status="Existing test splits have already been inspected during earlier development and root-cause diagnosis. This is not a new blind test.",
        inputs=list(INPUTS),
        temporal_semantics="statistics7, with age columns 0 and 2 transformed to exp2(-age/previously locked half_life); no signal overwrite or target correction",
        logistic=dict(
            ridge=10.0,
            offset_mode="zero",
            sample_std_floor=0.05,
            solver="Existing float64 sum-BCE+ridge trust-exact/Newton solver; train-only scaling; 1e-9 gradient tolerance",
        ),
        tree_grid=list(TREE_GRID),
        tree_common=dict(loss="log_loss", early_stopping=False, random_state=42),
        blend="One fixed probability average per input: .5*logistic10 + .5*that input's validation-AUC-best hgb7/hgb15. No fitted blend weights.",
        candidates_per_case=12,
        selection="Highest protocol-specific validation AUC; exact ties prefer lower capacity then lexical ID. No MRR/test/seed selection. All six validation locks are saved before opening new test feature archives.",
        fitting="Only original train positives and AUC negatives; no ranking candidates or validation labels used in fitting",
        train_mrr="Unavailable: existing train caches have no official ranking-negative feature suffix. Train MRR is null; no substituted candidate set is fabricated.",
        metrics="DD AUC pooled native scored events and source-best/1000-negative pessimistic MRR; UnderGS AUC unweighted snapshot mean and released NumPy event rank/100-negative MRR; MRR always unweighted snapshot mean. UnderGS scores float32.",
        cost_scope="All measured fit/score times reuse offline feature caches; graph initialization/extraction/maintenance and I/O outside the explicit timer are not measured. These times are not end-to-end service cost.",
        cases=cases(),
    )
    write_json(path, declaration)
    return declaration


def features(base, statistics, input_name, half_life):
    if input_name == "original17":
        return base
    temporal = statistics.astype(np.float32, copy=True)
    temporal[:, [0, 2]] = np.exp2(-temporal[:, [0, 2]] / np.float32(half_life))
    if input_name == "temporal7":
        return temporal
    if input_name == "union24":
        return np.concatenate((base, temporal), axis=1)
    raise ValueError(input_name)


def records(case, phase, full=False):
    manifest = json.loads((Path(case["feature_cache"]) / "manifest.json").read_text())
    for entry in manifest["index"]:
        if entry["phase"] != phase:
            continue
        with np.load(Path(case["feature_cache"]) / entry["file"]) as archive:
            npos, nneg = int(archive["positive_count"]), int(
                archive["auc_negative_count"]
            )
            count = npos + nneg
            base = archive["base"].astype(np.float32, copy=False)
            statistics = archive["statistics"].astype(np.float32, copy=False)
            if not full:
                base, statistics = base[:count].copy(), statistics[:count].copy()
            positive_sources = archive["positive_sources"].copy()
            rank_sources = archive["rank_sources"].copy()
        with np.load(Path(case["candidates"]) / entry["file"]) as query:
            if (
                query["positive"].shape[1] != npos
                or query["auc_negative"].shape[1] != nneg
            ):
                raise ValueError("Candidate/cache counts differ")
            np.testing.assert_array_equal(positive_sources, query["positive"][0])
            np.testing.assert_array_equal(rank_sources, query["rank_sources"])
            indices = (
                query["auc_positive_indices"].copy()
                if "auc_positive_indices" in query
                else np.arange(npos)
            )
        if base.shape[1] != 17 or statistics.shape != (len(base), 7):
            raise ValueError("Unexpected frozen feature shape")
        yield dict(
            time=int(entry["time"]),
            npos=npos,
            nneg=nneg,
            base=base,
            statistics=statistics,
            indices=indices,
            positive_sources=positive_sources,
            rank_sources=rank_sources,
        )


def probabilities(model, x):
    if model["family"] == "logistic":
        fitted = model["fitted"]
        normalized = (x.astype(np.float64) - fitted["mean"]) / fitted["scale"]
        return expit(
            normalized @ fitted["coefficient"][:-1] + fitted["coefficient"][-1]
        )
    if model["family"] == "hgb":
        return model["fitted"].predict_proba(x)[:, 1]
    if model["family"] == "pair_rank":
        return expit(model["fitted"].predict(x, thread_count=1))
    if model["family"] == "blend":
        return 0.5 * (
            probabilities(model["components"][0], x)
            + probabilities(model["components"][1], x)
        )
    raise ValueError(model["family"])


def apply_node_activity(x, statistics, ages, config):
    first = statistics[:, 1] == 0
    source_age, target_age = ages
    horizon = np.float32(config["half_life"])
    x[first, 23] = -np.exp2(-target_age[first] / horizon)
    if config["use_source"]:
        x[first, 18] = -np.exp2(-source_age[first] / horizon)
    return x


def predict_readout(state, base17, statistics7, protocol="undergs", node_ages=None):
    """Score fresh exact features with an already locked, saved readout.

    ``state = joblib.load(selected_readout_path)``. ``base17`` is the unchanged
    17-channel graph feature frame in ``evogfm.features`` order; statistics7 is
    TemporalHistory's age/count frame before recency conversion. Return one
    probability per ordered candidate. This function does no fitting, graph
    extraction, candidate sampling or cache access.
    """
    base17 = np.asarray(base17, dtype=np.float32)
    statistics7 = np.asarray(statistics7, dtype=np.float32)
    if (
        base17.ndim != 2
        or base17.shape[1] != 17
        or statistics7.shape != (len(base17), 7)
    ):
        raise ValueError("Expected aligned [N,17] graph and [N,7] history features")
    if len(base17) == 0:
        return np.empty(0, dtype=np.float32 if protocol == "undergs" else np.float64)
    x = features(base17, statistics7, state["model"]["input"], state["half_life"])
    if "node_activity" in state:
        if node_ages is None:
            raise ValueError(
                "This readout requires source and destination activity ages"
            )
        x = apply_node_activity(x, statistics7, node_ages, state["node_activity"])
    output = probabilities(state["model"], x)
    return output.astype(np.float32) if protocol == "undergs" else output


def model_capacity(model):
    if model["family"] == "logistic":
        return int(model["fitted"]["coefficient"].size)
    if model["family"] == "hgb":
        return int(
            sum(tree.nodes.size for row in model["fitted"]._predictors for tree in row)
        )
    if model["family"] == "pair_rank":
        return int((2 * model["fitted"].get_tree_leaf_counts() - 1).sum())
    return sum(model_capacity(row) for row in model["components"])


def auc_metrics(packs):
    pooled = float(
        roc_auc_score(
            np.concatenate([row["labels"] for row in packs]),
            np.concatenate([row["scores"] for row in packs]),
        )
    )
    return dict(
        pooled_auc=pooled,
        snapshot_mean_auc=float(np.mean([row["auc"] for row in packs])),
    )


def auc_pack(record, probability):
    npos, nneg = record["npos"], record["nneg"]
    labels = np.r_[np.ones(len(record["indices"])), np.zeros(nneg)]
    scores = np.r_[probability[record["indices"]], probability[npos : npos + nneg]]
    return dict(labels=labels, scores=scores, auc=float(roc_auc_score(labels, scores)))


def validation_auc(case, models):
    packs = {name: [] for name in models}
    for record in records(case, "validation"):
        for input_name in INPUTS:
            x = features(
                record["base"], record["statistics"], input_name, case["half_life"]
            )
            for name, model in models.items():
                if model["input"] != input_name:
                    continue
                score = probabilities(model, x)
                if case["protocol"] == "undergs":
                    score = score.astype(np.float32)
                packs[name].append(auc_pack(record, score))
    metric = "pooled_auc" if case["protocol"] == "decoupled" else "snapshot_mean_auc"
    return {
        name: dict(auc=auc_metrics(rows)[metric], **auc_metrics(rows))
        for name, rows in packs.items()
    }


def select_case(case, destination):
    lock_path = destination / "validation_lock.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = list(records(case, "train"))
    labels = np.concatenate(
        [np.r_[np.ones(row["npos"]), np.zeros(row["nneg"])] for row in train]
    )
    models, fit_audit = {}, {}
    for input_name in INPUTS:
        x = np.concatenate(
            [
                features(row["base"], row["statistics"], input_name, case["half_life"])
                for row in train
            ]
        )
        fitted, audit = fit_logistic(x, labels, ridge=10.0)
        name = f"{input_name}_logistic10"
        models[name] = dict(family="logistic", input=input_name, fitted=fitted)
        fit_audit[name] = audit
        for tree in TREE_GRID:
            name = f"{input_name}_{tree['name']}"
            settings = {key: value for key, value in tree.items() if key != "name"}
            settings.update(loss="log_loss", early_stopping=False, random_state=42)
            begin = time.perf_counter()
            fitted = HistGradientBoostingClassifier(**settings).fit(x, labels)
            models[name] = dict(
                family="hgb", input=input_name, fitted=fitted, settings=settings
            )
            fit_audit[name] = dict(
                fit_wall_s=time.perf_counter() - begin,
                train_rows=len(x),
                input_dimensions=x.shape[1],
                settings=settings,
            )
    val = validation_auc(case, models)
    for input_name in INPUTS:
        best_tree = max(
            (f"{input_name}_{tree['name']}" for tree in TREE_GRID),
            key=lambda name: (val[name]["auc"], -model_capacity(models[name]), name),
        )
        models[f"{input_name}_blend50"] = dict(
            family="blend",
            input=input_name,
            components=[models[f"{input_name}_logistic10"], models[best_tree]],
            component_ids=[f"{input_name}_logistic10", best_tree],
        )
    blend = {
        name: model for name, model in models.items() if model["family"] == "blend"
    }
    val.update(validation_auc(case, blend))
    selected = max(
        models, key=lambda name: (val[name]["auc"], -model_capacity(models[name]), name)
    )
    model_file = destination / "selected_readout.joblib"
    joblib.dump(
        dict(
            model=models[selected],
            half_life=case["half_life"],
            seed=42,
            selected_id=selected,
            feature_extractor_frozen=True,
        ),
        model_file,
        compress=3,
    )
    candidates = [
        dict(
            id=name,
            input=model["input"],
            family=model["family"],
            capacity_units=model_capacity(model),
            **val[name],
        )
        for name, model in models.items()
    ]
    lock = dict(
        protocol=case["protocol"],
        dataset=case["dataset"],
        locked_utc=now(),
        selected_id=selected,
        selected=next(row for row in candidates if row["id"] == selected),
        model_artifact=str(model_file),
        model_artifact_bytes=model_file.stat().st_size,
        selection="validation AUC only; no test or MRR selection",
        candidates=candidates,
        fit_audit=fit_audit,
        selection_wall_s=time.perf_counter() - started,
        cost_scope="Offline fit/validation over reused caches; not end-to-end graph cost",
    )
    write_json(lock_path, lock)
    write_csv(destination / "candidate_validation.csv", candidates)
    print(
        f"LOCK {case['protocol']}/{case['dataset']} {selected} valAUC={val[selected]['auc']:.6f}",
        flush=True,
    )
    return lock


def ranking_metrics(record, probability, protocol):
    npos, nneg = record["npos"], record["nneg"]
    positive, negative = probability[:npos], probability[npos + nneg :]
    if negative.size == 0:
        return dict(
            mrr=None,
            random_tie_expected_mrr=None,
            pessimistic_mrr=None,
            query_with_ties_fraction=None,
        )
    if protocol == "undergs":
        if negative.size % npos:
            raise ValueError("UnderGS ranking layout changed")
        matrix = negative.reshape(-1, npos).T
        combined = np.column_stack((positive, matrix))
        order = np.argsort(-combined, axis=1)
        ranks = np.argmax(order == 0, axis=1) + 1
    else:
        positive = np.asarray(
            [
                positive[record["positive_sources"] == node].max()
                for node in record["rank_sources"]
            ]
        )
        matrix = negative.reshape(len(positive), -1)
        ranks = 1 + (matrix >= positive[:, None]).sum(axis=1)
    greater = (matrix > positive[:, None]).sum(axis=1)
    ties = (matrix == positive[:, None]).sum(axis=1)
    harmonic = np.r_[0.0, np.cumsum(1 / np.arange(1, matrix.shape[1] + 2))]
    expected = (harmonic[greater + ties + 1] - harmonic[greater]) / (ties + 1)
    return dict(
        mrr=float(np.mean(1 / ranks)),
        random_tie_expected_mrr=float(expected.mean()),
        pessimistic_mrr=float(np.mean(1 / (greater + ties + 1))),
        query_with_ties_fraction=float(np.mean(ties > 0)),
    )


def evaluate(case, lock, phase, destination):
    state = joblib.load(lock["model_artifact"])
    model = state["model"]
    rows, packs = [], []
    scoring_s = 0.0
    for record in records(case, phase, full=(phase != "train")):
        begin = time.perf_counter()
        x = features(
            record["base"], record["statistics"], model["input"], case["half_life"]
        )
        probability = probabilities(model, x)
        if case["protocol"] == "undergs":
            probability = probability.astype(np.float32)
        scoring_s += time.perf_counter() - begin
        pack = auc_pack(record, probability)
        metric = ranking_metrics(record, probability, case["protocol"])
        rows.append(
            dict(
                time=record["time"],
                auc=pack["auc"],
                **metric,
                positive_events=record["npos"],
                auc_positive_events=len(record["indices"]),
                auc_negative_events=record["nneg"],
                ranking_candidates=len(probability) - record["npos"] - record["nneg"],
            )
        )
        packs.append(pack)
    aggregate = auc_metrics(packs)
    aggregate["auc"] = (
        aggregate["pooled_auc"]
        if case["protocol"] == "decoupled"
        else aggregate["snapshot_mean_auc"]
    )
    for key in (
        "mrr",
        "random_tie_expected_mrr",
        "pessimistic_mrr",
        "query_with_ties_fraction",
    ):
        available = [row[key] for row in rows if row[key] is not None]
        aggregate[key] = float(np.mean(available)) if available else None
    aggregate.update(
        snapshots=len(rows),
        score_pipeline_s=scoring_s,
        cost_scope="Selected readout and input concatenation on cached rows; excludes graph/cache creation, disk input and metric/output time",
        positive_events=sum(row["positive_events"] for row in rows),
        auc_positive_events=sum(row["auc_positive_events"] for row in rows),
        auc_negative_events=sum(row["auc_negative_events"] for row in rows),
    )
    if phase == "train":
        aggregate["mrr_unavailable_reason"] = (
            "Existing train caches contain no official ranking-negative feature suffix"
        )
    if phase == "validation":
        np.testing.assert_allclose(
            aggregate["auc"], lock["selected"]["auc"], rtol=0, atol=1e-12
        )
    write_json(destination / f"{phase}_metrics.json", dict(**aggregate, rows=rows))
    write_csv(destination / f"{phase}_snapshots.csv", rows)
    return aggregate


def baseline(case):
    directory = Path(case["baseline_directory"])
    if case["protocol"] == "decoupled":
        path = directory / "official/summary.json"
        replay = json.loads((directory / "reference_replay.json").read_text())
        if not replay["all_native_metrics_reproduced"]:
            raise ValueError("DD native baseline replay failed")
    else:
        path = directory / "causal_adaptation/undergs/summary.json"
    result = json.loads(path.read_text())
    return dict(
        name=(
            "DecoupledDGNN"
            if case["protocol"] == "decoupled"
            else "UnderGS (causal adaptation)"
        ),
        auc=result["auc"],
        mrr=result["mrr"],
        source=str(path),
    )


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    plan = predeclare(output)
    locks = []
    for case in plan["cases"]:
        locks.append(select_case(case, output / case["protocol"] / case["dataset"]))
    all_locks_path = output / "all_validation_locks.json"
    if not all_locks_path.exists():
        write_json(
            all_locks_path,
            dict(
                created_utc=now(),
                case_count=6,
                all_cases_selected_before_new_test_feature_read=True,
                locks=locks,
            ),
        )
    results = []
    for case, lock in zip(plan["cases"], locks):
        destination = output / case["protocol"] / case["dataset"]
        summary_path = destination / "summary.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text())
        else:
            metrics = {
                phase: evaluate(case, lock, phase, destination)
                for phase in ("train", "validation", "test")
            }
            reference = baseline(case)
            old_path = (
                ROOT
                / "runs/optimization/temporal_calibration_s42/locked_test"
                / case["protocol"]
                / case["dataset"]
                / "summary.json"
            )
            old = json.loads(old_path.read_text())["results"]["locked_selected"]
            result = dict(
                **case,
                selected_id=lock["selected_id"],
                seed=42,
                evaluation_status=plan["evaluation_status"],
                selected_model_artifact=lock["model_artifact"],
                selected_model_artifact_bytes=lock["model_artifact_bytes"],
                metrics=metrics,
                baseline=reference,
                previous_development=dict(auc=old["auc"], mrr=old["mrr"]),
                test_auc_change_pp=100 * (metrics["test"]["auc"] - old["auc"]),
                test_auc_gap_to_baseline_pp=100
                * (metrics["test"]["auc"] - reference["auc"]),
                quality_only=True,
                end_to_end_cost_measured=False,
            )
            write_json(summary_path, result)
        results.append(result)
        print(
            f"TEST {case['protocol']}/{case['dataset']} {result['selected_id']} "
            f"AUC={result['metrics']['test']['auc']:.6f} MRR={result['metrics']['test']['mrr']:.6f}",
            flush=True,
        )
    write_json(
        output / "summary.json",
        dict(
            status="complete",
            results=results,
            plan=str(output / "predeclared_plan.json"),
            all_validation_locks=str(all_locks_path),
            evaluation_status=plan["evaluation_status"],
            cost_scope=plan["cost_scope"],
        ),
    )
    flat = []
    for result in results:
        flat.append(
            dict(
                protocol=result["protocol"],
                dataset=result["dataset"],
                selected=result["selected_id"],
                baseline_auc=100 * result["baseline"]["auc"],
                previous_auc=100 * result["previous_development"]["auc"],
                train_auc=100 * result["metrics"]["train"]["auc"],
                validation_auc=100 * result["metrics"]["validation"]["auc"],
                test_auc=100 * result["metrics"]["test"]["auc"],
                baseline_mrr=result["baseline"]["mrr"],
                previous_mrr=result["previous_development"]["mrr"],
                validation_mrr=result["metrics"]["validation"]["mrr"],
                test_mrr=result["metrics"]["test"]["mrr"],
                readout_bytes=result["selected_model_artifact_bytes"],
            )
        )
    write_csv(output / "results.csv", flat)
    lines = [
        "# A0：轻量读出质量修复（开发基准）",
        "",
        "固定 seed 42；同一冻结特征缓存、时间划分、候选和可见边。每组在预声明的12个候选中只按 validation AUC 选择，六组全部锁定后读取 test。旧 test 已用于前期开发和根因诊断，本轮不能称为新的盲测。",
        "",
        "| 协议 | 数据集 | 选择的读出 | Train AUC | Val AUC | Baseline AUC | 旧 AUC | 当前 AUC | Baseline MRR | 当前 MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat:
        lines.append(
            f"| {row['protocol']} | {row['dataset']} | {row['selected']} | {row['train_auc']:.2f} | {row['validation_auc']:.2f} | {row['baseline_auc']:.2f} | {row['previous_auc']:.2f} | {row['test_auc']:.2f} | {row['baseline_mrr']:.4f} | {row['test_mrr']:.4f} |"
        )
    lines += [
        "",
        "AUC单位为%；MRR为原协议数值。DD使用pooled native scored-event AUC、source-best/1000负例MRR；UnderGS使用snapshot-mean AUC、官方NumPy排序/100负例MRR。不同协议的MRR不可直接比较。",
        "",
        "Baseline来自本地官方源码运行，不是论文发表均值。UnderGS此处为causal adaptation：先评分后摄入当前正边；不是原论文/原实现的当前正边预先可见轨道。DD已复现捕获的实际官方评分候选。",
        "",
        "训练缓存没有官方ranking负例特征，因此train MRR为null；val/test完整MRR与并列分数诊断已保存。",
        "",
        "仅新增轻量读出，不重新训练或提取冻结图特征。计时只覆盖离线缓存上的拟合/评分，不是端到端图处理成本。所选读出已保存为每组selected_readout.joblib。质量修复不能代替精确增量维护、完整资源计费和负载实验。",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/vldb/quality"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output.resolve())
