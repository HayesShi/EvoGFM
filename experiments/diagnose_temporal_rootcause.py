"""Fixed-input, fixed-solver diagnosis of temporal channel replacement.

This is an internal controlled diagnostic, not a new model or a selection run.
It consumes the six existing paired feature/candidate caches without extracting
features, changing candidates, loading a backbone, or evaluating ranking rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

# Set these before importing numerical libraries; the diagnostic shares the host.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


CONDITIONS = ("original17", "overwrite17", "temporal_only7", "union24")
OVERWRITE_INDICES = (0, 1, 2, 3, 10, 11, 12)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def declare(root, output):
    cases = []
    for protocol in ("decoupled", "undergs"):
        for dataset in ("bitcoin-alpha", "bitcoin-otc", "uci"):
            alignment = root / "runs/protocol_alignment" / protocol / dataset / "s42"
            cache = (
                alignment / "features_full"
                if protocol == "decoupled"
                else alignment / "causal_adaptation/frozen/features"
            )
            candidates = (
                alignment / "protocol"
                if protocol == "decoupled"
                else alignment / "candidates"
            )
            lock_path = (
                root
                / "runs/optimization/temporal_calibration_s42/local_statistics"
                / protocol
                / dataset
                / "validation_lock.json"
            )
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            configuration = lock["selected"]["configuration"]
            cases.append(
                dict(
                    protocol=protocol,
                    dataset=dataset,
                    half_life=float(configuration["half_life"]),
                    historical_configuration=configuration,
                    validation_lock=str(lock_path.resolve()),
                    feature_cache=str(cache.resolve()),
                    candidates=str(candidates.resolve()),
                )
            )
    declaration = dict(
        experiment="A temporal-channel root-cause controlled diagnostic",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        seed=42,
        threads=1,
        ridge=10.0,
        offset_mode="zero",
        conditions=list(CONDITIONS),
        temporal7="Existing past-only statistics7; age columns 0 and 2 become exp2(-age/locked_half_life). No case-specific correction.",
        overwrite17_indices=list(OVERWRITE_INDICES),
        union24="original base17 concatenated with exactly the same temporal7",
        normalization="Train-only column mean and sample std (ddof=1), std floor .05; append intercept",
        solver="Sum BCE + .5*ridge*L2, divided by train N; intercept penalty .1*ridge; scipy trust-exact gtol1e-10 maxiter200 + <=3 analytic Newton refinements; gradient infinity norm <=1e-9",
        fitting="All cached train positive and AUC-negative rows only. No ranking rows.",
        auc="DD pooled native scored-event probabilities (float64); UnderGS unweighted snapshot-mean probabilities (float32), matching existing protocol precision.",
        candidate_identity="Every condition uses the same ordered cached rows and labels; native DD auc_positive_indices retained during evaluation.",
        selection="No condition selection, no hyperparameter search, no post-test model choice. Historical final mixed-transform results are context only.",
        cases=cases,
    )
    # Must be committed before any feature/candidate test archive is opened.
    target = output / "predeclared_conditions.json"
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite a diagnostic declaration: {target}"
        )
    write_json(target, declaration)
    return declaration


def transform(base, statistics, half_life):
    # float32 arithmetic preserves the cached feature/TemporalHistory precision.
    temporal = statistics.astype(np.float32, copy=True)
    temporal[:, [0, 2]] = np.exp2(-temporal[:, [0, 2]] / np.float32(half_life))
    overwritten = base.copy()
    overwritten[:, OVERWRITE_INDICES] = temporal
    result = dict(
        original17=base,
        overwrite17=overwritten,
        temporal_only7=temporal,
        union24=np.concatenate((base, temporal), axis=1),
    )
    for name, dimensions in (
        ("original17", 17),
        ("overwrite17", 17),
        ("temporal_only7", 7),
        ("union24", 24),
    ):
        if (
            result[name].shape != (len(base), dimensions)
            or not np.isfinite(result[name]).all()
        ):
            raise ValueError(f"Invalid {name} features")
    np.testing.assert_array_equal(result["union24"][:, :17], base)
    np.testing.assert_array_equal(result["union24"][:, 17:], temporal)
    return result


def load_phase(case, phase, manifest):
    cache, candidates = Path(case["feature_cache"]), Path(case["candidates"])
    records = []
    for entry in manifest["index"]:
        if entry["phase"] != phase:
            continue
        with np.load(cache / entry["file"]) as archive:
            npos, nneg = int(archive["positive_count"]), int(
                archive["auc_negative_count"]
            )
            count = npos + nneg
            # Do not copy or evaluate the much larger ranking suffix.
            base = archive["base"][:count].astype(np.float32, copy=True)
            statistics = archive["statistics"][:count].astype(np.float32, copy=True)
        with np.load(candidates / entry["file"]) as query:
            if (
                query["positive"].shape[1] != npos
                or query["auc_negative"].shape[1] != nneg
            ):
                raise ValueError("Candidate/cache positive-negative counts differ")
            indices = (
                query["auc_positive_indices"].copy()
                if "auc_positive_indices" in query
                else np.arange(npos)
            )
        if base.shape != (count, 17) or statistics.shape != (count, 7):
            raise ValueError("Cache dimensions differ from declared inputs")
        if len(indices) and (indices.min() < 0 or indices.max() >= npos):
            raise ValueError("Invalid native AUC positive indices")
        records.append(
            dict(
                time=int(entry["time"]),
                npos=npos,
                nneg=nneg,
                base=base,
                statistics=statistics,
                indices=indices,
                labels=np.r_[np.ones(npos), np.zeros(nneg)],
            )
        )
    if not records:
        raise ValueError(f"No records for {case['protocol']} {case['dataset']} {phase}")
    return records


def fit(features, labels, ridge=10.0):
    started = time.perf_counter()
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0, ddof=1), 0.05)
    x = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    labels = np.asarray(labels, dtype=np.float64)
    penalty = np.full(x.shape[1], ridge)
    penalty[-1] *= 0.1

    def objective(coefficient):
        logits = x @ coefficient
        return float(
            np.sum(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * np.dot(coefficient * penalty, coefficient)
        ) / len(x)

    def gradient(coefficient):
        return (x.T @ (expit(x @ coefficient) - labels) + penalty * coefficient) / len(
            x
        )

    def hessian(coefficient):
        probability = expit(x @ coefficient)
        return (
            x.T @ ((probability * (1 - probability))[:, None] * x) + np.diag(penalty)
        ) / len(x)

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options=dict(gtol=1e-10, maxiter=200),
    )
    coefficient = result.x
    refinements = 0
    for _ in range(3):
        derivative = gradient(coefficient)
        if np.max(np.abs(derivative)) <= 1e-10:
            break
        candidate = coefficient - np.linalg.solve(hessian(coefficient), derivative)
        if np.max(np.abs(gradient(candidate))) >= np.max(np.abs(derivative)):
            break
        coefficient, refinements = candidate, refinements + 1
    error = float(np.max(np.abs(gradient(coefficient))))
    if error > 1e-9:
        raise RuntimeError(f"BCE solver gradient {error}: {result.message}")
    return dict(coefficient=coefficient, mean=mean, scale=scale), dict(
        train_rows=len(x),
        input_dimensions=features.shape[1],
        coefficients=x.shape[1],
        gradient_infinity_norm=error,
        iterations=int(result.nit),
        newton_refinements=refinements,
        optimizer_message=str(result.message),
        objective=float(objective(coefficient)),
        fit_wall_s=time.perf_counter() - started,
    )


def evaluate(records, fitted, case, condition):
    rows, labels_all, scores_all = [], [], []
    for record in records:
        features = transform(record["base"], record["statistics"], case["half_life"])[
            condition
        ]
        logits = (features.astype(np.float64) - fitted["mean"]) / fitted[
            "scale"
        ] @ fitted["coefficient"][:-1] + fitted["coefficient"][-1]
        probability = expit(logits)
        if case["protocol"] == "undergs":
            probability = probability.astype(np.float32)
        scores = np.r_[
            probability[record["indices"]],
            probability[record["npos"] : record["npos"] + record["nneg"]],
        ]
        labels = np.r_[np.ones(len(record["indices"])), np.zeros(record["nneg"])]
        auc = float(roc_auc_score(labels, scores))
        rows.append(
            dict(
                time=record["time"],
                auc=auc,
                cached_positive_rows=record["npos"],
                auc_positive_rows=len(record["indices"]),
                auc_negative_rows=record["nneg"],
            )
        )
        labels_all.append(labels)
        scores_all.append(scores)
    pooled = float(
        roc_auc_score(np.concatenate(labels_all), np.concatenate(scores_all))
    )
    snapshot_mean = float(np.mean([row["auc"] for row in rows]))
    return dict(
        auc=pooled if case["protocol"] == "decoupled" else snapshot_mean,
        pooled_auc=pooled,
        snapshot_mean_auc=snapshot_mean,
        snapshots=len(rows),
        auc_positive_rows=sum(row["auc_positive_rows"] for row in rows),
        auc_negative_rows=sum(row["auc_negative_rows"] for row in rows),
        rows=rows,
    )


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    declaration = declare(root, output)
    results = []
    for case in declaration["cases"]:
        case_started = time.perf_counter()
        identity = f"{case['protocol']}/{case['dataset']}"
        print(f"START {identity}", flush=True)
        destination = output / case["protocol"] / case["dataset"]
        destination.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(
            (Path(case["feature_cache"]) / "manifest.json").read_text()
        )
        train = load_phase(case, "train", manifest)
        labels = np.concatenate([row["labels"] for row in train])
        fitted, fit_audit = {}, {}
        for condition in CONDITIONS:
            features = np.concatenate(
                [
                    transform(row["base"], row["statistics"], case["half_life"])[
                        condition
                    ]
                    for row in train
                ]
            )
            fitted[condition], fit_audit[condition] = fit(features, labels)
            np.savez_compressed(
                destination / f"{condition}_readout.npz", **fitted[condition]
            )
        write_json(destination / "fit_audit.json", fit_audit)
        metrics = {condition: {} for condition in CONDITIONS}
        for phase in ("train", "validation", "test"):
            records = train if phase == "train" else load_phase(case, phase, manifest)
            for condition in CONDITIONS:
                metrics[condition][phase] = evaluate(
                    records, fitted[condition], case, condition
                )
        summary = dict(
            **case,
            seed=42,
            ridge=10.0,
            offset_mode="zero",
            diagnostic_only=True,
            conditions=list(CONDITIONS),
            fits=fit_audit,
            metrics=metrics,
            case_wall_s=time.perf_counter() - case_started,
        )
        write_json(destination / "summary.json", summary)
        results.append(summary)
        print(
            f"DONE {identity} "
            + "; ".join(
                f"{c}:"
                + "/".join(
                    f"{metrics[c][p]['auc']:.6f}"
                    for p in ("train", "validation", "test")
                )
                for c in CONDITIONS
            ),
            flush=True,
        )
        write_json(
            output / "progress.json", dict(completed_cases=len(results), total_cases=6)
        )
    compact = []
    for case in results:
        for condition in CONDITIONS:
            compact.append(
                dict(
                    protocol=case["protocol"],
                    dataset=case["dataset"],
                    condition=condition,
                    half_life=case["half_life"],
                    **{
                        f"{phase}_auc": case["metrics"][condition][phase]["auc"]
                        for phase in ("train", "validation", "test")
                    },
                )
            )
    write_json(
        output / "summary.json",
        dict(
            status="complete",
            diagnostic_only=True,
            predeclared_conditions=str(
                (output / "predeclared_conditions.json").resolve()
            ),
            results=compact,
            cases=results,
        ),
    )
    lines = [
        "# A：时间通道覆盖受控诊断",
        "",
        "固定 seed 42、ridge 10、zero offset 和同一 BCE 求解器。各条件复用相同候选与冻结特征；没有选参或按 test 选择条件。数值为 AUC（%），不是新方法主结果。",
        "",
        "| 协议 | 数据集 | 输入条件 | Train | Validation | Test |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in compact:
        lines.append(
            f"| {row['protocol']} | {row['dataset']} | {row['condition']} | "
            + " | ".join(
                f"{100 * row[f'{p}_auc']:.2f}" for p in ("train", "validation", "test")
            )
            + " |"
        )
    lines += [
        "",
        "original17：原始17维。overwrite17：将七个原始通道替换为时间统计。temporal_only7：仅相同七维时间统计。union24：原始17维拼接相同七维时间统计。年龄统一以各 case 已锁定的 half-life 转为 exp2 衰减。",
        "",
        "Decoupled 协议使用 pooled native scored-event AUC；UnderGS 使用 snapshot-mean AUC。ranking 候选没有参与拟合或评估。",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=Path("runs/vldb/diagnostics"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.root.resolve(), arguments.output.resolve())
