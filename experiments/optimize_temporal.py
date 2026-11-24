"""Validation-only selection of 17-channel temporal residuals, followed by locked test."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from evogfm.readout import predict_residual
from evogfm.temporal import (
    export_protocol,
    prepare_dataset,
    ranking_mrr,
    ranking_tie_diagnostics,
    replace_temporal,
)
from experiments.full_graph import load_checkpoint
from experiments.temporal_official import build_feature_cache, write_json


def cached_features(directory, entry, variant, half_life):
    archive = np.load(Path(directory) / entry["file"])
    base = torch.from_numpy(archive["base"])
    changed = replace_temporal(
        base, torch.from_numpy(archive["statistics"]), variant, half_life
    )
    return archive, base, changed


def fit_temporal_residual(features, offset, labels, ridge):
    """The released BCE/ridge objective, divided by N for stable stopping."""
    features = features.double()
    mean, scale = features.mean(0), features.std(0).clamp_min(0.05)
    normalized = (features - mean) / scale
    x = torch.cat(
        (normalized, torch.ones(len(normalized), 1, dtype=torch.float64)), dim=1
    ).numpy()
    offset, labels = offset.double().numpy(), labels.double().numpy()
    penalty = np.full(x.shape[1], float(ridge))
    penalty[-1] *= 0.1

    def objective(coefficient):
        logits = offset + x @ coefficient
        return float(
            np.sum(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * np.dot(coefficient * penalty, coefficient)
        ) / len(x)

    def gradient(coefficient):
        return (
            x.T @ (expit(offset + x @ coefficient) - labels) + penalty * coefficient
        ) / len(x)

    def hessian(coefficient):
        probability = expit(offset + x @ coefficient)
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
    # Near the optimum, objective-value rounding can stop trust-exact while
    # the analytic gradient still permits a more accurate Newton correction.
    for _ in range(3):
        derivative = gradient(coefficient)
        if np.max(np.abs(derivative)) <= 1e-10:
            break
        candidate = coefficient - np.linalg.solve(hessian(coefficient), derivative)
        if np.max(np.abs(gradient(candidate))) >= np.max(np.abs(derivative)):
            break
        coefficient = candidate
    error = float(np.max(np.abs(gradient(coefficient))))
    if error > 1e-9:
        raise RuntimeError(
            f"Temporal residual solver gradient {error}: {result.message}"
        )
    return torch.from_numpy(coefficient), mean, scale


@torch.no_grad()
def fit_configuration(directory, manifest, head, variant, half_life, ridge):
    all_features, all_offset, all_labels = [], [], []
    for entry in manifest["index"]:
        if entry["phase"] != "train":
            continue
        archive, base, features = cached_features(directory, entry, variant, half_life)
        npos = int(archive["positive_count"])
        all_features.append(features)
        # Preserve source-head semantics; only the existing linear calibrator
        # sees the replaced temporal channels.
        all_offset.append(head(base).flatten())
        all_labels.append(
            torch.cat((torch.ones(npos), torch.zeros(len(features) - npos)))
        )
    return fit_temporal_residual(
        torch.cat(all_features), torch.cat(all_offset), torch.cat(all_labels), ridge
    )


@torch.no_grad()
def evaluate_configuration(
    directory,
    manifest,
    head,
    fitted,
    variant,
    half_life,
    protocol,
    phase,
    tie_audit=False,
):
    started = time.perf_counter()
    rows = []
    for entry in manifest["index"]:
        if entry["phase"] != phase:
            continue
        archive, base, features = cached_features(directory, entry, variant, half_life)
        offset = torch.cat([head(block).flatten() for block in base.split(8192)])
        prediction = (
            predict_residual(features, offset, fitted).sigmoid().float().numpy()
        )
        npos, nneg = int(archive["positive_count"]), int(archive["auc_negative_count"])
        truth = np.r_[np.ones(npos), np.zeros(nneg)]
        row = dict(
            time=entry["time"],
            positives=npos,
            auc=float(roc_auc_score(truth, prediction[: npos + nneg])),
            ap=float(average_precision_score(truth, prediction[: npos + nneg])),
            mrr=ranking_mrr(
                protocol,
                prediction[:npos],
                prediction[npos + nneg :],
                archive["positive_sources"],
                archive["rank_sources"],
            ),
        )
        if tie_audit:
            row.update(
                ranking_tie_diagnostics(
                    protocol,
                    prediction[:npos],
                    prediction[npos + nneg :],
                    archive["positive_sources"],
                    archive["rank_sources"],
                )
            )
        rows.append(row)
    result = {
        key: float(np.mean([row[key] for row in rows])) for key in ("auc", "ap", "mrr")
    }
    result["snapshots"] = len(rows)
    if tie_audit:
        result.update(
            {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
                if key not in ("time", "positives", "auc", "ap", "mrr")
            }
        )
    result["evaluation_pipeline_s"] = time.perf_counter() - started
    return result, rows


def run_selection(
    protocol,
    dataset,
    seed,
    output,
    official_output,
    checkpoint,
    baselines,
    stage,
    threads=2,
    half_lives=(1.0, 4.0, 16.0),
    ridges=(0.1, 1.0, 10.0),
    reference_lock=None,
):
    torch.set_num_threads(threads)
    directory = Path(output) / protocol / dataset / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = directory / "validation_lock.json"
    if stage == "validation" and lock_file.exists():
        raise FileExistsError(f"Validation is already locked: {lock_file}")
    if stage == "test" and not lock_file.exists():
        raise FileNotFoundError(
            "Run --stage validation before generating test features"
        )
    randomness = (
        "training negative candidates depend on seed; source backbone and structure deterministic; UnderGS validation/test candidates use fixed seeds 0/2 per snapshot"
        if protocol == "undergs"
        else "all negative candidates use an independent seeded export stream; original training-dependent RNG state is not reproduced bitwise; source backbone and structure deterministic"
    )
    events = prepare_dataset(dataset, protocol, baselines=baselines)
    source_directory = Path(official_output) / protocol / dataset / f"s{seed}"
    protocol_directory = source_directory / "protocol"
    if (protocol_directory / "index.json").exists():
        index = json.loads((protocol_directory / "index.json").read_text())
    else:
        index = export_protocol(events, protocol, seed, protocol_directory)
    cache = source_directory / "features_full"
    manifest = build_feature_cache(
        events, index, protocol_directory, cache, checkpoint, stage, threads=threads
    )
    _, head, _ = load_checkpoint(checkpoint)
    if stage == "validation":
        candidates = [("original", 1.0, ridge) for ridge in ridges]
        candidates += list(
            itertools.product(
                ("recency", "recency_frequency", "pair_frequency_recency"),
                half_lives,
                ridges,
            )
        )
        candidates += [("pair_frequency", 1.0, ridge) for ridge in ridges]
        reference = None
        if reference_lock is not None:
            reference = json.loads(Path(reference_lock).read_text())
            if reference["protocol"] != protocol or reference["dataset"] != dataset:
                raise ValueError("Reference lock must match protocol and dataset")
            candidates = [
                (row["variant"], row["half_life"], row["ridge"])
                for row in (reference["baseline"], reference["selected"])
            ]
        trials = []
        fits = {}
        for variant, half_life, ridge in candidates:
            started = time.perf_counter()
            fitted = fit_configuration(cache, manifest, head, variant, half_life, ridge)
            fit_seconds = time.perf_counter() - started
            metric, _ = evaluate_configuration(
                cache,
                manifest,
                head,
                fitted,
                variant,
                half_life,
                protocol,
                "validation",
            )
            row = dict(
                variant=variant,
                half_life=half_life,
                ridge=ridge,
                fit_pipeline_s=fit_seconds,
                **metric,
            )
            trials.append(row)
            fits[(variant, half_life, ridge)] = fitted
            print(
                json.dumps(
                    dict(
                        action="validation_trial",
                        protocol=protocol,
                        dataset=dataset,
                        **row,
                    )
                ),
                flush=True,
            )
        baseline = max(
            (row for row in trials if row["variant"] == "original"),
            key=lambda row: row["mrr"],
        )
        selected = max(
            (row for row in trials if row["variant"] != "original"),
            key=lambda row: row["mrr"],
        )
        lock = dict(
            protocol=protocol,
            dataset=dataset,
            seed=seed,
            selection_metric=(
                "validation MRR"
                if reference is None
                else "fixed reference configurations; no selection on this seed"
            ),
            baseline=baseline,
            selected=selected,
            trials=trials,
            candidate_grid_revision="v2_pair_only_added_after_BA_development_test; BO/UCI validation untouched",
            reference_lock=(
                str(Path(reference_lock).resolve()) if reference_lock else None
            ),
            randomness=randomness,
            test_consulted=False,
            parameter_count_change=0,
            calibrator_input_dimensions=manifest["feature_dimensions"],
            offset="frozen source head always receives original 17 channels; original-size residual receives temporal replacement",
            residual_solver="released sum-BCE+ridge objective divided by N (same minimizer); float64 trust-exact with analytic Newton gradient refinement; average-objective gradient infinity norm <=1e-9",
            temporal_history_state_bytes=manifest["temporal_history_state_bytes"],
            cache_manifest=str((cache / "manifest.json").resolve()),
            observation_policy=manifest["observation_policy"],
            protocol_metadata=events.metadata,
        )
        torch.save(
            {
                name: fits[(row["variant"], row["half_life"], row["ridge"])]
                for name, row in (("baseline", baseline), ("selected", selected))
            },
            directory / "readouts.pt",
        )
        write_json(lock_file, lock)
        return lock
    lock = json.loads(lock_file.read_text())
    fitted = torch.load(
        directory / "readouts.pt", map_location="cpu", weights_only=False
    )
    results = {}
    for name in ("baseline", "selected"):
        row = lock[name]
        metric, per_snapshot = evaluate_configuration(
            cache,
            manifest,
            head,
            fitted[name],
            row["variant"],
            row["half_life"],
            protocol,
            "test",
        )
        results[name] = dict(configuration=row, **metric)
        with (directory / f"test_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_snapshot[0]))
            writer.writeheader()
            writer.writerows(per_snapshot)
    result = dict(
        protocol=protocol,
        dataset=dataset,
        seed=seed,
        results=results,
        validation_lock=str(lock_file.resolve()),
        test_used_for_selection=False,
        model_parameter_count_change=0,
        temporal_state_bytes=manifest["temporal_history_state_bytes"],
        scorer_cache_bytes=manifest.get("scorer_cache_bytes_final"),
        backbone_parameters=manifest["backbone_parameters"],
        head_parameters=manifest["head_parameters"],
        residual_parameters=manifest["feature_dimensions"] + 1,
        residual_state_bytes={
            name: sum(value.numel() * value.element_size() for value in fitted[name])
            for name in ("baseline", "selected")
        },
        feature_cache_disk_bytes=sum(
            (cache / entry["file"]).stat().st_size for entry in manifest["index"]
        ),
        maintenance_update_s=sum(
            row["update_s"] for row in manifest.get("maintenance_updates", [])
        ),
        reference_lock=lock.get("reference_lock"),
        randomness=lock.get("randomness", randomness),
        timing_scope="development CPU run; fit includes cache loading and frozen-head evaluation; evaluation includes cache loading, temporal replacement, readout and metrics; maintenance replay is recorded separately; no isolated system-throughput claim",
        protocol_metadata=events.metadata,
        observation_policy=manifest["observation_policy"],
        comparison_scope=(
            "official candidate/metric distribution; predict-before-ingest observation differs from current-positive-aware UnderGS code"
            if protocol == "undergs"
            else "official candidate distribution and per-source-best MRR; independent query RNG differs from original training-dependent RNG"
        ),
    )
    write_json(directory / "test_summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def evaluate_history_heuristics(protocol, dataset, seed, output, official_output):
    """Evaluate zero-parameter history-only controls on existing locked queries."""
    directory = Path(output) / protocol / dataset / f"s{seed}"
    lock_file = directory / "validation_lock.json"
    lock = json.loads(lock_file.read_text())
    half_life = float(lock["selected"]["half_life"])
    cache = Path(official_output) / protocol / dataset / f"s{seed}" / "features_full"
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["stage"] != "test":
        raise ValueError("Generate the locked test cache before evaluating heuristics")
    methods = {}
    for name in ("past_pair_frequency", "past_pair_recency"):
        started = time.perf_counter()
        rows = []
        for entry in manifest["index"]:
            if entry["phase"] != "test":
                continue
            with np.load(cache / entry["file"]) as archive:
                statistics = archive["statistics"]
                score = (
                    statistics[:, 1]
                    if name == "past_pair_frequency"
                    else np.exp2(-statistics[:, 0] / half_life)
                )
                npos, nneg = int(archive["positive_count"]), int(
                    archive["auc_negative_count"]
                )
                truth = np.r_[np.ones(npos), np.zeros(nneg)]
                rows.append(
                    dict(
                        time=entry["time"],
                        auc=float(roc_auc_score(truth, score[: npos + nneg])),
                        ap=float(average_precision_score(truth, score[: npos + nneg])),
                        mrr=ranking_mrr(
                            protocol,
                            score[:npos],
                            score[npos + nneg :],
                            archive["positive_sources"],
                            archive["rank_sources"],
                        ),
                        **ranking_tie_diagnostics(
                            protocol,
                            score[:npos],
                            score[npos + nneg :],
                            archive["positive_sources"],
                            archive["rank_sources"],
                        ),
                        constant_score_official_mrr=ranking_mrr(
                            protocol,
                            np.zeros(npos),
                            np.zeros(len(score) - npos - nneg),
                            archive["positive_sources"],
                            archive["rank_sources"],
                        ),
                    )
                )
        methods[name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "time"
        }
        methods[name].update(
            parameters=0,
            model_weights_bytes=0,
            snapshots=len(rows),
            half_life=half_life if name == "past_pair_recency" else None,
            evaluation_pipeline_s=time.perf_counter() - started,
        )
        with (directory / f"heuristic_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(
        project_title="Incremental Predictive Analytics over Evolving Graphs",
        protocol=protocol,
        dataset=dataset,
        seed=seed,
        methods=methods,
        foundation_model_used=False,
        statistics="column0 past directed pair age (infinity for unseen); column1 log1p past directed pair count",
        recency_half_life_source="existing selected validation lock; no new search",
        validation_lock=str(lock_file.resolve()),
        test_used_for_selection=False,
        observation_policy=manifest["observation_policy"],
        temporal_state_bytes=manifest["temporal_history_state_bytes"],
        memory_scope="common cached TemporalHistory accounting includes activity fields unused by these heuristics; not a minimal heuristic implementation",
        timing_scope="cached-history evaluation plus metrics, excludes stream maintenance",
        tie_diagnostic_scope="official mrr is retained exactly; random_tie_expected_mrr averages reciprocal rank uniformly over tied positions; pessimistic_mrr places positive behind all tied negatives; constant-score control is a metric diagnostic, not a tuned predictor",
    )
    write_json(directory / "history_heuristics.json", result)
    print(json.dumps(result), flush=True)
    return result


def audit_ranking_metrics(
    protocol, dataset, seed, output, official_output, checkpoint, threads=2
):
    """Audit the same frozen scores and queries without changing any model or lock."""
    torch.set_num_threads(threads)
    directory = Path(output) / protocol / dataset / f"s{seed}"
    lock_file = directory / "validation_lock.json"
    lock = json.loads(lock_file.read_text())
    fitted = torch.load(
        directory / "readouts.pt", map_location="cpu", weights_only=False
    )
    cache = Path(official_output) / protocol / dataset / f"s{seed}" / "features_full"
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["stage"] != "test":
        raise ValueError("Audit requires an existing locked test cache")
    _, head, _ = load_checkpoint(checkpoint)
    methods = {}
    for name in ("baseline", "selected"):
        configuration = lock[name]
        metric, rows = evaluate_configuration(
            cache,
            manifest,
            head,
            fitted[name],
            configuration["variant"],
            configuration["half_life"],
            protocol,
            "test",
            tie_audit=True,
        )
        methods[name] = dict(configuration=configuration, **metric)
        with (directory / f"metric_audit_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    heuristic = evaluate_history_heuristics(
        protocol, dataset, seed, output, official_output
    )
    methods.update(heuristic["methods"])
    constant_rows = []
    for entry in manifest["index"]:
        if entry["phase"] != "test":
            continue
        with np.load(cache / entry["file"]) as archive:
            npos, nneg = int(archive["positive_count"]), int(
                archive["auc_negative_count"]
            )
            positive = np.zeros(npos)
            negative = np.zeros(len(archive["statistics"]) - npos - nneg)
            constant_rows.append(
                dict(
                    auc=0.5,
                    ap=0.5,
                    mrr=ranking_mrr(
                        protocol,
                        positive,
                        negative,
                        archive["positive_sources"],
                        archive["rank_sources"],
                    ),
                    **ranking_tie_diagnostics(
                        protocol,
                        positive,
                        negative,
                        archive["positive_sources"],
                        archive["rank_sources"],
                    ),
                )
            )
    methods["constant_zero"] = {
        key: float(np.mean([row[key] for row in constant_rows]))
        for key in constant_rows[0]
    }
    methods["constant_zero"].update(parameters=0, snapshots=len(constant_rows))
    result = dict(
        project_title="Incremental Predictive Analytics over Evolving Graphs",
        protocol=protocol,
        dataset=dataset,
        seed=seed,
        methods=methods,
        validation_lock=str(lock_file.resolve()),
        model_or_lock_changed=False,
        original_official_mrr_preserved=True,
        metric_definition="pessimistic rank=1+sum(negative>=positive); uniform random-tie expected reciprocal rank averages over the tied rank positions",
        undergs_tie_note="released compute_mrr uses np.argsort on [positive, negatives...] and puts positive in array position zero, so unstable-sort tie ordering can favor constant/mostly-zero scores",
        protocol_metadata=manifest["dataset"],
        observation_policy=manifest["observation_policy"],
    )
    write_json(directory / "ranking_metric_audit.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["undergs", "decoupled"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["bitcoin-alpha"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261115])
    parser.add_argument(
        "--stage", choices=["validation", "test", "heuristics", "audit"], required=True
    )
    parser.add_argument("--output", default="runs/optimization/temporal_dev")
    parser.add_argument("--official-output", default="runs/official_a")
    parser.add_argument("--checkpoint", default="checkpoints/reference.pt")
    parser.add_argument("--baselines", default="baselines")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--reference-lock",
        help="Fit exactly the reference baseline/selected configurations; no new selection",
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        for seed in args.seeds:
            if args.stage == "heuristics":
                evaluate_history_heuristics(
                    args.protocol, dataset, seed, args.output, args.official_output
                )
                continue
            if args.stage == "audit":
                audit_ranking_metrics(
                    args.protocol,
                    dataset,
                    seed,
                    args.output,
                    args.official_output,
                    args.checkpoint,
                    args.threads,
                )
                continue
            run_selection(
                args.protocol,
                dataset,
                seed,
                args.output,
                args.official_output,
                args.checkpoint,
                args.baselines,
                args.stage,
                args.threads,
                reference_lock=args.reference_lock,
            )


if __name__ == "__main__":
    main()
