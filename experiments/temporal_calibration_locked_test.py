"""One authorized test replay of six final seed42 locks and fixed controls.

All training/validation replays must pass before any test candidate or feature
file is opened. There is no grid, seed, selection, or configuration override.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from threadpoolctl import threadpool_limits

from evogfm.readout import predict_residual
from evogfm.temporal import TemporalHistory, ranking_mrr, ranking_tie_diagnostics
from experiments.decoupled_protocol import aggregate as aggregate_dd
from experiments.full_graph import load_checkpoint
from experiments.temporal_calibration import (
    OUTPUT,
    ROOT,
    evaluate_validation,
    fit,
    input_paths,
    load_records,
    transformed,
)
from experiments.temporal_official import write_json


def validate_readout(fitted):
    if tuple(value.numel() for value in fitted) != (18, 17, 17):
        raise ValueError(
            "Locked readout must contain 18 coefficients and 17 means/scales"
        )
    size = sum(value.numel() * value.element_size() for value in fitted)
    if size != 416 or not all(value.dtype == torch.float64 for value in fitted):
        raise ValueError("Locked float64 calibrator state must be exactly 416 bytes")
    return size


def preflight_cases(output, head):
    cases, audit_rows = [], []
    paths = sorted((OUTPUT / "local_statistics").glob("*/*/validation_lock.json"))
    if len(paths) != 6:
        raise ValueError("Expected all six final validation locks")
    for path in paths:
        locked = json.loads(path.read_text())
        protocol, dataset = locked["protocol"], locked["dataset"]
        original_path = OUTPUT / protocol / dataset / "validation_lock.json"
        original = json.loads(original_path.read_text())
        selected = locked["selected"]
        if (
            locked["seed"] != 42
            or selected["configuration"].get("transform") == "local_graph_statistics"
        ):
            raise ValueError(
                "This finite test run expects the six already locked offset-family fallbacks"
            )
        selected_fit = torch.load(path.parent / "readouts.pt", weights_only=False)[
            selected["id"]
        ]
        original_fit = torch.load(
            original_path.parent / "readouts.pt", weights_only=False
        )["reference"]
        models = {
            "fixed_reference": {
                "configuration": original["reference"]["configuration"],
                "fitted": original_fit,
                "validation": original["reference"]["validation"],
            },
            "locked_selected": {
                "configuration": selected["configuration"],
                "fitted": selected_fit,
                "validation": selected["validation"],
            },
        }
        _, cache, candidates = input_paths(protocol, dataset)
        records, cache_audit = load_records(protocol, cache, candidates, head)
        model_audits = {}
        for name, model in models.items():
            validate_readout(model["fitted"])
            replay_fit = fit(records, model["configuration"])
            for replay, saved in zip(replay_fit, model["fitted"]):
                torch.testing.assert_close(replay, saved, rtol=1e-8, atol=1e-9)
            metric, _ = evaluate_validation(
                records, model["configuration"], model["fitted"], protocol
            )
            for key, expected in model["validation"].items():
                np.testing.assert_allclose(metric[key], expected, rtol=0, atol=1e-12)
            model_audits[name] = dict(
                configuration=model["configuration"],
                state_bytes=416,
                coefficient_count=18,
                fitted_max_abs_error=max(
                    float((a - b).abs().max())
                    for a, b in zip(replay_fit, model["fitted"])
                ),
                validation_max_abs_error=max(
                    abs(metric[key] - value)
                    for key, value in model["validation"].items()
                ),
                validation_replay=metric,
            )
        audit_rows.append(
            dict(
                protocol=protocol,
                dataset=dataset,
                models=model_audits,
                final_lock_path=str(path.resolve()),
                fixed_reference_lock_path=str(original_path.resolve()),
                cache_audit=cache_audit,
                test_candidate_or_feature_files_opened=0,
            )
        )
        cases.append(
            dict(
                protocol=protocol,
                dataset=dataset,
                cache=cache,
                candidates=candidates,
                final_lock_path=path,
                fixed_reference_lock_path=original_path,
                models=models,
            )
        )
        print(
            json.dumps(
                dict(
                    stage="train_validation_replay_passed",
                    protocol=protocol,
                    dataset=dataset,
                )
            ),
            flush=True,
        )
    # The gate is emitted only after every case passed, before test loading.
    write_json(
        output / "pretest_verification.json",
        dict(
            status="passed",
            cases=audit_rows,
            all_six_cases_passed_before_test_open=True,
            test_candidate_or_feature_files_opened=0,
        ),
    )
    return cases


def snapshot_metrics(predictions, archive, auc_indices, protocol):
    npos, nneg = int(archive["positive_count"]), int(archive["auc_negative_count"])
    scores = np.r_[predictions[auc_indices], predictions[npos : npos + nneg]]
    labels = np.r_[np.ones(len(auc_indices)), np.zeros(nneg)]
    metrics = dict(
        auc=float(roc_auc_score(labels, scores)),
        ap=float(average_precision_score(labels, scores)),
        mrr=ranking_mrr(
            protocol,
            predictions[:npos],
            predictions[npos + nneg :],
            archive["positive_sources"],
            archive["rank_sources"],
        ),
        **ranking_tie_diagnostics(
            protocol,
            predictions[:npos],
            predictions[npos + nneg :],
            archive["positive_sources"],
            archive["rank_sources"],
        ),
    )
    return metrics, labels, scores


@torch.no_grad()
def evaluate_case(case, head, output):
    protocol, dataset = case["protocol"], case["dataset"]
    destination = output / protocol / dataset
    completed = destination / "summary.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved["configurations"] != {
            name: value["configuration"] for name, value in case["models"].items()
        }:
            raise ValueError(
                "Existing one-time test has different locked configurations"
            )
        return saved
    metadata = json.loads((case["candidates"] / "metadata.json").read_text())
    index = json.loads((case["candidates"] / "index.json").read_text())
    history = TemporalHistory(metadata["nodes"])
    rows = {name: [] for name in case["models"]}
    labels_all, scores_all = {name: [] for name in rows}, {name: [] for name in rows}
    scoring_s = {name: 0.0 for name in rows}
    candidate_count = positive_count = auc_positive_count = negative_count = 0
    base_cache_numeric_peak = 0
    test_cache_disk_bytes = 0
    head_s = 0.0
    started = time.perf_counter()
    for entry in index:
        with np.load(case["candidates"] / entry["file"]) as queries:
            positive = queries["positive"]
            if entry["phase"] != "test":
                history.ingest(positive, entry["time"])
                continue
            cache_path = case["cache"] / entry["file"]
            with np.load(cache_path) as source:
                archive = {key: source[key] for key in source.files}
            pairs = np.concatenate(
                (positive, queries["auc_negative"], queries["ranking_negative"]), axis=1
            )
            if (
                len(archive["base"]) != pairs.shape[1]
                or int(archive["positive_count"]) != positive.shape[1]
                or int(archive["auc_negative_count"])
                != queries["auc_negative"].shape[1]
            ):
                raise ValueError(
                    "Test candidate boundaries differ from paired frozen cache"
                )
            np.testing.assert_array_equal(archive["positive_sources"], positive[0])
            np.testing.assert_array_equal(
                archive["rank_sources"], queries["rank_sources"]
            )
            np.testing.assert_array_equal(
                archive["statistics"], history.statistics(pairs, entry["time"]).numpy()
            )
            base = torch.from_numpy(archive["base"])
            if base.shape[1] != 17:
                raise ValueError("Paired frozen frame is not 17-dimensional")
            head_started = time.perf_counter()
            offset = torch.cat([head(block).flatten() for block in base.split(8192)])
            head_s += time.perf_counter() - head_started
            record = dict(base=base, statistics=torch.from_numpy(archive["statistics"]))
            auc_indices = (
                queries["auc_positive_indices"]
                if "auc_positive_indices" in queries
                else np.arange(len(positive[0]))
            )
            predictions_to_save = {}
            for name, model in case["models"].items():
                score_started = time.perf_counter()
                active_offset = (
                    torch.zeros_like(offset)
                    if model["configuration"].get("offset_mode", "frozen_source")
                    == "zero"
                    else offset
                )
                probability = predict_residual(
                    transformed(record, model["configuration"]),
                    active_offset,
                    model["fitted"],
                ).sigmoid()
                if protocol == "undergs":
                    probability = probability.float()
                probability = probability.numpy()
                scoring_s[name] += time.perf_counter() - score_started
                metric, labels, scores = snapshot_metrics(
                    probability, archive, auc_indices, protocol
                )
                rows[name].append(dict(time=entry["time"], **metric))
                labels_all[name].append(labels)
                scores_all[name].append(scores)
                predictions_to_save[name] = probability
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination / entry["file"],
                **predictions_to_save,
                positive_count=int(archive["positive_count"]),
                auc_negative_count=int(archive["auc_negative_count"]),
                auc_positive_indices=auc_indices,
            )
            candidate_count += pairs.shape[1]
            positive_count += positive.shape[1]
            auc_positive_count += len(auc_indices)
            negative_count += queries["auc_negative"].shape[1]
            base_cache_numeric_peak = max(
                base_cache_numeric_peak,
                archive["base"].nbytes + archive["statistics"].nbytes,
            )
            test_cache_disk_bytes += cache_path.stat().st_size
        # Both frozen models use the same preceding history; ingest once only
        # after both have scored all roles for this snapshot.
        history.ingest(positive, entry["time"])
    results = {}
    for name in rows:
        if protocol == "decoupled":
            results[name] = aggregate_dd(rows[name], labels_all[name], scores_all[name])
        else:
            results[name] = {
                key: float(np.mean([row[key] for row in rows[name]]))
                for key in rows[name][0]
                if key != "time"
            }
            results[name]["snapshots"] = len(rows[name])
    result = dict(
        protocol=protocol,
        dataset=dataset,
        seed=42,
        status="complete",
        configurations={
            name: value["configuration"] for name, value in case["models"].items()
        },
        results=results,
        snapshots=rows,
        deltas={
            key: results["locked_selected"][key] - results["fixed_reference"][key]
            for key in ("auc", "ap", "mrr")
        },
        final_lock_path=str(case["final_lock_path"].resolve()),
        fixed_reference_lock_path=str(case["fixed_reference_lock_path"].resolve()),
        source_feature_cache=str(case["cache"].resolve()),
        source_candidates=str(case["candidates"].resolve()),
        source_checkpoint=str(ROOT / "checkpoints/reference.pt"),
        candidate_rows=candidate_count,
        available_positive_events=positive_count,
        auc_scored_positive_events=auc_positive_count,
        auc_negative_events=negative_count,
        history_statistics_exactly_replayed=True,
        candidate_order_unchanged=True,
        visibility="past-only; positive snapshot ingested after all queries",
        auc_aggregation=(
            "pooled native scored events"
            if protocol == "decoupled"
            else "unweighted snapshot mean"
        ),
        mrr_aggregation=(
            "unweighted snapshot mean; source-best pessimistic rank"
            if protocol == "decoupled"
            else "unweighted snapshot mean; released NumPy event ranking, 100 negatives"
        ),
        feature_channels=17,
        coefficients_per_model=18,
        model_state_bytes_per_model=416,
        inference_models_per_operating_point=1,
        additional_local_statistics_index_bytes=0,
        temporal_history_state_bytes_final=history.state_bytes(),
        reused_feature_cache_disk_bytes=test_cache_disk_bytes,
        max_snapshot_base_statistics_numeric_bytes=base_cache_numeric_peak,
        resource_scope="cached-feature replay; includes history verification and output saving; original frozen graph construction/cache is reused and not remeasured",
        total_replay_s=time.perf_counter() - started,
        source_head_control_s=head_s,
        transform_and_readout_s=scoring_s,
        no_parameter_selection_after_test=True,
        old_locks_unchanged=True,
    )
    write_json(completed, result)
    print(
        json.dumps(
            dict(
                protocol=protocol,
                dataset=dataset,
                results=results,
                deltas=result["deltas"],
            )
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "locked_test")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        print(
            "One-time locked test already complete; no evaluation repeated.", flush=True
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "run_plan.json",
        dict(
            command=[sys.executable, *sys.argv],
            cwd=str(Path.cwd()),
            seed=42,
            cpu_threads=2,
            device="cpu",
            tested_models_per_case=2,
            final_configurations="local_statistics/validation_lock.json selected only",
            fixed_control="original temporal_calibration family reference, fixed before the new families",
            gating="all six train/validation replays and 416-byte states before any test files open",
            selection_after_test=False,
            no_new_search=True,
        ),
    )
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        _, head, _ = load_checkpoint(ROOT / "checkpoints/reference.pt")
        head.eval()
        cases = preflight_cases(args.output, head)
        results = [evaluate_case(case, head, args.output) for case in cases]
    rows = []
    for result in results:
        rows.append(
            dict(
                protocol=result["protocol"],
                dataset=result["dataset"],
                seed=42,
                reference_test_auc=result["results"]["fixed_reference"]["auc"],
                reference_test_mrr=result["results"]["fixed_reference"]["mrr"],
                selected_test_auc=result["results"]["locked_selected"]["auc"],
                selected_test_mrr=result["results"]["locked_selected"]["mrr"],
                delta_auc=result["deltas"]["auc"],
                delta_mrr=result["deltas"]["mrr"],
                coefficients=18,
                model_state_bytes=416,
                final_lock_path=result["final_lock_path"],
                result_path=str(
                    (
                        args.output
                        / result["protocol"]
                        / result["dataset"]
                        / "summary.json"
                    ).resolve()
                ),
            )
        )
    with (args.output / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "summary.json",
        dict(
            status="complete",
            cases=rows,
            all_six_pretest_replays_passed=True,
            no_test_based_selection=True,
            final_lock_paths=[str(case["final_lock_path"].resolve()) for case in cases],
            pretest_verification=str(
                (args.output / "pretest_verification.json").resolve()
            ),
        ),
    )


if __name__ == "__main__":
    main()
