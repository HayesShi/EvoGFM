"""Train/validation-only fixed-width calibration on paired seed42 caches.

No test-scoring entry point is provided. Existing candidates, frozen feature
frames, source head, native metric aggregation and tie ordering are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from threadpoolctl import threadpool_limits

from evogfm.readout import predict_residual
from evogfm.temporal import (
    TemporalHistory,
    ranking_mrr,
    ranking_tie_diagnostics,
    replace_temporal,
)
from evogfm.temporal_calibration import (
    PastLocalGraph,
    calibration_channels,
    local_calibration_channels,
    select_operating_point,
    selection_grid,
)
from experiments.decoupled_protocol import aggregate as aggregate_dd
from experiments.full_graph import load_checkpoint
from experiments.optimize_temporal import fit_temporal_residual
from experiments.temporal_official import write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/optimization/temporal_calibration_s42"


def input_paths(protocol, dataset):
    root = ROOT / "runs/protocol_alignment" / protocol / dataset / "s42"
    if protocol == "decoupled":
        return root, root / "features_full", root / "protocol"
    return root, root / "causal_adaptation/frozen/features", root / "candidates"


def reference_configuration(protocol, root):
    if protocol == "decoupled":
        path = root / "validation_frontier/summary.json"
        value = json.loads(path.read_text())["best_auc"]
        return {
            "id": "reference",
            "configuration": value["configuration"],
            "validation": value["validation"],
            "source": str(path),
        }
    # Read only the existing configuration, never any previously scored test
    # metric. This is the prior locked operating point, not a new seed search.
    path = root / "causal_adaptation/frozen/summary.json"
    value = json.loads(path.read_text())
    return {
        "id": "reference",
        "configuration": value["configuration"],
        "source": value["configuration_source"],
    }


def read_feature_record(cache, entry):
    if entry["phase"] not in ("train", "validation"):
        raise ValueError("This experiment reads train and validation features only")
    with np.load(cache / entry["file"]) as archive:
        return {key: archive[key] for key in archive.files}


@torch.no_grad()
def load_records(protocol, cache, candidates, head):
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["feature_dimensions"] != 17:
        raise ValueError("The paired cache must retain all 17 channels")
    source_checkpoint = manifest.get("checkpoint", manifest.get("source_checkpoint"))
    if (
        Path(source_checkpoint).resolve()
        != (ROOT / "checkpoints/reference.pt").resolve()
    ):
        raise ValueError("Paired features must use the unchanged reference checkpoint")
    if manifest["head_parameters"] != sum(value.numel() for value in head.parameters()):
        raise ValueError("Source head does not match the paired cache")
    if (
        protocol == "decoupled"
        and manifest["observation_policy"]
        != "predict-before-ingest; history contains t'<t only"
    ):
        raise ValueError("Expected past-only DD cache")
    if protocol == "undergs" and manifest["track"] != "causal_adaptation":
        raise ValueError("Expected the UnderGS causal track")
    metadata = json.loads((candidates / "metadata.json").read_text())
    if metadata["seed"] != 42:
        raise ValueError("This budget is restricted to seed42")
    entries = {
        entry["file"]: entry
        for entry in manifest["index"]
        if entry["phase"] in ("train", "validation")
    }
    index = json.loads((candidates / "index.json").read_text())
    history = TemporalHistory(metadata["nodes"])
    records = []
    totals = {"train": 0, "validation": 0}
    for position, entry in enumerate(index):
        if entry["phase"] == "test":
            break
        with np.load(candidates / entry["file"]) as queries:
            positive = queries["positive"]
            if position == 0:
                history.ingest(positive, entry["time"])
                continue
            if (
                entry["file"] not in entries
                or entries[entry["file"]]["phase"] != entry["phase"]
            ):
                raise ValueError("Paired candidate and feature manifests disagree")
            archive = read_feature_record(cache, entry)
            if (
                int(archive["positive_count"]) != positive.shape[1]
                or int(archive["auc_negative_count"])
                != queries["auc_negative"].shape[1]
            ):
                raise ValueError(
                    "Cached positive/negative boundaries differ from candidates"
                )
            pairs = np.concatenate(
                (
                    positive,
                    queries["auc_negative"],
                    (
                        queries["ranking_negative"]
                        if entry["phase"] == "validation"
                        else positive[:, :0]
                    ),
                ),
                axis=1,
            )
            if len(archive["base"]) != pairs.shape[1]:
                raise ValueError("Candidate counts differ from the paired cache")
            np.testing.assert_array_equal(archive["positive_sources"], positive[0])
            np.testing.assert_array_equal(
                archive["rank_sources"], queries["rank_sources"]
            )
            # Replay every allowed candidate's history statistics, proving that
            # the cache observes only earlier snapshots and retains row order.
            expected_statistics = history.statistics(pairs, entry["time"]).numpy()
            np.testing.assert_array_equal(archive["statistics"], expected_statistics)
            base = torch.from_numpy(archive["base"])
            records.append(
                dict(
                    entry=entry,
                    base=base,
                    statistics=torch.from_numpy(archive["statistics"]),
                    offset=torch.cat(
                        [head(block).flatten() for block in base.split(8192)]
                    ),
                    positive_count=int(archive["positive_count"]),
                    auc_negative_count=int(archive["auc_negative_count"]),
                    positive_sources=archive["positive_sources"],
                    rank_sources=archive["rank_sources"],
                    auc_positive_indices=(
                        queries["auc_positive_indices"]
                        if "auc_positive_indices" in queries
                        else np.arange(positive.shape[1])
                    ),
                )
            )
            totals[entry["phase"]] += len(base)
        history.ingest(positive, entry["time"])
    if len(records) != len(entries):
        raise ValueError("Not all train/validation feature snapshots were loaded")
    audit = dict(
        feature_dimensions=17,
        seed=42,
        candidate_row_counts=totals,
        history_statistics_exactly_replayed=True,
        candidate_order_unchanged=True,
        frozen_source_head_uses_original_base=True,
        source_checkpoint=manifest.get("checkpoint", manifest.get("source_checkpoint")),
        backbone_parameters=manifest["backbone_parameters"],
        head_parameters=manifest["head_parameters"],
        track="past_only" if protocol == "decoupled" else "causal_adaptation",
        train_snapshots=sum(row["entry"]["phase"] == "train" for row in records),
        validation_snapshots=sum(
            row["entry"]["phase"] == "validation" for row in records
        ),
        test_feature_or_candidate_files_opened=0,
        feature_fit="existing offline training-graph structural frame; no new backbone fitting",
    )
    return records, audit


def transformed(record, configuration):
    if configuration.get("transform") == "local_graph_statistics":
        return local_calibration_channels(
            record["base"],
            record["statistics"],
            record["local_statistics"],
            configuration["half_life"],
        )
    if "transform" in configuration:
        return calibration_channels(
            record["base"],
            record["statistics"],
            configuration["transform"],
            configuration["half_life"],
        )
    return replace_temporal(
        record["base"],
        record["statistics"],
        configuration["variant"],
        configuration["half_life"],
    )


def fit(records, configuration):
    train = [record for record in records if record["entry"]["phase"] == "train"]
    features = torch.cat([transformed(row, configuration) for row in train])
    labels = torch.cat(
        [
            torch.cat(
                (
                    torch.ones(row["positive_count"]),
                    torch.zeros(len(row["base"]) - row["positive_count"]),
                )
            )
            for row in train
        ]
    )
    offset = torch.cat([row["offset"] for row in train])
    if configuration.get("offset_mode", "frozen_source") == "zero":
        offset = torch.zeros_like(offset)
    return fit_temporal_residual(
        features,
        offset,
        labels,
        configuration["ridge"],
    )


@torch.no_grad()
def evaluate_validation(records, configuration, fitted, protocol):
    rows, all_labels, all_scores = [], [], []
    for record in records:
        if record["entry"]["phase"] != "validation":
            continue
        offset = record["offset"]
        if configuration.get("offset_mode", "frozen_source") == "zero":
            offset = torch.zeros_like(offset)
        predictions = predict_residual(
            transformed(record, configuration), offset, fitted
        ).sigmoid()
        # Preserve existing protocol precision as well as its ranking semantics.
        if protocol == "undergs":
            predictions = predictions.float()
        predictions = predictions.numpy()
        npos, nneg = record["positive_count"], record["auc_negative_count"]
        indices = record["auc_positive_indices"]
        labels = np.r_[np.ones(len(indices)), np.zeros(nneg)]
        scores = np.r_[predictions[indices], predictions[npos : npos + nneg]]
        all_labels.append(labels)
        all_scores.append(scores)
        rows.append(
            dict(
                time=record["entry"]["time"],
                auc=float(roc_auc_score(labels, scores)),
                ap=float(average_precision_score(labels, scores)),
                mrr=ranking_mrr(
                    protocol,
                    predictions[:npos],
                    predictions[npos + nneg :],
                    record["positive_sources"],
                    record["rank_sources"],
                ),
                **ranking_tie_diagnostics(
                    protocol,
                    predictions[:npos],
                    predictions[npos + nneg :],
                    record["positive_sources"],
                    record["rank_sources"],
                ),
            )
        )
    metrics = (
        aggregate_dd(rows, all_labels, all_scores)
        if protocol == "decoupled"
        else {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "time"
        }
    )
    metrics["snapshots"] = len(rows)
    return metrics, rows


def run_case(protocol, dataset, output, head):
    destination = output / protocol / dataset
    if (destination / "validation_lock.json").exists():
        return json.loads((destination / "validation_lock.json").read_text())
    root, cache, candidates = input_paths(protocol, dataset)
    reference = reference_configuration(protocol, root)
    grid = selection_grid(reference["configuration"]["half_life"])
    write_json(
        destination / "budget.json",
        dict(
            seed=42,
            cpu_threads=2,
            device="cpu",
            grid=grid,
            new_configurations=8,
            control_refits=0 if protocol == "decoupled" else 1,
            fixed_before_validation=True,
            selection="maximum validation AUC subject to MRR >= reference MRR; reference fallback",
            source_reference=reference,
            source_head="checkpoints/reference.pt; unchanged original 17-channel inputs",
            candidate_policy="reuse all paired native candidates in original order",
            test_evaluation_allowed=False,
            feature_channels=17,
            fitted_coefficients=18,
            trainable_parameter_increase=0,
            inference_models_per_point=1,
            transforms={
                "source_conditional_frequency": "log1p(pair_count / max(1, source_out_count)) in recency_frequency slot1",
                "target_corrected_recency": "exp2(-pair_age / half_life) / sqrt(1 + target_in_count) in recency_frequency slot0",
            },
        ),
    )
    start = time.perf_counter()
    records, audit = load_records(protocol, cache, candidates, head)
    write_json(destination / "cache_audit.json", audit)
    readouts = {}
    if protocol == "decoupled":
        readouts["reference"] = torch.load(
            root / "validation_frontier/readouts.pt", weights_only=False
        )["best_auc"]
    else:
        readouts["reference"] = fit(records, reference["configuration"])
        reference["validation"], reference_rows = evaluate_validation(
            records, reference["configuration"], readouts["reference"], protocol
        )
        write_json(
            destination / "reference_validation.json",
            dict(reference=reference, snapshots=reference_rows),
        )
    trials = []
    for index, configuration in enumerate(grid):
        trial_start = time.perf_counter()
        identity = f"trial{index:02d}"
        fitted = fit(records, configuration)
        validation, snapshots = evaluate_validation(
            records, configuration, fitted, protocol
        )
        row = dict(
            id=identity,
            configuration=configuration,
            validation=validation,
            elapsed_s=time.perf_counter() - trial_start,
        )
        readouts[identity] = fitted
        trials.append(row)
        write_json(destination / f"{identity}.json", dict(**row, snapshots=snapshots))
        print(json.dumps(dict(protocol=protocol, dataset=dataset, **row)), flush=True)
    torch.save(readouts, destination / "readouts.pt")
    selection = select_operating_point(trials, reference)
    result = dict(
        protocol=protocol,
        dataset=dataset,
        seed=42,
        reference=reference,
        trials=trials,
        **selection,
        source_cache=str(cache),
        source_candidates=str(candidates),
        source_checkpoint=str(ROOT / "checkpoints/reference.pt"),
        selection_split="validation",
        test_evaluated=False,
        feature_channels=17,
        fitted_coefficients=18,
        fitted_scalars_with_mean_scale=52,
        readout_bytes=416,
        inference_models_per_operating_point=1,
        added_trainable_parameters=0,
        elapsed_s=time.perf_counter() - start,
        cache_audit=audit,
    )
    write_json(destination / "validation_lock.json", result)
    return result


def run_mechanism_controls(output, head):
    """Matched legacy controls for selected transforms; never change a lock."""
    locks = [
        json.loads(path.read_text())
        for path in sorted(output.glob("*/*/validation_lock.json"))
    ]
    chosen = [
        lock for lock in locks if "transform" in lock["selected"]["configuration"]
    ]
    budget = [
        {
            "protocol": lock["protocol"],
            "dataset": lock["dataset"],
            "selected_id": lock["selected"]["id"],
            "configuration": {
                "variant": "recency_frequency",
                **{
                    key: lock["selected"]["configuration"][key]
                    for key in ("half_life", "ridge")
                },
            },
        }
        for lock in chosen
    ]
    write_json(
        output / "mechanism_controls_budget.json",
        dict(
            controls=budget,
            purpose="separate the selected channel transform from half-life/ridge changes",
            updates_configuration_locks=False,
            test_evaluated=False,
        ),
    )
    comparisons = []
    for lock, control in zip(chosen, budget):
        _, cache, candidates = input_paths(lock["protocol"], lock["dataset"])
        records, _ = load_records(lock["protocol"], cache, candidates, head)
        fitted = fit(records, control["configuration"])
        metric, snapshots = evaluate_validation(
            records, control["configuration"], fitted, lock["protocol"]
        )
        row = dict(
            **control,
            validation=metric,
            selected=lock["selected"],
            delta_auc_transform_only=lock["selected"]["validation"]["auc"]
            - metric["auc"],
            delta_mrr_transform_only=lock["selected"]["validation"]["mrr"]
            - metric["mrr"],
        )
        destination = output / lock["protocol"] / lock["dataset"]
        torch.save(fitted, destination / "mechanism_control.pt")
        write_json(
            destination / "mechanism_control.json", dict(**row, snapshots=snapshots)
        )
        comparisons.append(row)
    write_json(
        output / "mechanism_controls.json",
        dict(
            comparisons=comparisons,
            configuration_locks_unchanged=True,
            test_evaluated=False,
        ),
    )


def run_offset_ablation(output, head):
    """Matched source-offset ablation, independently locked from prior trials."""
    destination = output / "offset_ablation"
    locks = [
        json.loads(path.read_text())
        for path in sorted(output.glob("*/*/validation_lock.json"))
    ]
    plan = []
    for lock in locks:
        new_best = max(
            lock["trials"],
            key=lambda row: (row["validation"]["auc"], row["validation"]["mrr"]),
        )
        plan.append(
            dict(
                protocol=lock["protocol"],
                dataset=lock["dataset"],
                anchors=[lock["reference"], new_best],
            )
        )
    write_json(
        destination / "plan.json",
        dict(
            seed=42,
            cpu_threads=2,
            hypothesis="removed original channels remain in frozen head(base), causing a fixed-offset mismatch; direct calibration permits the same 18 coefficients to fit only the retained channels",
            cases=plan,
            offset_modes=["frozen_source", "zero"],
            matched_zero_offset_fits=2 * len(plan),
            frozen_source_controls="reuse identical prior fitted readouts and validation metrics",
            ridge_expansion_trigger="zero-offset matched AUC gain >=0.002 AND validation MRR >= original reference MRR",
            ridge_expansion="one best eligible zero-offset anchor per case, ridge in [0.1,1,10], skip already fitted ridge",
            max_additional_ridge_fits=2 * len(plan),
            selection="max validation AUC with MRR >= original reference; reference fallback",
            feature_channels=17,
            fitted_coefficients=18,
            added_trainable_parameters=0,
            inference_models_per_point=1,
            zero_offset_source_head_calls_at_inference=0,
            old_configuration_locks_unchanged=True,
            test_evaluation_allowed=False,
        ),
    )
    summaries = []
    for case, old_lock in zip(plan, locks):
        case_dir = destination / case["protocol"] / case["dataset"]
        if (case_dir / "validation_lock.json").exists():
            summaries.append(
                json.loads((case_dir / "validation_lock.json").read_text())
            )
            continue
        started = time.perf_counter()
        _, cache, candidates = input_paths(case["protocol"], case["dataset"])
        records, audit = load_records(case["protocol"], cache, candidates, head)
        old_readouts = torch.load(
            output / case["protocol"] / case["dataset"] / "readouts.pt",
            weights_only=False,
        )
        readouts, trials, matched = {}, [], []
        for anchor in case["anchors"]:
            frozen = dict(
                id=anchor["id"] + "_frozen_source",
                configuration=dict(
                    anchor["configuration"], offset_mode="frozen_source"
                ),
                validation=anchor["validation"],
            )
            readouts[frozen["id"]] = old_readouts[anchor["id"]]
            trials.append(frozen)
            configuration = dict(anchor["configuration"], offset_mode="zero")
            fitted = fit(records, configuration)
            metric, snapshots = evaluate_validation(
                records, configuration, fitted, case["protocol"]
            )
            row = dict(
                id=anchor["id"] + "_zero",
                configuration=configuration,
                validation=metric,
            )
            trials.append(row)
            readouts[row["id"]] = fitted
            contrast = dict(
                anchor=anchor["id"],
                delta_auc=metric["auc"] - anchor["validation"]["auc"],
                delta_mrr=metric["mrr"] - anchor["validation"]["mrr"],
                frozen_source=frozen,
                zero=row,
            )
            matched.append(contrast)
            write_json(
                case_dir / (row["id"] + ".json"), dict(**row, snapshots=snapshots)
            )
            print(
                json.dumps(
                    dict(
                        protocol=case["protocol"],
                        dataset=case["dataset"],
                        offset_matched=contrast,
                    )
                ),
                flush=True,
            )
        eligible = [
            item["zero"]
            for item in matched
            if item["delta_auc"] >= 0.002
            and item["zero"]["validation"]["mrr"]
            >= old_lock["reference"]["validation"]["mrr"]
        ]
        if eligible:
            best = max(
                eligible,
                key=lambda row: (row["validation"]["auc"], row["validation"]["mrr"]),
            )
            for ridge in (0.1, 1.0, 10.0):
                if ridge == best["configuration"]["ridge"]:
                    continue
                configuration = dict(best["configuration"], ridge=ridge)
                fitted = fit(records, configuration)
                metric, snapshots = evaluate_validation(
                    records, configuration, fitted, case["protocol"]
                )
                row = dict(
                    id=best["id"] + f"_ridge{ridge}",
                    configuration=configuration,
                    validation=metric,
                )
                trials.append(row)
                readouts[row["id"]] = fitted
                write_json(
                    case_dir / (row["id"] + ".json"), dict(**row, snapshots=snapshots)
                )
        reference = trials[0]
        selection = select_operating_point(trials[1:], reference)
        torch.save(readouts, case_dir / "readouts.pt")
        result = dict(
            protocol=case["protocol"],
            dataset=case["dataset"],
            seed=42,
            trials=trials,
            matched=matched,
            **selection,
            reference=reference,
            ridge_expansion_triggered=bool(eligible),
            elapsed_s=time.perf_counter() - started,
            source_cache=str(cache),
            source_candidates=str(candidates),
            cache_audit=audit,
            feature_channels=17,
            fitted_coefficients=18,
            readout_bytes=416,
            test_evaluated=False,
            original_locks_unchanged=True,
        )
        write_json(case_dir / "validation_lock.json", result)
        summaries.append(result)
    write_json(
        destination / "summary.json",
        dict(
            results=[
                {
                    key: row[key]
                    for key in (
                        "protocol",
                        "dataset",
                        "reference",
                        "selected",
                        "best_auc",
                        "matched",
                        "ridge_expansion_triggered",
                        "elapsed_s",
                    )
                }
                for row in summaries
            ],
            test_evaluated=False,
            original_locks_unchanged=True,
        ),
    )


def build_local_statistics(records, candidates, destination):
    metadata = json.loads((candidates / "metadata.json").read_text())
    index = json.loads((candidates / "index.json").read_text())
    lookup = {row["entry"]["file"]: row for row in records}
    graph = PastLocalGraph(metadata["nodes"])
    feature_directory = destination / "features"
    feature_directory.mkdir(parents=True, exist_ok=True)
    construction_s, update_s, numeric_bytes, disk_bytes = 0.0, 0.0, 0, 0
    max_errors = np.zeros(3)
    for position, entry in enumerate(index):
        if entry["phase"] == "test":
            break
        with np.load(candidates / entry["file"]) as queries:
            positive = queries["positive"]
            if position:
                record = lookup[entry["file"]]
                pairs = np.concatenate(
                    (
                        positive,
                        queries["auc_negative"],
                        (
                            queries["ranking_negative"]
                            if entry["phase"] == "validation"
                            else positive[:, :0]
                        ),
                    ),
                    axis=1,
                )
                started = time.perf_counter()
                local = graph.statistics(pairs)
                construction_s += time.perf_counter() - started
                # CN / RA / AA independently recomputed from the incremental
                # sets must agree with the corresponding existing base frame.
                expected = record["base"][:, [6, 7, 8]]
                actual = local[:, [0, 2, 1]]
                errors = (expected - actual).abs().amax(0).numpy()
                max_errors = np.maximum(max_errors, errors)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                record["local_statistics"] = local
                path = feature_directory / entry["file"]
                np.savez_compressed(
                    path,
                    local_statistics=local.numpy(),
                    positive_count=record["positive_count"],
                    auc_negative_count=record["auc_negative_count"],
                )
                numeric_bytes += local.numel() * local.element_size()
                disk_bytes += path.stat().st_size
        started = time.perf_counter()
        graph.ingest(positive)
        update_s += time.perf_counter() - started
    budget = dict(
        local_statistic_columns=5,
        local_cache_numeric_bytes=numeric_bytes,
        local_cache_compressed_disk_bytes=disk_bytes,
        graph_index_bytes_final=graph.state_bytes(),
        graph_index_memory_metric="recursive sys.getsizeof of Python adjacency sets and integer keys; not process RSS",
        undirected_edges_final=sum(len(row) for row in graph.neighbors) // 2,
        query_construction_s=construction_s,
        graph_update_s=update_s,
        new_coefficients=0,
        total_coefficients=18,
        calibration_numeric_bytes=416,
        local_state_cost_claim="additional index/cache storage reported separately; not zero-cost",
        cn_ra_aa_max_abs_error_vs_existing_frame=max_errors.tolist(),
        visibility="all candidates scored before current positive ingestion",
        test_candidate_or_feature_files_opened=0,
    )
    write_json(destination / "structure_budget.json", budget)
    return budget


def run_local_statistics(output, head):
    destination = output / "local_statistics"
    offset_locks = [
        json.loads(path.read_text())
        for path in sorted(
            (output / "offset_ablation").glob("*/*/validation_lock.json")
        )
    ]
    cases = [
        dict(
            protocol=lock["protocol"],
            dataset=lock["dataset"],
            reference=lock["selected"],
            half_life=lock["selected"]["configuration"]["half_life"],
        )
        for lock in offset_locks
    ]
    write_json(
        destination / "plan.json",
        dict(
            seed=42,
            cpu_threads=2,
            cases=cases,
            family="local_graph_statistics",
            first_ten_channels="unchanged paired frozen frame",
            final_seven_channels=[
                "log1p(directed pair frequency)",
                "forward recency",
                "log1p(CN)",
                "log1p(AdamicAdar)",
                "log1p(ResourceAllocation)",
                "Jaccard",
                "log1p(source degree * target degree)",
            ],
            graph="simple undirected previous-positive adjacency; duplicates collapse, self loops excluded",
            adamic_adar="sum 1/log(max(2, degree(common neighbor))); existing frozen-feature convention for self candidates",
            candidate_order="unchanged native candidates and source sorting; query role blind",
            half_life="fixed to each existing offset-selected validation lock",
            offset_mode="zero",
            ridges=[0.1, 1.0, 10.0],
            new_fits=3 * len(cases),
            selection="max validation AUC with MRR >= offset-selected reference; reference fallback",
            feature_channels=17,
            fitted_coefficients=18,
            inference_models_per_point=1,
            structural_index_and_cache_cost="measured separately from coefficients",
            test_evaluation_allowed=False,
            previous_locks_unchanged=True,
        ),
    )
    summaries = []
    for case in cases:
        case_dir = destination / case["protocol"] / case["dataset"]
        if (case_dir / "validation_lock.json").exists():
            summaries.append(
                json.loads((case_dir / "validation_lock.json").read_text())
            )
            continue
        started = time.perf_counter()
        _, cache, candidates = input_paths(case["protocol"], case["dataset"])
        records, audit = load_records(case["protocol"], cache, candidates, head)
        structure_budget = build_local_statistics(records, candidates, case_dir)
        old_readouts = torch.load(
            output
            / "offset_ablation"
            / case["protocol"]
            / case["dataset"]
            / "readouts.pt",
            weights_only=False,
        )
        reference = dict(case["reference"], id="reference")
        readouts = {"reference": old_readouts[case["reference"]["id"]]}
        trials = []
        for ridge in (0.1, 1.0, 10.0):
            configuration = dict(
                transform="local_graph_statistics",
                half_life=case["half_life"],
                ridge=ridge,
                offset_mode="zero",
            )
            fitted = fit(records, configuration)
            metric, snapshots = evaluate_validation(
                records, configuration, fitted, case["protocol"]
            )
            row = dict(
                id=f"local_ridge{ridge}", configuration=configuration, validation=metric
            )
            trials.append(row)
            readouts[row["id"]] = fitted
            write_json(
                case_dir / (row["id"] + ".json"), dict(**row, snapshots=snapshots)
            )
            print(
                json.dumps(
                    dict(protocol=case["protocol"], dataset=case["dataset"], **row)
                ),
                flush=True,
            )
        torch.save(readouts, case_dir / "readouts.pt")
        result = dict(
            protocol=case["protocol"],
            dataset=case["dataset"],
            seed=42,
            reference=reference,
            trials=trials,
            **select_operating_point(trials, reference),
            elapsed_s=time.perf_counter() - started,
            cache_audit=audit,
            structure_budget=structure_budget,
            source_cache=str(cache),
            source_candidates=str(candidates),
            feature_channels=17,
            fitted_coefficients=18,
            test_evaluated=False,
            previous_locks_unchanged=True,
        )
        write_json(case_dir / "validation_lock.json", result)
        summaries.append(result)
    write_json(
        destination / "summary.json",
        dict(
            results=[
                {
                    key: row[key]
                    for key in (
                        "protocol",
                        "dataset",
                        "reference",
                        "selected",
                        "best_auc",
                        "structure_budget",
                        "elapsed_s",
                    )
                }
                for row in summaries
            ],
            test_evaluated=False,
            previous_locks_unchanged=True,
        ),
    )


def aggregate_results(output):
    """Mechanical per-family summary of six locked validation operating points."""
    model, head, _ = load_checkpoint(ROOT / "checkpoints/reference.pt")
    backbone_bytes = sum(
        value.numel() * value.element_size() for value in model.parameters()
    )
    head_bytes = sum(
        value.numel() * value.element_size() for value in head.parameters()
    )
    rows = []
    for family, directory in (
        ("temporal_calibration", output),
        ("offset_ablation", output / "offset_ablation"),
        ("local_statistics", output / "local_statistics"),
    ):
        for path in sorted(directory.glob("*/*/validation_lock.json")):
            lock = json.loads(path.read_text())
            reference, selected = lock["reference"], lock["selected"]
            fitted = torch.load(path.parent / "readouts.pt", weights_only=False)[
                selected["id"]
            ]
            model_state_bytes = sum(
                value.numel() * value.element_size() for value in fitted
            )
            configuration = selected["configuration"]
            uses_source_head = (
                configuration.get("offset_mode", "frozen_source") != "zero"
            )
            _, cache, _ = input_paths(lock["protocol"], lock["dataset"])
            manifest = json.loads((cache / "manifest.json").read_text())
            paired_disk_bytes = sum(
                (cache / entry["file"]).stat().st_size
                for entry in manifest["index"]
                if entry["phase"] in ("train", "validation")
            )
            structure = lock.get("structure_budget", {})
            selected_local = configuration.get("transform") == "local_graph_statistics"
            rows.append(
                dict(
                    family=family,
                    protocol=lock["protocol"],
                    dataset=lock["dataset"],
                    seed=42,
                    reference_val_auc=reference["validation"]["auc"],
                    reference_val_mrr=reference["validation"]["mrr"],
                    selected_val_auc=selected["validation"]["auc"],
                    selected_val_mrr=selected["validation"]["mrr"],
                    delta_auc=selected["validation"]["auc"]
                    - reference["validation"]["auc"],
                    delta_mrr=selected["validation"]["mrr"]
                    - reference["validation"]["mrr"],
                    params=int(fitted[0].numel()),
                    model_state_bytes=model_state_bytes,
                    model_state_scope="18 coefficients + 17 training means + 17 training scales; float64",
                    frozen_backbone_weight_bytes=backbone_bytes,
                    frozen_source_head_weight_bytes=(
                        head_bytes if uses_source_head else 0
                    ),
                    selected_weight_and_readout_bytes=backbone_bytes
                    + model_state_bytes
                    + (head_bytes if uses_source_head else 0),
                    reused_train_validation_feature_cache_disk_bytes=paired_disk_bytes,
                    selected_additional_graph_index_bytes=(
                        structure.get("graph_index_bytes_final", 0)
                        if selected_local
                        else 0
                    ),
                    family_graph_index_bytes=structure.get(
                        "graph_index_bytes_final", 0
                    ),
                    family_structure_cache_numeric_bytes=structure.get(
                        "local_cache_numeric_bytes", 0
                    ),
                    family_structure_cache_disk_bytes=structure.get(
                        "local_cache_compressed_disk_bytes", 0
                    ),
                    graph_cache_scope="additional family index/cache; existing frozen graph state is reused, not claimed zero",
                    selected_id=selected["id"],
                    selected_configuration=json.dumps(configuration, sort_keys=True),
                    select_rule="maximum validation AUC with MRR >= this family's reference; reference fallback",
                    selection_path=str(path.resolve()),
                    readout_path=str((path.parent / "readouts.pt").resolve()),
                    test_evaluated=False,
                )
            )
    with (output / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "aggregate.json",
        dict(
            rows=rows,
            families=3,
            cases_per_family=6,
            seed=42,
            train_validation_only=True,
            test_evaluated=False,
            additional_fits=dict(
                temporal_transforms=48,
                undergs_reference_controls=3,
                selected_transform_mechanism_controls=2,
                zero_offset_ablation=24,
                local_statistics=18,
            ),
            total_new_calibrator_fits=95,
            frozen_backbone_retraining=0,
            stopped_after_preplanned_local_statistics=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("decoupled", "undergs"),
        default=["decoupled", "undergs"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("uci", "bitcoin-alpha", "bitcoin-otc"),
        default=["uci", "bitcoin-alpha", "bitcoin-otc"],
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--mechanism-controls",
        action="store_true",
        help="Only evaluate matched legacy controls for already locked new transforms",
    )
    parser.add_argument(
        "--offset-ablation",
        action="store_true",
        help="Run the separately locked matched frozen-source versus zero-offset experiment",
    )
    parser.add_argument(
        "--local-statistics",
        action="store_true",
        help="Run the final preplanned exact local-statistics family only",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Export existing locks and resource budgets without fitting or evaluation",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.aggregate_only:
            aggregate_results(args.output)
            return
        _, head, _ = load_checkpoint(ROOT / "checkpoints/reference.pt")
        head.eval()
        if args.local_statistics:
            run_local_statistics(args.output, head)
            return
        if args.offset_ablation:
            run_offset_ablation(args.output, head)
            return
        if args.mechanism_controls:
            run_mechanism_controls(args.output, head)
            return
        summaries = []
        for protocol in args.protocols:
            for dataset in args.datasets:
                result = run_case(protocol, dataset, args.output, head)
                summaries.append(
                    {
                        key: result[key]
                        for key in (
                            "protocol",
                            "dataset",
                            "seed",
                            "reference",
                            "selected",
                            "best_auc",
                            "elapsed_s",
                        )
                    }
                )
        write_json(
            args.output / "summary.json",
            dict(results=summaries, test_evaluated=False, seed=42),
        )


if __name__ == "__main__":
    main()
