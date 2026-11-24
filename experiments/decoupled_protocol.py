"""Evaluate the released full model and frozen predictor on identical queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from evogfm.decoupled_protocol import (
    first_source_negatives,
    prepare_released_events,
    released_auc_indices,
)
from evogfm.readout import predict_residual
from evogfm.temporal import export_protocol, ranking_mrr, ranking_tie_diagnostics
from experiments.full_graph import load_checkpoint
from experiments.optimize_temporal import cached_features, fit_configuration
from experiments.temporal_official import build_feature_cache, write_json


def prepare(dataset, seed, directory):
    events = prepare_released_events(dataset)
    protocol = directory / "protocol"
    if not (protocol / "index.json").exists():
        export_protocol(events, "decoupled", seed, protocol)
    return events


def inject_native_queries(events, directory):
    protocol = directory / "protocol"
    index = json.loads((protocol / "index.json").read_text())
    reference_rows, truth_all, score_all = [], [], []
    for entry in index:
        if entry["phase"] != "test":
            continue
        path = protocol / entry["file"]
        with np.load(path) as original:
            fields = {key: original[key] for key in original.files}
        native_path = directory / "official/shared_queries" / entry["file"]
        with np.load(native_path) as archive:
            native = {key: archive[key] for key in archive.files}
        selected, sources = first_source_negatives(native["ranking_negative_all"][0])
        positive = native["positive"]
        auc_indices = released_auc_indices(positive.shape[1])
        np.testing.assert_array_equal(native["auc_positive"], positive[:, auc_indices])
        fields.update(
            positive=positive,
            auc_negative=native["auc_negative"],
            ranking_negative=native["ranking_negative_all"][:, selected],
            rank_sources=sources,
            auc_positive_indices=auc_indices,
        )
        np.savez_compressed(path, **fields)
        entry.update(
            positives=positive.shape[1],
            ranking_candidates=len(selected),
            auc_positive_events=len(auc_indices),
        )
        scores = np.r_[native["auc_positive_scores"], native["auc_negative_scores"]]
        truth = np.r_[
            np.ones(len(auc_indices)), np.zeros(native["auc_negative"].shape[1])
        ]
        score_all.append(scores)
        truth_all.append(truth)
        mrr = ranking_mrr(
            "decoupled",
            native["positive_scores"],
            native["ranking_negative_scores_all"][selected],
            positive[0],
            sources,
        )
        np.testing.assert_allclose(mrr, native["native_mrr"], atol=1e-7, rtol=0)
        reference_rows.append(
            {
                "time": entry["time"],
                "mrr": mrr,
                "auc": float(roc_auc_score(truth, scores)),
                "ap": float(average_precision_score(truth, scores)),
                **ranking_tie_diagnostics(
                    "decoupled",
                    native["positive_scores"],
                    native["ranking_negative_scores_all"][selected],
                    positive[0],
                    sources,
                ),
            }
        )
    result = aggregate(reference_rows, truth_all, score_all)
    native_result = json.loads((directory / "official/summary.json").read_text())
    for name in ("auc", "ap", "mrr"):
        np.testing.assert_allclose(result[name], native_result[name], atol=1e-7, rtol=0)
    write_json(protocol / "index.json", index)
    write_json(
        directory / "reference_replay.json",
        {
            "all_native_metrics_reproduced": True,
            "result": result,
            "rows": reference_rows,
            "official_auc_event_coverage": {
                "scored": sum(
                    row["auc_positive_events"]
                    for row in index
                    if row["phase"] == "test"
                ),
                "available": sum(
                    row["positives"] for row in index if row["phase"] == "test"
                ),
            },
            "metadata": events.metadata,
        },
    )
    return index, result


def aggregate(rows, labels, scores):
    output = {
        "auc": float(roc_auc_score(np.concatenate(labels), np.concatenate(scores))),
        "ap": float(
            average_precision_score(np.concatenate(labels), np.concatenate(scores))
        ),
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "snapshot_mean_auc": float(np.mean([row["auc"] for row in rows])),
        "snapshot_mean_ap": float(np.mean([row["ap"] for row in rows])),
        "snapshots": len(rows),
    }
    for name in ("pessimistic_mrr", "random_tie_expected_mrr", "mean_tied_rank_mrr"):
        output[name] = float(np.mean([row[name] for row in rows]))
    return output


@torch.no_grad()
def evaluate(cache, manifest, protocol, head, fitted, configuration, phase):
    rows, labels_all, scores_all = [], [], []
    for entry in manifest["index"]:
        if entry["phase"] != phase:
            continue
        archive, base, changed = cached_features(
            cache, entry, configuration["variant"], configuration["half_life"]
        )
        offset = torch.cat([head(block).flatten() for block in base.split(8192)])
        predictions = predict_residual(changed, offset, fitted).sigmoid().numpy()
        npos, nneg = int(archive["positive_count"]), int(archive["auc_negative_count"])
        with np.load(protocol / entry["file"]) as queries:
            indices = (
                queries["auc_positive_indices"]
                if "auc_positive_indices" in queries
                else np.arange(npos)
            )
        scores = np.r_[predictions[indices], predictions[npos : npos + nneg]]
        labels = np.r_[np.ones(len(indices)), np.zeros(nneg)]
        labels_all.append(labels)
        scores_all.append(scores)
        rows.append(
            {
                "time": entry["time"],
                "auc": float(roc_auc_score(labels, scores)),
                "ap": float(average_precision_score(labels, scores)),
                "mrr": ranking_mrr(
                    "decoupled",
                    predictions[:npos],
                    predictions[npos + nneg :],
                    archive["positive_sources"],
                    archive["rank_sources"],
                ),
                **ranking_tie_diagnostics(
                    "decoupled",
                    predictions[:npos],
                    predictions[npos + nneg :],
                    archive["positive_sources"],
                    archive["rank_sources"],
                ),
            }
        )
    return aggregate(rows, labels_all, scores_all), rows


def paired(dataset, seed, directory, checkpoint, threads):
    events = prepare(dataset, seed, directory)
    index, reference = inject_native_queries(events, directory)
    cache = directory / "features_full"
    manifest = build_feature_cache(
        events,
        index,
        directory / "protocol",
        cache,
        checkpoint,
        stage="test",
        threads=threads,
    )
    _, head, _ = load_checkpoint(checkpoint)
    lock_path = directory / "validation_lock.json"
    readouts_path = directory / "readouts.pt"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        fitted = torch.load(readouts_path, weights_only=False)
    else:
        # Preserve the prior validation-selected channel definitions; only ridge
        # is selected again because the corrected UCI training set differs.
        previous = (
            Path("runs/optimization/temporal_dev/decoupled")
            / dataset
            / "s20261115/validation_lock.json"
        )
        previous_lock = json.loads(previous.read_text())
        lock, fitted = {
            "seed": seed,
            "configuration_source": str(previous),
            "test_used_for_selection": False,
            "variants_fixed_from_prior_development": True,
        }, {}
        for name in ("baseline", "selected"):
            configuration = previous_lock[name]
            trials = []
            for ridge in (0.1, 1.0, 10.0):
                fit = fit_configuration(
                    cache,
                    manifest,
                    head,
                    configuration["variant"],
                    configuration["half_life"],
                    ridge,
                )
                candidate = {
                    "variant": configuration["variant"],
                    "half_life": configuration["half_life"],
                    "ridge": ridge,
                }
                metric, _ = evaluate(
                    cache,
                    manifest,
                    directory / "protocol",
                    head,
                    fit,
                    candidate,
                    "validation",
                )
                trials.append((metric["mrr"], candidate, fit, metric))
            _, best, fitted[name], validation = max(trials, key=lambda item: item[0])
            lock[name] = dict(best, validation=validation)
        torch.save(fitted, readouts_path)
        write_json(lock_path, lock)
    results = {"DecoupledDGNN": reference}
    for name in ("baseline", "selected"):
        metric, rows = evaluate(
            cache,
            manifest,
            directory / "protocol",
            head,
            fitted[name],
            lock[name],
            "test",
        )
        results[name] = metric
        write_json(directory / f"test_{name}.json", rows)
    write_json(
        directory / "paired_summary.json",
        {
            "project": "Incremental Predictive Analytics over Evolving Graphs",
            "dataset": dataset,
            "seed": seed,
            "results": results,
            "configuration": lock,
            "data": events.metadata,
            "shared_candidates": "captured actual released-code test forwards, including original AUC batch slicing and first1000/source ranking candidates",
            "visibility": "past snapshots only; source-head offset unchanged; calibrator width17 and18 coefficients unchanged",
            "new_trainable_parameters": 0,
            "timing_comparison": None,
        },
    )
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["uci", "bitcoin-alpha", "bitcoin-otc"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action", choices=["prepare", "paired"], default="prepare")
    parser.add_argument("--checkpoint", default="checkpoints/reference.pt")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    directory = (
        Path("runs/protocol_alignment/decoupled") / args.dataset / f"s{args.seed}"
    )
    if args.action == "prepare":
        print(
            json.dumps(prepare(args.dataset, args.seed, directory).metadata), flush=True
        )
    else:
        paired(args.dataset, args.seed, directory, args.checkpoint, args.threads)


if __name__ == "__main__":
    main()
