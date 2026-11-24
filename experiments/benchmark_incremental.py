"""Replay the locked predictive model with three incremental maintenance policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from evogfm.data import canonical_edges
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.readout import predict_residual
from evogfm.temporal import (
    TemporalHistory,
    prepare_dataset,
    ranking_mrr,
    ranking_tie_diagnostics,
    replace_temporal,
)
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


TITLE = "Incremental Predictive Analytics over Evolving Graphs"


def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.official_output / args.protocol / args.dataset / f"s{args.seed}"
    protocol_dir = source / "protocol"
    index = json.loads((protocol_dir / "index.json").read_text())
    lock = json.loads((args.lock / "validation_lock.json").read_text())
    if lock["dataset"] != args.dataset or lock["protocol"] != args.protocol:
        raise ValueError("The locked readout must match the replayed dataset/protocol")
    selected = lock["selected"]
    fitted = torch.load(args.lock / "readouts.pt", weights_only=False)["selected"]
    events = prepare_dataset(args.dataset, args.protocol, baselines=args.baselines)
    model, head, _ = load_checkpoint(args.checkpoint)
    mask = events.phase == "train"
    train_graph = canonical_edges(
        torch.from_numpy(np.stack((events.source[mask], events.destination[mask]))),
        events.n_nodes,
    )
    raw = frozen_structural_features(
        train_graph, events.n_nodes, model.config["input_dim"]
    )
    encoded = model.encode_features(raw)
    first = index[0]
    warm = np.load(protocol_dir / first["file"])["positive"]
    summaries = []
    full_predictions = {}
    for method in args.methods:
        visible = canonical_edges(torch.from_numpy(warm), events.n_nodes)
        known = (visible[0] * events.n_nodes + visible[1]).numpy()
        scorer = MaintainedLinkScorer(
            model,
            head,
            raw,
            raw,
            visible,
            method=method,
            period=args.period,
            threshold=args.threshold,
            encoded=encoded,
        )
        history = TemporalHistory(events.n_nodes)
        history.ingest(warm, first["time"])
        trajectory, maintenance, payload = [], [], {}
        for entry in index[1:]:
            query = np.load(protocol_dir / entry["file"])
            positive = query["positive"]
            if entry["phase"] == "test":
                pairs = np.concatenate(
                    (positive, query["auc_negative"], query["ranking_negative"]), axis=1
                )
                started = time.perf_counter()
                with torch.no_grad():
                    base = torch.cat(
                        [
                            scorer.features(block)
                            for block in torch.from_numpy(pairs).split(1024, dim=1)
                        ]
                    )
                    statistics = history.statistics(pairs, entry["time"])
                    features = replace_temporal(
                        base, statistics, selected["variant"], selected["half_life"]
                    )
                    offset = torch.cat(
                        [head(block).flatten() for block in base.split(8192)]
                    )
                    probability = (
                        predict_residual(features, offset, fitted)
                        .sigmoid()
                        .float()
                        .numpy()
                    )
                query_s = time.perf_counter() - started
                npos, nneg = positive.shape[1], query["auc_negative"].shape[1]
                truth = np.r_[np.ones(npos), np.zeros(nneg)]
                if method == "full":
                    full_predictions[entry["time"]] = probability
                reference = full_predictions[entry["time"]]
                trajectory.append(
                    dict(
                        method=method,
                        time=entry["time"],
                        phase="test",
                        positives=npos,
                        scored_pairs=len(probability),
                        query_s=query_s,
                        auc=float(roc_auc_score(truth, probability[: npos + nneg])),
                        mrr=ranking_mrr(
                            args.protocol,
                            probability[:npos],
                            probability[npos + nneg :],
                            positive[0],
                            query["rank_sources"],
                        ),
                        **ranking_tie_diagnostics(
                            args.protocol,
                            probability[:npos],
                            probability[npos + nneg :],
                            positive[0],
                            query["rank_sources"],
                        ),
                        mean_probability_error_vs_full=float(
                            np.mean(np.abs(probability - reference))
                        ),
                        max_probability_error_vs_full=float(
                            np.max(np.abs(probability - reference))
                        ),
                    )
                )
                payload[f"t{entry['time']}"] = probability
            started = time.perf_counter()
            history.ingest(positive, entry["time"])
            visible, added, known = ingest_edges(
                visible, torch.from_numpy(positive), events.n_nodes, known
            )
            update = scorer.update(visible, added)
            maintenance.append(
                dict(
                    method=method,
                    time=entry["time"],
                    phase=entry["phase"],
                    stream_transition_s=time.perf_counter() - started,
                    **update,
                )
            )
        np.savez_compressed(args.output / f"{method}_predictions.npz", **payload)
        relevant = [row for row in maintenance if row["phase"] == "test"]
        result = dict(
            method=method,
            snapshots=len(trajectory),
            auc=float(np.mean([row["auc"] for row in trajectory])),
            mrr=float(np.mean([row["mrr"] for row in trajectory])),
            pessimistic_mrr=float(
                np.mean([row["pessimistic_mrr"] for row in trajectory])
            ),
            test_update_s=sum(row["update_s"] for row in relevant),
            test_stream_transition_s=sum(
                row["stream_transition_s"] for row in relevant
            ),
            test_query_s=sum(row["query_s"] for row in trajectory),
            probability_mae_vs_full=float(
                np.mean([row["mean_probability_error_vs_full"] for row in trajectory])
            ),
            probability_max_error_vs_full=max(
                row["max_probability_error_vs_full"] for row in trajectory
            ),
            scorer_cache_bytes=scorer.cache_bytes(),
            temporal_history_bytes=history.state_bytes(),
            parameters=sum(p.numel() for p in model.parameters())
            + sum(p.numel() for p in head.parameters())
            + fitted[0].numel(),
        )
        summaries.append(result)
        for suffix, rows in (("updates", maintenance), ("queries", trajectory)):
            with (args.output / f"{method}_{suffix}.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
                )
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(result), flush=True)
    full = summaries[0]
    expected = json.loads((args.lock / "test_summary.json").read_text())["results"][
        "selected"
    ]
    agreement = {key: abs(full[key] - expected[key]) for key in ("auc", "mrr")}
    if max(agreement.values()) > 1e-6:
        raise RuntimeError(
            f"Full replay disagrees with the original locked evaluation: {agreement}"
        )
    for row in summaries:
        row["update_speedup_vs_full"] = full["test_update_s"] / row["test_update_s"]
        row["transition_query_speedup_vs_full"] = (
            full["test_stream_transition_s"] + full["test_query_s"]
        ) / (row["test_stream_transition_s"] + row["test_query_s"])
    report = dict(
        title=TITLE,
        name_zh="面向演化图的增量式预测分析",
        dataset=args.dataset,
        protocol=args.protocol,
        seed=args.seed,
        threads=args.threads,
        locked_readout=str(args.lock.resolve()),
        locked_configuration=selected,
        period=args.period,
        threshold=args.threshold,
        results=summaries,
        full_replay_metric_difference=agreement,
        observation_policy="test queries precede ingestion of that snapshot; static structural frame fitted on training graph only",
        timing="Serial methods, CPU two threads by default; query scoring includes historical statistics and readout, excludes file I/O/metrics; stream transition includes ingestion and maintenance. Shared-machine concurrent jobs may affect timings.",
        complexity="Local refresh still rebuilds sparse adjacency and copies caches; no O(delta E) total-complexity claim",
        source_checkpoint=str(Path(args.checkpoint).resolve()),
        paper_table_comparability="Same candidate/evaluation distribution, causal observation; released UnderGS incorporates current positives before scoring",
    )
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bitcoin-otc")
    parser.add_argument(
        "--protocol", default="undergs", choices=["undergs", "decoupled"]
    )
    parser.add_argument("--seed", default=20261115, type=int)
    parser.add_argument("--official-output", type=Path, default=Path("runs/official_a"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoints/reference.pt")
    parser.add_argument("--baselines", default="baselines")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "exact_incremental", "error_aware"],
        choices=MaintainedLinkScorer.METHODS,
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    if args.methods[0] != "full":
        parser.error("full must be the first method for the reference comparison")
    run(args)


if __name__ == "__main__":
    main()
