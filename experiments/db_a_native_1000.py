"""GPU replay of the frozen predictor on captured DecoupledDGNN 1,000-negative queries."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evogfm.data import canonical_edges
from evogfm.decoupled_protocol import prepare_released_events
from evogfm.incremental import ingest_edges
from evogfm.shared_attention_cuda import CudaLinkScorer
from evogfm.temporal import TemporalHistory, ranking_mrr, ranking_tie_diagnostics
from evogfm.three_hop_csr import three_hop_walk_counts
from experiments.db_a_execution_suite import write_json
from experiments.db_a_native import DEFAULT_CHECKPOINT, WORKSPACE
from experiments.db_a_native_trace import identity
from experiments.db_a_quality import predict_readout
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


@torch.no_grad()
def replay(dataset, output):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = WORKSPACE / "runs/protocol_alignment/decoupled" / dataset / "s42"
    protocol = baseline / "protocol"
    reference = json.loads(
        (baseline / "reference_replay.json").read_text(encoding="utf-8")
    )
    assert reference["all_native_metrics_reproduced"]
    index = json.loads((protocol / "index.json").read_text(encoding="utf-8"))
    readout_file = (
        WORKSPACE / "runs/vldb/quality/decoupled" / dataset / "selected_readout.joblib"
    )
    readout = joblib.load(readout_file)
    events = prepare_released_events(
        dataset, WORKSPACE / "data/temporal", WORKSPACE / "baselines"
    )
    n = events.n_nodes
    model, head, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    train = events.phase == "train"
    pairs_train = torch.from_numpy(
        np.stack((events.source[train], events.destination[train]))
    )
    train_edges = canonical_edges(pairs_train, n)
    raw = frozen_structural_features(train_edges, n, 100)
    history = TemporalHistory(n)
    previous = []
    for row in index:
        if row["phase"] == "test":
            continue
        with np.load(protocol / row["file"]) as data:
            positive = data["positive"]
        previous.append(positive)
        history.ingest(positive, row["time"])
    visible = canonical_edges(torch.from_numpy(np.concatenate(previous, axis=1)), n)
    known = (visible[0] * n + visible[1]).numpy()
    scorer = CudaLinkScorer(
        model,
        head,
        raw,
        visible,
        resident_scores=True,
        gpu_propagation=True,
        path_counts=three_hop_walk_counts,
        tile_edges=32768,
        async_indices=True,
    )
    rows, all_truth, all_scores = [], [], []
    started = time.perf_counter()
    for row in (entry for entry in index if entry["phase"] == "test"):
        with np.load(protocol / row["file"]) as data:
            positive, negative, ranking = (
                data[key] for key in ("positive", "auc_negative", "ranking_negative")
            )
            positive_auc, sources = data["auc_positive_indices"], data["rank_sources"]
        if ranking.shape[1] != 1000 * len(sources):
            raise ValueError(
                "Native ranking requires 1,000 fixed negatives for each source"
            )
        results = []
        for pairs in torch.from_numpy(
            np.concatenate((positive, negative, ranking), axis=1)
        ).split(2048, dim=1):
            base = scorer.features(pairs).numpy()
            statistics = history.statistics(pairs.numpy(), row["time"]).numpy()
            results.append(
                predict_readout(readout, base, statistics, protocol="decoupled")
            )
        prediction = np.concatenate(results)
        pos_end, auc_end = positive.shape[1], positive.shape[1] + negative.shape[1]
        roc_scores = np.r_[prediction[positive_auc], prediction[pos_end:auc_end]]
        roc_labels = np.arange(len(roc_scores)) < len(positive_auc)
        all_truth.append(roc_labels)
        all_scores.append(roc_scores)
        rank_args = (
            "decoupled",
            prediction[:pos_end],
            prediction[auc_end:],
            positive[0],
            sources,
        )
        rows.append(
            dict(
                time=row["time"],
                sources=len(sources),
                positives=pos_end,
                ranking_candidates=ranking.shape[1],
                mrr=ranking_mrr(*rank_args),
                **ranking_tie_diagnostics(*rank_args),
            )
        )
        history.ingest(positive, row["time"])
        visible, inserted, known = ingest_edges(
            visible, torch.from_numpy(positive), n, known
        )
        scorer.update(visible, inserted)
        if len(rows) % 5 == 0:
            write_json(
                output / "status.json",
                {
                    "phase": "native test ranking",
                    "completed_snapshots": len(rows),
                    "scored_ranking_negatives": sum(
                        row["ranking_candidates"] for row in rows
                    ),
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
            )
    write_json(
        output / "result.json",
        dict(
            dataset=dataset,
            seed=42,
            method="Ours locked native readout",
            gpu=torch.cuda.get_device_name(),
            native_official_result=reference["result"],
            reference_artifact=str(baseline / "reference_replay.json"),
            result=dict(
                mrr=float(np.mean([row["mrr"] for row in rows])),
                auc=float(
                    roc_auc_score(np.concatenate(all_truth), np.concatenate(all_scores))
                ),
                ap=float(
                    average_precision_score(
                        np.concatenate(all_truth), np.concatenate(all_scores)
                    )
                ),
            ),
            scoring_s=time.perf_counter() - started,
            rows=rows,
            readout=identity(readout_file),
            readout_id=readout["selected_id"],
            candidate_directory=str(protocol),
            contract="Released candidates, 1,000 negatives per source; best positive per source and pessimistic ties; snapshot macro MRR",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("bitcoin-alpha", "bitcoin-otc", "uci"), required=True
    )
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replay(args.dataset, args.output)


if __name__ == "__main__":
    main()
