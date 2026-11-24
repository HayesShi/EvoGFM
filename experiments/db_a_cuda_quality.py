"""Replay the existing held-out UnderGS candidate sets on CUDA, seed 42."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from evogfm.data import canonical_edges
from evogfm.incremental import ingest_edges
from evogfm.shared_attention_cuda import CudaLinkScorer
from evogfm.temporal import TemporalHistory, prepare_dataset
from evogfm.three_hop_csr import three_hop_walk_counts
from evogfm.undergs_protocol import CausalNeighborhood, restore_runtime, score_metrics
from experiments.db_a_native import DEFAULT_CHECKPOINT, WORKSPACE
from experiments.db_a_native_trace import write_json
from experiments.db_a_quality import predict_readout
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features
from experiments.temporal_official import original_undergs_modules


def arrays(directory, entry):
    with np.load(directory / entry["file"]) as data:
        return [data[name] for name in ("positive", "auc_negative", "ranking_negative")]


def observed_candidates(events, index, candidates):
    seen = np.zeros(events.n_nodes, dtype=np.bool_)
    rng = np.random.default_rng(42)
    queries = {}
    for entry in index:
        with np.load(candidates / entry["file"]) as pack:
            positive = pack["positive"]
        if entry["phase"] == "test":
            active = np.flatnonzero(seen)
            negative = np.empty((100, positive.shape[1]), dtype=np.int64)
            for node in np.unique(positive[0]):
                group = np.flatnonzero(positive[0] == node)
                allowed = np.setdiff1d(active, np.r_[node, positive[1, group]])
                if len(allowed) < 100:
                    raise ValueError(
                        "Fewer than 100 previously observed eligible destinations"
                    )
                for row in group:
                    negative[:, row] = rng.choice(allowed, 100, replace=False)
            rank = np.stack((np.tile(positive[0], 100), negative.ravel()))
            queries[entry["time"]] = positive, rank, rank
        seen[positive] = True
    return queries


def observed_metrics(positive, negative, pos_pairs, neg_pairs):
    metrics = score_metrics(positive, negative, positive, negative)
    pos = np.asarray(positive).reshape(-1, 1)
    neg = np.asarray(negative).reshape(100, -1).T
    precedes = neg_pairs[1].reshape(100, -1).T < pos_pairs[1, :, None]
    ranks = 1 + ((neg > pos) | ((neg == pos) & precedes)).sum(1)
    metrics["mrr"] = float(np.mean(1 / ranks))
    return metrics


@torch.no_grad()
def ours(events, index, candidates, readout, overrides):
    model, head, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    train = events.phase == "train"
    edges = torch.from_numpy(
        np.stack((events.source[train], events.destination[train]))
    )
    raw = frozen_structural_features(
        canonical_edges(edges, events.n_nodes), events.n_nodes, 100
    )
    before = [row for row in index if row["phase"] != "test"]
    previous = [arrays(candidates, row)[0] for row in before]
    visible = canonical_edges(
        torch.from_numpy(np.concatenate(previous, axis=1)), events.n_nodes
    )
    known = (visible[0] * events.n_nodes + visible[1]).numpy()
    history = TemporalHistory(events.n_nodes)
    for entry, positive in zip(before, previous):
        history.ingest(positive, entry["time"])
    scorer = CudaLinkScorer(
        model,
        head,
        raw,
        visible,
        resident_scores=True,
        path_counts=three_hop_walk_counts,
        tile_edges=32768,
    )
    for entry in index:
        if entry["phase"] != "test":
            continue
        positive, auc_neg, rank_neg = (
            overrides[entry["time"]] if overrides else arrays(candidates, entry)
        )
        pairs = np.concatenate(
            (positive, rank_neg) if overrides else (positive, auc_neg, rank_neg), axis=1
        )
        parts = []
        for request in torch.from_numpy(pairs).split(2048, dim=1):
            features = scorer.features(request).numpy()
            history_features = history.statistics(
                request.numpy(), entry["time"]
            ).numpy()
            node_ages = (
                entry["time"] - history.last_node[request.numpy()]
                if "node_activity" in readout
                else None
            )
            parts.append(
                predict_readout(
                    readout, features, history_features, node_ages=node_ages
                )
            )
        result = np.concatenate(parts)
        offset = positive.shape[1]
        end = offset + auc_neg.shape[1]
        rank = result[offset:] if overrides else result[end:]
        if overrides:
            yield observed_metrics(result[:offset], rank, positive, rank_neg)
        else:
            yield score_metrics(
                result[:offset], result[offset:end], result[:offset], rank
            )
        history.ingest(positive, entry["time"])
        visible, added, known = ingest_edges(
            visible, torch.from_numpy(positive), events.n_nodes, known
        )
        scorer.update(visible, added)


@torch.no_grad()
def undergs(events, index, candidates, checkpoint, overrides):
    module, util, _ = original_undergs_modules(WORKSPACE / "baselines")
    config = argparse.Namespace(
        n_nodes=events.n_nodes + 1,
        n_edges=len(events.source) + 1,
        topk=16,
        lambda_1=0.1,
        lambda_2=0.05,
        filter_ratio=0.3,
        device=torch.device("cuda"),
    )
    model = (
        module.UnderGS(
            None, events.edge_features, config.device, 0.1, 128, 8, "gcn", config
        )
        .cuda()
        .eval()
    )
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(saved["parameters"])
    restore_runtime(model, saved["runtime"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    np.random.set_state(saved["numpy_rng"])
    neighborhood = CausalNeighborhood(model, util)
    for entry in index:
        if entry["phase"] != "test":
            continue
        positive, auc_neg, rank_neg = (
            overrides[entry["time"]] if overrides else arrays(candidates, entry)
        )
        selected = events.events_at(entry["time"])
        source, destination = positive + 1
        times, edges = events.time[selected], events.event_index[selected]
        auc_positive, auc_negative = neighborhood.score(
            source, destination, auc_neg[1] + 1, times, edges
        )
        if overrides:
            rank_positive, rank_negative = auc_positive, auc_negative
        else:
            rank_positive, rank_negative = neighborhood.score(
                source, destination, rank_neg[1] + 1, times, edges
            )
        neighborhood.observe(source, destination, times, edges)
        model.embedding_module.tcns.snapshot_decay(config.lambda_2)
        predictions = [
            value.cpu().numpy()
            for value in (auc_positive, auc_negative, rank_positive, rank_negative)
        ]
        if overrides:
            yield observed_metrics(predictions[0], predictions[1], positive, rank_neg)
        else:
            yield score_metrics(*predictions)


def measure(label, generator):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    rows = list(generator)
    torch.cuda.synchronize()
    return {
        "method": label,
        "auc": float(np.mean([r["auc"] for r in rows])),
        "mrr": float(np.mean([r["mrr"] for r in rows])),
        "snapshots": len(rows),
        "elapsed_s": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("bitcoin-alpha", "bitcoin-otc", "uci"), required=True
    )
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-negatives", action="store_true")
    args = parser.parse_args()
    root = WORKSPACE / "runs/protocol_alignment/undergs" / args.dataset / "s42"
    candidates = root / "candidates"
    index = json.loads((candidates / "index.json").read_text(encoding="utf-8"))
    events = prepare_dataset(
        args.dataset, "undergs", WORKSPACE / "data/temporal", WORKSPACE / "baselines"
    )
    overrides = (
        observed_candidates(events, index, candidates)
        if args.observed_negatives
        else None
    )
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    model_path = root / "causal_adaptation/undergs/pretest_state.pt"
    base = measure(
        "UnderGS causal adaptation",
        undergs(events, index, candidates, model_path, overrides),
    )
    print(json.dumps(base), flush=True)
    torch.cuda.empty_cache()
    readout = joblib.load(args.readout)
    ours_result = measure("Ours", ours(events, index, candidates, readout, overrides))
    print(json.dumps(ours_result), flush=True)
    write_json(
        args.output,
        {
            "dataset": args.dataset,
            "seed": 42,
            "gpu": torch.cuda.get_device_name(),
            "readout_training_seed": readout.get(
                "training_seed", readout.get("seed", 42)
            ),
            "scope": "Existing causal UnderGS snapshot test split, previously used for development",
            "candidate_protocol": (
                "100 distinct observed destinations, same AUC and MRR candidates"
                if overrides
                else "Existing 1:1 AUC and 1:100 MRR candidates"
            ),
            "candidates": str(candidates),
            "readout": str(args.readout.resolve()),
            "results": [base, ours_result],
        },
    )


if __name__ == "__main__":
    main()
