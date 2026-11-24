"""Real temporal-stream pilot: Incremental Predictive Analytics over Evolving Graphs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from evogfm.data import canonical_edges
from evogfm.features import adjacency
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.readout import fit_residual, predict_residual
from experiments.full_graph import load_checkpoint


SOURCES = {
    "bitcoin-alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "bitcoin-otc": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
}


def download_events(dataset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    url = SOURCES[dataset]
    path = directory / url.rsplit("/", 1)[-1]
    if not path.exists():
        # Direct connection keeps this reproducible on machines with stale proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
    with gzip.open(path, "rt") as stream:
        events = np.loadtxt(stream, delimiter=",")
    ids = np.unique(events[:, :2].astype(np.int64))
    endpoints = np.searchsorted(ids, events[:, :2].astype(np.int64))
    order = np.argsort(events[:, 3], kind="stable")
    return endpoints[order], events[order, 3].astype(np.int64), ids, path


def edges_before(endpoints, time_bins, stop, n):
    return canonical_edges(torch.from_numpy(endpoints[time_bins < stop].T.copy()), n)


def snapshot_queries(endpoints, time_bins, step, visible, n, rng, cap, negatives):
    positive = canonical_edges(
        torch.from_numpy(endpoints[time_bins == step].T.copy()), n
    )
    # Repeated historical interactions remain valid event-prediction positives.
    if cap > 0 and positive.shape[1] > cap:
        positive = positive[
            :, torch.from_numpy(rng.choice(positive.shape[1], cap, replace=False))
        ]
    forbidden = set((visible[0] * n + visible[1]).tolist())
    current = canonical_edges(
        torch.from_numpy(endpoints[time_bins == step].T.copy()), n
    )
    forbidden.update((current[0] * n + current[1]).tolist())
    groups = []
    for source, target in positive.T.tolist():
        candidates = []
        taken = set()
        while len(candidates) < negatives:
            for other in rng.integers(0, n, max(64, 2 * negatives)):
                other = int(other)
                low, high = sorted((source, other))
                if (
                    source != other
                    and low * n + high not in forbidden
                    and other not in taken
                ):
                    candidates.append(other)
                    taken.add(other)
                    if len(candidates) == negatives:
                        break
        groups.append(
            torch.tensor([[source] * negatives, candidates], dtype=torch.long)
        )
    sampled = torch.cat(groups, dim=1) if groups else positive[:, :0]
    return positive, sampled


def frozen_structural_features(edges, n, dimension):
    """Degree/neighbor-degree histogram fitted only on the training prefix."""
    matrix = adjacency(edges, n).tocsr()
    degree = np.asarray(matrix.sum(1)).ravel()
    bins = min(dimension - 5, 24)
    boundary = np.geomspace(1, max(2.0, degree.max()), bins)
    assigned = np.minimum(bins - 1, np.searchsorted(boundary, degree))
    onehot = np.eye(bins, dtype=np.float32)[assigned]
    onehot[degree == 0] = 0
    histogram = matrix @ onehot
    result = np.zeros((n, dimension), dtype=np.float32)
    result[:, :bins] = np.log1p(histogram)
    result[:, bins] = np.log1p(degree)
    result[:, bins + 1] = np.log1p(matrix @ degree) / np.maximum(1, degree)
    result[:, bins + 2] = degree > 0
    result[:, bins + 3] = np.log1p(
        np.asarray(matrix.multiply(matrix.T).sum(axis=1)).ravel()
    )
    result[:, bins + 4] = 1
    result /= np.maximum(1, result.sum(1, keepdims=True))
    return torch.from_numpy(result)


@torch.no_grad()
def query_features(scorer, pairs):
    return torch.cat([scorer.features(block) for block in pairs.split(512, dim=1)])


@torch.no_grad()
def scores(scorer, pairs, fitted):
    features = query_features(scorer, pairs)
    return predict_residual(features, scorer.head(features).flatten(), fitted)


def atomic_json(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, default="bitcoin-alpha")
    parser.add_argument("--checkpoint", default="checkpoints/reference.pt")
    parser.add_argument("--data", default="data/temporal")
    parser.add_argument("--output", default="runs/idea_a/bitcoin-alpha")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--snapshot-days", type=int, default=7)
    parser.add_argument(
        "--query-cap",
        type=int,
        default=128,
        help="0 retains every current-snapshot positive",
    )
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--max-test-snapshots", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=MaintainedLinkScorer.METHODS,
        default=list(MaintainedLinkScorer.METHODS),
    )
    args = parser.parse_args()
    if "full" not in args.methods:
        parser.error("include --methods full to retain a prediction-error reference")
    if args.negatives < 1 or args.period < 1 or args.snapshot_days < 1:
        parser.error("negatives, period and snapshot-days must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    endpoints, timestamps, ids, source_file = download_events(
        args.dataset, Path(args.data)
    )
    bins = (timestamps - timestamps.min()) // (args.snapshot_days * 86400)
    count = int(bins.max()) + 1
    bin_offsets = np.searchsorted(bins, np.arange(count + 1))
    train_stop, validation_stop = int(0.7 * count), int(0.8 * count)
    stop = (
        count
        if not args.max_test_snapshots
        else min(count, validation_stop + args.max_test_snapshots)
    )
    model, head, checkpoint = load_checkpoint(args.checkpoint)
    feature_edges = edges_before(endpoints, bins, train_stop - 1, len(ids))
    raw = frozen_structural_features(feature_edges, len(ids), model.config["input_dim"])
    encode_start = time.perf_counter()
    with torch.no_grad():
        encoded = model.encode_features(raw)
    encoded_s = time.perf_counter() - encode_start
    config = dict(
        vars(args),
        title="Incremental Predictive Analytics over Evolving Graphs",
        upstream_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        model_config=model.config,
        source_url=SOURCES[args.dataset],
        source_file=str(source_file.resolve()),
        nodes=len(ids),
        raw_events=len(timestamps),
        snapshots=count,
        train_stop=train_stop,
        validation_stop=validation_stop,
        stream_stop=stop,
        feature_freeze_snapshot=train_stop - 1,
        encoded_once_s=encoded_s,
        device="cpu",
        graph_semantics="undirected simple cumulative visible graph; signed ratings ignored",
        task="predict next weekly rating interactions, including repeated historical pairs",
        negatives_protocol=f"{args.negatives} distinct destination corruptions per positive, excluding visible and target-snapshot positives; no later positives consulted",
        node_universe="fixed transductive node IDs; topology features use training prefix only",
        status="single-seed temporal feasibility pilot; not an UnderGS reproduction",
        refresh_trigger="relative Frobenius attention-program drift; no predictive-loss bound",
        quality_aggregation="unweighted means over nonempty test snapshots; per-query predictions retained",
        update_time_aggregation="update_s covers scored snapshots; stream_update_s_mean also includes empty target snapshots",
        timed_work="shared_ingest_s ingests the preceding event bin, excludes known edges, and merges a canonical visible graph; update_s maintains model caches; query_s prepares channels and scores fixed query pairs",
        untimed_work="download, initial feature encoding, calibration, query sampling, and prediction serialization",
        cache_measurement="distinct numeric storage reachable from scorer caches, including GraphState, moments, intermediate hidden states, input/edge arrays, degrees and sparse indices; NumPy/PyTorch aliases deduplicated by storage interval; excludes model/head weights, Python objects and allocator overhead",
        unused_backbone_skipped=not model.config.get("joint_task_backbone", False),
    )
    atomic_json(output / "configuration.json", config)
    # The final training snapshot provides labels; features precede those labels.
    support_pos, support_neg = snapshot_queries(
        endpoints, bins, train_stop - 1, feature_edges, len(ids), rng, 512, 1
    )
    support = torch.cat((support_pos, support_neg), dim=1)
    labels = torch.cat(
        (torch.ones(support_pos.shape[1]), torch.zeros(support_neg.shape[1]))
    )
    calibration = MaintainedLinkScorer(
        model, head, raw, raw, feature_edges, encoded=encoded
    )
    fitted_features = query_features(calibration, support)
    with torch.no_grad():
        fitted = fit_residual(
            fitted_features, head(fitted_features).flatten(), labels, ridge=1.0
        )
    torch.save(
        dict(fit=fitted, pairs=support, labels=labels), output / "calibration.pt"
    )
    methods = {
        method: MaintainedLinkScorer(
            model,
            head,
            raw,
            raw,
            feature_edges,
            method,
            args.period,
            args.threshold,
            encoded=encoded,
        )
        for method in args.methods
    }
    previous = feature_edges
    known_keys = (previous[0] * len(ids) + previous[1]).numpy()
    rows = []
    update_rows = []
    trajectory = []
    score_file = output / "predictions.jsonl"
    score_file.write_text("", encoding="utf-8")
    fieldnames = [
        "snapshot",
        "phase",
        "method",
        "visible_edges",
        "added_edges",
        "queries",
        "auc",
        "mrr",
        "score_mae_vs_full",
        "score_max_error_vs_full",
        "cache_bytes",
        "query_s",
        "shared_ingest_s",
        "maintenance_plus_query_s",
        "statistics_s",
        "spectral_s",
        "graph_index_s",
        "affinity_s",
        "propagation_s",
        "backbone_s",
        "update_s",
        "refreshed",
        "drift",
        "affected_fraction",
    ]
    with (output / "measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(train_stop, stop):
            ingest_start = time.perf_counter()
            incoming = torch.from_numpy(
                endpoints[bin_offsets[step - 1] : bin_offsets[step]].T.copy()
            )
            visible, added, known_keys = ingest_edges(
                previous, incoming, len(ids), known_keys
            )
            shared_ingest_s = time.perf_counter() - ingest_start
            updates = {
                method: scorer.update(visible, added)
                for method, scorer in methods.items()
            }
            for method, values in updates.items():
                if not np.isfinite(values["drift"]):
                    values["drift"] = None
                update_rows.append(
                    dict(
                        snapshot=step,
                        phase="validation" if step < validation_stop else "test",
                        method=method,
                        shared_ingest_s=shared_ingest_s,
                        query_s=0.0,
                        maintenance_plus_query_s=values["update_s"],
                        **values,
                    )
                )
            positive, negative = snapshot_queries(
                endpoints,
                bins,
                step,
                visible,
                len(ids),
                rng,
                args.query_cap,
                args.negatives,
            )
            previous = visible
            if positive.shape[1] == 0:
                trajectory.append(dict(snapshot=step, positives=0, updates=updates))
                continue
            pairs = torch.cat((positive, negative), dim=1)
            predictions, query_time = {}, {}
            for method, scorer in methods.items():
                start = time.perf_counter()
                predictions[method] = scores(scorer, pairs, fitted)
                query_time[method] = time.perf_counter() - start
                matching_update = next(
                    row for row in reversed(update_rows) if row["method"] == method
                )
                matching_update["query_s"] = query_time[method]
                matching_update["maintenance_plus_query_s"] += query_time[method]
            truth = np.r_[np.ones(positive.shape[1]), np.zeros(negative.shape[1])]
            for method, prediction in predictions.items():
                npos = positive.shape[1]
                negatives = prediction[npos:].reshape(npos, args.negatives)
                rank = (
                    1
                    + 0.5
                    * (
                        (negatives >= prediction[:npos, None]).sum(1)
                        + (negatives > prediction[:npos, None]).sum(1)
                    ).double()
                )
                reference = predictions.get("full", prediction)
                row = dict(
                    snapshot=step,
                    phase="validation" if step < validation_stop else "test",
                    method=method,
                    visible_edges=visible.shape[1],
                    added_edges=added.shape[1],
                    queries=npos,
                    auc=float(roc_auc_score(truth, prediction.numpy())),
                    mrr=float((1 / rank).mean()),
                    score_mae_vs_full=float((prediction - reference).abs().mean()),
                    score_max_error_vs_full=float((prediction - reference).abs().max()),
                    cache_bytes=methods[method].cache_bytes(),
                    query_s=query_time[method],
                    shared_ingest_s=shared_ingest_s,
                    maintenance_plus_query_s=updates[method]["update_s"]
                    + query_time[method],
                    **updates[method],
                )
                writer.writerow(row)
                rows.append(row)
            stream.flush()
            with score_file.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        dict(
                            snapshot=step,
                            pairs=pairs.T.tolist(),
                            positive_count=positive.shape[1],
                            scores={
                                name: prediction.tolist()
                                for name, prediction in predictions.items()
                            },
                        )
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "snapshot": step,
                        "phase": rows[-1]["phase"],
                        "queries": positive.shape[1],
                        "methods": {
                            method: {
                                key: next(
                                    row[key]
                                    for row in reversed(rows)
                                    if row["method"] == method
                                )
                                for key in (
                                    "auc",
                                    "mrr",
                                    "update_s",
                                    "score_mae_vs_full",
                                    "refreshed",
                                )
                            }
                            for method in methods
                        },
                    }
                ),
                flush=True,
            )
    summary = {}
    with (output / "updates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(update_rows[0]))
        writer.writeheader()
        writer.writerows(update_rows)
    for method in methods:
        selected = [
            row for row in rows if row["phase"] == "test" and row["method"] == method
        ]
        if selected:
            summary[method] = {
                key: float(np.mean([row[key] for row in selected]))
                for key in (
                    "auc",
                    "mrr",
                    "update_s",
                    "query_s",
                    "score_mae_vs_full",
                    "refreshed",
                    "cache_bytes",
                    "statistics_s",
                    "spectral_s",
                    "graph_index_s",
                    "affinity_s",
                    "propagation_s",
                    "backbone_s",
                )
            }
            summary[method]["test_snapshots"] = len(selected)
            stream_updates = [
                row
                for row in update_rows
                if row["phase"] == "test" and row["method"] == method
            ]
            summary[method]["stream_update_s_mean"] = float(
                np.mean([row["update_s"] for row in stream_updates])
            )
            summary[method]["stream_update_s_total"] = float(
                sum(row["update_s"] for row in stream_updates)
            )
            summary[method]["stream_maintenance_plus_query_s_mean"] = float(
                np.mean([row["maintenance_plus_query_s"] for row in stream_updates])
            )
            summary[method]["stream_shared_ingest_s_mean"] = float(
                np.mean([row["shared_ingest_s"] for row in stream_updates])
            )
            summary[method]["stream_test_snapshots"] = len(stream_updates)
            summary[method]["test_positive_queries"] = int(
                sum(row["queries"] for row in selected)
            )
    atomic_json(output / "summary.json", summary)
    # Empty target snapshots still contribute update work to the stream log.
    (output / "empty_snapshots.json").write_text(
        json.dumps(trajectory, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
