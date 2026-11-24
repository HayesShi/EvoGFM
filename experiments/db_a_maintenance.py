"""Exact full/eager/demand/hybrid replay over real timestamped graph updates.

Queries are explicitly generated service workloads, not captured production logs.
This measures maintenance mechanics independently of predictive-model selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.temporal import TemporalHistory
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf8",
    )
    temporary.replace(path)


def workload(n, version, kind, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, version, 721]))
    if kind == "mixed":
        # The switch is a generated workload property, not future information
        # supplied to the scheduler; it sees only each current request batch.
        kind = ("sparse", "hotspot", "dense")[version % 3]
    if kind == "sparse":
        return [
            torch.from_numpy(
                np.vstack((np.repeat(rng.integers(n), 8), rng.integers(n, size=8)))
            ).long()
        ]
    if kind == "hotspot":
        pairs = torch.from_numpy(
            np.vstack((np.repeat(seed % n, 32), rng.integers(n, size=32)))
        ).long()
        return [pairs] * 8
    if kind == "dense":
        return [
            torch.from_numpy(rng.integers(n, size=(2, 512))).long() for _ in range(4)
        ]
    raise ValueError(kind)


def replay(args, dataset):
    root = args.root
    protocol = root / "runs/protocol_alignment/undergs" / dataset / "s42/candidates"
    index = json.loads((protocol / "index.json").read_text(encoding="utf8"))
    events = [(row, np.load(protocol / row["file"])["positive"]) for row in index]
    n = max(int(x.max()) for _, x in events if x.size) + 1
    train = [x for row, x in events if row["phase"] == "train"]
    stream = [(row, x) for row, x in events if row["phase"] != "train"]
    if args.max_versions:
        stream = stream[: args.max_versions]
    initial = canonical_edges(torch.from_numpy(np.concatenate(train, axis=1)), n)
    model, head, _ = load_checkpoint(args.checkpoint)
    start = time.perf_counter()
    raw = frozen_structural_features(initial, n, model.config["input_dim"])
    encoded = model.encode_features(raw)
    quality_readout, readout_bytes = None, 0
    if args.quality is not None:
        import joblib
        from experiments.db_a_quality import predict_readout

        quality_readout = joblib.load(
            args.quality / "undergs" / dataset / "selected_readout.joblib"
        )
        metadata = json.loads((args.quality / "metadata_verification.json").read_text())
        readout_bytes = next(
            row["unique_numpy_payload_bytes"]
            for row in metadata["cases"]
            if row["protocol"] == "undergs" and row["dataset"] == dataset
        )
    feature_setup = time.perf_counter() - start
    declaration = dict(
        dataset=dataset,
        seed=args.seed,
        protocol="UnderGS timestamp bins",
        model_configuration=model.config,
        model_parameters=sum(p.numel() for p in model.parameters()),
        head_parameters=sum(p.numel() for p in head.parameters()),
        versions=len(stream),
        n_nodes=n,
        initial_edges=initial.shape[1],
        queries="Generated uniform sparse, within-version repeated hotspot, dense, or a periodic mixture of these; every method has identical queries.",
        methods=args.methods,
        repeats=args.repeats,
        threads=args.threads,
        materialize_fraction=args.materialize_fraction,
        encoding="Fixed train-prefix structural inputs and full pretrained backbone; no future edges in encoding.",
        equivalence="Feature allclose rtol=2e-4, atol=2e-4; score absolute error recorded. Full reference uses full-edge float64 statistics.",
        timing="Wall clock includes input encoding, initial graph state preparation, graph event ingestion, maintenance and every query feature/head. Data file loading, metric computation and prediction serialization excluded equally. Shared Windows host; CPU other projects may compete.",
        scope=(
            "Full released 17-channel model plus selected union24 HGBT and seven temporal channels; includes history ingestion, feature extraction and selected readout scoring."
            if quality_readout is not None
            else "Full released 17-channel model, no fitted temporal quality readout; maintained-feature and source-head numerical parity, not a new prediction-quality comparison."
        ),
        memory="Numeric state payload; dense cache allocation retained by lazy prototype. Process peak RSS not yet isolated.",
    )
    output = args.output / dataset
    output.mkdir(parents=True, exist_ok=True)
    save(output / "declaration.json", declaration)
    reports = []
    for kind in args.workloads:
        for repeat in range(args.repeats):
            references = {}
            for method in declaration["methods"]:
                initial_start = time.perf_counter()
                if method in ("full", "eager_incremental"):
                    scorer = MaintainedLinkScorer(
                        model,
                        head,
                        raw,
                        raw,
                        initial,
                        method="full" if method == "full" else "exact_incremental",
                        encoded=encoded,
                    )
                else:
                    scorer = DemandLinkScorer(
                        model,
                        head,
                        raw,
                        initial,
                        encoded=encoded,
                        policy={
                            "lazy_cached": "lazy",
                            "hybrid_coverage": "hybrid",
                            "hybrid_closure": "closure",
                        }[method],
                        materialize_fraction=args.materialize_fraction,
                    )
                history = None
                if quality_readout is not None:
                    history = TemporalHistory(n)
                    for entry, incoming in events:
                        if entry["phase"] == "train":
                            history.ingest(incoming, entry["time"])
                initialization_s = time.perf_counter() - initial_start
                visible = initial.clone()
                known = (visible[0] * n + visible[1]).numpy()
                update_s, query_s, latencies = 0.0, 0.0, []
                rows_evaluated, max_feature_error, max_score_error = 0, 0.0, 0.0
                valid, trajectory = True, []
                for version, (entry, incoming) in enumerate(stream):
                    batches = workload(n, version, kind, args.seed)
                    for batch_id, pairs in enumerate(batches):
                        start = time.perf_counter()
                        features = scorer.features(pairs)
                        if quality_readout is None:
                            prediction = head(features).flatten().numpy()
                        else:
                            statistics = history.statistics(
                                pairs.numpy(), entry["time"]
                            ).numpy()
                            prediction = predict_readout(
                                quality_readout, features.numpy(), statistics
                            )
                        duration = time.perf_counter() - start
                        latencies.append(duration)
                        query_s += duration
                        rows_evaluated += pairs.shape[1]
                        key = version, batch_id
                        feature_array = features.numpy().copy()
                        if method == "full":
                            references[key] = feature_array, prediction.copy()
                        else:
                            expected, score = references[key]
                            max_feature_error = max(
                                max_feature_error,
                                float(np.max(np.abs(feature_array - expected))),
                            )
                            max_score_error = max(
                                max_score_error,
                                float(np.max(np.abs(prediction - score))),
                            )
                            valid = valid and bool(
                                np.allclose(
                                    feature_array, expected, rtol=2e-4, atol=2e-4
                                )
                            )
                            if quality_readout is not None:
                                valid = valid and bool(
                                    np.allclose(prediction, score, rtol=1e-6, atol=1e-6)
                                )
                    # Every graph batch is ingested, including final outstanding work.
                    start = time.perf_counter()
                    visible, added, known = ingest_edges(
                        visible, torch.from_numpy(incoming), n, known
                    )
                    if history is not None:
                        history.ingest(incoming, entry["time"])
                    scorer.update(visible, added)
                    duration = time.perf_counter() - start
                    update_s += duration
                    trajectory.append(
                        dict(
                            version=version,
                            time=entry["time"],
                            phase=entry["phase"],
                            added_edges=added.shape[1],
                            update_and_ingest_s=duration,
                        )
                    )
                result = dict(
                    dataset=dataset,
                    workload=kind,
                    repeat=repeat,
                    method=method,
                    versions=len(stream),
                    scored_pairs=rows_evaluated,
                    shared_input_setup_s=feature_setup,
                    initialization_s=initialization_s,
                    update_and_ingest_s=update_s,
                    query_s=query_s,
                    total_s=feature_setup + initialization_s + update_s + query_s,
                    query_p95_ms=float(np.quantile(latencies, 0.95)) * 1000,
                    query_p99_ms=float(np.quantile(latencies, 0.99)) * 1000,
                    max_feature_abs_error=max_feature_error,
                    max_prediction_abs_error=max_score_error,
                    feature_allclose=valid,
                    numeric_state_bytes=scorer.cache_bytes(),
                    readout_numeric_bytes=readout_bytes,
                    temporal_history_bytes=(
                        history.state_bytes() if history is not None else 0
                    ),
                    evaluated_rows=getattr(scorer, "counters", None),
                )
                full = next(
                    (
                        r
                        for r in reports
                        if r["workload"] == kind
                        and r["repeat"] == repeat
                        and r["method"] == "full"
                    ),
                    result,
                )
                result["speedup_vs_full"] = full["total_s"] / result["total_s"]
                reports.append(result)
                save(
                    output / "progress.json",
                    dict(declaration=declaration, results=reports),
                )
                save(output / f"{kind}_r{repeat}_{method}_updates.json", trajectory)
                print(json.dumps(result), flush=True)
                del scorer
                if not valid:
                    raise RuntimeError(
                        "Demand evaluation failed numerical parity; see progress.json"
                    )
    save(output / "summary.json", dict(declaration=declaration, results=reports))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/reference.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["bitcoin-alpha", "bitcoin-otc", "uci"]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["sparse", "hotspot", "dense", "mixed"],
        default=["sparse", "hotspot", "dense"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-versions", type=int, default=0)
    parser.add_argument("--materialize-fraction", type=float, default=0.35)
    parser.add_argument("--quality", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "full",
            "eager_incremental",
            "lazy_cached",
            "hybrid_coverage",
            "hybrid_closure",
        ],
        choices=[
            "full",
            "eager_incremental",
            "lazy_cached",
            "hybrid_coverage",
            "hybrid_closure",
        ],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.methods[0] != "full":
        parser.error("The full method must run first for numerical parity.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for dataset in args.datasets:
            replay(args, dataset)


if __name__ == "__main__":
    main()
