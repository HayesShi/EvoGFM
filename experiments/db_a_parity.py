"""Connect the selected quality readout to current-version demand features.

Checks deterministically spread candidate rows in every validation/test snapshot
against the original quality cache. This is a bridge correctness check, not a
replacement for full ranking evaluation (which is in db_a_quality).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import joblib
import numpy as np
import torch

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.temporal import TemporalHistory
from experiments.db_a_quality import predict_readout
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


@torch.no_grad()
def run(args, dataset):
    source = Path("runs/protocol_alignment/undergs") / dataset / "s42"
    candidates = source / "candidates"
    cache = source / "causal_adaptation/frozen/features"
    index = json.loads((candidates / "index.json").read_text())
    training = [
        np.load(candidates / row["file"])["positive"]
        for row in index
        if row["phase"] == "train"
    ]
    metadata = json.loads((candidates / "metadata.json").read_text())
    n = metadata["nodes"]
    edges = canonical_edges(torch.from_numpy(np.concatenate(training, axis=1)), n)
    model, head, _ = load_checkpoint("checkpoints/reference.pt")
    raw = frozen_structural_features(edges, n, model.config["input_dim"])
    encoded = model.encode_features(raw)
    eager = MaintainedLinkScorer(model, head, raw, raw, edges, encoded=encoded)
    demand = DemandLinkScorer(
        model, head, raw, edges, encoded=encoded, policy="closure"
    )
    readout = joblib.load(
        args.quality / "undergs" / dataset / "selected_readout.joblib"
    )
    history = TemporalHistory(n)
    known = (edges[0] * n + edges[1]).numpy()
    output = []
    for entry in index:
        with np.load(candidates / entry["file"]) as archive:
            positive = archive["positive"]
            if entry["phase"] == "train":
                history.ingest(positive, entry["time"])
                continue
            pairs = np.concatenate(
                (positive, archive["auc_negative"], archive["ranking_negative"]), axis=1
            )
        selected = np.unique(
            np.linspace(0, pairs.shape[1] - 1, min(args.rows, pairs.shape[1])).astype(
                np.int64
            )
        )
        pair = pairs[:, selected]
        statistics = history.statistics(pair, entry["time"]).numpy()
        ef = eager.features(torch.from_numpy(pair)).numpy()
        df = demand.features(torch.from_numpy(pair)).numpy()
        with np.load(cache / entry["file"]) as original:
            cf = original["base"][selected]
            cs = original["statistics"][selected]
        ep = predict_readout(readout, ef, statistics)
        dp = predict_readout(readout, df, statistics)
        cp = predict_readout(readout, cf, cs)
        result = dict(
            time=entry["time"],
            phase=entry["phase"],
            rows=len(selected),
            eager_cache_max_feature_error=float(np.max(np.abs(ef - cf))),
            demand_eager_max_feature_error=float(np.max(np.abs(df - ef))),
            temporal_max_error=float(
                np.max(
                    np.abs(
                        statistics[np.isfinite(statistics) & np.isfinite(cs)]
                        - cs[np.isfinite(statistics) & np.isfinite(cs)]
                    )
                )
            ),
            demand_cache_max_probability_error=float(np.max(np.abs(dp - cp))),
            eager_cache_max_probability_error=float(np.max(np.abs(ep - cp))),
            same_tree_probabilities=bool(np.array_equal(dp, cp)),
            numeric_parity=bool(
                np.allclose(df, cf, rtol=2e-4, atol=2e-4)
                and np.allclose(statistics, cs, rtol=1e-6, atol=1e-6)
            ),
        )
        output.append(result)
        history.ingest(positive, entry["time"])
        edges, added, known = ingest_edges(edges, torch.from_numpy(positive), n, known)
        eager.update(edges, added)
        demand.update(edges, added)
    report = dict(
        dataset=dataset,
        snapshots=len(output),
        rows=sum(x["rows"] for x in output),
        all_numeric_parity=all(x["numeric_parity"] for x in output),
        same_tree_probabilities=all(x["same_tree_probabilities"] for x in output),
        max_probability_error=max(
            x["demand_cache_max_probability_error"] for x in output
        ),
        max_feature_error=max(x["eager_cache_max_feature_error"] for x in output),
        results=output,
        scope="Fixed spread of candidate rows in every non-training snapshot; complete test quality metrics are the separate full db_a_quality evaluation.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{dataset}.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    if not report["all_numeric_parity"]:
        raise RuntimeError("Current version/cache mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["bitcoin-alpha", "bitcoin-otc", "uci"]
    )
    parser.add_argument("--quality", type=Path, default=Path("runs/vldb/quality"))
    parser.add_argument("--output", type=Path, default=Path("runs/vldb/parity"))
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for dataset in args.datasets:
        run(args, dataset)


if __name__ == "__main__":
    main()
