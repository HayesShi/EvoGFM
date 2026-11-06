"""Evaluate official HeaRT hard negatives with the frozen EvoGFM checkpoint.

Place official train/valid/test positives, heart_*_samples.npy and gnn_feature
under data/heart/<dataset>/. No candidate regeneration or split substitution.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from evogfm.data import seed_all, canonical_edges, save_json
from evogfm.features import (
    preprocess,
    contexts,
    task_features,
    pair_context,
    pair_channels,
    attention,
    node_representations,
)
from evogfm.evaluation import fit_link_readout
from evogfm.readout import predict_residual, hard_negative_metrics
from evogfm.training import load_checkpoint


def evaluate(
    checkpoint="runs/source/best.pt",
    data_root="data",
    dataset="citeseer",
    split="valid",
    output="runs/heart",
    batch_size=2048,
    ablation="none",
    threads=4,
    negative_mode="uniform",
):
    seed_all(42)
    torch.set_num_threads(threads)
    folder = Path(data_root) / "heart" / dataset
    features = torch.load(
        folder / "gnn_feature", map_location="cpu", weights_only=False
    )["entity_embedding"].float()
    edges = torch.from_numpy(np.loadtxt(folder / "train_pos.txt", dtype=np.int64)).T
    edges = canonical_edges(edges, len(features))
    positives = torch.from_numpy(
        np.loadtxt(folder / f"{split}_pos.txt", dtype=np.int64)
    )
    negatives = torch.from_numpy(np.load(folder / f"heart_{split}_samples.npy"))
    if (
        negatives.ndim != 3
        or negatives.shape[0] != len(positives)
        or negatives.shape[2] != 2
    ):
        raise ValueError(
            "Expected official negatives with shape [positive_count, candidate_count, 2]"
        )
    raw, x = preprocess(dict(x=features))
    model, head, ckpt = load_checkpoint(checkpoint)
    if ablation == "zero-inverse":
        model.inverse.set_theta(torch.zeros_like(model.inverse.theta()))
    # All readout fitting stays inside the training graph, as in public LP.
    fitted = fit_link_readout(
        model, head, raw, x, edges, observation_rate=0.85, negative_mode=negative_mode
    )
    score_parts = []
    with torch.no_grad():
        state = model.state(x, edges, 0.85)
        parts, h = node_representations(model, state, raw, edges, hops=2)
        visible = contexts(edges, len(raw), edges[:, :0])["visible"]
        prepared = (attention(model, state, visible), parts + [h])
        content = pair_context(raw, edges)
        pairs = torch.cat((positives, negatives.reshape(-1, 2))).T
        for batch in tqdm(
            pairs.split(batch_size, dim=1), desc=f"HeaRT {dataset} {split}"
        ):
            ctx = contexts(edges, len(raw), batch)
            base = pair_channels(raw, edges, batch, prepared=content)
            f, _ = task_features(
                model, state, raw, edges, batch, ctx, base, prepared=prepared
            )
            score_parts.append(
                predict_residual(f, head(f).squeeze(1), fitted["fitted"])
            )
    scores = torch.cat(score_parts)
    positive = scores[: len(positives), None]
    negative = scores[len(positives) :].reshape(negatives.shape[:2])
    row = dict(
        dataset=dataset,
        protocol="official HeaRT",
        split=split,
        ablation=ablation,
        seed=42,
        negative_mode=negative_mode,
        source_epoch=ckpt["epoch"],
        positive_pairs=len(positives),
        negatives_per_positive=negative.shape[1],
        **hard_negative_metrics(positive, negative),
    )
    save_json(Path(output) / f"{dataset}_{split}_{ablation}.json", row)
    print(row, flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--dataset", default="citeseer", choices=["cora", "citeseer", "pubmed"]
    )
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--output", default="runs/heart")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ablation", choices=["none", "zero-inverse"], default="none")
    parser.add_argument(
        "--negative-mode", choices=["uniform", "mixed"], default="uniform"
    )
    evaluate(**vars(parser.parse_args()))
