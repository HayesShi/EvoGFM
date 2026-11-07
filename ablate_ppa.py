"""Score both PPA HeaRT ablations in one exact shared-topology pass."""

import argparse
from pathlib import Path
import time
import torch
from tqdm import tqdm
from evogfm.data import save_json, seed_all
from evogfm.features import preprocess
from evogfm.ogb_data import load_ogb_links, heart_candidates
from evogfm.link_inference import LinkScorer, fit_large_links
from evogfm.readout import hard_negative_metrics
from evogfm.training import load_checkpoint


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    graph, splits = load_ogb_links("ogbl-ppa", args.data_root)
    raw, x = preprocess(graph)
    edges = graph["edges"]
    del graph
    model, head, _ = load_checkpoint(args.checkpoint)
    model.inverse.set_theta(torch.zeros_like(model.inverse.theta()))
    fitted, selections = {}, {}
    for variant in ("zero-inverse", "propagation"):
        cache = Path(args.calibration_dir or output) / f"{variant}_calibration.pt"
        if cache.exists():
            stored = torch.load(cache, weights_only=False)
            if stored["checkpoint"] != str(Path(args.checkpoint).resolve()):
                raise ValueError("Calibration checkpoint mismatch")
            fitted[variant], selections[variant] = stored["fitted"], stored["selection"]
        else:
            fitted[variant], selections[variant] = fit_large_links(
                model, head, raw, x, edges, variant
            )
            torch.save(
                dict(
                    fitted=fitted[variant],
                    selection=selections[variant],
                    checkpoint=str(Path(args.checkpoint).resolve()),
                ),
                cache,
            )
    save_json(
        output / "configuration.json",
        dict(
            seed=42,
            checkpoint=args.checkpoint,
            variants=list(fitted),
            selection=selections,
            protocol="released PPA HeaRT; all100000x500 per split",
            optimization="Only label-free base-channel computations shared between controls; separately fitted heads",
        ),
    )
    scorer = LinkScorer(model, head, raw, x, edges)
    rows = []
    for split in ("valid", "test") if args.split == "both" else (args.split,):
        begin = time.perf_counter()
        candidates, indices = heart_candidates(
            args.heart_archive, "ogbl-ppa", split, args.data_root
        )
        positives = splits[split]["edge"].T[:, indices]
        if candidates.shape != (100000, 500, 2) or positives.shape != (2, 100000):
            raise ValueError("Unexpected released PPA query shape")
        positive = scorer.score(positives, fitted, args.batch_size)
        sums = {name: {} for name in fitted}
        count = 0
        for start in tqdm(range(0, 100000, 16), desc=f"PPA joint ablations {split}"):
            stop = min(start + 16, 100000)
            pairs = torch.from_numpy(candidates[start:stop].reshape(-1, 2).T)
            negative = scorer.score(pairs, fitted, args.batch_size).reshape(
                stop - start, 500, 2
            )
            for column, variant in enumerate(fitted):
                metrics = hard_negative_metrics(
                    positive[start:stop, column], negative[:, :, column]
                )
                for key, value in metrics.items():
                    sums[variant][key] = (
                        sums[variant].get(key, 0.0) + (stop - start) * value
                    )
            count += stop - start
            if start % 1600 == 0:
                save_json(
                    output / "progress.json",
                    dict(
                        split=split,
                        done=count,
                        total=100000,
                        variants=list(fitted),
                        seconds=time.perf_counter() - begin,
                    ),
                )
        for variant in fitted:
            rows.append(
                dict(
                    variant=variant,
                    split=split,
                    positive_pairs=count,
                    negatives_per_positive=500,
                    **{key: value / count for key, value in sums[variant].items()},
                    shared_seconds=time.perf_counter() - begin,
                )
            )
        save_json(
            output / "results.json",
            dict(
                seed=42,
                selection=selections,
                rows=rows,
                full_reference="runs/v8/heart_ogbl-ppa/results.json",
            ),
        )
        print("RESULT", rows[-2:], flush=True)
        del candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--heart-archive", default="../work/heart_archive_v7")
    parser.add_argument("--output", default="runs/round345/ppa")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--split", choices=["valid", "test", "both"], default="both")
    parser.add_argument(
        "--calibration-dir", help="Reuse this checkpoint's separately fitted controls"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    evaluate(parser.parse_args())
