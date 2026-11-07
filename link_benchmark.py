"""Full official OGB link prediction, optionally with released HeaRT candidates."""

import argparse
import gc
from pathlib import Path
import time

import torch
from tqdm import tqdm

from evogfm.data import seed_all, save_json, canonical_edges
from evogfm.ogb_data import load_ogb_links, heart_candidates
from evogfm.features import preprocess
from evogfm.link_inference import LinkScorer, fit_large_links
from evogfm.readout import hard_negative_metrics
from evogfm.training import load_checkpoint


def positive_pairs(part):
    return (
        torch.stack((part["source_node"], part["target_node"]))
        if "source_node" in part
        else part["edge"].T
    )


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    graph, splits = load_ogb_links(args.dataset, args.data_root)
    raw, x = preprocess(graph)
    edges = graph["edges"]
    del graph
    model, head, metadata = load_checkpoint(args.checkpoint)
    if args.ablation == "zero-inverse":
        model.inverse.set_theta(torch.zeros_like(model.inverse.theta()))
    cache = (
        Path(args.calibration_cache)
        if args.calibration_cache
        else output / "calibration.pt"
    )
    if cache.exists():
        stored = torch.load(cache, weights_only=False)
        if (
            stored["checkpoint"] != str(Path(args.checkpoint).resolve())
            or stored["dataset"] != args.dataset
            or stored.get("ablation", "full") != args.ablation
        ):
            raise ValueError("Calibration cache identity mismatch")
        fitted, selection = stored["fitted"], stored["selection"]
    else:
        fitted, selection = fit_large_links(
            model, head, raw, x, edges, ablation=args.ablation
        )
        torch.save(
            dict(
                fitted=fitted,
                selection=selection,
                dataset=args.dataset,
                checkpoint=str(Path(args.checkpoint).resolve()),
                ablation=args.ablation,
            ),
            cache,
        )
    save_json(
        output / "FROZEN.json",
        dict(
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            source_graphs=metadata["source_graphs"],
            seed=42,
            ablation=args.ablation,
            selection=selection,
            source_epoch=metadata["epoch"],
            protocol="official OGB; collab test includes validation edges",
            node_features=(
                "degree indicators capped at63"
                if args.dataset == "ogbl-ddi"
                else "original OGB attributes"
            ),
            calibration="four full masked training graphs; 4096 fit / 2048 holdout pairs; uniform nonedges",
        ),
    )
    rows = []
    scorer = None
    for split in ("valid", "test"):
        if split == "valid" or args.dataset == "ogbl-collab":
            if scorer is not None:
                del scorer
                gc.collect()
            visible = (
                edges
                if split == "valid"
                else canonical_edges(
                    torch.cat((edges, positive_pairs(splits["valid"])), dim=1), len(raw)
                )
            )
            print("Preparing complete inference graph", args.dataset, split, flush=True)
            scorer = LinkScorer(model, head, raw, x, visible, ablation=args.ablation)
        positives = positive_pairs(splits[split])
        if args.heart_archive:
            candidates, indices = heart_candidates(
                args.heart_archive, args.dataset, split, args.data_root
            )
            if indices is not None:
                positives = positives[:, indices]
            if (
                candidates.ndim != 3
                or candidates.shape[0] != positives.shape[1]
                or candidates.shape[-1] != 2
            ):
                raise ValueError("Released HeaRT candidate/positive shape mismatch")
            candidates = torch.from_numpy(candidates)
            protocol = (
                "official HeaRT; released PPA positive subset"
                if args.dataset == "ogbl-ppa"
                else "official HeaRT"
            )
        elif "target_node_neg" in splits[split]:
            candidates = splits[split]["target_node_neg"]
            protocol = "official OGB citation2 target candidates"
        else:
            candidates = splits[split]["edge_neg"].T
            protocol = "official OGB shared negative pool"
        print(
            "Scoring",
            args.dataset,
            split,
            "positive pairs",
            positives.shape[1],
            "negative shape",
            tuple(candidates.shape),
            flush=True,
        )
        positive = scorer.score(positives, fitted, args.batch_size)
        if args.heart_archive or "target_node_neg" in splits[split]:
            sums, count = {}, 0
            # Batches preserve each positive's complete released candidate list.
            for start in tqdm(
                range(0, positives.shape[1], 16), desc=f"{args.dataset} {split} ranks"
            ):
                stop = min(start + 16, positives.shape[1])
                if args.heart_archive:
                    pairs = candidates[start:stop].reshape(-1, 2).T
                    width = candidates.shape[1]
                else:
                    targets = candidates[start:stop]
                    width = targets.shape[1]
                    pairs = torch.stack(
                        (
                            positives[0, start:stop, None]
                            .expand_as(targets)
                            .reshape(-1),
                            targets.reshape(-1),
                        )
                    )
                negative = scorer.score(pairs, fitted, args.batch_size).reshape(
                    stop - start, width
                )
                metrics = hard_negative_metrics(positive[start:stop], negative)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0) + (stop - start) * value
                count += stop - start
                if start % 1600 == 0:
                    save_json(
                        output / f"progress_{split}.json",
                        dict(
                            done=count,
                            total=positives.shape[1],
                            seconds=time.perf_counter() - begin,
                        ),
                    )
            metrics = {key: value / count for key, value in sums.items()}
        else:
            negative = scorer.score(candidates, fitted, args.batch_size)
            k = {"ogbl-collab": 50, "ogbl-ppa": 100, "ogbl-ddi": 20}[args.dataset]
            threshold = negative.topk(k).values[-1]
            metrics = {
                f"hits{k}_percent": 100 * float((positive > threshold).double().mean())
            }
        row = dict(
            dataset=args.dataset,
            split=split,
            protocol=protocol,
            positive_pairs=positives.shape[1],
            **metrics,
            seconds=time.perf_counter() - begin,
        )
        rows.append(row)
        save_json(
            output / "results.json",
            dict(
                seed=42,
                checkpoint=args.checkpoint,
                ablation=args.ablation,
                selection=selection,
                rows=rows,
            ),
        )
        print("RESULT", row, flush=True)
        del candidates
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ogbl-collab", "ogbl-ppa", "ogbl-citation2", "ogbl-ddi"],
    )
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--heart-archive",
        help="Official .tar.gz, or directory of complete consecutive one-MiB range files",
    )
    parser.add_argument(
        "--calibration-cache",
        help="Reuse the same checkpoint's training-only calibration",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--ablation", choices=["full", "zero-inverse", "propagation"], default="full"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    evaluate(parser.parse_args())
