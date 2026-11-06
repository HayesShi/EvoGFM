"""Frozen EvoGFM graph classification on TU PROTEINS, nested 10-fold SVM."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from evogfm.data import load_proteins, save_json, seed_all
from evogfm.features import preprocess, node_representations
from evogfm.training import load_checkpoint


def graph_vectors(model, graphs):
    full, statistics = [], []
    with torch.no_grad():
        for graph in tqdm(graphs, desc="PROTEINS graph encoding"):
            raw, x = preprocess(graph)
            edges = graph["edges"]
            state = model.state(x, edges)
            parts, h = node_representations(model, state, raw, edges, hops=4)
            nodes = torch.cat(parts + [h], dim=1)
            pooled = torch.cat(
                (nodes.mean(0), nodes.std(0, unbiased=False), nodes.max(0).values)
            )
            degree = torch.bincount(edges.flatten(), minlength=len(raw)).float()
            histogram = torch.histogram(
                degree,
                bins=torch.tensor(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0, float("inf")]
                ),
            )[0] / len(raw)
            basic = torch.cat(
                (
                    raw.mean(0),
                    raw.std(0, unbiased=False),
                    raw.max(0).values,
                    torch.tensor(
                        [
                            np.log1p(len(raw)),
                            np.log1p(edges.shape[1]),
                            degree.mean(),
                            degree.std(unbiased=False),
                            degree.max(),
                        ]
                    ),
                    histogram,
                )
            )
            full.append(torch.cat((basic, pooled)))
            statistics.append(basic)
    return torch.stack(full).numpy(), torch.stack(statistics).numpy()


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, _, checkpoint = load_checkpoint(args.checkpoint)
    graphs, labels = load_proteins(args.data_root)
    cache = output / "embeddings.pt"
    if cache.exists():
        stored = torch.load(cache, weights_only=False)
        if stored["checkpoint"] != str(Path(args.checkpoint).resolve()):
            raise ValueError(
                "Embedding cache belongs to a different checkpoint; use a new output folder"
            )
        full, basic = stored["full"], stored["statistics"]
    else:
        begin = time.perf_counter()
        full, basic = graph_vectors(model, graphs)
        torch.save(
            dict(
                full=full,
                statistics=basic,
                checkpoint=str(Path(args.checkpoint).resolve()),
                seconds=time.perf_counter() - begin,
            ),
            cache,
        )
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y = labels.numpy()
    result = dict(
        seed=42,
        dataset="PROTEINS",
        graphs=len(graphs),
        source_epoch=checkpoint["epoch"],
        node_features="one-hot node labels; no node attributes",
        protocol="nested stratified 10-fold; inner 5-fold C selection; fold standard deviation",
        rows=[],
    )
    save_json(
        output / "configuration.json",
        dict(
            outer_folds=10,
            inner_folds=5,
            C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            seed=42,
        ),
    )
    for variant, features in [("statistics", basic), ("evogfm", full)]:
        folds = []
        for fold, (train, test) in enumerate(
            tqdm(list(outer.split(features, y)), desc=f"PROTEINS {variant} CV"), 1
        ):
            classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
            fitted = GridSearchCV(
                classifier,
                {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=1,
            )
            fitted.fit(features[train], y[train])
            folds.append(
                dict(
                    fold=fold,
                    accuracy_percent=100 * float(fitted.score(features[test], y[test])),
                    C=fitted.best_params_["svc__C"],
                )
            )
            print(variant, fold, folds[-1], flush=True)
        values = [r["accuracy_percent"] for r in folds]
        result["rows"].append(
            dict(
                variant=variant,
                accuracy_percent=float(np.mean(values)),
                fold_std_percent=float(np.std(values)),
                folds=folds,
            )
        )
        save_json(output / "results.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/proteins")
    parser.add_argument("--threads", type=int, default=4)
    evaluate(parser.parse_args())
