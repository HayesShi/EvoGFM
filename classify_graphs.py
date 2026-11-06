"""Frozen EvoGFM graph classification on TU datasets, nested 10-fold SVM."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm

from evogfm.data import load_tu, save_json, seed_all
from evogfm.features import preprocess, node_representations
from evogfm.training import load_checkpoint


class GraphScale(TransformerMixin, BaseEstimator):
    """Balance statistics and pooled encoder blocks using training-fold scales."""

    def __init__(self, split, weight=1.0):
        self.split, self.weight = split, weight

    def fit(self, x, y=None):
        self.scaler_ = StandardScaler().fit(x)
        return self

    def transform(self, x):
        z = self.scaler_.transform(x)
        return np.concatenate(
            (
                z[:, : self.split] / np.sqrt(self.split),
                self.weight * z[:, self.split :] / np.sqrt(z.shape[1] - self.split),
            ),
            axis=1,
        )


def fit_balanced(features, labels, split):
    """Inner CV with each fold's kernel computed once for all C values."""
    penalties = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    weights = [0.1, 1.0, 10.0]
    scores = np.zeros((len(weights), len(penalties)))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in inner.split(features, labels):
        for index, weight in enumerate(weights):
            scaler = GraphScale(split, weight).fit(features[train])
            x, v = scaler.transform(features[train]), scaler.transform(features[valid])
            variance = x.var()
            gamma = 1 / (x.shape[1] * variance) if variance else 1.0
            kernel = rbf_kernel(x, gamma=gamma)
            query = rbf_kernel(v, x, gamma=gamma)
            for column, penalty in enumerate(penalties):
                fitted = SVC(kernel="precomputed", C=penalty).fit(kernel, labels[train])
                scores[index, column] += fitted.score(query, labels[valid])
    best = np.unravel_index(scores.argmax(), scores.shape)
    parameters = dict(graphscale__weight=weights[best[0]], svc__C=penalties[best[1]])
    classifier = make_pipeline(GraphScale(split), SVC(kernel="rbf"))
    classifier.set_params(**parameters).fit(features, labels)
    return classifier, parameters


def graph_vectors(model, graphs, name="PROTEINS"):
    full, statistics = [], []
    with torch.no_grad():
        for graph in tqdm(graphs, desc=f"{name} graph encoding"):
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
    graphs, labels = load_tu(args.dataset, args.data_root)
    cache = output / "embeddings.pt"
    if cache.exists():
        stored = torch.load(cache, weights_only=False)
        if stored["checkpoint"] != str(Path(args.checkpoint).resolve()):
            raise ValueError(
                "Embedding cache belongs to a different checkpoint; use a new output folder"
            )
        if stored.get("dataset", "PROTEINS") != args.dataset:
            raise ValueError("Use a separate output folder for each dataset")
        full, basic = stored["full"], stored["statistics"]
    else:
        begin = time.perf_counter()
        full, basic = graph_vectors(model, graphs, args.dataset)
        torch.save(
            dict(
                full=full,
                statistics=basic,
                checkpoint=str(Path(args.checkpoint).resolve()),
                dataset=args.dataset,
                seconds=time.perf_counter() - begin,
            ),
            cache,
        )
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y = labels.numpy()
    result = dict(
        seed=42,
        dataset=args.dataset,
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
            balanced_readout=args.balanced_readout,
            block_weights=[0.1, 1.0, 10.0] if args.balanced_readout else [],
        ),
    )
    for variant, features in [("statistics", basic), ("evogfm", full)]:
        folds = []
        for fold, (train, test) in enumerate(
            tqdm(list(outer.split(features, y)), desc=f"{args.dataset} {variant} CV"), 1
        ):
            classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
            grid = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
            if variant == "evogfm" and args.balanced_readout:
                fitted, parameters = fit_balanced(
                    features[train], y[train], basic.shape[1]
                )
            else:
                fitted = GridSearchCV(
                    classifier,
                    grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    n_jobs=1,
                )
                fitted.fit(features[train], y[train])
                parameters = fitted.best_params_
            folds.append(
                dict(
                    fold=fold,
                    accuracy_percent=100 * float(fitted.score(features[test], y[test])),
                    parameters=parameters,
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
    parser.add_argument(
        "--dataset", choices=["PROTEINS", "NCI1", "MUTAG", "DD"], default="PROTEINS"
    )
    parser.add_argument("--balanced-readout", action="store_true")
    evaluate(parser.parse_args())
