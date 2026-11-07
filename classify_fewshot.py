"""GP2F's published 90%-test graph episode, with frozen EvoGFM and SVM."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from classify_graphs import GraphScale, graph_vectors
from evogfm.data import load_tu, save_json, seed_all
from evogfm.training import load_checkpoint


def evaluate(args):
    seed_all(42)
    torch.set_num_threads(args.threads)
    graphs, labels = load_tu(args.dataset, args.data_root)
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(graphs), generator=generator)
    cut = int(0.9 * len(graphs))
    test, remaining = order[:cut], order[cut:]
    support = torch.cat(
        [remaining[labels[remaining] == value][:50] for value in labels.unique()]
    )
    support = support[torch.randperm(len(support), generator=generator)]
    output = Path(args.output)
    save_json(
        output / "episode.json",
        dict(
            seed=42,
            dataset=args.dataset,
            sampling="GP2F data.py graph_sample_and_save; random90% test then class cap50",
            support=support.tolist(),
            test=test.tolist(),
            support_per_class=torch.bincount(labels[support]).tolist(),
            penalties=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            weights=[0.1, 1, 10],
            fallback=dict(C=1.0, weight=1.0),
            selection="support-only stratified CV AUROC; <=5 folds",
        ),
    )
    rows = []
    y = labels.numpy()
    for variant in ["full", "zero-inverse", "propagation", "statistics"]:
        begin = time.perf_counter()
        model, _, _ = load_checkpoint(args.checkpoint)
        if variant == "zero-inverse":
            model.inverse.set_theta(torch.zeros_like(model.inverse.theta()))
        cache = Path("runs/mutag_v7/embeddings.pt")
        if (
            args.dataset == "MUTAG"
            and variant in ("full", "statistics")
            and cache.exists()
        ):
            stored = torch.load(cache, weights_only=False)
            if stored["checkpoint"] != str(Path(args.checkpoint).resolve()):
                raise ValueError("Wrong source checkpoint in graph embeddings")
            full, basic = stored["full"], stored["statistics"]
        else:
            full, basic = graph_vectors(model, graphs, args.dataset, variant)
        features = basic if variant == "statistics" else full
        scaler = (
            StandardScaler() if variant == "statistics" else GraphScale(basic.shape[1])
        )
        classifier = make_pipeline(scaler, SVC(kernel="rbf"))
        parameters = {"svc__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        if variant != "statistics":
            parameters["graphscale__weight"] = [0.1, 1, 10]
        folds = min(5, int(torch.bincount(labels[support]).min()))
        if len(np.unique(y[support])) != 2:
            raise ValueError("Episode does not contain both training classes")
        if folds >= 2:
            inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            fitted = GridSearchCV(
                classifier, parameters, scoring="roc_auc", cv=inner, n_jobs=1
            ).fit(features[support], y[support])
            selected = fitted.best_params_
        else:
            fitted = classifier.fit(features[support], y[support])
            selected = dict(C=1.0, weight=1.0, reason="too few support examples for CV")
        row = dict(
            variant=variant,
            metric="AUROC",
            test_percent=100
            * float(roc_auc_score(y[test], fitted.decision_function(features[test]))),
            support=len(support),
            test=len(test),
            inner_folds=folds,
            selected=selected,
            seconds=time.perf_counter() - begin,
        )
        rows.append(row)
        save_json(
            output / "results.json",
            dict(
                seed=42,
                dataset=args.dataset,
                protocol="GP2F sampling; one seed42 episode",
                checkpoint=args.checkpoint,
                rows=rows,
            ),
        )
        print("RESULT", row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=3)
    evaluate(parser.parse_args())
