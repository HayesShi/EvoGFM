"""Serializable ridge heads for fixed-head retention, AP and regression."""

import torch
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


def predict_linear(features, fitted):
    return torch.cat(
        [
            (block.double() - fitted["mean"]) @ fitted["weight"] + fitted["bias"]
            for block in features.split(1024)
        ]
    ).float()


def metric_score(prediction, labels, ids, metric, classes=None):
    values = prediction[ids]
    truth = labels[ids].numpy()
    if metric == "accuracy":
        return float((classes[values.argmax(1)] == labels[ids]).double().mean())
    if metric == "auc":
        return float(roc_auc_score(truth, (values[:, 1] - values[:, 0]).numpy()))
    if metric == "ap":
        return float(average_precision_score(truth, values.flatten().numpy()))
    if metric == "r2":
        return float(r2_score(truth, values.flatten().numpy()))
    raise ValueError(metric)


def fit_linear(
    features,
    labels,
    train,
    val=None,
    metric="accuracy",
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    """Fit with training labels; select regularization using validation only."""
    if metric in ("accuracy", "auc"):
        classes = labels[train].unique(sorted=True)
        target = (labels[train, None] == classes[None, :]).double()
    else:
        classes = None
        target = labels[train, None].double()
    bias = target.mean(0)
    scale = target.std(0, unbiased=False).clamp_min(1e-8) if metric == "r2" else 1.0
    target = (target - bias) / scale
    training = features[train].double()
    mean = training.mean(0)
    a = training - mean
    dual = len(train) < features.shape[1]
    gram = a @ a.T if dual else a.T @ a
    rhs = target if dual else a.T @ target
    values, vectors = torch.linalg.eigh(gram)
    transformed = vectors.T @ rhs
    candidates = []
    best = None
    if val is None and len(alphas) != 1:
        raise ValueError("A fixed head without validation needs one alpha")
    for alpha in alphas:
        weight = vectors @ (transformed / (values.clamp_min(0) + alpha)[:, None])
        if dual:
            weight = a.T @ weight
        fitted = dict(
            mean=mean,
            weight=weight * scale,
            bias=bias,
            classes=classes,
            alpha=alpha,
            metric=metric,
        )
        if val is None:
            return fitted, []
        score = (features[val].double() - mean) @ fitted["weight"] + bias
        quality = metric_score(
            score, labels[val], torch.arange(len(val)), metric, classes
        )
        candidates.append(dict(alpha=alpha, validation_score=quality))
        if best is None or quality > best[0]:
            best = quality, fitted
    return best[1], candidates
