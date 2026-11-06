"""Ridge node classification and a small offset-logistic link readout."""

import torch
import torch.nn.functional as F


def ridge(h, y, train, val, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Read only train/validation labels. Return fitted scores, not test metrics."""
    h = h.double()
    mu = h[train].mean(0)
    centered = h - mu
    a = centered[train]
    classes = y[train].unique(sorted=True)
    target = (y[train, None] == classes[None, :]).double()
    bias = target.mean(0)
    target = target - bias
    dual = len(train) < h.shape[1]
    gram = a @ a.T if dual else a.T @ a
    rhs = target if dual else a.T @ target
    values, vectors = torch.linalg.eigh(gram)
    rhs = vectors.T @ rhs
    best = None
    for alpha in alphas:
        weight = vectors @ (rhs / (values.clamp_min(0) + alpha)[:, None])
        if dual:
            weight = a.T @ weight
        score = centered @ weight + bias
        acc = float((classes[score[val].argmax(1)] == y[val]).double().mean())
        if best is None or acc > best[0]:
            best = (acc, alpha, score)
    return best[2].float(), dict(
        validation_percent=100 * best[0], alpha=best[1], dimensions=h.shape[1]
    )


def hits100(pos, neg):
    # Exact OGB evaluator convention: ties to the K-th negative are misses.
    if len(neg) < 100:
        return 1.0
    return float((pos > torch.topk(neg, 100).values[-1]).float().mean())


def sampled_negatives(n, train_edges, count, generator):
    # Exclude training positives only, matching information used by NCN sampler.
    forbidden = set((train_edges[0] * n + train_edges[1]).tolist())
    chosen = set()
    while len(chosen) < count:
        draw = torch.randint(n, (2, count * 2), generator=generator)
        for a, b in draw.T.tolist():
            a, b = min(a, b), max(a, b)
            if a != b and a * n + b not in forbidden:
                chosen.add(a * n + b)
            if len(chosen) == count:
                break
    keys = torch.tensor(sorted(chosen))
    return torch.stack((keys // n, keys % n))


def node_masks(y):
    g = torch.Generator().manual_seed(42)
    train = []
    val = []
    for label in y.unique():
        ids = torch.where(y == label)[0]
        ids = ids[torch.randperm(len(ids), generator=g)]
        cut = int(0.6 * len(ids))
        train.append(ids[:cut])
        val.append(ids[cut : int(0.8 * len(ids))])
    return torch.cat(train), torch.cat(val)


def episode(y, pool, g, shots=5, queries=20):
    support = []
    query = []
    for label in y[pool].unique():
        ids = pool[y[pool] == label]
        ids = ids[torch.randperm(len(ids), generator=g)]
        support.append(ids[:shots])
        query.append(ids[shots : shots + queries])
    return torch.cat(support), torch.cat(query)


def episode_loss(h, y, support, query, alpha=0.1):
    classes = y[support].unique(sorted=True)
    target = (y[support, None] == classes[None, :]).float()
    a = h[support]
    b = h[query]
    mean = a.mean(0)
    a = a - mean
    b = b - mean
    bias = target.mean(0)
    target = target - bias
    solution = torch.linalg.solve(a @ a.T + alpha * torch.eye(len(a)), target)
    score = 8 * (b @ a.T @ solution + bias)
    truth = (y[query, None] == classes[None, :]).long().argmax(1)
    return F.cross_entropy(score, truth)


def new_head(backbone=False):
    return torch.nn.Sequential(
        torch.nn.Linear(17 if backbone else 16, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )


def fit_residual(features, offset, labels, ridge=1.0, steps=8):
    """An offset-logistic linear readout; graph encoder/inverse remain frozen."""
    features = features.double()
    offset = offset.double()
    labels = labels.double()
    mean = features.mean(0)
    scale = features.std(0).clamp_min(0.05)
    x = (features - mean) / scale
    x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1)
    coefficient = torch.zeros(x.shape[1], dtype=x.dtype)
    penalty = torch.eye(x.shape[1], dtype=x.dtype) * ridge
    penalty[-1, -1] = ridge * 0.1
    for step in range(steps):
        p = (offset + x @ coefficient).sigmoid()
        gradient = x.T @ (p - labels) + penalty @ coefficient
        metric = x.T @ ((p * (1 - p))[:, None] * x) + penalty
        update = torch.linalg.solve(metric, gradient)
        # Damped Newton with the calibration objective only.
        previous = float(
            F.binary_cross_entropy_with_logits(
                offset + x @ coefficient, labels, reduction="sum"
            )
            + 0.5 * coefficient @ penalty @ coefficient
        )
        for rate in [1.0, 0.5, 0.25, 0.1, 0.0]:
            candidate = coefficient - rate * update
            value = float(
                F.binary_cross_entropy_with_logits(
                    offset + x @ candidate, labels, reduction="sum"
                )
                + 0.5 * candidate @ penalty @ candidate
            )
            if value <= previous:
                coefficient = candidate
                break
    return coefficient, mean, scale


def predict_residual(features, offset, state):
    coefficient, mean, scale = state
    x = (features.double() - mean) / scale
    return offset.double() + x @ coefficient[:-1] + coefficient[-1]


def nc_logits(parts, y, support, query):
    h = torch.cat(parts, dim=1)
    classes = y[support].unique(sorted=True)
    target = (y[support, None] == classes[None, :]).float()
    a = h[support]
    b = h[query]
    mean = a.mean(0)
    a = a - mean
    b = b - mean
    bias = target.mean(0)
    target = target - bias
    if len(a) < a.shape[1]:
        solution = torch.linalg.solve(a @ a.T + 0.1 * torch.eye(len(a)), target)
        logits = 8 * (b @ a.T @ solution + bias)
    else:
        solution = torch.linalg.solve(
            a.T @ a + 0.1 * torch.eye(a.shape[1]), a.T @ target
        )
        logits = 8 * (b @ solution + bias)
    truth = (y[query, None] == classes[None, :]).long().argmax(1)
    return logits, truth
