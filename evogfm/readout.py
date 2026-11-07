"""Ridge node classification and a small offset-logistic link readout."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from .feature_views import join_features, quadratic_features, rbf_features


@dataclass
class RidgeFit:
    features: object
    mean: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    validation: torch.Tensor
    info: dict

    def predict(self):
        return torch.cat(
            [
                ((block.double() - self.mean) @ self.weight + self.bias).float()
                for block in self.features.split(1024)
            ]
        )


def fit_ridge(
    h,
    y,
    train,
    val,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    metric="accuracy",
    device="cpu",
):
    """Fit every alpha, selecting from validation rows before full-graph prediction."""
    # Keep wide graph representations in their original dtype. Only the
    # training system and one prediction block need double precision at once.
    classes = y[train].unique(sorted=True)
    if metric not in ("accuracy", "auc"):
        raise ValueError("metric must be accuracy or auc")
    if metric == "auc" and len(classes) != 2:
        raise ValueError("ROC-AUC readout selection requires two training classes")
    # Tiny systems stay on CPU; large systems use the selected device in
    # float64, including all five regularization candidates.
    device = device if min(len(train), h.shape[1]) > 512 else "cpu"
    target = (y[train, None] == classes[None, :]).double().to(device)
    bias = target.mean(0)
    target = target - bias
    dual = len(train) < h.shape[1]
    if not dual and len(train) * h.shape[1] > 8_000_000:
        mu = torch.zeros(h.shape[1], dtype=torch.float64, device=device)
        for ids in train.split(1024):
            mu += h[ids].double().to(device).sum(0) / len(train)
        gram = torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64, device=device)
        rhs = torch.zeros(h.shape[1], len(classes), dtype=torch.float64, device=device)
        offset = 0
        for ids in train.split(1024):
            block = h[ids].double().to(device) - mu
            gram += block.T @ block
            rhs += block.T @ target[offset : offset + len(ids)]
            offset += len(ids)
    else:
        training = h[train].double().to(device)
        mu = training.mean(0)
        a = training - mu
        gram = a @ a.T if dual else a.T @ a
        rhs = target if dual else a.T @ target
    values, vectors = torch.linalg.eigh(gram)
    rhs = vectors.T @ rhs
    weights = []
    for alpha in alphas:
        weight = vectors @ (rhs / (values.clamp_min(0) + alpha)[:, None])
        weights.append(a.T @ weight if dual else weight)
    joined = torch.cat(weights, dim=1)
    validation = (
        torch.cat(
            [(h[ids].double().to(device) - mu) @ joined for ids in val.split(1024)]
        ).reshape(len(val), len(alphas), len(classes))
        + bias
    ).cpu()
    quality = [
        (
            float(roc_auc_score(y[val], score[:, 1] - score[:, 0]))
            if metric == "auc"
            else float((classes[score.argmax(1)] == y[val]).double().mean())
        )
        for score in validation.unbind(1)
    ]
    chosen = max(range(len(alphas)), key=lambda index: quality[index])
    info = dict(
        validation_percent=100
        * float((classes[validation[:, chosen].argmax(1)] == y[val]).double().mean()),
        alpha=alphas[chosen],
        dimensions=h.shape[1],
    )
    if metric == "auc":
        info["validation_auc_percent"] = 100 * quality[chosen]
    return RidgeFit(
        h,
        mu.cpu(),
        weights[chosen].cpu(),
        bias.cpu(),
        validation[:, chosen].float(),
        info,
    )


def ridge(h, y, train, val, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), metric="accuracy"):
    """Read only train/validation labels. Return fitted scores, not test metrics."""
    fitted = fit_ridge(h, y, train, val, alphas, metric)
    return fitted.predict(), fitted.info


def _multiscale_fit(
    parts, values, y, train, val, metric="accuracy", kernel_device="cpu"
):
    """Select a small fixed family of graph filters on validation labels.

    The quadratic sketch approximates a degree-two polynomial kernel. Its
    random projections are fixed; only the final ridge coefficients are fit.
    """
    specs = [
        ("current", 2, 1.0, False),
        ("light_values", 2, 0.1, False),
        ("multiscale", 4, 0.5, False),
        ("quadratic", 4, 0.5, True),
    ]
    best = None
    records = []
    for name, hops, weight, quadratic in specs:
        features = join_features(parts[: hops + 1], values, weight)
        if quadratic:
            features = quadratic_features(features, 512, device=kernel_device)
        fitted = fit_ridge(features, y, train, val, metric=metric, device=kernel_device)
        info = fitted.info
        value = info[
            "validation_auc_percent" if metric == "auc" else "validation_percent"
        ]
        records.append(dict(variant=name, **info))
        if best is None or value > best[0]:
            best = (value, fitted, dict(variant=name, **info))
    return best[1], dict(**best[2], candidates=records)


def multiscale_ridge(
    parts, values, y, train, val, metric="accuracy", kernel_device="cpu"
):
    fitted, info = _multiscale_fit(parts, values, y, train, val, metric, kernel_device)
    return fitted.predict(), info


def diffuse_labels(scores, y, train, transition, alpha, steps=20):
    """Restart at model predictions and clamp only labeled training nodes."""
    import numpy as np

    if isinstance(transition, torch.Tensor):
        device = transition.device
        prior = (8 * scores).softmax(1).to(device)
        classes = y[train].unique(sorted=True)
        target = (y[train, None] == classes[None, :]).float().to(device)
        support = train.to(device)
        offsets = transition.crow_indices()
        isolated = offsets[1:] == offsets[:-1]
        probabilities = prior.clone()
        for _ in range(steps):
            probabilities = (1 - alpha) * prior + alpha * torch.sparse.mm(
                transition, probabilities
            )
            probabilities[isolated] = prior[isolated]
            probabilities[support] = target
        return probabilities.cpu()

    prior = (8 * scores).softmax(1).numpy()
    classes = y[train].unique(sorted=True)
    target = (y[train, None] == classes[None, :]).numpy().astype(np.float32)
    probabilities = prior.copy()
    isolated = np.asarray(transition.sum(1)).ravel() == 0
    for _ in range(steps):
        probabilities = (1 - alpha) * prior + alpha * (transition @ probabilities)
        probabilities[isolated] = prior[isolated]
        probabilities[train.numpy()] = target
    return torch.from_numpy(probabilities)


def correct_labels(scores, y, train, transition, alpha=0.8, steps=20):
    """Diffuse training prediction errors, then rescale using training errors only."""
    import numpy as np

    if isinstance(transition, torch.Tensor):
        device = transition.device
        prior = (8 * scores).softmax(1)
        classes = y[train].unique(sorted=True)
        residual = torch.zeros_like(prior)
        residual[train] = (y[train, None] == classes[None, :]).float() - prior[train]
        residual = residual.to(device)
        support = train.to(device)
        error = residual.clone()
        for _ in range(steps):
            error = (1 - alpha) * residual + alpha * torch.sparse.mm(transition, error)
        magnitude = np.abs(error[support].cpu().numpy()).mean()
        scale = np.abs(residual[support].cpu().numpy()).mean() / max(
            float(magnitude), 1e-8
        )
        corrected = (prior.to(device) + float(scale) * error).clamp_min(1e-8)
        corrected /= corrected.sum(1, keepdim=True)
        return corrected.cpu()

    prior = (8 * scores).softmax(1).numpy()
    classes = y[train].unique(sorted=True)
    residual = np.zeros_like(prior)
    residual[train.numpy()] = (y[train, None] == classes[None, :]).numpy() - prior[
        train.numpy()
    ]
    error = residual.copy()
    for _ in range(steps):
        error = (1 - alpha) * residual + alpha * (transition @ error)
    magnitude = np.abs(error[train.numpy()]).mean()
    scale = np.abs(residual[train.numpy()]).mean() / max(float(magnitude), 1e-8)
    corrected = np.maximum(prior + scale * error, 1e-8)
    corrected /= corrected.sum(1, keepdims=True)
    return torch.from_numpy(corrected)


def nystrom_features(features, gamma, landmarks=256):
    """Label-free Nyström features for an RBF kernel on normalized representations."""
    x = F.normalize(features, dim=1)
    ids = torch.randperm(len(x), generator=torch.Generator().manual_seed(42))[
        :landmarks
    ]
    centers = x[ids]
    cross = (-gamma * (2 - 2 * x @ centers.T).clamp_min(0)).exp()
    gram = (-gamma * (2 - 2 * centers @ centers.T).clamp_min(0)).exp().double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    basis = eigenvectors / eigenvalues.clamp_min(1e-6).sqrt()[None, :]
    return cross @ basis.float()


def enhanced_ridge(
    parts,
    values,
    y,
    train,
    val,
    transition,
    metric="accuracy",
    advanced=False,
    kernel_device="cpu",
):
    """Multiscale ridge, a larger quadratic sketch and train-label diffusion.

    A new component must improve validation by more than 0.1 percentage point;
    otherwise the smaller selected readout is retained.
    """
    classes = y[train].unique(sorted=True)

    def measure(scores):
        if metric == "auc":
            return 100 * float(roc_auc_score(y[val], scores[val, 1] - scores[val, 0]))
        return 100 * float((classes[scores[val].argmax(1)] == y[val]).double().mean())

    fitted, info = _multiscale_fit(parts, values, y, train, val, metric, kernel_device)

    def measure_validation(scores):
        if metric == "auc":
            return 100 * float(roc_auc_score(y[val], scores[:, 1] - scores[:, 0]))
        return 100 * float((classes[scores.argmax(1)] == y[val]).double().mean())

    best_value = measure_validation(fitted.validation)
    best_info = dict(variant=info["variant"], alpha=info["alpha"], diffusion=0.0)
    records = [dict(**best_info, validation_score_percent=best_value)]
    features = join_features(parts[:5], values, 0.5)
    expanded = quadratic_features(features, 2048, device=kernel_device)
    candidate = fit_ridge(expanded, y, train, val, metric=metric, device=kernel_device)
    score_value = measure_validation(candidate.validation)
    configuration = dict(
        variant="quadratic2048", alpha=candidate.info["alpha"], diffusion=0.0
    )
    records.append(dict(**configuration, validation_score_percent=score_value))
    if score_value > best_value + 0.1:
        fitted, best_value, best_info = candidate, score_value, configuration
    if advanced:
        for gamma in (1.0, 4.0):
            candidate = fit_ridge(
                rbf_features(features, gamma, device=kernel_device),
                y,
                train,
                val,
                metric=metric,
                device=kernel_device,
            )
            score_value = measure_validation(candidate.validation)
            configuration = dict(
                variant=f"rbf{gamma:g}", alpha=candidate.info["alpha"], diffusion=0.0
            )
            records.append(dict(**configuration, validation_score_percent=score_value))
            if score_value > best_value + 0.1:
                fitted, best_value, best_info = candidate, score_value, configuration
    scores = fitted.predict()
    if kernel_device != "cpu":
        transition = torch.sparse_csr_tensor(
            torch.from_numpy(transition.indptr).to(kernel_device),
            torch.from_numpy(transition.indices).to(kernel_device),
            torch.from_numpy(transition.data).to(kernel_device),
            size=transition.shape,
        )
    base = scores
    base_info = best_info.copy()
    for alpha in (0.5, 0.9):
        candidate = diffuse_labels(base, y, train, transition, alpha)
        score_value = measure(candidate)
        configuration = dict(base_info, diffusion=alpha)
        records.append(dict(**configuration, validation_score_percent=score_value))
        if score_value > best_value + 0.1:
            scores, best_value, best_info = candidate, score_value, configuration
    if advanced:
        corrected = correct_labels(base, y, train, transition)
        corrected_logits = corrected.clamp_min(1e-8).log() / 8
        for alpha in (0.0, 0.5, 0.9):
            candidate = (
                corrected
                if alpha == 0
                else diffuse_labels(corrected_logits, y, train, transition, alpha)
            )
            score_value = measure(candidate)
            configuration = dict(base_info, correction=True, diffusion=alpha)
            records.append(dict(**configuration, validation_score_percent=score_value))
            if score_value > best_value + 0.1:
                scores, best_value, best_info = candidate, score_value, configuration
    return scores, dict(
        **best_info,
        validation_percent=100
        * float((classes[scores[val].argmax(1)] == y[val]).double().mean()),
        **({"validation_auc_percent": best_value} if metric == "auc" else {}),
        candidates=records,
    )


def hits100(pos, neg):
    # Exact OGB evaluator convention: ties to the K-th negative are misses.
    if len(neg) < 100:
        return 1.0
    return float((pos > torch.topk(neg, 100).values[-1]).float().mean())


def hard_negative_metrics(positive, negative):
    """OGB/HeaRT ranks, averaging optimistic and pessimistic ties."""
    positive = positive.reshape(-1, 1)
    rank = (
        1
        + 0.5 * ((negative >= positive).sum(1) + (negative > positive).sum(1)).double()
    )
    return dict(
        mrr_percent=100 * float((1 / rank).mean()),
        **{
            f"hits{k}_percent": 100 * float((rank <= k).double().mean())
            for k in (10, 20, 50, 100)
        },
    )


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
