import pytest
import torch
from sklearn.metrics import roc_auc_score

from evogfm.readout import (
    ridge,
    hard_negative_metrics,
    diffuse_labels,
    correct_labels,
    nystrom_features,
)


def test_auc_selection_matches_direct_solve_and_ignores_test_labels():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(80, 8, generator=generator, dtype=torch.float64)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.5).long()
    train, val = torch.arange(40), torch.arange(40, 60)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    scores, info = ridge(x, y, train, val, alphas=alphas, metric="auc")
    centered = x - x[train].mean(0)
    target = torch.nn.functional.one_hot(y[train], 2).double()
    bias = target.mean(0)
    expected = []
    for alpha in alphas:
        weight = torch.linalg.solve(
            centered[train].T @ centered[train] + alpha * torch.eye(8),
            centered[train].T @ (target - bias),
        )
        prediction = centered @ weight + bias
        auc = roc_auc_score(y[val], prediction[val, 1] - prediction[val, 0])
        expected.append((auc, alpha, prediction))
    best = max(expected, key=lambda row: row[0])
    assert info["alpha"] == best[1]
    assert info["validation_auc_percent"] == pytest.approx(100 * best[0])
    assert torch.allclose(scores.double(), best[2], atol=1e-6)
    changed = y.clone()
    changed[60:] = 1 - changed[60:]
    repeated, repeated_info = ridge(x, changed, train, val, alphas=alphas, metric="auc")
    assert torch.equal(repeated, scores) and repeated_info == info


def test_heart_averages_tied_ranks():
    # No ties: rank 1; four ties: rank 3; twelve larger candidates: rank 13.
    positive = torch.tensor([2.0, 1.0, 0.0])
    negative = torch.tensor([[0.0] * 12, [1.0] * 4 + [0.0] * 8, [1.0] * 12])
    metrics = hard_negative_metrics(positive, negative)
    assert metrics["mrr_percent"] == pytest.approx(100 * (1 + 1 / 3 + 1 / 13) / 3)
    assert metrics["hits10_percent"] == pytest.approx(200 / 3)
    assert metrics["hits20_percent"] == 100


def test_diffusion_uses_only_train_labels_and_preserves_isolated_predictions():
    import scipy.sparse as sp

    transition = sp.csr_matrix(
        ([1.0, 1.0], ([0, 2], [2, 0])), shape=(4, 4), dtype="float32"
    )
    scores = torch.zeros(4, 2)
    train = torch.tensor([0, 1])
    y = torch.tensor([0, 1, 1, 0])
    actual = diffuse_labels(scores, y, train, transition, alpha=0.5)
    changed = y.clone()
    changed[2:] = 1 - changed[2:]
    assert torch.equal(
        actual, diffuse_labels(scores, changed, train, transition, alpha=0.5)
    )
    assert torch.allclose(actual[2], torch.tensor([0.75, 0.25]))
    assert torch.equal(actual[3], torch.tensor([0.5, 0.5]))


def test_error_correction_ignores_nontraining_labels_and_preserves_isolates():
    import scipy.sparse as sp

    transition = sp.csr_matrix(
        ([1.0, 1.0], ([0, 2], [2, 0])), shape=(4, 4), dtype="float32"
    )
    scores = torch.tensor([[0.0, 0.5], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])
    labels = torch.tensor([0, 1, 0, 0])
    train = torch.tensor([0, 1])
    first = correct_labels(scores, labels, train, transition)
    changed = labels.clone()
    changed[2:] = 1 - changed[2:]
    assert torch.equal(first, correct_labels(scores, changed, train, transition))
    assert torch.allclose(first.sum(1), torch.ones(4))
    assert torch.allclose(first[3], (8 * scores[3]).softmax(0))
    assert first[2, 0] > (8 * scores[2]).softmax(0)[0]


def test_nystrom_handles_repeated_landmarks_without_nonfinite_features():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    actual = nystrom_features(features, gamma=1.0, landmarks=3)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0], actual[1])
    expected = torch.exp(-torch.cdist(features, features).square())
    assert torch.allclose(actual @ actual.T, expected, atol=1e-4)
