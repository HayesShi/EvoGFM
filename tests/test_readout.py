import pytest
import torch
from sklearn.metrics import roc_auc_score

from evogfm.readout import ridge, hard_negative_metrics


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
