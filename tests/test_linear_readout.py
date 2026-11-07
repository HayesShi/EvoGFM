import torch
from evogfm.linear_readout import fit_linear, predict_linear, metric_score
from evogfm.readout import ridge


def test_fixed_head_matches_existing_ridge_primal_and_dual():
    generator = torch.Generator().manual_seed(4)
    for dimensions in (8, 60):
        h = torch.randn(70, dimensions, generator=generator)
        y = torch.arange(70) % 3
        train, val = torch.arange(35), torch.arange(35, 70)
        fitted, _ = fit_linear(h, y, train, alphas=(0.1,))
        expected, _ = ridge(h, y, train, val, alphas=(0.1,))
        torch.testing.assert_close(predict_linear(h, fitted), expected)
        original = fitted["weight"].clone()
        predict_linear(h + 1.0, fitted)
        torch.testing.assert_close(fitted["weight"], original)


def test_regression_head_and_selection_ignore_test_targets():
    h = torch.arange(40).float()[:, None]
    y = 3 * h[:, 0] + 7
    train, val = torch.arange(20), torch.arange(20, 30)
    fitted, candidates = fit_linear(h, y, train, val, metric="r2")
    changed = y.clone()
    changed[30:] += 1e6
    same, records = fit_linear(h, changed, train, val, metric="r2")
    assert candidates == records
    torch.testing.assert_close(fitted["weight"], same["weight"])
    assert (
        metric_score(predict_linear(h, fitted), y, torch.arange(30, 40), "r2") > 0.999
    )


def test_ap_uses_average_precision():
    prediction = torch.tensor([[0.9], [0.8], [0.7], [0.1]])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert abs(metric_score(prediction, y, torch.arange(4), "ap") - 5 / 6) < 1e-8
