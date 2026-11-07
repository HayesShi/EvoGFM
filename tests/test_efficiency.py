"""Full candidate-search parity after delaying unselected full-graph predictions."""

import pytest
import torch
import scipy.sparse as sp

from evogfm.feature_views import FeatureView
from evogfm.readout import fit_ridge, enhanced_ridge


def test_fit_reads_only_support_and_validation_before_prediction():
    torch.manual_seed(12)
    x = torch.randn(101, 23)
    touched = []

    def get(ids):
        touched.extend(torch.arange(len(x))[ids].tolist())
        return x[ids]

    features = FeatureView(len(x), x.shape[1], get)
    train, val = torch.arange(19), torch.arange(19, 39)
    fitted = fit_ridge(features, torch.arange(101) % 3, train, val)
    assert set(touched) <= set(torch.cat((train, val)).tolist())
    assert fitted.predict().shape == (101, 3)


def test_advanced_selection_matches_eager_candidate_predictions(monkeypatch):
    import evogfm.readout as readout

    torch.manual_seed(42)
    parts = [torch.randn(71, 5) for _ in range(5)]
    values = torch.randn(71, 7)
    labels = torch.arange(71) % 3
    train, val = torch.arange(17), torch.arange(17, 40)
    transition = sp.eye(71, format="csr", dtype="float32")
    actual, selected = enhanced_ridge(
        parts, values, labels, train, val, transition, advanced=True
    )
    original_fit = readout.fit_ridge

    def eager(*args, **kwargs):
        fitted = original_fit(*args, **kwargs)
        fitted.validation = fitted.predict()[val]
        return fitted

    monkeypatch.setattr(readout, "fit_ridge", eager)
    expected, previous = enhanced_ridge(
        parts, values, labels, train, val, transition, advanced=True
    )
    assert selected == previous
    torch.testing.assert_close(actual, expected)


def test_sparse_tensor_diffusion_and_correction_match_scipy():
    from evogfm.readout import diffuse_labels, correct_labels

    torch.manual_seed(43)
    matrix = torch.rand(27, 27)
    matrix[matrix < 0.8] = 0
    matrix[0] = 0
    matrix /= matrix.sum(1, keepdim=True).clamp_min(1)
    scipy_matrix = sp.csr_matrix(matrix.numpy())
    tensor_matrix = matrix.to_sparse_csr()
    scores = torch.randn(27, 3)
    labels, train = torch.arange(27) % 3, torch.arange(9)
    for function in (diffuse_labels, correct_labels):
        expected = function(scores, labels, train, scipy_matrix, alpha=0.8)
        actual = function(scores, labels, train, tensor_matrix, alpha=0.8)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_offloaded_backbone_and_propagation_match_complete_cpu_graph():
    from evogfm.model import InverseLawGFM
    from evogfm.data import canonical_edges
    from evogfm.features import node_representations

    torch.manual_seed(91)
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=2,
        heads=2,
        rank=4,
        frame_aligned=True,
        input_encoder="identity",
    ).eval()
    x, raw = torch.randn(41, 8), torch.randn(41, 15)
    edges = canonical_edges(torch.randint(41, (2, 130)), 41)
    with torch.no_grad():
        state = model.state(x, edges)
        expected = node_representations(model, state, raw, edges)
        actual = node_representations(model, state, raw, edges, inference_device="cuda")
    for a, b in zip(expected[0] + [expected[1]], actual[0] + [actual[1]]):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)
