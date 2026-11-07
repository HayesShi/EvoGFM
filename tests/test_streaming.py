"""Numerical equivalence of bounded-memory inference and full tensor formulas."""

import torch

from evogfm.model import ProgrammedLayer, InverseLawGFM
from evogfm.features import weighted_propagate, covariance_svd_basis
from evogfm.feature_views import FeatureView, quadratic_features, rbf_features
from evogfm.readout import ridge


def test_large_edge_quadrature_stays_in_bounds():
    model = InverseLawGFM(d=8, rank=4, hidden=16, heads=2, layers=1)
    # A strided view exercises 61M edge IDs without allocating a large graph.
    edges = torch.tensor([[0], [1]]).expand(2, 61_859_012)
    state = model.state(torch.randn(7, 100), edges)
    assert torch.isfinite(state.spectrum).all()


def test_three_hop_repeated_endpoints():
    import numpy as np
    from evogfm.features import adjacency
    from evogfm.data import canonical_edges
    from evogfm.link_inference import three_hop_entries

    torch.manual_seed(42)
    matrix = adjacency(canonical_edges(torch.randint(47, (2, 199)), 47), 47)
    left_ids = np.repeat([2, 5], 31)
    right_ids = np.arange(len(left_ids)) % 47
    for a, b in (
        (left_ids, right_ids),
        (right_ids, left_ids),
        (np.arange(40), np.arange(40) % 39),
    ):
        left, right = matrix[a], matrix[b]
        expected = np.asarray((left @ matrix).multiply(right).sum(1)).ravel()
        np.testing.assert_array_equal(
            three_hop_entries(matrix, a, b, left, right), expected
        )


def test_shared_link_content_matches_dense_propagation():
    import torch.nn.functional as functional
    from evogfm.data import canonical_edges
    from evogfm.features import adjacency, pair_context

    torch.manual_seed(42)
    x = torch.randn(47, 11)
    edges = canonical_edges(torch.randint(47, (2, 199)), 47)
    normalized = functional.normalize(x, dim=1)
    _, _, raw, p1, p2 = pair_context(x, edges, normalized=normalized)
    matrix = torch.from_numpy(adjacency(edges, 47).toarray()) + torch.eye(47)
    inv = matrix.sum(1).rsqrt()
    matrix = inv[:, None] * matrix * inv[None, :]
    expected1 = functional.normalize((matrix @ x.double()).float(), dim=1)
    expected2 = functional.normalize((matrix @ expected1.double()).float(), dim=1)
    assert raw.data_ptr() == normalized.data_ptr()
    torch.testing.assert_close(p1, expected1)
    torch.testing.assert_close(p2, expected2)


def test_chunked_graph_attention():
    torch.manual_seed(42)
    layer = ProgrammedLayer(16, 8, 2, 4, frame_aligned=True).eval()
    h, z = torch.randn(31, 16), torch.randn(31, 8)
    left = torch.linalg.qr(torch.randn(8, 4)).Q
    s = torch.randn(4)
    edges = torch.randint(31, (2, 173))
    with torch.no_grad():
        full = layer.forward_graph(h, left, s, left, edges, z)
        streamed = layer.forward_graph_chunked(h, left, s, left, edges, z, batch_size=7)
    torch.testing.assert_close(full, streamed, rtol=1e-5, atol=1e-6)


def test_sparse_aggregation():
    torch.manual_seed(42)
    values = torch.randn(37, 512)
    edges = torch.randint(37, (2, 16000))
    weights = torch.rand(edges.shape[1]) / 1000
    full = torch.zeros_like(values).index_add_(
        0, edges[0], weights[:, None] * values[edges[1]]
    )
    with torch.no_grad():
        streamed = weighted_propagate(values, edges, weights)
    torch.testing.assert_close(full, streamed, rtol=2e-5, atol=1e-6)


def test_full_gram_svd():
    torch.manual_seed(42)
    x = torch.randn(401, 37)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    basis = covariance_svd_basis(x, 12)
    torch.testing.assert_close(
        basis @ basis.T, vh[:12].T @ vh[:12], rtol=2e-4, atol=2e-5
    )


def test_lazy_kernel_and_ridge():
    torch.manual_seed(42)
    dense = torch.randn(123, 17)
    lazy = FeatureView(123, 17, lambda ids: dense[ids])
    y = torch.arange(123) % 3
    for transform in (
        lambda x: quadratic_features(x, 31),
        lambda x: rbf_features(x, 1.0, 19),
    ):
        a, b = transform(dense), transform(lazy)
        torch.testing.assert_close(
            a, torch.cat(list(b.split(23))), rtol=2e-4, atol=2e-5
        )
        score_a, info_a = ridge(a, y, torch.arange(29), torch.arange(29, 65))
        score_b, info_b = ridge(b, y, torch.arange(29), torch.arange(29, 65))
        assert info_a["alpha"] == info_b["alpha"]
        torch.testing.assert_close(score_a, score_b, rtol=2e-4, atol=2e-5)


def test_prepared_link_channels_and_head_streaming():
    from evogfm.data import canonical_edges
    from evogfm.features import (
        contexts,
        pair_channels,
        task_features,
        node_representations,
        label_transition,
    )
    from evogfm.link_inference import LinkScorer

    torch.manual_seed(42)
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=2,
        heads=2,
        rank=4,
        frame_aligned=True,
        input_encoder="identity",
        joint_task_backbone=True,
    ).eval()
    head = torch.nn.Linear(17, 1).eval()
    raw = torch.rand(8, 31).T
    edges = canonical_edges(torch.randint(31, (2, 89)), 31)
    pairs = torch.randint(31, (2, 17))
    with torch.no_grad():
        scorer = LinkScorer(model, head, raw, raw, edges)
        assert all(part.is_contiguous() for part in scorer.parts)
        state = model.state(raw, edges)
        _, _, transition = node_representations(
            model, state, raw, edges, hops=2, return_transition=True
        )
        expected_transition = label_transition(model, state, edges)
        torch.testing.assert_close(
            torch.from_numpy(transition.toarray()),
            torch.from_numpy(expected_transition.toarray()),
        )
        full, _ = task_features(
            model,
            state,
            raw,
            edges,
            pairs,
            contexts(edges, 31, pairs),
            pair_channels(raw, edges, pairs),
        )
        torch.testing.assert_close(scorer.features(pairs), full, rtol=2e-4, atol=3e-5)
        from evogfm.readout import fit_residual

        features = scorer.features(pairs)
        fitted = fit_residual(
            features, head(features).squeeze(1), torch.arange(len(features)) % 2
        )
        torch.testing.assert_close(
            scorer.score(pairs, fitted, batch_size=5),
            scorer.score(pairs, fitted, batch_size=5, group_queries=False),
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            scorer.score(pairs.flip(0), fitted, batch_size=17, group_queries=False),
            scorer.score(pairs, fitted, batch_size=5, group_queries=False),
            rtol=2e-5,
            atol=2e-6,
        )
        layer = model.layers[0]
        left, s, right, _ = model.attention_program(state)
        h = model.up(state.z)
        visible = contexts(edges, 31, pairs)["visible"]
        expected = layer.forward_graph(h, left, s, right, visible, state.z)
        actual = layer.forward_graph_by_head(
            h, left, s, right, visible, state.z, batch_size=7
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_official_candidate_tar_stream(tmp_path):
    import io
    import tarfile
    import numpy as np
    from evogfm.ogb_data import heart_candidates

    candidates = np.arange(24).reshape(3, 4, 2)
    buffer = io.BytesIO()
    np.save(buffer, candidates)
    path = tmp_path / "heart.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("dataset/ogbl-ddi/heart_valid_samples.npy")
        member.size = buffer.tell()
        buffer.seek(0)
        archive.addfile(member, buffer)
    values, indices = heart_candidates(path, "ogbl-ddi", "valid")
    np.testing.assert_array_equal(values, candidates)
    assert indices is None


def test_streamed_primal_statistics():
    torch.manual_seed(42)
    x = torch.randn(20003, 401)
    y = torch.arange(len(x)) % 3
    train, val = torch.arange(20000), torch.arange(20000, 20003)
    score, _ = ridge(x, y, train, val, alphas=(0.1,))
    a = x[train].double()
    mu = a.mean(0)
    a -= mu
    target = torch.nn.functional.one_hot(y[train], 3).double()
    bias = target.mean(0)
    weight = torch.linalg.solve(
        a.T @ a + 0.1 * torch.eye(x.shape[1]), a.T @ (target - bias)
    )
    expected = ((x.double() - mu) @ weight + bias).float()
    torch.testing.assert_close(score, expected, rtol=2e-5, atol=1e-6)
