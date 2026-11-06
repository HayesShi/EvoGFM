import torch

from evogfm.model import InverseLawGFM
from evogfm.online import OnlineNewton, TaskCoefficientBank


def test_resolved_inverse_is_affine_and_aligned_dense_equals_graph():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    n = 24
    pairs = torch.cartesian_prod(torch.arange(n), torch.arange(n)).T
    # Partial moment graph; complete attention graph for equality verification.
    edges = pairs[:, (pairs[0] < pairs[1]) & ((pairs[0] + pairs[1]) % 3 == 0)]
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=2,
        heads=2,
        rank=6,
        k=8,
        dropout=0.0,
        inverse_basis="conditional_hat",
        frame_aligned=True,
    )
    state = model.state(torch.randn(n, 8), edges)
    design = model.relation_design(state, pairs)
    assert torch.allclose(
        design @ model.inverse.theta(), model.relation_logits(state, pairs), atol=1e-5
    )
    q, _ = model.inverse.features(state)
    assert (q >= 0).all() and torch.allclose(q.sum(1), torch.ones(8), atol=1e-6)
    actual = model.embed_graph(state, pairs)
    expected = model.embed_block(state, torch.arange(n))
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    actual[:, 0].sum().backward()
    assert (
        model.inverse.c.grad is not None and torch.isfinite(model.inverse.c.grad).all()
    )
    assert model.layers[0].inputs[0].weight.grad is None


def test_aligned_kernel_preserves_rotated_coordinate_system():
    torch.manual_seed(42)
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=1,
        heads=2,
        rank=6,
        dropout=0.0,
        inverse_basis="conditional_hat",
        frame_aligned=True,
    )
    n = 16
    x = torch.randn(n, 8)
    edges = torch.combinations(torch.arange(n), r=2).T
    state = model.state(x, edges)
    left, s, right, _ = model.programmed(state)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    z = state.z
    rotated = z @ rotation
    before = (z @ left * s) @ (z @ right).T
    after = (rotated @ (rotation.T @ left) * s) @ (rotated @ (rotation.T @ right)).T
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)


def test_identity_encoder_reestimates_the_same_kernel_after_rotation():
    # Rotate the INPUT and independently re-estimate both whitening and moment.
    # V/FFN and its arbitrary value projection are not covered by this claim.
    torch.manual_seed(42)
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=1,
        heads=2,
        rank=8,
        k=8,
        dropout=0.0,
        inverse_basis="conditional_hat",
        frame_aligned=True,
        input_encoder="identity",
        whiten_floor=0.05,
        attention_normalization="frobenius",
    )
    x = torch.randn(60, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    edges = torch.combinations(torch.arange(60), r=2).T[:, ::4]
    matrices = []
    for features in [x, x @ rotation]:
        state = model.state(features, edges)
        left, s, right, _ = model.attention_program(state)
        matrices.append((state.z @ left * s) @ (state.z @ right).T)
        assert torch.allclose(s.square().sum(), torch.tensor(1.0), atol=2e-6)
        pairs = edges[:, :30]
        assert torch.allclose(
            model.relation_design(state, pairs) @ model.inverse.theta(),
            model.relation_logits(state, pairs),
            atol=1e-5,
        )
    assert torch.allclose(matrices[0], matrices[1], atol=5e-5, rtol=5e-5)
    model.eval()
    state = model.state(x, edges)
    complete = torch.cartesian_prod(torch.arange(60), torch.arange(60)).T
    assert torch.allclose(
        model.embed_graph(state, complete),
        model.embed_block(state, torch.arange(60)),
        atol=3e-5,
        rtol=3e-5,
    )


def test_supplied_task_curvature_is_accumulated():
    initial = torch.tensor([0.6, 0.4], dtype=torch.float64)
    gradient = torch.tensor([0.2, -0.1], dtype=torch.float64)
    curvature = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    ons = OnlineNewton(initial, [0.0, 0.0], [1.0, 1.0], eta=0.1, ridge=1.0)
    after = ons.step_gradient(gradient, curvature=curvature)
    expected = initial - 0.1 * torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) + curvature, gradient
    )
    assert torch.allclose(after, expected, atol=1e-10)
    assert torch.allclose(
        torch.from_numpy(ons.metric), torch.eye(2, dtype=torch.float64) + curvature
    )


def test_inverse_weighted_common_neighbors_match_dense_operator():
    from evogfm.features import contexts, task_features, attention
    from evogfm.features import pair_channels

    torch.manual_seed(42)
    raw = torch.randn(12, 8)
    edges = torch.combinations(torch.arange(12), r=2).T[:, ::3]
    pairs = torch.tensor([[0, 0, 3, 4, 7], [1, 5, 9, 8, 11]])
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=1,
        heads=2,
        rank=6,
        k=8,
        input_encoder="identity",
        inverse_basis="conditional_hat",
        frame_aligned=True,
        attention_normalization="frobenius",
        dropout=0.0,
    )
    state = model.state(raw, edges)
    ctx = contexts(edges, len(raw), pairs)
    base = pair_channels(raw, edges, pairs)
    features, _ = task_features(model, state, raw, edges, pairs, ctx, base)
    w = attention(model, state, ctx["visible"])
    dense = w.new_zeros(len(raw), len(raw))
    dense[ctx["visible"][0], ctx["visible"][1]] = w
    expected = (
        (dense @ dense.T)[pairs[0], pairs[1]]
        * ctx["degree"][pairs[0]]
        * ctx["degree"][pairs[1]]
    )
    assert torch.allclose(features[:, 14], expected.log1p(), atol=1e-6)
    uniform, _ = task_features(model, state, raw, edges, pairs, ctx, base, uniform=True)
    assert torch.allclose(uniform[:, 14], base[:, 6], atol=1e-6)
    features[:, 10:].square().sum().backward()
    assert model.inverse.c.grad is not None and model.inverse.c.grad.norm() > 1e-7


def test_offset_readout_adapts_against_an_inaccurate_source_prior():
    from evogfm.readout import fit_residual, predict_residual

    torch.manual_seed(42)
    features = torch.randn(200, 5)
    labels = (features[:, 0] - 0.3 * features[:, 1] > 0).float()
    offset = -features[:, 0]
    fitted = fit_residual(features, offset, labels, ridge=1.0)
    before = torch.nn.functional.binary_cross_entropy_with_logits(offset, labels)
    after = torch.nn.functional.binary_cross_entropy_with_logits(
        predict_residual(features, offset, fitted), labels.double()
    )
    assert after < before * 0.5


def test_task_bank_shares_inverse_weights_but_retains_task_coefficients():
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=1,
        heads=2,
        rank=6,
        k=8,
        inverse_basis="conditional_hat",
        frame_aligned=True,
        joint_task_backbone=True,
    )
    restored = InverseLawGFM(**model.config)
    restored.load_state_dict(model.state_dict())
    assert restored.config == model.config
    initial = model.inverse.theta().detach().clone()
    lower, upper = model.inverse.bounds()
    bank = TaskCoefficientBank(initial, lower, upper)
    network_before = model.inverse.net[0].weight.detach().clone()
    bank.activate(model.inverse, "lp")
    model.inverse.set_theta(initial * 1.1)
    bank.record(model.inverse, "lp")
    bank.activate(model.inverse, "nc")
    assert torch.allclose(model.inverse.theta(), initial)
    bank.activate(model.inverse, "lp")
    assert torch.allclose(model.inverse.theta(), initial * 1.1)
    assert torch.equal(model.inverse.net[0].weight, network_before)
    assert set(bank.state_dict()["tasks"]) == {"nc", "lp"}
    assert len(bank.optimizers) == 2  # No graph/domain key is created.
    loaded = TaskCoefficientBank(initial, lower, upper, eta=0.5)
    loaded.load_task("lp", bank.state_dict()["tasks"]["lp"])
    assert loaded.optimizers["lp"].eta == 1.0
    assert (loaded.optimizers["lp"].upper == bank.optimizers["lp"].upper).all()
