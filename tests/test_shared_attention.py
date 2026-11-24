"""Full feature-path parity, strong head-fused baseline, and budget contracts."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from types import SimpleNamespace

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.readout import new_head
from evogfm.features import attention
from evogfm.shared_attention import SharedAttentionLinkScorer
from experiments.full_graph import FullGraphGFM


@torch.no_grad()
@pytest.mark.parametrize("budget", [0, 32768, 64 * 1024**2, 256 * 1024**2])
def test_full_model_changed_path_and_noop(budget):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = FullGraphGFM(
        input_dim=100,
        d=64,
        hidden=128,
        heads=4,
        rank=16,
        layers=3,
        frame_aligned=True,
        mode="coupled",
        dropout=0.1,
        inverse_basis="conditional_hat",
        attention_normalization="frobenius",
        joint_task_backbone=True,
    ).eval()
    raw = torch.randn(96, 100)
    head = new_head(True).eval()
    edges = canonical_edges(torch.randint(0, 90, (2, 380)), len(raw))
    ours = SharedAttentionLinkScorer(
        model, head, raw, edges, shared_budget_bytes=budget, tile_edges=37
    )
    reference = DemandLinkScorer(model, head, raw, edges)
    queries = [
        torch.tensor([[0, 2, 11, 20, 94], [7, 18, 3, 30, 95]]),
        torch.tensor([[8, 16, 42], [11, 38, 50]]),
        torch.stack((torch.arange(96), torch.arange(96).flip(0))),
    ]
    for version in range(3):
        for pairs in queries:
            expected, actual = reference.features(pairs), ours.features(pairs)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(
                head(actual), head(expected), atol=3e-5, rtol=3e-5
            )
            stats = ours.cache_stats()
            assert stats["resident_bytes"] <= budget
            assert stats["peak_bytes"] <= budget
            assert stats["invalid_bytes"] == 0
            assert (
                stats["resident_bytes"]
                == stats["metadata_bytes"] + stats["payload_bytes"]
            )
            assert stats["index_bytes"] == sum(
                stats[k]
                for k in (
                    "edge_index_bytes",
                    "projection_index_bytes",
                    "fixed_value_index_bytes",
                )
            )
            assert stats["index_bytes"] <= stats["metadata_bytes"]
        before = ours.counters["shared_score_evaluations"]
        values_before = ours.counters["backbone_value_projection_rows"]
        ours.update(edges, edges[:, :0])
        ours.features(queries[0])
        assert ours.counters["shared_score_evaluations"] == before
        assert ours.counters["backbone_value_projection_rows"] == values_before
        removed = edges[:, :2]
        added = torch.tensor([[90 + version], [93 + version]])
        edges = canonical_edges(torch.cat((edges[:, 2:], added), 1), len(raw))
        ours.update(edges, added, removed)
        reference.update(edges, added, removed)
    if budget:
        assert ours.counters["shared_score_hits"] > 0
        assert ours.counters["shared_value_hits"] > 0
    else:
        assert ours.counters["shared_score_hits"] == 0
        assert (
            ours.counters["shared_score_evaluations"]
            == ours.counters["shared_planned_score_consumptions"]
        )
    if budget == 64 * 1024**2:
        assert ours.counters["shared_score_recomputations"] == 0


@torch.no_grad()
def test_zero_and_signed_tiny_amplitudes_preserve_weighted_branch():
    torch.manual_seed(42)
    model = FullGraphGFM(
        input_dim=8,
        d=8,
        hidden=16,
        heads=2,
        rank=4,
        layers=2,
        frame_aligned=True,
        mode="coupled",
        dropout=0,
        inverse_basis="conditional_hat",
        attention_normalization="frobenius",
        joint_task_backbone=True,
    ).eval()
    raw = torch.randn(16, 8)
    edges = canonical_edges(torch.randint(0, 16, (2, 50)), len(raw))
    head = new_head(True).eval()
    ours = SharedAttentionLinkScorer(model, head, raw, edges, tile_edges=7)
    reference = DemandLinkScorer(model, head, raw, edges)
    left, _, right, bias = ours.program
    amplitude = torch.tensor([0.0, 1e-30, -1e-30, 0.5])
    ours.program = reference.program = (left, amplitude, right, bias)
    pairs = torch.tensor([[0, 3, 6], [1, 7, 9]])
    torch.testing.assert_close(
        ours.features(pairs), reference.features(pairs), atol=3e-5, rtol=3e-5
    )
    assert ours.counters["shared_score_correction_edges"] > 0


@torch.no_grad()
def test_hub_softmax_preserves_released_reduction_order():
    # A real-trace failure came from hub rows: an algebraically valid pairwise
    # denominator changes FP32 probabilities by 1.6e-4 on this dynamic range.
    count = 65536
    base = torch.linspace(0, -12, count)
    state = SimpleNamespace(z=torch.cat((torch.ones(1), base))[:, None])
    temperature_parameter = torch.tensor([np.log(np.expm1(0.95))], dtype=torch.float32)
    model = SimpleNamespace(
        layers=[SimpleNamespace(log_temperature=temperature_parameter)],
        attention_program=lambda _: (
            torch.ones(1, 1),
            torch.ones(1),
            torch.ones(1, 1),
            None,
        ),
    )
    visible = torch.stack(
        (torch.zeros(count, dtype=torch.long), torch.arange(1, count + 1))
    )
    expected = attention(model, state, visible)
    csr = sp.csr_matrix(
        (
            np.ones(count, dtype=np.float32),
            np.arange(1, count + 1),
            np.array([0, count]),
        ),
        shape=(1, count + 1),
    )
    temperature = float(
        torch.nn.functional.softplus(temperature_parameter).mean() + 0.05
    )
    actual = SharedAttentionLinkScorer._softmax(
        base.numpy(), temperature, csr, np.zeros(count, dtype=np.int64)
    )
    torch.testing.assert_close(
        torch.from_numpy(actual), expected, rtol=1e-7, atol=1e-12
    )
