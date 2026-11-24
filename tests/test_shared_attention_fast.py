"""Small full-path checks for dense admission, fallback, and version lifetime."""

import pytest
import torch

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.shared_attention import SharedAttentionLinkScorer
from evogfm.shared_attention_fast import SharedAttentionFastLinkScorer
from evogfm.readout import new_head
from experiments.full_graph import FullGraphGFM


def fixture(full=True):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = FullGraphGFM(
        input_dim=100 if full else 8,
        d=64 if full else 8,
        hidden=128 if full else 16,
        heads=4 if full else 2,
        rank=16 if full else 4,
        layers=2,
        frame_aligned=True,
        mode="coupled",
        dropout=0.1,
        inverse_basis="conditional_hat",
        attention_normalization="frobenius",
        joint_task_backbone=True,
    ).eval()
    raw = torch.randn(96 if full else 64, 100 if full else 8)
    edges = (
        canonical_edges(torch.randint(0, len(raw) - 3, (2, 380)), len(raw))
        if full
        else torch.stack((torch.arange(63), torch.arange(1, 64)))
    )
    return model, new_head(True).eval(), raw, edges


@torch.no_grad()
@pytest.mark.parametrize("budget", [0, 32768, 64 * 1024**2, 256 * 1024**2])
def test_full_predictor_dense_and_fallback_match(budget):
    model, head, raw, edges = fixture()
    scorer = SharedAttentionFastLinkScorer(
        model, head, raw, edges, shared_budget_bytes=budget, tile_edges=37
    )
    reference = DemandLinkScorer(model, head, raw, edges)
    v3 = SharedAttentionLinkScorer(
        model, head, raw, edges, shared_budget_bytes=budget, tile_edges=37
    )
    requests = [
        torch.tensor([[0, 3, 16, 94], [1, 7, 19, 95]]),
        torch.tensor([[2, 8, 20], [6, 11, 25]]),
        torch.stack((torch.arange(96), torch.arange(96).flip(0))),
    ]
    for version in range(3):
        for pairs in requests:
            actual, expected = scorer.features(pairs), reference.features(pairs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(actual, v3.features(pairs), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(
                head(actual), head(expected), rtol=1e-4, atol=1e-5
            )
            stats = scorer.cache_stats()
            assert stats["resident_bytes"] <= budget and stats["peak_bytes"] <= budget
            assert (
                stats["payload_bytes"] + stats["metadata_bytes"]
                == stats["resident_bytes"]
            )
            if stats["dense_mode"]:
                assert stats["resident_bytes"] == stats["dense_required_bytes"]
                assert stats["projection_bytes"] == stats["index_bytes"] == 0
        removed = edges[:, :2]
        added = torch.tensor([[93 + version], [version]])
        edges = canonical_edges(torch.cat((edges[:, 2:], added), 1), len(raw))
        for implementation in (scorer, reference, v3):
            implementation.update(edges, added, removed)
    if budget >= 64 * 1024**2:
        assert scorer.counters["shared_planning_s"] == 0
        assert scorer.counters["shared_score_recomputations"] == 0
    if budget == 0:
        assert (
            scorer.counters["shared_score_evaluations"]
            == v3.counters["shared_score_evaluations"]
        )


@torch.no_grad()
def test_graph_growth_releases_dense_before_fallback_and_can_return():
    model, head, raw, initial = fixture(full=False)
    scorer = SharedAttentionFastLinkScorer(
        model, head, raw, initial, shared_budget_bytes=16000, tile_edges=19
    )
    assert scorer.cache_stats()["dense_mode"]
    pairs = torch.tensor([[0, 3, 8], [4, 7, 12]])
    dense_edges = torch.triu_indices(len(raw), len(raw), offset=1)
    initial_keys = initial[0] * len(raw) + initial[1]
    added = dense_edges[
        :, ~torch.isin(dense_edges[0] * len(raw) + dense_edges[1], initial_keys)
    ]
    scorer.features(pairs)
    scorer.update(dense_edges, added)
    assert not scorer.cache_stats()["dense_mode"]
    reference = DemandLinkScorer(model, head, raw, dense_edges)
    torch.testing.assert_close(
        scorer.features(pairs), reference.features(pairs), atol=1e-5, rtol=1e-4
    )
    scorer.update(initial, initial[:, :0], added)
    assert scorer.cache_stats()["dense_mode"]
    reference = DemandLinkScorer(model, head, raw, initial)
    torch.testing.assert_close(
        scorer.features(pairs), reference.features(pairs), atol=1e-5, rtol=1e-4
    )
    assert scorer.cache_stats()["peak_bytes"] <= 16000


@torch.no_grad()
def test_noop_keeps_dense_scores_and_topology_update_keeps_fixed_values():
    model, head, raw, edges = fixture(full=False)
    scorer = SharedAttentionFastLinkScorer(model, head, raw, edges)
    pairs = torch.tensor([[0, 3, 8], [4, 7, 12]])
    expected = scorer.features(pairs)
    score_cache, value_cache = scorer._edge_cache, scorer._value_cache
    evaluations = scorer.counters["shared_score_evaluations"]
    scorer.update(edges, edges[:, :0])
    torch.testing.assert_close(scorer.features(pairs), expected)
    assert scorer._edge_cache is score_cache
    assert scorer._value_cache is value_cache
    assert scorer.counters["shared_score_evaluations"] == evaluations
    added = torch.tensor([[0], [63]])
    updated = canonical_edges(torch.cat((edges, added), 1), len(raw))
    scorer.update(updated, added)
    assert scorer._value_cache is value_cache
    assert scorer._edge_cache is not score_cache
    assert not scorer._edge_cache.valid.any()
    before = scorer.counters["shared_value_hits"]
    scorer.features(pairs)
    assert scorer.counters["shared_value_hits"] > before
