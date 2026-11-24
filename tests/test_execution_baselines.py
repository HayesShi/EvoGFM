"""Exact operator controls and LRU eviction across graph-version changes."""

from collections import OrderedDict
import numpy as np
import pytest
import torch

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.execution_baselines import (
    DemandOnlyLinkScorer,
    FullGraphRefreshLinkScorer,
    SharedLRULinkScorer,
    _LRUTable,
)
from evogfm.shared_attention import SharedAttentionLinkScorer
from evogfm.shared_attention_fast import SharedAttentionFastLinkScorer
from evogfm.readout import new_head
from experiments.full_graph import FullGraphGFM


def test_lru_batch_operations_match_ordered_dictionary():
    cache = _LRUTable(4096 + 3 * 25, width=1)
    assert len(cache.keys) == 3
    reference = OrderedDict()
    requests = ([0, 1], [2, 0], [3, 4, 1, 5], [5, 3], [7, 8], [0, 7, 8])
    for keys in requests:
        keys = np.array(keys, dtype=np.int64)
        expected_hit = []
        for key in keys.tolist():
            expected_hit.append(key in reference)
            if key in reference:
                reference.move_to_end(key)
        hit, values = cache.get(keys)
        np.testing.assert_array_equal(hit, expected_hit)
        np.testing.assert_array_equal(values[:, 0], keys[hit] * 2)
        missing = keys[~hit]
        cache.put(missing, (missing * 2).astype(np.float32)[:, None])
        for key in missing.tolist():
            reference[key] = key * 2
            reference.move_to_end(key)
            while len(reference) > 3:
                reference.popitem(last=False)
        actual = cache.keys[np.argsort(cache.recency)].tolist()
        assert [k for k in actual if k >= 0] == list(reference)
        assert cache.bytes == 4096 + 3 * 25


@torch.no_grad()
@pytest.mark.parametrize("budget", [48 * 1024, 64 * 1024**2])
def test_five_execution_paths_match_full_neighbors_and_updates(budget):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = FullGraphGFM(
        input_dim=100,
        d=100,
        hidden=128,
        heads=4,
        rank=16,
        layers=2,
        frame_aligned=True,
        mode="coupled",
        dropout=0.1,
        inverse_basis="conditional_hat",
        attention_normalization="frobenius",
        joint_task_backbone=True,
    ).eval()
    raw = torch.randn(160, 100)
    edges = canonical_edges(torch.randint(0, 157, (2, 1300)), len(raw))
    head = new_head(True).eval()
    reference = DemandLinkScorer(model, head, raw, edges)
    implementations = [
        FullGraphRefreshLinkScorer,
        DemandOnlyLinkScorer,
        SharedAttentionLinkScorer,
        SharedLRULinkScorer,
        SharedAttentionFastLinkScorer,
    ]
    scorers = [
        cls(
            model,
            head,
            raw,
            edges,
            shared_budget_bytes=(0 if cls is SharedAttentionLinkScorer else budget),
            tile_edges=37,
        )
        for cls in implementations
    ]
    requests = [
        torch.tensor([[0, 3, 12, 158], [6, 18, 21, 159]]),
        torch.tensor([[1, 7, 32], [9, 19, 48]]),
        torch.stack((torch.arange(160), torch.arange(160).flip(0))),
    ]
    for version in range(3):
        for pairs in requests:
            expected = reference.features(pairs)
            for scorer in scorers:
                actual = scorer.features(pairs)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(
                    head(actual), head(expected), atol=1e-5, rtol=1e-4
                )
                stats = scorer.cache_stats()
                assert stats["peak_bytes"] <= stats["budget_bytes"]
                assert (
                    stats["resident_bytes"]
                    == stats["payload_bytes"] + stats["metadata_bytes"]
                )
        for scorer in scorers:
            previous = scorer.counters["shared_score_evaluations"]
            scorer.update(edges, edges[:, :0])
            scorer.features(requests[0])
            assert scorer.counters["shared_score_evaluations"] == previous
        if version < 2:
            removed = edges[:, :2]
            added = torch.tensor([[157 + version], [159]])
            edges = canonical_edges(torch.cat((edges[:, 2:], added), 1), len(raw))
            for scorer in [reference] + scorers:
                scorer.update(edges, added, removed)
    assert scorers[0].counters["moment_rebuilds"] == 3
    assert scorers[0].counters["materializations"] == 3
    assert (
        scorers[1].counters["shared_score_evaluations"]
        > scorers[2].counters["shared_score_evaluations"]
    )
    assert scorers[1].counters["backbone_rows"] == scorers[2].counters["backbone_rows"]
    assert (
        scorers[1].counters["backbone_edges"] == scorers[2].counters["backbone_edges"]
    )
    if budget > 48 * 1024:
        assert scorers[3].cache_stats()["dense_mode"]
        assert (
            scorers[3].counters["shared_score_evaluations"]
            == scorers[4].counters["shared_score_evaluations"]
        )
