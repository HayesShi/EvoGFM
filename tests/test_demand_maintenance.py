"""Prediction parity across demand cache hits and edge insertion/deletion."""

import pytest
import torch

from evogfm.data import canonical_edges
from evogfm.demand_maintenance import DemandLinkScorer
from evogfm.incremental import MaintainedLinkScorer
from evogfm.readout import new_head
from experiments.full_graph import FullGraphGFM


@torch.no_grad()
@pytest.mark.parametrize("policy", ["lazy", "hybrid", "closure"])
def test_complete_features_survive_global_change_and_cache_reuse(policy):
    torch.manual_seed(42)
    model = FullGraphGFM(
        input_dim=8,
        d=8,
        hidden=16,
        heads=2,
        rank=4,
        layers=2,
        frame_aligned=True,
        dropout=0,
        joint_task_backbone=True,
    ).eval()
    head = new_head(True).eval()
    raw = torch.rand(24, 8)
    edges = canonical_edges(
        torch.tensor(
            [
                [0, 0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 16],
                [1, 2, 3, 3, 4, 5, 6, 7, 8, 11, 13, 18],
            ]
        ),
        len(raw),
    )
    demand = DemandLinkScorer(
        model, head, raw, edges, policy=policy, materialize_fraction=0.25
    )
    # A sparse request, a cache hit, then a request covering all nodes.
    requests = [
        torch.tensor([[0, 2], [3, 23]]),
        torch.tensor([[0, 2], [3, 23]]),
        torch.stack((torch.arange(24), torch.arange(24).flip(0))),
    ]
    for version in range(3):
        full = MaintainedLinkScorer(model, head, raw, raw, edges)
        for pairs in requests:
            expected, actual = full.features(pairs), demand.features(pairs)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(
                head(actual), head(expected), rtol=2e-4, atol=2e-4
            )
        removed = edges[:, :1]
        added = torch.tensor([[14 + version], [20 + version]])
        edges = canonical_edges(torch.cat((edges[:, 1:], added), dim=1), len(raw))
        demand.update(edges, added, removed)
    assert demand.counters["materializations"] == (0 if policy == "lazy" else 3)
