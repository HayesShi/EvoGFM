import warnings

import torch

from evogfm.data import canonical_edges, ncn_split
from evogfm.evaluation import fit_link_readout
from evogfm.model import InverseLawGFM
from evogfm.readout import new_head


def test_public_split_preserves_planetoid_order_and_disjoint_positives():
    n = 40
    edges = torch.combinations(torch.arange(n), r=2).T[:, ::5]
    graph = dict(x=torch.randn(n, 8), edges=edges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        split = ncn_split(graph)
    # Planetoid upper edges are destination-sorted before the seed-42 shuffle.
    ordered = edges[:, (edges[1] * n + edges[0]).argsort()]
    generator = torch.Generator().manual_seed(42)
    shuffled = ordered[:, torch.randperm(edges.shape[1], generator=generator)]
    count = int(0.2 * edges.shape[1])
    expected = canonical_edges(shuffled[:, count : 2 * count], n).T
    assert torch.equal(split["test"]["edge"], expected)
    parts = [
        set((p["edge"][:, 0] * n + p["edge"][:, 1]).tolist()) for p in split.values()
    ]
    assert not any(parts[i] & parts[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    assert set.union(*parts) == set((edges[0] * n + edges[1]).tolist())
    for part in ["valid", "test"]:
        negative = split[part]["edge_neg"]
        assert not set((negative[:, 0] * n + negative[:, 1]).tolist()) & set.union(
            *parts
        )


def test_masked_readout_keeps_backbone_and_inverse_frozen():
    torch.manual_seed(42)
    n = 60
    edges = torch.combinations(torch.arange(n), r=2).T[:, ::10]
    raw = torch.randn(n, 8)
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
        joint_task_backbone=True,
        dropout=0.0,
    )
    model.eval()
    head = new_head(True)
    head.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = fit_link_readout(model, head, raw, raw, edges)
    assert len(result["fitted"][0]) == 18
    assert result["unique_train_pairs"] > 0 and result["unique_holdout_pairs"] > 0
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
