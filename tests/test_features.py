import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

from evogfm.features import (
    adjacency,
    pair_channels,
    propagate,
    contexts,
    task_features,
    node_representations,
    attention,
    pair_context,
)
from evogfm.model import InverseLawGFM


def test_query_structural_channels_equal_full_matrix_products():
    torch.manual_seed(42)
    n = 35
    edges = torch.combinations(torch.arange(n), r=2).T[:, ::4]
    x = torch.randn(n, 8)
    pairs = torch.randint(n, (2, 80))
    actual = pair_channels(x, edges, pairs, batch_size=7)
    a = adjacency(edges, n)
    degree = np.asarray(a.sum(1)).ravel()
    matrices = [
        a @ a,
        a @ sp.diags(1 / np.maximum(1, degree)) @ a,
        a @ sp.diags(1 / np.log(np.maximum(2, degree))) @ a,
        a @ a @ a,
    ]
    expected = torch.stack(
        [
            torch.tensor(
                np.asarray(
                    matrix[pairs[0].numpy().copy(), pairs[1].numpy().copy()]
                ).ravel(),
                dtype=torch.float32,
            ).log1p()
            for matrix in matrices
        ],
        dim=1,
    )
    assert torch.allclose(actual[:, 6:], expected, atol=1e-6)
    raw = F.normalize(x, dim=1)
    one = F.normalize(propagate(x, edges), dim=1)
    assert torch.allclose(actual[:, 0], (raw[pairs[0]] * raw[pairs[1]]).sum(1))
    assert torch.allclose(actual[:, 1], (one[pairs[0]] * one[pairs[1]]).sum(1))


def test_cached_query_batches_preserve_full_features_and_node_readout():
    torch.manual_seed(42)
    n = 30
    raw = torch.randn(n, 8)
    edges = torch.combinations(torch.arange(n), r=2).T[:, ::3]
    pairs = torch.randint(n, (2, 40))
    model = InverseLawGFM(
        input_dim=8,
        d=8,
        hidden=16,
        layers=1,
        heads=2,
        rank=6,
        k=8,
        dropout=0.0,
        frame_aligned=True,
    ).eval()
    model.config["joint_task_backbone"] = True
    with torch.no_grad():
        state = model.state(raw, edges)
        ctx = contexts(edges, n, pairs)
        base = pair_channels(raw, edges, pairs)
        expected, original_parts = task_features(
            model, state, raw, edges, pairs, ctx, base
        )
        parts, h = node_representations(model, state, raw, edges, hops=4)
        for actual, original in zip(parts[:3] + [h], original_parts):
            assert torch.allclose(actual, original, atol=1e-6)
        prepared = (attention(model, state, ctx["visible"]), parts[:3] + [h])
        content = pair_context(raw, edges)
        result = []
        for batch in pairs.split(7, dim=1):
            features, _ = task_features(
                model,
                state,
                raw,
                edges,
                batch,
                contexts(edges, n, batch),
                pair_channels(raw, edges, batch, prepared=content),
                prepared=prepared,
            )
            result.append(features)
        assert torch.allclose(torch.cat(result), expected, atol=1e-6)
