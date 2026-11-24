"""Numerical contracts for evolving-graph maintenance."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from evogfm.data import canonical_edges
from evogfm.incremental import (
    FullEdgeMoments,
    MaintainedLinkScorer,
    ingest_edges,
    numeric_payload_bytes,
    program_matrix,
)
from evogfm.link_inference import LinkScorer
from evogfm.model import GraphState
from experiments.full_graph import FullGraphGFM
from experiments.idea_a import edges_before
from evogfm.readout import new_head


def fixture(backbone=False):
    torch.manual_seed(17)
    model = FullGraphGFM(
        input_dim=8,
        d=8,
        hidden=16,
        heads=2,
        rank=4,
        layers=3,
        frame_aligned=True,
        dropout=0.0,
        joint_task_backbone=backbone,
    )
    model.eval()
    raw = torch.rand(20, 8)
    edges = canonical_edges(
        torch.tensor(
            [
                [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
                [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 13, 18],
            ]
        ),
        20,
    )
    return model, new_head(backbone), raw, edges


@torch.no_grad()
def test_moments_add_and_delete_match_full_graph():
    model, _, raw, edges = fixture()
    z = model.encode_features(raw)
    moments = FullEdgeMoments.build(z, edges)
    added = torch.tensor([[2, 14], [10, 19]])
    removed = edges[:, :2]
    moments.update(added, removed)
    updated = canonical_edges(torch.cat((edges[:, 2:], added), dim=1), len(raw))
    exact = model.state(raw, updated)
    result = moments.state()
    torch.testing.assert_close(result.spectrum, exact.spectrum, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        program_matrix(model, result),
        program_matrix(model, exact),
        rtol=1e-4,
        atol=1e-4,
    )


@torch.no_grad()
@pytest.mark.parametrize("backbone", [False, True])
def test_link_channels_match_released_scorer_and_local_refresh(backbone):
    model, head, raw, edges = fixture(backbone)
    pairs = torch.tensor([[0, 1, 2, 7, 10], [3, 5, 6, 9, 19]])
    maintained = MaintainedLinkScorer(
        model, head, raw, raw, edges, "error_aware", 100, 1e10
    )
    released = LinkScorer(model, head, raw, raw, edges)
    torch.testing.assert_close(
        maintained.features(pairs), released.features(pairs), rtol=1e-4, atol=1e-4
    )
    added = torch.tensor([[1, 8], [4, 15]])
    removed = edges[:, :1]
    changed = canonical_edges(torch.cat((edges[:, 1:], added), dim=1), len(raw))
    times = maintained.update(changed, added, removed)
    assert times["refreshed"] == 0
    # A full cache rebuild under the same frozen spectral state is the local oracle.
    reference = MaintainedLinkScorer(model, head, raw, raw, edges)
    reference.state = maintained.state
    reference._full_prepare(changed)
    np.testing.assert_allclose(
        maintained.weighted.toarray(),
        reference.weighted.toarray(),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        maintained.features(pairs), reference.features(pairs), rtol=1e-4, atol=1e-4
    )


def test_numeric_payload_deduplicates_cross_library_storage_and_dataclasses():
    array = np.arange(30, dtype=np.float32)
    tensor = torch.from_numpy(array)
    sliced = torch.from_numpy(array[5:15])
    sparse = sp.eye(4, format="csr", dtype=np.float32)
    state = GraphState(tensor, sliced, tensor[:3], sliced, tensor[:1], tensor[1:8])
    moments = FullEdgeMoments(tensor, tensor[3:12], 4, tensor[2:9])
    expected = (
        array.nbytes + sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
    )
    cached = dict(
        state=state,
        moments=moments,
        raw=array[::-1],
        nested=[sliced, sparse, tensor.numpy()],
    )
    assert numeric_payload_bytes(cached) == expected
    extra = np.arange(7, dtype=np.int64)
    cached["new_numeric_field"] = extra
    assert numeric_payload_bytes(cached) == expected + extra.nbytes


def test_batch_ingest_matches_all_temporal_prefix_oracles():
    events = np.array(
        [[0, 1], [1, 0], [2, 2], [3, 4], [0, 1], [4, 3], [1, 5], [5, 1], [2, 5]]
    )
    bins = np.array([0, 0, 0, 1, 1, 3, 3, 4, 4])
    previous = torch.empty((2, 0), dtype=torch.long)
    known = np.empty(0, dtype=np.int64)
    for step in range(1, 6):
        incoming = torch.from_numpy(events[bins == step - 1].T.copy())
        visible, added, known = ingest_edges(previous, incoming, 6, known)
        torch.testing.assert_close(visible, edges_before(events, bins, step, 6))
        old_keys = (previous[0] * 6 + previous[1]).numpy()
        assert not np.isin((added[0] * 6 + added[1]).numpy(), old_keys).any()
        previous = visible
