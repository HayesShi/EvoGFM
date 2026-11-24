"""Fixed-width transformation and validation-selection contracts."""

import pytest
import torch
import numpy as np

from evogfm.temporal import TemporalHistory, replace_temporal
from evogfm.temporal_calibration import (
    PastLocalGraph,
    calibration_channels,
    local_calibration_channels,
    select_operating_point,
    selection_grid,
)
from experiments.temporal_calibration import fit, read_feature_record
from experiments.temporal_calibration_locked_test import validate_readout


def test_transform_formulas_and_query_role_invariance():
    history = TemporalHistory(4)
    history.ingest([[0, 0, 0, 2], [1, 1, 2, 1]], 1)
    pairs = [[0, 0, 1], [1, 1, 3]]
    statistics = history.statistics(pairs, 3)
    base = torch.arange(51).reshape(3, 17).float()
    base[1] = base[0]
    unchanged = base.clone()
    common = replace_temporal(base, statistics, "recency_frequency", 2)
    conditional = calibration_channels(
        base, statistics, "source_conditional_frequency", 2
    )
    corrected = calibration_channels(base, statistics, "target_corrected_recency", 2)
    assert conditional.shape == corrected.shape == (3, 17)
    torch.testing.assert_close(conditional[0, 1], torch.tensor(2 / 3).log1p())
    torch.testing.assert_close(corrected[0, 0], torch.tensor(0.25))
    assert corrected[2, 0] == conditional[2, 1] == 0
    torch.testing.assert_close(
        conditional[:, [i for i in range(17) if i != 1]],
        common[:, [i for i in range(17) if i != 1]],
    )
    torch.testing.assert_close(corrected[:, 1:], common[:, 1:])
    assert torch.equal(conditional[0], conditional[1])
    order = torch.tensor([2, 0, 1])
    assert torch.equal(
        calibration_channels(
            base[order], statistics[order], "source_conditional_frequency", 2
        ),
        conditional[order],
    )
    assert torch.equal(base, unchanged)


def test_selection_preserves_mrr_and_has_eight_novel_configs():
    reference = dict(id="reference", validation=dict(auc=0.8, mrr=0.3))
    trials = [
        dict(id="auc_only", validation=dict(auc=0.9, mrr=0.29)),
        dict(id="both", validation=dict(auc=0.85, mrr=0.31)),
    ]
    selected = select_operating_point(trials, reference)
    assert selected["selected"]["id"] == "both"
    assert selected["best_auc"]["id"] == "auc_only"
    assert select_operating_point(trials[:1], reference)["selected"] == reference
    assert len(selection_grid(1)) == len(selection_grid(16)) == 8
    assert all("variant" not in row for row in selection_grid(1))


def test_test_feature_files_are_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match="train and validation"):
        read_feature_record(tmp_path, {"phase": "test", "file": "absent.npz"})


def test_direct_calibration_has_eighteen_coefficients_and_ignores_frozen_offset():
    torch.manual_seed(42)
    record = dict(
        entry={"phase": "train"},
        base=torch.rand(12, 17),
        statistics=torch.zeros(12, 7),
        offset=torch.randn(12),
        positive_count=6,
    )
    configuration = dict(
        transform="source_conditional_frequency",
        half_life=4,
        ridge=1,
        offset_mode="zero",
    )
    first = fit([record], configuration)
    second = fit([dict(record, offset=record["offset"] * 100 + 25)], configuration)
    assert first[0].numel() == 18
    assert validate_readout(first) == 416
    with pytest.raises(ValueError, match="416 bytes"):
        validate_readout(tuple(value.float() for value in first))
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_exact_incremental_local_statistics_and_fixed_channel_boundary():
    graph = PastLocalGraph(5)
    graph.ingest([[0, 0, 1, 2, 0, 4], [1, 2, 2, 3, 1, 4]])
    # Duplicate edge and self loop do not alter the simple undirected graph.
    assert [len(row) for row in graph.neighbors] == [2, 2, 3, 1, 0]
    pairs = np.array([[0, 0, 0, 4], [1, 1, 3, 4]])
    before = [row.copy() for row in graph.neighbors]
    values = graph.statistics(pairs)
    torch.testing.assert_close(
        values[0],
        torch.tensor(
            [np.log1p(1), np.log1p(1 / np.log(3)), np.log1p(1 / 3), 1 / 3, np.log1p(4)],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(values[0], values[1])
    assert torch.equal(graph.statistics(pairs[:, [2, 0, 3, 1]]), values[[2, 0, 3, 1]])
    assert graph.neighbors == before
    assert torch.equal(values[3], torch.zeros(5))
    base, temporal = torch.randn(4, 17), torch.zeros(4, 7)
    out = local_calibration_channels(base, temporal, values, 4)
    assert out.shape == (4, 17)
    assert torch.equal(out[:, :10], base[:, :10])
    assert torch.equal(out[:, 12:], values)
    graph.ingest([[0], [3]])
    assert not torch.equal(graph.statistics(pairs), values)
    assert graph.state_bytes() > 0
