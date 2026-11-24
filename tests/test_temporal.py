"""Check candidate and metric parity directly against frozen public functions."""

import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from evogfm.temporal import (
    TemporalHistory,
    decoupled_candidates,
    ranking_mrr,
    ranking_tie_diagnostics,
    undergs_destinations,
)


BASELINES = Path(__file__).resolve().parents[1] / "baselines"


def original_definitions(file, names):
    if not file.is_file():
        pytest.skip(f"Optional author source is missing: {file}")
    tree = ast.parse(file.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    scope = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), "exec"), scope)
    return scope


def test_undergs_public_sampler_and_ranking_parity():
    util = original_definitions(BASELINES / "UnderGS/code/util.py", {"RandEdgeSampler"})
    source, destination = np.array([1, 2, 2, 3]), np.array([2, 1, 4, 4])
    sampler = util["RandEdgeSampler"](source, destination, seed=2)
    _, expected = sampler.sample(23)
    np.testing.assert_array_equal(
        undergs_destinations(source, destination, 23, 2), expected
    )
    evaluation = original_definitions(
        BASELINES / "UnderGS/code/evaluation.py", {"compute_mrr"}
    )
    pos = np.array([0.3, 0.3, 0.7])
    neg = np.array([0.3, 0.2, 0.8, 0.2, 0.3, 0.1, 0.5, 0.5, 0.7])
    assert ranking_mrr("undergs", pos, neg, source[:3], source[:3]) == evaluation[
        "compute_mrr"
    ](pos, neg)


def test_decoupled_public_negative_generation_parity():
    original = original_definitions(
        BASELINES / "DecoupledDGNN/link_prediction_alongtime.py",
        {"gen_negative_edges", "edge_index_difference"},
    )
    source, destination = np.array([1, 1, 3]), np.array([2, 4, 6])
    np.random.seed(37)
    expected_source, expected_destination = original["gen_negative_edges"](
        source, destination, 20, 5
    )
    expected = []
    for node in np.unique(source):
        expected.extend(np.flatnonzero(expected_source == node)[:5])
    actual, groups = decoupled_candidates(
        source, destination, 20, 5, np.random.RandomState(37)
    )
    np.testing.assert_array_equal(
        actual, np.stack((expected_source[expected], expected_destination[expected]))
    )
    np.testing.assert_array_equal(groups, [1, 3])
    # Best positive per sender, pessimistic ties, then mean across senders.
    assert (
        ranking_mrr("decoupled", [0.2, 0.7, 0.4], [0.7, 0.6, 0.8, 0.4], source, groups)
        == (1 / 2 + 1 / 3) / 2
    )


def test_decoupled_public_source_best_mrr_parity(monkeypatch):
    original = original_definitions(
        BASELINES / "DecoupledDGNN/link_prediction_alongtime.py",
        {"gen_negative_edges", "edge_index_difference", "Execute"},
    )
    scatter = types.ModuleType("torch_scatter")

    def scatter_max(src, index, dim_size):
        output = src.new_full((dim_size,), -torch.inf)
        output.scatter_reduce_(0, index.long(), src, reduce="amax", include_self=True)
        return output, None

    scatter.scatter_max = scatter_max
    monkeypatch.setitem(sys.modules, "torch_scatter", scatter)

    class Features:
        def get_edges_feats(self, source, destination, *args, **kwargs):
            return (
                torch.from_numpy(source)[:, None],
                torch.from_numpy(destination)[:, None],
            )

    class Model:
        def get_edges_embedding(self, source, destination):
            return ((source * 11 + destination) % 10).float() / 10

    oracle = original["Execute"].__new__(original["Execute"])
    oracle.edge_helper, oracle.model, oracle.device, oracle.window_size = (
        Features(),
        Model(),
        "cpu",
        20,
    )
    positive = np.array([[1, 2], [1, 4], [3, 6]])
    np.random.seed(37)
    expected, _ = oracle.eval_mrr_and_recall(
        positive, np.zeros(3), 20, num_neg_per_node=5
    )
    negative, sources = decoupled_candidates(
        positive[:, 0], positive[:, 1], 20, 5, np.random.RandomState(37)
    )
    positive_score = (
        Model()
        .get_edges_embedding(
            torch.from_numpy(positive[:, :1]), torch.from_numpy(positive[:, 1:])
        )
        .numpy()
    )
    negative_score = (
        Model()
        .get_edges_embedding(
            torch.from_numpy(negative[:1].T), torch.from_numpy(negative[1:].T)
        )
        .numpy()
    )
    assert (
        abs(
            ranking_mrr(
                "decoupled", positive_score, negative_score, positive[:, 0], sources
            )
            - expected
        )
        < 1e-7
    )


def test_temporal_channels_use_past_events_and_keep_input_width():
    history = TemporalHistory(6)
    pairs = np.array([[1, 1, 3], [2, 2, 1]])
    features = torch.zeros(3, 17)
    assert torch.equal(history.channels(pairs, 0)[:, :4], torch.zeros(3, 4))
    history.ingest(pairs, 0)
    channels = history.channels(pairs, 4, half_life=4)
    torch.testing.assert_close(channels[:, 0], torch.full((3,), 0.5))
    torch.testing.assert_close(channels[:2, 1], torch.full((2,), float(np.log(3))))
    assert history.replace(features, pairs, 4).shape == features.shape
    assert sum(value.numel() for value in history.replace(features, pairs, 4)) == 3 * 17
    original = torch.arange(51, dtype=torch.float32).reshape(3, 17)
    frequency = history.replace(original, pairs, 4, "pair_frequency")
    torch.testing.assert_close(frequency[:, :16], original[:, :16])
    torch.testing.assert_close(frequency[:, 16], channels[:, 1])
    pair_only = history.replace(original, pairs, 4, "pair_frequency_recency")
    untouched = [column for column in range(17) if column not in (10, 11)]
    torch.testing.assert_close(pair_only[:, untouched], original[:, untouched])
    torch.testing.assert_close(pair_only[:, [10, 11]], channels[:, [1, 0]])


def test_ranking_ties_match_uniform_and_pessimistic_controls():
    diagnostic = ranking_tie_diagnostics(
        "undergs", [0, 1], [0, 0, 0, 1], [0, 1], [0, 1]
    )
    assert (
        abs(
            diagnostic["random_tie_expected_mrr"]
            - ((1 + 1 / 2 + 1 / 3) / 3 + (1 + 1 / 2) / 2) / 2
        )
        < 1e-12
    )
    assert abs(diagnostic["pessimistic_mrr"] - (1 / 3 + 1 / 2) / 2) < 1e-12
