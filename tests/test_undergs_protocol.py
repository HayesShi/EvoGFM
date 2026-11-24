from pathlib import Path

import numpy as np
import pytest
import torch

from evogfm.temporal import undergs_destinations
from evogfm.undergs_protocol import CausalNeighborhood, runtime_state, score_metrics
from experiments.temporal_official import original_undergs_modules


def author_modules():
    baselines = Path(__file__).resolve().parents[1] / "baselines"
    if not (baselines / "UnderGS/code/undergs_model.py").is_file():
        pytest.skip("Optional UnderGS author source is missing")
    return original_undergs_modules(baselines)


def test_candidates_are_per_event_and_match_released_sampler():
    _, util, evaluation = author_modules()
    sources = np.array([1, 1, 2, 3])
    destinations = np.array([2, 3, 1, 2])
    sampler = util.RandEdgeSampler(sources, destinations, seed=2)
    _, expected = sampler.sample(400)
    np.testing.assert_array_equal(
        expected, undergs_destinations(sources, destinations, 400, 2)
    )
    positive = np.array([0.5, 0.3, 0.5, 0.1])
    negative = np.random.RandomState(42).choice([0.1, 0.3, 0.5], 400)
    metrics = score_metrics(positive, negative[:4], positive, negative)
    assert metrics["mrr"] == evaluation.compute_mrr(positive, negative)
    assert metrics["pessimistic_mrr"] <= metrics["random_tie_expected_mrr"]


def test_all_ties_report_random_and_pessimistic_controls():
    metrics = score_metrics(np.ones(2), np.ones(2), np.ones(2), np.ones(200))
    assert metrics["pessimistic_mrr"] == 1 / 101
    assert np.isclose(
        metrics["random_tie_expected_mrr"], (1 / np.arange(1, 102)).mean()
    )


def test_release_current_edges_are_temporary_and_causal_history_persists():
    from argparse import Namespace

    module, util, _ = author_modules()
    args = Namespace(n_nodes=5, topk=2, lambda_1=0.1, lambda_2=0.05, filter_ratio=0.0)
    torch.manual_seed(42)
    model = module.UnderGS(
        None, np.zeros((4, 1), np.float32), torch.device("cpu"), 0.0, 8, 2, "gcn", args
    )
    store = model.embedding_module.tcns
    nodes = np.array([1, 2, 3], dtype=np.int32)
    args_update = (nodes, np.array([1, 2]), 1, np.array([1]), 1, np.array([0]))
    neighbors, _, _, weights = store.update_tcns(*args_update)
    assert weights[:2].sum() > 0
    assert neighbors[0, 0] == 2
    assert weights[2].sum() == 0
    assert store.tcns_weights.sum() == 0
    bridge = CausalNeighborhood(model, util)
    assert bridge.update(*args_update)[3].sum() == 0
    bridge.observe(np.array([1]), np.array([2]), np.array([1]), np.array([1]))
    before = store.backup()
    past = bridge.update(
        np.array([1, 3, 2], dtype=np.int32),
        np.array([1, 3]),
        2,
        np.array([2]),
        1,
        np.array([0]),
    )
    assert past[0][0, 0] == 2
    assert past[3][2].sum() > 0  # A negative query gets its true past neighbors.
    torch.testing.assert_close(before["tcns_keys"], store.tcns_keys)
    assert past[3][1].sum() == 0
    model.eval()
    before_score = runtime_state(model)
    with torch.no_grad():
        first_score = bridge.score(
            np.array([1]), np.array([3]), np.array([4]), np.array([2]), np.array([2])
        )
        second_score = bridge.score(
            np.array([1]), np.array([3]), np.array([4]), np.array([2]), np.array([3])
        )
    torch.testing.assert_close(first_score, second_score, rtol=0, atol=0)
    torch.testing.assert_close(before_score, runtime_state(model), rtol=0, atol=0)
