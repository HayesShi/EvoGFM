"""Check the actual released map/split reconciliation and AUC batch slicing."""

import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evogfm.decoupled_protocol import (
    first_source_negatives,
    prepare_released_events,
    released_auc_indices,
)


def test_released_uci_map_keys_and_raw_order(tmp_path):
    baseline = Path(__file__).resolve().parents[1] / "baselines/DecoupledDGNN"
    author_script = baseline / "prepare_data/preprocess_data_DTDG.py"
    raw = baseline.parent.parent / "data/temporal/CollegeMsg.txt.gz"
    if not (author_script.is_file() and raw.is_file()):
        pytest.skip("The optional DecoupledDGNN/UCI fixture is missing")
    tree = ast.parse(author_script.read_text())
    definition = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_time_edge_map"
    ]
    scope = {"np": np, "pickle": pickle}
    exec(
        compile(
            ast.Module(body=definition, type_ignores=[]),
            "released_preprocessing",
            "exec",
        ),
        scope,
    )
    x = prepare_released_events(
        "uci",
        data=Path(__file__).resolve().parents[1] / "data/temporal",
        baselines=baseline.parent,
    )
    frame = pd.DataFrame({"u": x.source, "i": x.destination, "ts_str": x.time})
    target = tmp_path / "map.pkl"
    scope["build_time_edge_map"](frame, target)
    with target.open("rb") as handle:
        mapping = pickle.load(handle)
    assert len(mapping) == x.metadata["snapshots"] == 87
    assert x.metadata["snapshot_counts"] == {"train": 62, "validation": 10, "test": 15}
    assert x.metadata["dropped_event_indices"] == [1, 2]
    for timestamp, edges in mapping.items():
        selected = x.time == timestamp
        np.testing.assert_array_equal(
            edges.T, np.stack((x.source[selected], x.destination[selected]))
        )
        assert np.all(np.diff(x.event_index[selected]) > 0)


def test_released_batch_slicing_and_source_major_selection():
    np.testing.assert_array_equal(released_auc_indices(3000), np.arange(2048))
    indices, sources = first_source_negatives(np.array([1, 1, 1, 3, 3, 3]), 2)
    np.testing.assert_array_equal(indices, [0, 1, 3, 4])
    np.testing.assert_array_equal(sources, [1, 3])
