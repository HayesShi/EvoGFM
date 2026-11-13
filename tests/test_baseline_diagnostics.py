"""Scientific consistency checks for the input diagnostic and aggregation."""

import pandas as pd
import pytest
import torch

from experiments.airport_inputs import structural_input
from scripts.summarize_node_results import GRAPHS, markdown_tables, summarize


@pytest.mark.parametrize("policy", ["degree64", "degree_quantile64"])
def test_structural_encoding_is_equivariant_to_node_renumbering(policy):
    # Includes repeated degrees and nodes above the degree64 cap.
    n = 80
    edges = torch.stack([torch.zeros(n - 1, dtype=torch.long), torch.arange(1, n)])
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(123))
    original = structural_input(edges, n, policy)
    renumbered = structural_input(permutation[edges], n, policy)
    torch.testing.assert_close(renumbered[permutation], original)
    assert torch.equal(original.sum(1), torch.ones(n))


def test_overall_sd_preserves_paired_seed_cancellation():
    rows = []
    for index, graph in enumerate(sorted(GRAPHS)):
        for seed in [42, 43, 44]:
            value = 50 + (seed - 43) * (1 if index < 5 else -1)
            rows.append(
                dict(
                    method="complete",
                    graph=graph,
                    shots=5,
                    seed=seed,
                    accuracy_percent=value,
                )
            )
    frame = pd.DataFrame(rows)
    per_graph, overall = summarize(frame)
    assert (per_graph["std"] == 1).all()
    assert overall.iloc[0]["mean"] == 50
    assert overall.iloc[0]["std"] == 0
    assert "| complete |  | 50.00 ± 0.00 |  |" in markdown_tables(per_graph, overall)
    partial_graphs, partial = summarize(
        frame[frame.graph != sorted(GRAPHS)[0]].assign(method="GraphPFN")
    )
    assert partial.empty
    assert "| GraphPFN |  |  |  |" in markdown_tables(partial_graphs, partial)
