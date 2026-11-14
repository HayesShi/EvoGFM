"""A fixed configuration must not silently reselect seeds during reporting."""

import pandas as pd

from scripts.export_current_results import select_rows


def test_export_retains_configured_seed_for_every_method():
    config = {"datasets": {"example": {"input": "original", "seeds": {1: 42, 5: 43}}}}
    rows = []
    for method in ["EvoGFM", "GraphPFN"]:
        for shots in [1, 5]:
            for seed in [42, 43, 44]:
                rows.append(
                    dict(
                        method=method,
                        graph="example",
                        policy="original",
                        seed=seed,
                        shots=shots,
                        accuracy_percent=seed,
                    )
                )
    # Seed 44 has the highest score; a report must still use the fixed choices.
    selected = select_rows(config, pd.DataFrame(rows))
    assert len(selected) == 4
    assert set(selected[selected.shots == 1].seed) == {42}
    assert set(selected[selected.shots == 5].seed) == {43}
    assert set(selected.method) == {"EvoGFM", "GraphPFN"}
