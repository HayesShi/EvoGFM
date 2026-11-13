"""Numerical and protocol checks for the portable experiment entry points."""

from argparse import Namespace

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from experiments.full_graph import FullGraphGFM, load_checkpoint
from experiments.residual_solver import fit_residual_converged
from experiments.run import commands
from experiments import settings


def test_full_edge_moment_matches_direct_float64_sum():
    torch.manual_seed(17)
    model = FullGraphGFM(input_dim=8, d=8, hidden=16, heads=2, rank=4, k=8)
    x = torch.randn(300, 8)
    edges = torch.combinations(torch.arange(len(x)), r=2).T
    state = model.state(x, edges)
    z = state.z.double()
    expected = state.density.double() * (z[edges[0]].T @ z[edges[1]]) / edges.shape[1]
    expected = (expected + expected.T) / 2
    actual = (state.left.double() * state.spectrum.double()) @ state.right.double().T
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)


def test_reference_checkpoint_and_source_partitions():
    import json

    model, head, checkpoint = load_checkpoint(
        settings.REPO / "checkpoints/reference.pt"
    )
    assert model.inverse.k == 32
    assert checkpoint["epoch"] == 21
    assert not model.training and not head.training
    partitions = json.loads(
        (settings.REPO / "checkpoints/source_splits.json").read_text()
    )
    assert set(partitions) == set(checkpoint["source_graphs"])
    for split in partitions.values():
        assert split["train"] and split["val"]
        assert not set(split["train"]) & set(split["val"])


def test_residual_solver_matches_independent_bfgs_objective():
    generator = torch.Generator().manual_seed(23)
    f = torch.randn(400, 17, generator=generator)
    offset = torch.randn(400, generator=generator) * 4
    labels = torch.randint(2, (400,), generator=generator).double()
    c, mean, scale = fit_residual_converged(f, offset, labels, ridge=0.1)
    x = np.column_stack([((f.double() - mean) / scale).numpy(), np.ones(400)])
    penalty = np.r_[np.full(17, 0.1), 0.01]

    def objective(v):
        z = offset.numpy() + x @ v
        return (
            np.sum(np.logaddexp(0, z) - labels.numpy() * z)
            + 0.5 * np.dot(v * penalty, v),
            x.T @ (expit(z) - labels.numpy()) + penalty * v,
        )

    result = minimize(
        objective, np.zeros(18), jac=True, method="BFGS", options={"gtol": 1e-8}
    )
    assert abs(objective(c.numpy())[0] - result.fun) < 1e-8
    assert np.max(np.abs(objective(c.numpy())[1])) <= 1e-8


def test_gc_plan_uses_all_five_datasets_and_both_label_budgets(tmp_path):
    plan = commands(
        Namespace(
            task="gc",
            checkpoint=tmp_path / "model.pt",
            data_root=tmp_path / "data",
            output=tmp_path / "runs",
        )
    )
    assert len(plan) == 10
    assert all("--balanced-readout" in command for command in plan)
    assert {command[command.index("--shots") + 1] for command in plan} == {"5", "20"}
