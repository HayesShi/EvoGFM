"""Converged support-only offset-logistic readout; original prediction tuple.

Uses the original residual-readout objective with a converged Newton solve.
"""

import numpy as np
from scipy.special import expit
import torch


def fit_residual_converged(features, offset, labels, ridge=1.0):
    """Fit the original sum-BCE + ridge objective using float64 Armijo Newton.

    Returns ``(coefficient, mean, scale)`` for the original ``predict_residual``.
    Like the released readout, the current interface uses CPU tensors. Only the
    supplied support data enter the objective or numerical stopping rule. The
    intercept penalty is 0.1 * ridge. Non-convergence raises RuntimeError.
    """
    features = features.detach().double()
    offset = offset.detach().double()
    labels = labels.detach().double()
    mean = features.mean(0)
    scale = features.std(0).clamp_min(0.05)
    x = (features - mean) / scale
    x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1).numpy()
    offset = offset.numpy()
    labels = labels.numpy()
    penalty = np.full(x.shape[1], float(ridge))
    penalty[-1] = float(ridge) * 0.1

    def objective_gradient(coefficient):
        logits = offset + x @ coefficient
        objective = float(
            np.sum(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * np.dot(coefficient * penalty, coefficient)
        )
        gradient = x.T @ (expit(logits) - labels) + penalty * coefficient
        return objective, gradient

    coefficient = np.zeros(x.shape[1])
    for _ in range(200):
        objective, gradient = objective_gradient(coefficient)
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise RuntimeError(
                "Residual solver encountered a non-finite support objective or gradient"
            )
        if np.max(np.abs(gradient)) <= 1e-8:
            return torch.from_numpy(coefficient), mean, scale
        probability = expit(offset + x @ coefficient)
        hessian = x.T @ ((probability * (1 - probability))[:, None] * x) + np.diag(
            penalty
        )
        direction = np.linalg.solve(hessian, -gradient)
        slope = np.dot(gradient, direction)
        if not np.isfinite(slope) or slope >= 0:
            raise RuntimeError(
                "Residual solver could not form a descending Newton direction"
            )
        rate = 1.0
        for _ in range(80):
            candidate = coefficient + rate * direction
            candidate_objective, _ = objective_gradient(candidate)
            if candidate_objective <= objective + 1e-4 * rate * slope + 1e-12:
                coefficient = candidate
                break
            rate *= 0.5
        else:
            raise RuntimeError("Residual solver exhausted Armijo backtracking")
    _, gradient = objective_gradient(coefficient)
    raise RuntimeError(
        f"Residual solver did not converge in 200 iterations; "
        f"support gradient infinity norm = {np.max(np.abs(gradient)):.6g}"
    )
