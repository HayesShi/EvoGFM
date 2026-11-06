import numpy as np
import torch
from scipy.optimize import minimize


class OnlineNewton:
    """Box-constrained ONS with a metric (not coordinate-clipping) projection."""

    def __init__(self, theta, lower, upper, eta=1.0, ridge=1.0, projection_tol=1e-6):
        self.theta = theta.detach().cpu().double().numpy().copy()
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.metric = np.eye(len(self.theta)) * ridge
        self.eta = eta
        self.projection_tol = projection_tol
        self.updates = 0
        self.last_kkt = 0.0

    def step(self, features, labels, weights):
        x = features.detach().cpu().double().numpy()
        y = labels.detach().cpu().double().numpy()
        w = weights.detach().cpu().double().numpy()
        from scipy.special import expit

        gradient = x.T @ (w * (expit(x @ self.theta) - y)) / len(y)
        return self.step_gradient(torch.from_numpy(gradient), like=features)

    def step_gradient(self, gradient, like=None, curvature=None):
        """Curvature-precondition an actual task gradient in online coordinates.

        For a nonlinear task this is an ONS-style optimizer, not an invocation
        of the exp-concave logistic regret guarantee used by step().
        """
        reference = gradient if like is None else like
        gradient = gradient.detach().cpu().double().numpy()
        if gradient.shape != self.theta.shape or not np.isfinite(gradient).all():
            raise ValueError("invalid online gradient")
        if curvature is None:
            self.metric += np.outer(gradient, gradient)
        else:
            update = curvature.detach().cpu().double().numpy()
            if update.shape != self.metric.shape or not np.isfinite(update).all():
                raise ValueError("invalid curvature")
            update = (update + update.T) / 2
            if np.linalg.eigvalsh(update).min() < -1e-8:
                raise ValueError("curvature must be positive semidefinite")
            self.metric += update
        target = self.theta - self.eta * np.linalg.solve(self.metric, gradient)
        if np.all(target >= self.lower) and np.all(target <= self.upper):
            projected = target
            self.last_kkt = 0.0
        else:
            scale = max(1.0, np.linalg.norm(self.metric, ord=2))
            metric = self.metric / scale

            def objective(v):
                delta = v - target
                return 0.5 * delta @ metric @ delta, metric @ delta

            result = minimize(
                objective,
                np.clip(self.theta, self.lower, self.upper),
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(self.lower, self.upper)),
                options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
            )
            projected = result.x
            grad = metric @ (projected - target)
            residual = projected - np.clip(projected - grad, self.lower, self.upper)
            self.last_kkt = float(np.abs(residual).max())
            if self.last_kkt > self.projection_tol:
                raise RuntimeError(
                    f"ONS metric projection KKT residual {self.last_kkt:.3g}; {result.message}"
                )
        self.theta = projected
        self.updates += 1
        return torch.tensor(projected, dtype=reference.dtype, device=reference.device)

    def state_dict(self):
        return dict(
            theta=self.theta,
            metric=self.metric,
            updates=self.updates,
            last_kkt=self.last_kkt,
            lower=self.lower,
            upper=self.upper,
            eta=self.eta,
        )


class TaskCoefficientBank:
    """Task-conditioned calibration of ONE source-pretrained inverse family.

    Entries are keyed by task type, never by graph/domain. Each holds only an
    inverse coefficient vector and its online metric; backbone weights are shared.
    """

    def __init__(self, initial, lower, upper, tasks=("nc", "lp"), eta=1.0, ridge=0.1):
        self.prior = initial.detach().clone()
        self.optimizers = {
            task: OnlineNewton(initial, lower, upper, eta=eta, ridge=ridge)
            for task in tasks
        }

    def activate(self, inverse, task):
        value = torch.as_tensor(
            self.optimizers[task].theta,
            device=self.prior.device,
            dtype=self.prior.dtype,
        )
        inverse.set_theta(value)

    def record(self, inverse, task, metric=None):
        optimizer = self.optimizers[task]
        optimizer.theta = inverse.theta().detach().double().cpu().numpy().copy()
        if metric is not None:
            optimizer.metric = np.asarray(metric, dtype=np.float64).copy()

    def state_dict(self):
        return dict(
            prior=self.prior,
            tasks={
                task: optimizer.state_dict()
                for task, optimizer in self.optimizers.items()
            },
        )

    def load_task(self, task, state):
        optimizer = self.optimizers[task]
        optimizer.theta = np.asarray(state["theta"], dtype=np.float64).copy()
        optimizer.metric = np.asarray(state["metric"], dtype=np.float64).copy()
        optimizer.updates = int(state["updates"])
        optimizer.last_kkt = float(state.get("last_kkt", 0.0))
        if "lower" in state:
            optimizer.lower = np.asarray(state["lower"], dtype=np.float64).copy()
        if "upper" in state:
            optimizer.upper = np.asarray(state["upper"], dtype=np.float64).copy()
        if "eta" in state:
            optimizer.eta = float(state["eta"])


def continual_metrics(matrix):
    """R[t,j], t=0..T; diagonal entry R[j+1,j] is post-learning task j."""
    r = np.asarray(matrix, dtype=np.float64)
    t = r.shape[1]
    if r.shape[0] != t + 1:
        raise ValueError("R must have T+1 rows and T columns")
    if not np.isfinite(r).all():
        raise ValueError("missing or nonfinite performance cells")
    forgetting = []
    bwt = []
    for j in range(t - 1):
        forgetting.append(float(np.max(r[j + 1 :, j]) - r[t, j]))
        bwt.append(float(r[t, j] - r[j + 1, j]))
    return {
        "final_average": float(r[-1].mean()),
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "backward_transfer": float(np.mean(bwt)) if bwt else 0.0,
    }
