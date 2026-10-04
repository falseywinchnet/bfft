"""Matrix-free, fixed-metric Arnoldi and finite discrete-time transport.

No fixed-point inverse is formed. A fresh local derivative is acquired at each
restart. Smooth mirror maps and full-state splitting maps use the same code.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


def identity(v):
    return v


def metric_norm(v, metric=identity):
    squared = float(v @ metric(v))
    if not np.isfinite(squared) or squared < -1e-12 * max(1., float(v @ v)):
        raise FloatingPointError("invalid metric norm")
    return float(np.sqrt(max(0., squared)))


@dataclass
class Linearization:
    next_state: np.ndarray
    action: Callable
    metric: Callable = identity


@dataclass
class Arnoldi:
    Q: np.ndarray
    H: np.ndarray
    remainder: np.ndarray
    beta: float
    actions: int
    metric: Callable

    def coordinates(self, horizon):
        """beta * sum_{j=0}^{horizon-1} H**j e1, including H eigenvalue 1."""
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        y = np.zeros(self.H.shape[0])
        source = np.zeros_like(y)
        if len(y):
            source[0] = self.beta
        for _ in range(horizon):
            y = source + self.H @ y
        return y

    def displacement(self, horizon):
        return self.Q @ self.coordinates(horizon)


def arnoldi(action, residual, depth=4, metric=identity, tolerance=1e-13):
    """Twice-reorthogonalized Arnoldi in a frozen SPD metric.

    Store the full floating-point remainder E=AQ-QH. This both handles happy
    breakdown and lets the error diagnostic account for earlier roundoff.
    An action must be linear: finite secants depending on the direction do
    not meet this contract and cannot simply replace Jacobian actions.
    """
    r = np.asarray(residual, dtype=float)
    if r.ndim != 1 or depth < 1:
        raise ValueError("one-dimensional residual and positive depth required")
    beta = metric_norm(r, metric)
    if beta == 0:
        return Arnoldi(np.empty((len(r), 0)), np.empty((0, 0)),
                       np.empty((len(r), 0)), beta, 0, metric)
    qs = [r / beta]
    mqs = [metric(qs[0])]
    aq = []
    H = np.zeros((depth, depth))
    for j in range(min(depth, len(r))):
        a = np.asarray(action(qs[j]), dtype=float)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("nonfinite tangent action")
        aq.append(a.copy())
        v = a.copy()
        for _ in range(2):
            for i in range(j + 1):
                coefficient = float(mqs[i] @ v)
                H[i, j] += coefficient
                v -= coefficient * qs[i]
        h = metric_norm(v, metric)
        if j + 1 == min(depth, len(r)) or h <= tolerance * max(1., metric_norm(a, metric)):
            break
        H[j + 1, j] = h
        qs.append(v / h)
        mqs.append(metric(qs[-1]))
    k = len(aq)
    Q = np.column_stack(qs[:k])
    H = H[:k, :k]
    return Arnoldi(Q, H, np.column_stack(aq) - Q @ H, beta, k, metric)


def local_arnoldi(model, state, depth=4, natural_metric=True):
    lin = model.linearize(state)
    basis = arnoldi(lin.action, lin.next_state - state, depth,
                    lin.metric if natural_metric else identity)
    return basis, lin


def finite_flow(model, state, depth=4, horizon=16, natural_metric=True):
    basis, lin = local_arnoldi(model, state, depth, natural_metric)
    return state + basis.displacement(horizon), basis, lin


def shadow_diagnostic(model, state, basis, lin, horizon):
    """Exact path defects for verification, NOT a free runtime certificate.

    A diagnostic evaluates every predicted point. It is deliberately excluded
    from accelerated runs and has its full map/action count reported.
    """
    r = lin.next_state - state
    actual = state.copy()
    predicted = state.copy()
    row = []
    for i in range(horizon):
        y = basis.coordinates(i)
        displacement = basis.Q @ y
        next_prediction = state + basis.displacement(i + 1)
        nonlinear = model.step(state + displacement) - lin.next_state - lin.action(displacement)
        compression = basis.remainder @ y
        defect = model.step(state + displacement) - next_prediction
        actual = model.step(actual)
        predicted = next_prediction
        row.append({
            "step": i,
            "nonlinear": metric_norm(nonlinear, basis.metric),
            "compression": metric_norm(compression, basis.metric),
            "defect": metric_norm(defect, basis.metric),
            "decomposition_error": metric_norm(defect - nonlinear - compression, basis.metric),
        })
    return {"rows": row,
            "terminal_error": metric_norm(actual - predicted, basis.metric),
            "ordinary_displacement": metric_norm(actual - state, basis.metric),
            "diagnostic_map_calls": 3 * horizon,
            "diagnostic_action_calls": horizon}


class Anderson:
    """Regularized type-II Anderson comparator in the model's full coordinates.

    No novelty claim and no objective guard: failures remain visible. This is
    not the guarded AA-BPG algorithm of Mai and Johansson.
    """
    def __init__(self, depth=4, regularization=1e-10):
        self.depth = depth
        self.regularization = regularization
        self.states = []
        self.residuals = []

    def advance(self, state, next_state):
        residual = next_state - state
        self.states.append(state.copy())
        self.residuals.append(residual.copy())
        self.states = self.states[-self.depth - 1:]
        self.residuals = self.residuals[-self.depth - 1:]
        if len(self.states) < 2:
            return next_state
        dX = np.diff(np.column_stack(self.states), axis=1)
        dR = np.diff(np.column_stack(self.residuals), axis=1)
        # Residual-scale floor prevents roundoff-sized differences between
        # constant residuals from producing arbitrary, enormous coefficients.
        ridge = self.regularization * max(float(np.sum(dR * dR)),
                                           float(residual @ residual),
                                           np.finfo(float).tiny)
        A = np.vstack((dR, np.sqrt(ridge) * np.eye(dR.shape[1])))
        rhs = np.concatenate((residual, np.zeros(dR.shape[1])))
        gamma = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return next_state - (dX + dR) @ gamma
