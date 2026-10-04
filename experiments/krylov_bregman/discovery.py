"""Discover candidate finite transport from current actions and live-map probes.

Sparse probes FALSIFY candidates; they do not certify the unobserved path.
No optimum, objective, future ordinary trajectory, or spectral labels enter
selection. All acquisition and unsuccessful probes remain charged.
"""
from dataclasses import dataclass
import numpy as np
from .core import Arnoldi, identity


def esp(values, order):
    """Elementary symmetric polynomial, O(n*order), nonnegative inputs."""
    values = np.asarray(values, dtype=float)
    if order < 0 or np.any(values < 0):
        raise ValueError('nonnegative values and order required')
    e = np.zeros(order + 1); e[0] = 1.
    for x in values:
        for j in range(order, 0, -1):
            e[j] += x * e[j-1]
    return float(e[order])


def rank_witness(matrix, rank):
    """e_(rank+1)(sigma^2/||matrix||_F^2); zero iff rank <= rank in exact arithmetic.

    Small positive values are NOT a numerical rank certificate. The complete
    singular spectrum is returned so a gap or conditioning is not concealed.
    """
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular * singular
    if energy.sum():
        energy /= energy.sum()
    return {'esp': esp(energy, rank+1), 'singular_values': singular.tolist()}


def arnoldi_stream(action, residual, maximum_depth=12):
    """Incremental Euclidean Arnoldi; yield without paying for future actions."""
    beta = float(np.linalg.norm(residual))
    if beta == 0:
        return
    qs = [residual / beta]; aq = []; H = np.zeros((maximum_depth, maximum_depth))
    for j in range(min(maximum_depth, len(residual))):
        a = np.asarray(action(qs[j])); aq.append(a.copy()); v = a.copy()
        for _ in range(2):
            for i in range(j+1):
                coefficient = float(qs[i] @ v)
                H[i,j] += coefficient; v -= coefficient * qs[i]
        h = float(np.linalg.norm(v))
        Q = np.column_stack(qs[:j+1]); small = H[:j+1,:j+1].copy()
        closed = h <= 1e-13 * max(1., float(np.linalg.norm(a)))
        yield Arnoldi(Q, small, np.column_stack(aq)-Q@small, beta, j+1, identity), closed
        if closed or j+1 == min(maximum_depth, len(residual)):
            break
        H[j+1,j] = h; qs.append(v/h)


@dataclass
class Discovery:
    candidate: np.ndarray
    horizon: int
    depth: int
    map_calls: int
    tangent_actions: int
    accepted: bool
    sampled_score: float
    trials: list


def discover(model, state, depths=(2,4,8,12), horizons=(64,32,16,8),
             tolerance=.05, minimum_work_gain=1.5):
    """First depth / longest horizon passing live-path and work screens.

    Probes at quarter, half, and penultimate predicted states; reject as soon
    as one falsifies. The m*max(defect)/displacement score would bound relative
    displacement error for a nonexpansive map IF it bounded ALL path defects.
    Here it is merely a sampled screen. Accepted state is followed by one
    ordinary settling step by the caller; that future call is budgeted here.
    """
    lin = model.linearize(state); calls = 1; actions = 0; trials = []
    best_score = float('inf')
    for basis, closed in arnoldi_stream(lin.action, lin.next_state-state, max(depths)):
        actions = basis.actions
        if actions not in depths and not closed:
            continue
        cache = {}
        for m in horizons:
            indices = sorted(set([m//4, m//2, m-1]))
            missing = sum(i not in cache for i in indices)
            if m < minimum_work_gain * (calls+actions+missing+1):
                continue
            delta = basis.displacement(m); scale = max(np.linalg.norm(delta), 1e-30)
            score = 0.
            for i in indices:
                if i not in cache:
                    vi = basis.displacement(i)
                    vp = basis.displacement(i+1)
                    defect = model.step(state+vi)-(state+vp); calls += 1
                    cache[i] = float(np.linalg.norm(defect))
                score = max(score, m*cache[i]/scale)
                if not np.isfinite(score) or score > tolerance:
                    break
            best_score = min(best_score, score)
            passed = bool(np.isfinite(score) and score <= tolerance)
            trials.append({'depth':actions,'horizon':m,'score':float(score),
                           'passed':passed,'algebraic_closure':bool(closed),
                           'work_including_settling':calls+actions+1})
            if passed:
                return Discovery(state+delta,m,actions,calls,actions,True,score,trials)
        if closed:
            break
    return Discovery(lin.next_state,1,actions,calls,actions,False,best_score,trials)
