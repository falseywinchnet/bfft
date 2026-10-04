"""Experimental finite transport closures for positive-kernel Sinkhorn.

General kernels: weighted tangent or quadratic conditional-moment closure,
with a complete-state trajectory bound and an actual marginal-error gate.
Known exact two-block kernels: projective matrix powers.
No fixed-point linear solve, fitted optimum, or uncharged kernel oracle.
"""
from dataclasses import dataclass
import time
import numpy as np


def qnorm(v):
    """Infinity norm modulo the constant scaling gauge."""
    return float(np.ptp(v) / 2)


class Kernel:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(self.matrix)) or np.any(self.matrix <= 0):
            raise ValueError('The kernel must be finite and strictly positive.')
        self.calls = 0
        self.vector_products = 0

    def apply(self, v, transpose=False):
        self.calls += 1
        self.vector_products += 1 if v.ndim == 1 else v.shape[1]
        return (self.matrix.T if transpose else self.matrix) @ v


@dataclass
class State:
    y: np.ndarray
    f: np.ndarray
    d: np.ndarray
    v: np.ndarray
    u: np.ndarray
    kv: np.ndarray
    ktu: np.ndarray
    c: np.ndarray
    residual: float


def evaluate(kernel, a, b, y):
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    v = np.exp(y)
    kv = kernel.apply(v)
    u = a / kv
    ktu = kernel.apply(u, transpose=True)
    f = np.log(b) - np.log(ktu)
    d = f-y
    c = v*ktu
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise FloatingPointError('Scaling left the supported floating-point range.')
    return State(y, f, d, v, u, kv, ktu, c,
                 float(np.max(np.abs(c/b-1))))


def conditional(kernel, state, values, reverse=False):
    vector = values.ndim == 1
    scale = state.u if reverse else state.v
    denominator = state.ktu if reverse else state.kv
    if vector:
        return kernel.apply(scale*values, transpose=reverse)/denominator
    return kernel.apply(scale[:, None]*values, transpose=reverse)/denominator[:, None]


@dataclass
class Model:
    U: np.ndarray
    Jcols: np.ndarray
    r: np.ndarray
    H: np.ndarray
    second: np.ndarray
    reduced_second: np.ndarray
    pairs: list
    d: np.ndarray
    quadratic: bool

    def full_and_reduced(self, coords):
        full = self.d + self.Jcols@coords
        reduced = self.r + self.H@coords
        if self.quadratic:
            products = np.array([coords[i]*coords[j]*(.5 if i == j else 1.)
                                 for i, j in self.pairs])
            full = full + self.second@products
            reduced = reduced + self.reduced_second@products
        return full, reduced

    def reduced_derivative(self, coords):
        out = self.H.copy()
        if self.quadratic:
            for p, (i, j) in enumerate(self.pairs):
                out[:, i] += self.reduced_second[:, p]*coords[j]
                if i != j:
                    out[:, j] += self.reduced_second[:, p]*coords[i]
        return out


def prepare(kernel, state, depth=4, quadratic=True):
    """Weighted Krylov acquisition, followed by actual conditional moments."""
    weights = state.c/state.c.sum()
    root = np.sqrt(weights)
    w = root*state.d
    w -= root*(root@w)
    basis, columns, pcolumns = [], [], []
    for _ in range(depth):
        # Twice reorthogonalize, including the known constant gauge mode.
        for _ in range(2):
            w -= root*(root@w)
            for col in basis:
                w -= col*(col@w)
        size = np.linalg.norm(w)
        if size < 1e-13:
            break
        z = w/size
        basis.append(z)
        direction = z/root
        pd = conditional(kernel, state, direction)
        jd = conditional(kernel, state, pd, reverse=True)
        pcolumns.append(pd)
        columns.append(jd)
        w = root*jd
    if not basis:
        return None
    U = np.column_stack(basis)/root[:, None]
    JU = np.column_stack(columns)
    PU = np.column_stack(pcolumns)
    projection = U.T*weights[None, :]
    pairs = [(i, j) for i in range(U.shape[1]) for j in range(i, U.shape[1])]
    second = np.empty((len(state.y), 0))
    reduced_second = np.empty((U.shape[1], 0))
    if quadratic:
        products = np.column_stack([U[:, i]*U[:, j] for i, j in pairs])
        pproducts = conditional(kernel, state, products)
        row_products = np.column_stack([PU[:, i]*PU[:, j] for i, j in pairs])
        second = conditional(kernel, state, pproducts-2*row_products, reverse=True)
        second += np.column_stack([JU[:, i]*JU[:, j] for i, j in pairs])
        reduced_second = projection@second
    return Model(U, JU, projection@state.d, projection@JU, second,
                 reduced_second, pairs, state.d.copy(), quadratic)


def rollout(model, max_horizon=128, relative_budget=.05):
    """Bound the whole trajectory, including subspace escape, in gauge norm.

    Exact arithmetic bound: quadratic Taylor remainder <= 11 osc(s)^3/96.
    It is combined with the full-vector projection defect at EVERY step.
    No kernel evaluations occur in this rollout.
    """
    coords = np.zeros(len(model.r))
    bound = projection_bound = remainder_bound = 0.
    best = None
    for h in range(1, max_horizon+1):
        s = model.U@coords
        spread = float(np.ptp(s))
        # Unrestricted diagnostic rollouts can diverge. Stop before a
        # cubic bound itself overflows; ordinary admission stops far earlier.
        if not np.isfinite(spread) or spread > 1e100:
            break
        full, next_coords = model.full_and_reduced(coords)
        next_s = model.U@next_coords
        projection_error = qnorm(full-next_s)
        nonlinear_error = (11/96*spread**3 if model.quadratic else spread**2/8)
        projection_bound += projection_error
        remainder_bound += nonlinear_error
        bound = projection_bound+remainder_bound
        if not np.isfinite(bound) or bound > relative_budget*max(qnorm(next_s), 1e-15):
            break
        best = dict(displacement=next_s.copy(), horizon=h, bound=bound,
                    projection_bound=projection_bound, remainder_bound=remainder_bound,
                    relative_bound=bound/max(qnorm(next_s), 1e-300), coords=next_coords.copy())
        coords = next_coords
    return best


def solve(matrix, a, b, method='ordinary', tolerance=1e-9, max_evaluations=20000,
          depth=4, max_horizon=128, relative_budget=.05, cooldown=8, y0=None):
    """Return scaling coordinates and a real full-kernel marginal certificate.

    All evaluations, model acquisition, rejected trials, bounds, and checks
    are inside the timer. Input kernel creation is common to every method.
    """
    start = time.perf_counter()
    kernel = Kernel(matrix)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.any(a <= 0) or np.any(b <= 0) or not np.isclose(a.sum(), b.sum()):
        raise ValueError('Marginals must be positive and have equal mass.')
    if method not in ('ordinary', 'linear', 'quadratic', 'quadratic2'):
        raise ValueError(method)
    if method == 'quadratic2':
        depth = 2
    quadratic = method.startswith('quadratic')
    y = np.zeros(len(b)) if y0 is None else np.asarray(y0, dtype=float).copy()
    state = evaluate(kernel, a, b, y)
    evaluations, acquisitions, accepted, rejected = 1, 0, 0, 0
    horizons, bounds = [], []
    wait = cooldown
    min_horizon = 32 if quadratic and depth > 2 else 16
    while state.residual > tolerance and evaluations < max_evaluations:
        trial = None
        if method != 'ordinary' and wait <= 0:
            acquisitions += 1
            model = prepare(kernel, state, depth, quadratic)
            proposal = None if model is None else rollout(model, max_horizon, relative_budget)
            if proposal is not None and proposal['horizon'] >= min_horizon:
                trial = evaluate(kernel, a, b, state.y+proposal['displacement'])
                evaluations += 1
                if trial.residual < state.residual:
                    accepted += 1
                    horizons.append(proposal['horizon'])
                    bounds.append(proposal['relative_bound'])
                    state = trial
                    wait = 0
                    continue
                rejected += 1
            wait = cooldown
        state = evaluate(kernel, a, b, state.f)
        evaluations += 1
        wait -= 1
    return dict(y=state.y, u=state.u, v=state.v, residual=state.residual,
                converged=state.residual <= tolerance, seconds=time.perf_counter()-start,
                evaluations=evaluations, acquisitions=acquisitions, accepted=accepted,
                rejected=rejected, horizons=horizons, relative_bounds=bounds,
                kernel_calls=kernel.calls, kernel_vector_products=kernel.vector_products)


def mobius_matrix(C, A, B):
    p, q = C[0]
    r, s = C[1]
    ratio = B[0]/B[1]
    return np.array([[ratio*(q*A[0]*r+s*A[1]*p), ratio*q*s*sum(A)],
                     [p*r*sum(A), p*A[0]*s+r*A[1]*q]])


def projective_power(M, w, horizon):
    """Normalized binary powering; scales of matrices/vectors are irrelevant."""
    M = M/np.max(M)
    w = w/np.sum(w)
    while horizon:
        if horizon & 1:
            w = M@w
            w /= np.sum(w)
        horizon //= 2
        if horizon:
            M = M@M
            M /= np.max(M)
    return w


def solve_blocks(C, g, h, row_factor, col_factor, a, b, method='power',
                 tolerance=1e-9, max_steps=100000, y0=None):
    """Known EXACT two-group structure, with setup and reconstruction charged.

    The ordinary comparator uses the SAME exact two-group reduction.
    No dense-kernel work is needed or claimed saved by the power comparison.
    """
    start = time.perf_counter()
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2) or np.any(C <= 0):
        raise ValueError('Requires a positive 2x2 block kernel.')
    if method not in ('power', 'ordinary'):
        raise ValueError(method)
    A, B = np.bincount(g, weights=a), np.bincount(h, weights=b)
    v0 = np.ones(len(b)) if y0 is None else np.exp(y0-np.mean(y0))
    w = np.bincount(h, weights=col_factor*v0)
    # Enter the invariant manifold before reconstructing the full v.
    w = B/(C.T@(A/(C@w)))
    w /= w.sum()
    steps, checks, jumps = 1, 0, 0
    M = mobius_matrix(C, A, B)
    horizon = 16
    while True:
        cw = C@w
        kt = C.T@(A/cw)
        residual = float(np.max(np.abs(w*kt/B-1)))
        checks += 1
        if residual <= tolerance or steps >= max_steps:
            break
        if method == 'ordinary':
            w = B/kt
            w /= w.sum()
            steps += 1
        else:
            count = min(horizon, max_steps-steps)
            w = projective_power(M, w, count)
            steps += count
            jumps += 1
            horizon *= 2
    v = (b/col_factor)*(w/B)[h]
    # Return the row-normalized coupling's pair, matching general solve().
    u = a/(row_factor*(C@w)[g])
    y = np.log(v)
    return dict(y=y, u=u, v=v, residual=residual, converged=residual <= tolerance,
                seconds=time.perf_counter()-start, ordinary_horizon=steps,
                checks=checks, jumps=jumps)
