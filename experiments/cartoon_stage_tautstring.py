#!/usr/bin/env python3
"""Does an exactly-solved subproblem find the jump set faster than a shrink?

`notes/cartoon_stage_problem_statement.md` says the measured difficulty is
that the jump set is discovered slowly -- Jaccard 0.88 at the shipped 24
passes, and 0.05 through pass 16 on a synthetic field. Every method we use
reaches that combinatorial object only through pointwise thresholding. The
proposal is that a method whose subproblems are solved *exactly* decides the
jump set instead of approaching it.

The test is made apples-to-apples by running both methods on the **same**
functional. Exact 1-D TV solves the anisotropic problem, so split Bregman is
run with a componentwise shrink rather than its isotropic one. Both then
minimize

    (1/2)||u - g||^2 + lam * ( sum |dx u| + sum |dy u| ),   lam = 1/c

so any difference is method, not problem.

* Method A: split Bregman, the kernel's algorithm with an anisotropic shrink.
* Method B: Douglas-Rachford alternating **exact** 1-D TV along rows and
  columns, by taut string.

The taut string is certified by its duality gap before any result is used.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/cartoon_stage_tautstring.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import gallery  # noqa: E402

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True) if njit is not None else _identity
_parallel = njit(cache=True, parallel=True) if njit is not None else _identity


# ----------------------------------------------------------------------
# Exact 1-D total variation by taut string
# ----------------------------------------------------------------------

@_compile
def _tv1d(y, lam, out, cumulative, path):
    """min_x (1/2)||x - y||^2 + lam * sum |x_{i+1} - x_i|, exactly.

    The dual is |X_i - Y_i| <= lam on the cumulative sums with the last one
    pinned, and the primal solution is the increments of the taut string
    through that tube.  The string leaves its anchor on the steepest feasible
    slope and bends the moment the feasible slope interval empties, at
    whichever earlier point closed it.
    """
    n = y.size
    if n == 1:
        out[0] = y[0]
        return
    total = 0.0
    for i in range(n):
        total += y[i]
        cumulative[i] = total

    anchor = -1
    anchor_value = 0.0
    while anchor < n - 1:
        best_low = -1e300
        best_high = 1e300
        low_at = -1
        high_at = -1
        bend = -1
        bend_low = False
        j = anchor + 1
        while j <= n - 1:
            span = j - anchor
            if j == n - 1:
                low = (cumulative[j] - anchor_value) / span
                high = low
            else:
                low = (cumulative[j] - lam - anchor_value) / span
                high = (cumulative[j] + lam - anchor_value) / span
            if low > best_high:
                bend = high_at
                bend_low = False
                break
            if high < best_low:
                bend = low_at
                bend_low = True
                break
            if low > best_low:
                best_low = low
                low_at = j
            if high < best_high:
                best_high = high
                high_at = j
            j += 1
        if bend < 0:
            slope = (cumulative[n - 1] - anchor_value) / (n - 1 - anchor)
            for t in range(anchor + 1, n):
                path[t] = anchor_value + slope * (t - anchor)
            anchor = n - 1
        else:
            slope = best_low if bend_low else best_high
            for t in range(anchor + 1, bend + 1):
                path[t] = anchor_value + slope * (t - anchor)
            anchor_value = path[bend]
            anchor = bend

    out[0] = path[0]
    for i in range(1, n):
        out[i] = path[i] - path[i - 1]


@_parallel
def _tv_rows(a, lam, out):
    h, w = a.shape
    for i in prange(h):
        work = np.empty(w, dtype=np.float64)
        cumulative = np.empty(w, dtype=np.float64)
        path = np.empty(w, dtype=np.float64)
        _tv1d(np.ascontiguousarray(a[i]), lam, work, cumulative, path)
        for j in range(w):
            out[i, j] = work[j]


def tv_rows(a, lam):
    out = np.empty_like(a)
    _tv_rows(np.ascontiguousarray(a), lam, out)
    return out


def tv_cols(a, lam):
    return tv_rows(np.ascontiguousarray(a.T), lam).T


def certify(y, lam, x):
    """Duality gap of the 1-D solution.  Zero certifies optimality.

    The dual of min (1/2)||x-y||^2 + lam||Dx||_1 has value
    (1/2)||y||^2 - (1/2)||x||^2 at the primal-feasible x, so the gap reduces
    to <x, x - y> + lam*||Dx||_1 with no adjoint sign convention to get
    wrong.  Sanity check by hand: y = (0, 1), lam = 0.1 gives x = (0.1, 0.9),
    <x, x-y> = -0.08 and lam||Dx||_1 = 0.08.

    Dual feasibility is reported separately from the tube residual
    p = cumsum(y) - cumsum(x), which must lie in [-lam, lam].
    """
    differences = np.diff(x)
    p = (np.cumsum(y) - np.cumsum(x))[:-1]
    feasibility = float(np.max(np.abs(p)) - lam)
    gap = float(np.dot(x, x - y) + lam * np.sum(np.abs(differences)))
    # Normalize by the problem's own scale, not by the primal value: on a
    # constant input the primal is pure rounding and dividing by it turns
    # 1e-14 into 1.
    scale = float(0.5 * np.dot(y, y)) + 1e-30
    return feasibility, gap, scale


# ----------------------------------------------------------------------
# Method A: split Bregman, anisotropic shrink
# ----------------------------------------------------------------------

def _grad(u):
    """Forward differences with free ends, matching the 1-D solver.

    The taut string treats each line independently with no wraparound, so
    split Bregman must use the same boundary condition or the two methods
    are minimizing different functionals and the comparison is void.
    """
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div(px, py):
    """The negative adjoint of `_grad`."""
    out = np.zeros_like(px)
    out[:, 0] = px[:, 0]
    out[:, 1:] = px[:, 1:] - px[:, :-1]
    out[0, :] += py[0, :]
    out[1:, :] += py[1:, :] - py[:-1, :]
    return out


def _laplacian_symbol(h, w):
    lx = 2.0 * np.cos(np.pi * np.arange(w) / w) - 2.0
    ly = 2.0 * np.cos(np.pi * np.arange(h) / h) - 2.0
    return ly[:, None] + lx[None, :]


def _solve_neumann(rhs, c, eta):
    from scipy.fft import dctn, idctn
    h, w = rhs.shape
    symbol = 1.0 / (c - eta * _laplacian_symbol(h, w))
    return idctn(dctn(rhs, type=2, norm="ortho") * symbol,
                 type=2, norm="ortho")


def check_operators(h=37, w=53):
    """div(grad(u)) must be the Laplacian the DCT symbol claims."""
    from scipy.fft import dctn, idctn
    rng = np.random.default_rng(1)
    u = rng.standard_normal((h, w))
    direct = _div(*_grad(u))
    spectral = idctn(dctn(u, type=2, norm="ortho") *
                     _laplacian_symbol(h, w), type=2, norm="ortho")
    return float(np.max(np.abs(direct - spectral)) /
                 max(float(np.max(np.abs(spectral))), 1e-30))


def split_bregman_aniso(g, c, eta, iterations):
    bx = np.zeros_like(g)
    by = np.zeros_like(g)
    px = np.zeros_like(g)
    py = np.zeros_like(g)
    theta = 1.0 / eta
    trace = []
    for _ in range(iterations):
        u = _solve_neumann(c * g - eta * _div(px, py), c, eta)
        gx, gy = _grad(u)
        tx, ty = gx + bx, gy + by
        dx = np.sign(tx) * np.maximum(np.abs(tx) - theta, 0.0)
        dy = np.sign(ty) * np.maximum(np.abs(ty) - theta, 0.0)
        bx, by = tx - dx, ty - dy
        px, py = dx - bx, dy - by
        trace.append(u.copy())
    return trace


# ----------------------------------------------------------------------
# Method B: Douglas-Rachford over exact 1-D solves
# ----------------------------------------------------------------------

def douglas_rachford(g, c, iterations, gamma=1.0):
    """prox of (fidelity + row TV) alternated with prox of column TV."""
    lam = 1.0 / c
    z = g.copy()
    trace = []
    for _ in range(iterations):
        target = (gamma * g + z) / (1.0 + gamma)
        u = tv_rows(target, gamma * lam / (1.0 + gamma))
        v = tv_cols(2.0 * u - z, gamma * lam)
        z = z + v - u
        trace.append(u.copy())
    return trace


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------

def objective(u, g, c):
    lam = 1.0 / c
    gx, gy = _grad(u)
    return float(0.5 * np.sum((u - g) ** 2) +
                 lam * (np.sum(np.abs(gx)) + np.sum(np.abs(gy))))


def jump_set(u, cut):
    gx, gy = _grad(u)
    return (np.abs(gx) + np.abs(gy)) > cut


def jaccard(a, b):
    return float(np.count_nonzero(a & b)) / max(np.count_nonzero(a | b), 1)


def compare(name, g, c, eta, iterations=64):
    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    bregman = split_bregman_aniso(g, c, eta, iterations)
    bregman_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rachford = douglas_rachford(g, c, iterations)
    rachford_s = time.perf_counter() - t0

    reference = min([bregman[-1], rachford[-1]],
                    key=lambda u: objective(u, g, c))
    deep = douglas_rachford(g, c, 3 * iterations)[-1]
    if objective(deep, g, c) < objective(reference, g, c):
        reference = deep
    scale = float(np.linalg.norm(reference))
    magnitude = np.abs(_grad(reference)[0]) + np.abs(_grad(reference)[1])
    cut = float(np.percentile(magnitude, 95.0))
    target = jump_set(reference, cut)
    print(f"  objective at {iterations} iterations: "
          f"split Bregman {objective(bregman[-1], g, c):.6e}   "
          f"Douglas-Rachford {objective(rachford[-1], g, c):.6e}")
    print(f"  wall time: split Bregman {bregman_s:.2f}s   "
          f"Douglas-Rachford {rachford_s:.2f}s "
          f"({rachford_s / max(iterations, 1) * 1e3:.0f} ms/iter vs "
          f"{bregman_s / max(iterations, 1) * 1e3:.0f})")

    print(f"  {'iter':>5s} {'Bregman err':>12s} {'B jaccard':>10s}   "
          f"{'DR err':>12s} {'DR jaccard':>11s}")
    marks = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
    rows = []
    for k in range(iterations):
        be = float(np.linalg.norm(bregman[k] - reference)) / scale
        re = float(np.linalg.norm(rachford[k] - reference)) / scale
        bj = jaccard(jump_set(bregman[k], cut), target)
        rj = jaccard(jump_set(rachford[k], cut), target)
        rows.append((k + 1, be, bj, re, rj))
        if k + 1 in marks:
            print(f"  {k + 1:5d} {be:12.3e} {bj:10.4f}   "
                  f"{re:12.3e} {rj:11.4f}")

    for level in (0.90, 0.95, 0.98):
        b = next((r[0] for r in rows if r[2] >= level), None)
        d = next((r[0] for r in rows if r[4] >= level), None)
        speed = (f"{b / d:.1f}x fewer" if b and d and d else "n/a")
        print(f"    jaccard {level:.2f}: split Bregman {b}, "
              f"Douglas-Rachford {d}   ({speed} iterations)")
    return rows


def main():
    print("== certifying the exact 1-D solver ==")
    rng = np.random.default_rng(17)
    worst_gap = 0.0
    worst_feasible = -1e30
    for trial in range(200):
        n = int(rng.integers(2, 400))
        y = rng.standard_normal(n) * rng.uniform(0.1, 50.0)
        if trial % 3 == 0:
            y = np.repeat(rng.standard_normal(max(n // 8, 1)) * 20.0,
                          8)[:n].astype(np.float64)
        n = y.size
        lam = float(rng.uniform(0.01, 40.0))
        out = np.empty(n)
        _tv1d(np.ascontiguousarray(y), lam, out,
              np.empty(n), np.empty(n))
        feasible, gap, primal = certify(y, lam, out)
        worst_gap = max(worst_gap, abs(gap) / max(primal, 1e-30))
        worst_feasible = max(worst_feasible, feasible / max(lam, 1e-30))
    print(f"  200 random problems: worst relative duality gap "
          f"{worst_gap:.3e}, worst dual infeasibility "
          f"{worst_feasible:.3e}")
    assert worst_gap < 1e-10 and worst_feasible < 1e-10
    operator_gap = check_operators()
    print(f"  certified optimal; div(grad) matches its DCT symbol to "
          f"{operator_gap:.2e}, so both methods minimize the same "
          f"functional\n")
    assert operator_gap < 1e-12

    side = 512
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:side, 0:side]
    y, x = yy / (side - 1), xx / (side - 1)
    synthetic = np.zeros((side, side))
    for cy, cx, r, v in ((0.30, 0.35, 0.22, 200.0),
                         (0.68, 0.62, 0.18, 60.0)):
        synthetic[(y - cy) ** 2 + (x - cx) ** 2 < r * r] = v
    synthetic += 40.0 * x
    synthetic += 18.0 * np.sin(2 * np.pi * 26 * x) * np.cos(
        2 * np.pi * 21 * y)
    synthetic += 6.0 * rng.standard_normal((side, side))
    synthetic = np.clip(synthetic, 0.0, 255.0)

    compare("synthetic cartoon + texture", synthetic, 0.05, 0.10)
    camera = np.asarray(gallery.load("camera"), dtype=np.float64)
    compare("cameraman", camera, 0.05, 0.10)


if __name__ == "__main__":
    main()
