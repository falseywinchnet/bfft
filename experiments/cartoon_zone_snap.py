#!/usr/bin/env python3
"""An isotropic globally-decisive step: snap the flat-zone levels.

The functional stays exactly isotropic. Nothing here approximates
sqrt(ux^2 + uy^2) by anything.

The diagnosis in `notes/cartoon_stage_problem_statement.md` was that split
Bregman's late error is the *levels of the flat zones* -- low frequency, in
the interiors -- and that its only global operator is the resolvent, which
smooths rather than decides. The taut string beat it by making a decision
propagate a whole line at once; that needed anisotropy, and the picture said
no.

But the flat zones are themselves a coarse grid, and unlike a geometric
coarsening they respect edges exactly, because they are bounded by them.
Restricting u to be constant on the current zones turns the fine problem into

    min_a  (c/2) sum_k n_k (a_k - mean_k g)^2  +  sum_{j~k} P_jk |a_j - a_k|

a weighted total variation on the zone adjacency graph with a few thousand
unknowns instead of a few hundred thousand. Solving it sets every plateau
level at once. `P_jk` is the shared boundary length and `n_k` the zone size,
both counted from the partition.

This is a coarse-grid correction whose coarse space is the current jump set.
It is used as a *proposal*: the true fine isotropic objective is evaluated
before and after, and the snap is kept only if it decreases. So the scheme
cannot converge anywhere the isotropic problem does not.

What this file measures is only the premise -- given the jump set that split
Bregman has at pass p, how much objective does an exact level assignment buy,
and how many Bregman passes is that worth?

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_zone_snap.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import gallery  # noqa: E402
from bfft.effects import srgb_to_lab  # noqa: E402

from cartoon_stage_tautstring import _div, _grad, _solve_neumann  # noqa: E402


def isotropic_objective(u, g, c):
    gx, gy = _grad(u)
    return float(0.5 * c * np.sum((u - g) ** 2) +
                 np.sum(np.sqrt(gx * gx + gy * gy)))


def split_bregman_iso(g, c, eta, iterations):
    """The shipped algorithm, isotropic shrink, Neumann boundaries."""
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
        magnitude = np.sqrt(tx * tx + ty * ty)
        shrink = np.where(
            magnitude > theta,
            1.0 - theta / np.maximum(magnitude, 1e-300), 0.0)
        dx, dy = tx * shrink, ty * shrink
        bx, by = tx - dx, ty - dy
        px, py = dx - bx, dy - by
        trace.append(u.copy())
    return trace


def flat_zones(u, percentile):
    """Connected components of the pixel graph with strong links cut."""
    h, w = u.shape
    index = np.arange(h * w).reshape(h, w)
    right = np.abs(u[:, 1:] - u[:, :-1]).ravel()
    down = np.abs(u[1:, :] - u[:-1, :]).ravel()
    cut = float(np.percentile(np.concatenate([right, down]), percentile))
    ra = index[:, :-1].ravel()[right <= cut]
    rb = index[:, 1:].ravel()[right <= cut]
    da = index[:-1, :].ravel()[down <= cut]
    db = index[1:, :].ravel()[down <= cut]
    rows = np.concatenate([ra, da])
    cols = np.concatenate([rb, db])
    graph = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(h * w, h * w)).tocsr()
    count, labels = connected_components(graph, directed=False)
    return count, labels.reshape(h, w)


def zone_problem(labels, count, g, c):
    """Sizes, targets, and the adjacency with shared boundary lengths."""
    flat = labels.ravel()
    sizes = np.bincount(flat, minlength=count).astype(np.float64)
    sums = np.bincount(flat, weights=g.ravel(), minlength=count)
    means = sums / np.maximum(sizes, 1.0)

    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :], labels[1:, :])):
        differing = a != b
        low = np.minimum(a[differing], b[differing])
        high = np.maximum(a[differing], b[differing])
        pairs.append(low.astype(np.int64) * count + high.astype(np.int64))
    keys, lengths = np.unique(np.concatenate(pairs), return_counts=True)
    edge_a = (keys // count).astype(np.int64)
    edge_b = (keys % count).astype(np.int64)
    return sizes, means, edge_a, edge_b, lengths.astype(np.float64)


def solve_zone_tv(sizes, means, edge_a, edge_b, lengths, c,
                  iterations=4000):
    """Weighted TV on the zone graph, by Chambolle-Pock.

    Small enough -- thousands of unknowns, not hundreds of thousands -- that
    running it far past convergence costs nothing.
    """
    n = sizes.size
    m = edge_a.size
    if m == 0:
        return means.copy()
    degree = np.bincount(np.concatenate([edge_a, edge_b]), minlength=n)
    norm = np.sqrt(2.0 * max(float(degree.max()), 1.0))
    tau = 1.0 / norm
    sigma = 1.0 / norm

    a = means.copy()
    bar = a.copy()
    p = np.zeros(m)
    for _ in range(iterations):
        p = p + sigma * (bar[edge_b] - bar[edge_a])
        p = np.clip(p, -lengths, lengths)
        flux = np.zeros(n)
        np.add.at(flux, edge_a, -p)
        np.add.at(flux, edge_b, p)
        previous = a
        a = (a - tau * flux + tau * c * sizes * means) / (
            1.0 + tau * c * sizes)
        bar = 2.0 * a - previous
    return a


def zone_correction(u, g, c, labels, count, iterations=4000):
    """Additive coarse correction: shift each zone, keep its interior.

    The replacement form above fails because a ROF solution is *not*
    piecewise constant on its zones -- Pikachu's body and the cameraman's sky
    carry smooth shading the fidelity term wants, and flattening a zone
    destroys it.  Multigrid corrects additively.  So solve for offsets,

        u_new = u + sum_k delta_k * 1_{Z_k}

    which leaves every interior untouched and changes the gradient only on
    zone boundaries.  With r = u - g and Delta_jk the current mean jump
    across the boundary between j and k, the coarse problem is

        min_d  (c/2) sum_k n_k (d_k + mean_k r)^2
               + sum_{j~k} P_jk |Delta_jk + d_k - d_j|

    a shifted fused lasso on the zone graph.  It is a model, not the truth,
    so the fine isotropic objective decides whether the step is taken.
    """
    flat = labels.ravel()
    residual = (u - g).ravel()
    sizes = np.bincount(flat, minlength=count).astype(np.float64)
    mean_residual = (np.bincount(flat, weights=residual, minlength=count) /
                     np.maximum(sizes, 1.0))

    keys = []
    jumps = []
    for a, b, ua, ub in ((labels[:, :-1], labels[:, 1:],
                          u[:, :-1], u[:, 1:]),
                         (labels[:-1, :], labels[1:, :],
                          u[:-1, :], u[1:, :])):
        differing = a != b
        first, second = a[differing], b[differing]
        step = (ub - ua)[differing]
        swap = first > second
        low = np.where(swap, second, first).astype(np.int64)
        high = np.where(swap, first, second).astype(np.int64)
        keys.append(low * count + high)
        jumps.append(np.where(swap, -step, step))
    keys = np.concatenate(keys)
    jumps = np.concatenate(jumps)
    order = np.argsort(keys, kind="stable")
    keys, jumps = keys[order], jumps[order]
    unique, start = np.unique(keys, return_index=True)
    lengths = np.diff(np.append(start, keys.size)).astype(np.float64)
    mean_jump = np.add.reduceat(jumps, start) / lengths
    edge_a = (unique // count).astype(np.int64)
    edge_b = (unique % count).astype(np.int64)

    n, m = count, edge_a.size
    if m == 0:
        return np.zeros(n)
    degree = np.bincount(np.concatenate([edge_a, edge_b]), minlength=n)
    norm = np.sqrt(2.0 * max(float(degree.max()), 1.0))
    tau = sigma = 1.0 / norm
    delta = np.zeros(n)
    bar = delta.copy()
    p = np.zeros(m)
    for _ in range(iterations):
        p = np.clip(
            p + sigma * (mean_jump + bar[edge_b] - bar[edge_a]),
            -lengths, lengths)
        flux = np.zeros(n)
        np.add.at(flux, edge_a, -p)
        np.add.at(flux, edge_b, p)
        previous = delta
        delta = (delta - tau * flux - tau * c * sizes * mean_residual) / (
            1.0 + tau * c * sizes)
        bar = 2.0 * delta - previous
    return delta


def report(name, g, c, eta, iterations=64, percentile=95.0):
    print(f"\n=== {name} ===")
    trace = split_bregman_iso(g, c, eta, iterations)
    objectives = [isotropic_objective(u, g, c) for u in trace]
    best = objectives[-1]
    print(f"  isotropic objective: pass 1 {objectives[0]:.6e}   "
          f"pass {iterations} {best:.6e}")
    print(f"  {'pass':>5s} {'objective':>13s} {'zones':>7s} "
          f"{'replace':>13s} {'correct':>13s} {'alpha':>6s} "
          f"{'worth':>6s} {'time':>7s}")
    for p in (4, 8, 12, 16, 24, 32):
        u = trace[p - 1]
        started = time.perf_counter()
        count, labels = flat_zones(u, percentile)
        sizes, means, ea, eb, lengths = zone_problem(labels, count, g, c)
        replaced = isotropic_objective(
            solve_zone_tv(sizes, means, ea, eb, lengths, c)[labels], g, c)
        delta = zone_correction(u, g, c, labels, count)
        before = objectives[p - 1]
        best, best_alpha = before, 0.0
        for alpha in (1.0, 0.5, 0.25, 0.125):
            candidate = isotropic_objective(u + alpha * delta[labels], g, c)
            if candidate < best:
                best, best_alpha = candidate, alpha
        elapsed = time.perf_counter() - started
        equivalent = next(
            (k + 1 for k in range(iterations) if objectives[k] <= best),
            None)
        mark = "" if best_alpha > 0.0 else "  (rejected)"
        print(f"  {p:5d} {before:13.6e} {count:7d} {replaced:13.6e} "
              f"{best:13.6e} {best_alpha:6.3f} "
              f"{str(equivalent):>6s} {elapsed:6.2f}s{mark}")


def main():
    rgb = gallery.load("pikachu")
    light = np.ascontiguousarray(
        srgb_to_lab(np.asarray(rgb, dtype=np.float64))[..., 0] * 255.0)
    light = light[::2, ::2]
    report(f"pikachu {light.shape[0]}x{light.shape[1]}", light, 0.05, 0.10)

    camera = np.asarray(gallery.load("camera"), dtype=np.float64)
    report(f"cameraman {camera.shape[0]}x{camera.shape[1]}",
           camera, 0.05, 0.10)


if __name__ == "__main__":
    main()
