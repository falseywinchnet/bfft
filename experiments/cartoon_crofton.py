#!/usr/bin/env python3
"""Isotropy is a discretization, not a binary. Measure where we actually are.

The visual A/B rejected 4-neighbour anisotropic TV for staircasing. But the
shipped "isotropic" TV is a 4-point forward-difference discretization, and
that is *also* grid-biased -- measured, it puts 57.1% of its edge energy
within 10 degrees of an axis where the target has 51.8%. So the axis is not
isotropic-versus-anisotropic. It is metrication quality, and there are three
points on it, not two.

The Cauchy-Crofton construction gives the third. For a set of lattice
directions e_k with lengths l_k and angular spacings dtheta_k,

    TV(u) ~= sum_k w_k * sum_{lines along e_k} |du|,    w_k = dtheta_k/(2*l_k)

converges to the *true* Euclidean total variation as the direction set grows
(Boykov-Kolmogorov weights). Four directions already halve the anisotropy of
two; eight roughly halves it again.

The point: every term is a one-dimensional total variation along a family of
disjoint lattice lines, so **every subproblem is still solved exactly by the
taut string**. The mechanism that beat split Bregman by 2.8-3.9x on jump-set
discovery survives. What changes is only how many directions it decides along.

So the question this file answers is: does a Crofton cartoon land *closer to
the target's orientation statistics than the shipped isotropic one*? If it
does, it is not a retreat from isotropy -- it is a better discretization of
it, obtained with exact subproblems.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_crofton.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import srgb_to_lab  # noqa: E402

from cartoon_stage_tautstring import _tv1d  # noqa: E402
from cartoon_visual_ab import axis_share  # noqa: E402

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_parallel = njit(cache=True, parallel=True) if njit is not None else _identity


@_parallel
def _tv_segments(data, starts, lengths, lam, out):
    """Exact 1-D total variation on each of many variable-length lines."""
    for i in prange(starts.size):
        start = starts[i]
        n = lengths[i]
        work = np.empty(n, dtype=np.float64)
        cumulative = np.empty(n, dtype=np.float64)
        path = np.empty(n, dtype=np.float64)
        _tv1d(np.ascontiguousarray(data[start:start + n]), lam, work,
              cumulative, path)
        for j in range(n):
            out[start + j] = work[j]


def line_plan(shape, direction):
    """Order the pixels into the disjoint lattice lines along `direction`.

    Lines along (dy, dx) are the level sets of x*dy - y*dx, and position
    within a line is x*dx + y*dy.  Sorting on that pair lists every line
    contiguously, which is what the segment solver wants.
    """
    h, w = shape
    dy, dx = direction
    yy, xx = np.mgrid[0:h, 0:w]
    key = (xx * dy - yy * dx).ravel()
    position = (xx * dx + yy * dy).ravel()
    order = np.lexsort((position, key)).astype(np.int64)
    boundaries = np.flatnonzero(np.diff(key[order])) + 1
    starts = np.concatenate([[0], boundaries]).astype(np.int64)
    lengths = np.diff(np.concatenate(
        [starts, [order.size]])).astype(np.int64)
    keep = lengths >= 2
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return order, inverse, starts[keep], lengths[keep]


def crofton_weights(directions):
    """Boykov-Kolmogorov weights: dtheta / (2 * length)."""
    angles = np.array([np.arctan2(dy, dx) % np.pi for dy, dx in directions])
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(np.concatenate(
        [sorted_angles, [sorted_angles[0] + np.pi]]))
    spacing = np.empty_like(gaps)
    spacing[0] = 0.5 * (gaps[0] + gaps[-1])
    spacing[1:] = 0.5 * (gaps[1:] + gaps[:-1])
    lengths = np.array([np.hypot(dy, dx) for dy, dx in directions])[order]
    weights = spacing / (2.0 * lengths)
    out = np.empty(len(directions))
    out[order] = weights
    return out


DIRECTIONS = {
    2: [(0, 1), (1, 0)],
    4: [(0, 1), (1, 1), (1, 0), (1, -1)],
    8: [(0, 1), (1, 2), (1, 1), (2, 1), (1, 0), (2, -1), (1, -1), (1, -2)],
}


def crofton_rof(g, c, directions, iterations=40):
    """Parallel proximal splitting: fidelity plus one term per direction.

    PPXA over `K + 1` functions.  Each directional prox is a family of exact
    1-D solves, so no term is ever approached by thresholding.
    """
    shape = g.shape
    weights = crofton_weights(directions)
    plans = [line_plan(shape, d) for d in directions]
    k = len(directions) + 1
    flat_g = g.ravel()

    states = [flat_g.copy() for _ in range(k)]
    u = flat_g.copy()
    gamma = 1.0
    for _ in range(iterations):
        proximals = []
        # fidelity term, weight 1/k inside PPXA
        proximals.append(
            (states[0] + gamma * k * c * flat_g) / (1.0 + gamma * k * c))
        for index, (direction, weight) in enumerate(
                zip(directions, weights)):
            order, inverse, starts, lengths = plans[index]
            source = np.ascontiguousarray(states[index + 1][order])
            out = source.copy()
            _tv_segments(source, starts, lengths,
                         float(gamma * k * weight), out)
            proximals.append(out[inverse])
        mean = np.mean(proximals, axis=0)
        for index in range(k):
            states[index] = states[index] + 2.0 * mean - u - proximals[index]
        u = mean
    return u.reshape(shape)


def main():
    rgb = gallery.load("pikachu")
    light = np.ascontiguousarray(
        srgb_to_lab(np.asarray(rgb, dtype=np.float64))[..., 0] * 255.0)
    print(f"pikachu {light.shape[0]}x{light.shape[1]}")
    print(f"  {'variant':34s} {'axis-aligned edge energy':>24s} "
          f"{'gap to target':>14s} {'time':>8s}")
    target_share = axis_share(light)
    print(f"  {'target':34s} {target_share * 100:23.1f}% "
          f"{'-':>14s}")

    t0 = time.perf_counter()
    iso, _ = bfft.meyer_split(light, lam=0.05, mu=40.0, passes=24, threads=4)
    iso_s = time.perf_counter() - t0
    share = axis_share(iso)
    print(f"  {'shipped isotropic (4-point)':34s} {share * 100:23.1f}% "
          f"{(share - target_share) * 100:+13.1f}  {iso_s * 1e3:6.0f} ms")

    for count in (2, 4, 8):
        directions = DIRECTIONS[count]
        t0 = time.perf_counter()
        cartoon = crofton_rof(light, 0.05, directions)
        elapsed = time.perf_counter() - t0
        share = axis_share(cartoon)
        label = (f"Crofton, {count} directions"
                 if count > 2 else "Crofton, 2 directions (anisotropic)")
        print(f"  {label:34s} {share * 100:23.1f}% "
              f"{(share - target_share) * 100:+13.1f}  "
              f"{elapsed * 1e3:6.0f} ms")


if __name__ == "__main__":
    main()
