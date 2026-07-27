#!/usr/bin/env python3
"""Component 3: the per-cell ridge scan.

Baseline: 153 ms, the most expensive warm component in the sigma round.
For each of 16 angles it makes four full-image passes and three
`np.bincount` calls, each allocating an `n * bins` accumulator -- 48 arrays
of 98,400 doubles built and thrown away per call.

Formal target: **the scan is one sinogram, not sixteen histograms.**  The
quantity being maximized,

    g(theta, c) = sum_x phi(x) R(x) sign(n_theta . (x - x_i) - c)

is the cumulative Radon transform of the weighted residual, restricted to a
cell.  Every angle reads the same pixel and the same residual, so the loop
order is wrong: iterating pixels outermost and angles innermost turns 16
sweeps of the image into one, removes every temporary, and -- because
consecutive pixels usually share an owner -- keeps the whole accumulator for
the live cell inside 16 KB of cache.

The sign basis has constant norm inside a cell, which is why the optimal
threshold for a fixed angle is just the extremum of a cumulative sum and no
search is needed in the first place.  That was already true in the baseline;
this only stops paying for it sixteen times.

Tie-breaking is replicated exactly -- first angle wins, first bin wins --
so the chosen axis and offset are identical, not merely equivalent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture  # noqa: E402

from claude_trial_sigma import LAB_WEIGHTS  # noqa: E402

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True, fastmath=False) if njit is not None else _identity


@_compile
def _scan(owner, weight, residual, dx, dy, cosines, sines, spacing,
          n, angles, bins, span, lab_weights):
    """One pass over pixels, all angles innermost."""
    stride = angles * bins * 3
    acc = np.zeros(n * stride, dtype=np.float64)
    mass = np.zeros(n, dtype=np.float64)
    total = np.zeros((n, 3), dtype=np.float64)
    scale = bins / (2.0 * span)

    for p in range(len(owner)):
        cell = owner[p]
        if cell < 0:
            continue
        phi = weight[p]
        r0 = phi * residual[p, 0]
        r1 = phi * residual[p, 1]
        r2 = phi * residual[p, 2]
        mass[cell] += phi
        total[cell, 0] += r0
        total[cell, 1] += r1
        total[cell, 2] += r2
        px = dx[p] / spacing
        py = dy[p] / spacing
        base = cell * stride
        for a in range(angles):
            projection = px * cosines[a] + py * sines[a]
            # int() truncates toward zero, matching the baseline's astype
            # before the clip; the clip makes the two agree on negatives.
            index = int((projection + span) * scale)
            if index < 0:
                index = 0
            elif index >= bins:
                index = bins - 1
            slot = base + (a * bins + index) * 3
            acc[slot] += r0
            acc[slot + 1] += r1
            acc[slot + 2] += r2

    best_score = np.zeros(n, dtype=np.float64)
    best_angle = np.zeros(n, dtype=np.int64)
    best_bin = np.zeros(n, dtype=np.int64)
    for cell in range(n):
        cell_mass = mass[cell]
        if cell_mass < 1e-9:
            cell_mass = 1e-9
        base = cell * stride
        top = 0.0
        top_angle = 0
        top_bin = 0
        seen = False
        for a in range(angles):
            run0 = 0.0
            run1 = 0.0
            run2 = 0.0
            local_top = 0.0
            local_bin = 0
            for b in range(bins):
                slot = base + (a * bins + b) * 3
                run0 += acc[slot]
                run1 += acc[slot + 1]
                run2 += acc[slot + 2]
                c0 = total[cell, 0] - 2.0 * run0
                c1 = total[cell, 1] - 2.0 * run1
                c2 = total[cell, 2] - 2.0 * run2
                value = (lab_weights[0] * c0 * c0 +
                         lab_weights[1] * c1 * c1 +
                         lab_weights[2] * c2 * c2) / cell_mass
                if b == 0 or value > local_top:
                    local_top = value
                    local_bin = b
            if not seen or local_top > top:
                top = local_top
                top_angle = a
                top_bin = local_bin
                seen = True
        best_score[cell] = top
        best_angle[cell] = top_angle
        best_bin[cell] = top_bin
    return best_score, best_angle, best_bin


def measure_ridge_fast(model, residual, ctx):
    n, spacing = ctx["n"], ctx["spacing"]
    bins, angles = model.ridge_bins, model.ridge_angles
    span = 2.5
    weight = ctx["w1"]
    sx = model.seeds[model.owner, 0]
    sy = model.seeds[model.owner, 1]
    dx = model.xf - sx
    dy = model.yf - sy
    thetas = np.linspace(0.0, np.pi, angles, endpoint=False)
    score, angle_index, bin_index = _scan(
        model.owner.astype(np.int64), weight,
        np.ascontiguousarray(residual.reshape(-1, 3)), dx, dy,
        np.cos(thetas), np.sin(thetas), spacing,
        n, angles, bins, span, LAB_WEIGHTS)
    model.ridge_axis = thetas[angle_index]
    model.ridge_offset = (
        (bin_index + 0.5) / bins * (2.0 * span) - span)
    model.ridge_score = score
    return score


def main():
    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400)):
        model = fixture(image, side, cells)
        ctx = model.assemble(4.0)
        solved = model.solve_exact(model.base_lab, ctx)
        residual = model.base_lab - solved["field"]
        print(f"\n{image} {side} px / {cells} cells, "
              f"{model.ridge_angles} angles x {model.ridge_bins} bins")

        measure_ridge_fast(model, residual, ctx)
        base_time, _ = bench(
            "baseline per-angle bincount",
            lambda: model.measure_ridge(residual, ctx))
        axis_a = model.ridge_axis.copy()
        offset_a = model.ridge_offset.copy()
        score_a = model.ridge_score.copy()
        fast_time, _ = bench(
            "fused single pass over pixels",
            lambda: measure_ridge_fast(model, residual, ctx))
        axis_b = model.ridge_axis.copy()
        offset_b = model.ridge_offset.copy()
        score_b = model.ridge_score.copy()
        print(f"  speedup {base_time / fast_time:.1f}x")

        axis_same = int(np.sum(axis_a != axis_b))
        offset_same = int(np.sum(offset_a != offset_b))
        relative = float(np.max(np.abs(score_a - score_b)) /
                         max(float(np.max(score_a)), 1e-30))
        print(f"    axis disagreements {axis_same} / {len(axis_a)}   "
              f"offset disagreements {offset_same}   "
              f"max relative score gap {relative:.3e}")
        assert axis_same == 0 and offset_same == 0
        assert relative < 1e-12


if __name__ == "__main__":
    main()
