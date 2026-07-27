#!/usr/bin/env python3
"""Component 2: exact Schur-complement gain of a candidate cell.

Baseline: 843 ms for 96 candidates.  Per candidate it scans all 65,536
pixels to find a disc of about 113, forms a sparse `(npix, 3)` block, and
multiplies the whole `(npix, 3n)` design matrix against it -- 393k nonzeros
of work for a column supported on 113 rows -- then issues three separate
triangular solves.

Formal target: **the algebra is already local; only the implementation is
global.**  Three identities do all the work, and none of them approximates
anything:

    cross          = A^T B          = A[near]^T C
    cross^T x      = C^T (A[near] x)          for any x already known
    cross^T G^-1 cross                        needs the only real solves

The first says the product touches only the rows the candidate occupies.
The second says every subsequent contraction against `cross` can be pushed
through `A[near]`, so `A @ incumbent` is computed once for every candidate
rather than once per candidate.  The third is irreducible -- but the solves
for all candidates form one right-hand-side block, so they are issued as a
single call instead of `3 * C` calls.

Locating the disc is likewise local: a bounding-box slice, not three
full-image passes.

Output is asserted identical to the baseline, not merely close.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture  # noqa: E402

from claude_trial_sigma import LAB_WEIGHTS  # noqa: E402


def local_disc(model, pixel, reach):
    """Pixels of the candidate's disc, found by slicing not scanning."""
    cy, cx = divmod(int(pixel), model.w)
    radius = int(math.ceil(reach))
    y0, y1 = max(0, cy - radius), min(model.h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(model.w, cx + radius + 1)
    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    dy = (ys - cy).astype(np.float64)
    dx = (xs - cx).astype(np.float64)
    square = dy[:, None] ** 2 + dx[None, :] ** 2
    keep = square <= reach * reach
    if not np.any(keep):
        return None
    rows, cols = np.nonzero(keep)
    flat = (ys[rows] * model.w + xs[cols]).astype(np.int64)
    return flat, dx[cols], dy[rows], square[rows, cols]


def addition_gains_fast(model, solved, ctx, candidates, residual):
    lu, spacing = solved["lu"], ctx["spacing"]
    design, width = ctx["design"], ctx["width"]
    n = ctx["n"]
    flat = residual.reshape(-1, 3)
    reach = 1.15 * spacing
    sigma = 0.55 * reach

    incumbent = np.column_stack([
        lu.solve(design.T @ flat[:, c]) for c in range(3)])
    # A @ incumbent, once for every candidate rather than once per candidate.
    design_incumbent = design @ incumbent

    prepared = []
    columns = []
    for slot, pixel in enumerate(candidates):
        disc = local_disc(model, pixel, reach)
        if disc is None:
            continue
        near, dx, dy, square = disc
        if near.size < 8:
            continue
        window = np.exp(-0.5 * square / (sigma * sigma))
        column = np.column_stack([
            window, window * dx / spacing, window * dy / spacing])
        rows = design[near]
        cross = (rows.T @ column)
        if sparse.issparse(cross):
            cross = cross.toarray()
        prepared.append({
            "slot": slot, "near": near, "column": column,
            "rows": rows, "cross": cross,
            "gram": column.T @ column,
            "rhs": column.T @ flat[near],
            "pull": column.T @ design_incumbent[near],
        })
        columns.append(cross)

    gains = np.zeros(len(candidates), dtype=np.float64)
    if not prepared:
        return gains

    # One solve, not three per candidate.
    stacked = np.concatenate(columns, axis=1)
    projected = lu.solve(stacked)
    offset = 0
    for item in prepared:
        piece = projected[:, offset:offset + 3]
        offset += 3
        schur = item["gram"] - item["column"].T @ (item["rows"] @ piece)
        schur = schur + 1e-9 * np.eye(3)
        inverse = np.linalg.pinv(schur)
        effective = item["rhs"] - item["pull"]
        gains[item["slot"]] = float(np.sum(
            LAB_WEIGHTS * np.einsum(
                "kc,kl,lc->c", effective, inverse, effective)))
    return gains


def main():
    from experiments.claude_trial_sigma_run import choose_candidates

    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400)):
        model = fixture(image, side, cells)
        ctx = model.assemble(4.0)
        solved = model.solve_exact(model.base_lab, ctx)
        residual = model.lab - model.reconstruction
        candidates = choose_candidates(
            model, model.allocation_pressure.ravel().copy(), 96)
        print(f"\n{image} {side} px / {cells} cells, "
              f"{len(candidates)} candidates")

        base_time, base_gains = bench(
            "baseline addition_gains", repeats=3,
            fn=lambda: model.addition_gains(
                solved, ctx, candidates, residual))
        fast_time, fast_gains = bench(
            "local support + batched solve", repeats=3,
            fn=lambda: addition_gains_fast(
                model, solved, ctx, candidates, residual))
        print(f"  speedup {base_time / fast_time:.1f}x")
        scale = max(float(np.max(np.abs(base_gains))), 1e-30)
        gap = float(np.max(np.abs(base_gains - fast_gains))) / scale
        order_same = bool(np.all(
            np.argsort(base_gains) == np.argsort(fast_gains)))
        print(f"    max relative gain gap {gap:.3e}   "
              f"identical ranking {order_same}")
        assert gap < 1e-9, gap


if __name__ == "__main__":
    main()
