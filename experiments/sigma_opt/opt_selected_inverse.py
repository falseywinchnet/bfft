#!/usr/bin/env python3
"""Component 6: the deletion price, exactly, for every cell.

Baseline (`SigmaVoronoi.deletion_costs`): 48 Rademacher probes estimate
`diag(G^-1)`, a diagonal-only reading of the price ranks the cells, and the
cheapest `n/12` get exact 3-column solves.  Two approximations stacked --
a stochastic diagonal, then a diagonal reading of a 3x3 form -- to produce a
*shortlist*, after which the answer is exact but only for 200 cells of 2,400.

Formal target: **Takahashi's equations / Erisman-Tinney 1975.**  If
`G = L D L^T` then the entries of `G^-1` that lie on the sparsity pattern of
`L` satisfy a closed recurrence among themselves:

    Z_ij = delta_ij / d_j  -  sum_{k>j, L_kj != 0} Z_ik L_kj,     i >= j

Read it right to left and the content is: every `Z_ik` this needs is already
on the pattern (the fill-path lemma), so no entry outside the pattern is ever
required and no solve is ever issued.  One backward sweep over the factor
computes *all* of them, in the flop count of the factorization itself.

The 3x3 diagonal blocks we need are entries of `G` -- three unknowns of the
same cell are coupled by construction -- so they are on the pattern of `L` by
definition, and selected inversion returns them exactly.  There is no
shortlist because there is no cost to pricing everything.

The factor already exists: `solve_exact` keeps the `splu` object, and
`L`, `U`, `perm_r`, `perm_c` come straight off it.  With
`permc_spec="MMD_AT_PLUS_A"`, `SymmetricMode` and `diag_pivot_thresh=0`
SuperLU does not pivot, so `perm_r == perm_c` and `U == D L^T` -- both are
asserted here rather than assumed.

Also measured, because it was the other candidate: the Johnson-Lindenstrauss
/ leverage-score estimator of Spielman-Srivastava and Drineas-Mahoney, which
is what Rademacher probing *should* have been.  Writing
`G^-1 = (B G^-1)^T (B G^-1)` with `B = [A ; Lambda^1/2]` turns every entry of
`G^-1` into an inner product of two columns of a fixed matrix, so a random
projection of its rows preserves them to `(1 +- eps)` -- a relative
guarantee, at exactly the same number of solves as blind probing.  It wins on
every tail statistic, and it is still an estimate; selected inversion is
exact and cheaper.  Numbers in `notes/study_selected_inverse.md`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture  # noqa: E402

from claude_trial_sigma import LAB_WEIGHTS  # noqa: E402

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


# -- the recurrence ----------------------------------------------------

@njit(cache=True, fastmath=False)
def _takahashi(Lp, Li, Lx, dinv, Zx, pos, acc):
    """Fill `Zx` -- the inverse on the pattern of `L` -- in one backward pass.

    `L` is unit lower triangular in CSC with the diagonal stored first in
    every column; `Zx` shares its pattern.  `pos` is an all -1 scratch of
    length n used to scatter one column at a time, and is left all -1.

    Returns 0 on success, or 1 if an entry the recurrence asked for was not
    on the pattern -- which cannot happen for a true Cholesky pattern and is
    checked rather than trusted.
    """
    n = len(Lp) - 1
    for j in range(n - 1, -1, -1):
        start = Lp[j]
        end = Lp[j + 1]
        base = start + 1                       # skip the unit diagonal
        m = end - base
        for a in range(m):
            acc[a] = 0.0
        for a in range(m):
            col = Li[base + a]                 # a column index > j
            la = Lx[base + a]
            lo = Lp[col]
            hi = Lp[col + 1]
            for t in range(lo, hi):            # scatter column `col` of Z
                pos[Li[t]] = t
            for b in range(a, m):
                t = pos[Li[base + b]]
                if t < 0:
                    for u in range(lo, hi):
                        pos[Li[u]] = -1
                    return 1
                v = Zx[t]                      # Z[r_b, r_a] == Z[r_a, r_b]
                acc[a] += Lx[base + b] * v
                if b != a:
                    acc[b] += la * v
            for t in range(lo, hi):
                pos[Li[t]] = -1
        s = dinv[j]
        for a in range(m):
            val = -acc[a]
            Zx[base + a] = val
            s -= Lx[base + a] * val
        Zx[start] = s
    return 0


@njit(cache=True)
def _gather_blocks(Zp, Zi, Zx, perm, n_cells, width, out):
    """Read the `width x width` diagonal blocks of `G^-1` out of `Z`.

    `Z` holds the inverse of the *permuted* matrix on the lower triangle, so
    unknown `i` of the original system sits at row `perm[i]`.
    """
    for c in range(n_cells):
        for k in range(width):
            a = perm[width * c + k]
            for l in range(k, width):
                b = perm[width * c + l]
                if a >= b:
                    row, col = a, b
                else:
                    row, col = b, a
                lo = Zp[col]
                hi = Zp[col + 1]
                found = -1
                while lo < hi:                 # binary search, Zi sorted
                    mid = (lo + hi) // 2
                    if Zi[mid] == row:
                        found = mid
                        break
                    elif Zi[mid] < row:
                        lo = mid + 1
                    else:
                        hi = mid
                if found < 0:
                    return 1
                out[c, k, l] = Zx[found]
                out[c, l, k] = Zx[found]
    return 0


def selected_inverse_blocks(lu, n_cells, width, check=False):
    """Exact `width x width` diagonal blocks of `G^-1`, no solves.

    `lu` is the `splu` object `solve_exact` already built.
    """
    L = lu.L
    if not sparse.isspmatrix_csc(L):
        L = L.tocsc()
    if not L.has_sorted_indices:
        L.sort_indices()
    Lp = L.indptr.astype(np.int64)
    Li = L.indices.astype(np.int64)
    Lx = L.data.astype(np.float64)
    n = L.shape[0]
    # The diagonal must lead each column for the kernel's `base = start + 1`.
    if not np.all(Li[Lp[:-1]] == np.arange(n)):
        raise RuntimeError("SuperLU L is not diagonal-first in CSC")
    dinv = 1.0 / lu.U.diagonal()

    Zx = np.zeros(Lx.shape[0], dtype=np.float64)
    pos = np.full(n, -1, dtype=np.int64)
    acc = np.zeros(n, dtype=np.float64)
    status = _takahashi(Lp, Li, Lx, dinv, Zx, pos, acc)
    if status:
        raise RuntimeError("Takahashi asked for an entry off the pattern")

    perm = np.asarray(lu.perm_r, dtype=np.int64)
    out = np.zeros((n_cells, width, width), dtype=np.float64)
    if _gather_blocks(Lp, Li, Zx, perm, n_cells, width, out):
        raise RuntimeError("a diagonal block entry was off the pattern")
    if check:
        return out, (Lp, Li, Zx)
    return out


def deletion_costs_selected(solved, ctx):
    """Exact deletion price for *every* cell, from one selected inversion.

    Same quantity the baseline computes on its shortlist -- `c^T S c` with
    `S = [(G^-1)_ii]^-1` -- but for all `n` cells and with no probing stage,
    so there is no shortlist to be wrong about.
    """
    n, width = ctx["n"], ctx["width"]
    blocks = selected_inverse_blocks(solved["lu"], n, width)
    coeff = solved["coeff"]
    inv = np.linalg.pinv(blocks)
    return np.einsum(
        "c,ikc,ikl,ilc->i", LAB_WEIGHTS, coeff, inv, coeff), blocks


# -- the estimator it replaces, done properly ---------------------------

def jl_blocks(lu, design, reg, n_cells, width, k=48, seed=7):
    """Johnson-Lindenstrauss estimate of the same blocks, `k` solves.

    `G^-1 = M^T M` with `M = B G^-1`, `B = [A ; Lambda^1/2]`, so block
    `(G^-1)_ii` is a Gram matrix of three columns of `M` and a random
    projection of `M`'s rows preserves it to `(1 +- eps)` with
    `k = O(log n / eps^2)`.  Contrast with `z * (G^-1 z)`, which is unbiased
    but whose error is *additive* in the off-diagonal mass and carries no
    relative guarantee at all.
    """
    rng = np.random.default_rng(seed)
    npix, dim = design.shape
    sign = rng.integers(0, 2, size=(k, npix + dim)).astype(np.float64)
    sign = (2.0 * sign - 1.0) / np.sqrt(k)
    proj = np.asarray(sign[:, :npix] @ design) + sign[:, npix:] * np.sqrt(reg)
    W = lu.solve(np.ascontiguousarray(proj.T))          # (dim, k)
    W = W.reshape(n_cells, width, k)
    return np.einsum("iak,ibk->iab", W, W)


def rademacher_diag(lu, n_cells, width, probes=48, seed=20260726):
    """The baseline's own estimator, isolated."""
    rng = np.random.default_rng(seed)
    dim = width * n_cells
    est = np.zeros(dim, dtype=np.float64)
    for _ in range(probes):
        z = 2.0 * rng.integers(0, 2, size=dim).astype(np.float64) - 1.0
        est += z * lu.solve(z)
    return (est / probes).reshape(n_cells, width)


# -- ground truth -------------------------------------------------------

def exact_blocks_by_solves(lu, n_cells, width, chunk=300):
    """Every diagonal block by explicit unit-vector solves.  The reference."""
    dim = width * n_cells
    blocks = np.zeros((n_cells, width, width), dtype=np.float64)
    for start in range(0, dim, chunk):
        stop = min(start + chunk, dim)
        unit = np.zeros((dim, stop - start))
        unit[np.arange(start, stop), np.arange(stop - start)] = 1.0
        columns = lu.solve(unit)
        for j in range(start, stop):
            cell, k = divmod(j, width)
            blocks[cell, :, k] = columns[
                width * cell: width * cell + width, j - start]
    return 0.5 * (blocks + blocks.transpose(0, 2, 1))


def prices_from_blocks(blocks, coeff):
    inv = np.linalg.pinv(blocks)
    return np.einsum("c,ikc,ikl,ilc->i", LAB_WEIGHTS, coeff, inv, coeff)


# -- report -------------------------------------------------------------

def _relative(estimate, truth):
    return np.abs(estimate - truth) / np.maximum(np.abs(truth), 1e-300)


def _stats(name, estimate, truth):
    rel = _relative(estimate, truth)
    print(f"    {name:<26s} median {np.median(rel):7.4f}  "
          f"p90 {np.percentile(rel, 90):7.4f}  max {np.max(rel):9.4f}  "
          f"neg {int(np.sum(estimate <= 0)):4d}")
    return rel


def run(image, side, cells):
    model = fixture(image, side, cells)
    ctx = model.assemble(4.0)
    solved = model.solve_exact(model.base_lab, ctx)
    lu, n, width = solved["lu"], ctx["n"], ctx["width"]
    coeff = solved["coeff"]
    dim = width * n
    print(f"\n{image} {side} px / {cells} cells   "
          f"unknowns {dim}  G nnz {solved['gram'].nnz}  "
          f"L nnz {lu.L.nnz}  factor {model.solve_stats['factor_ms']:.1f} ms")

    # the factorization is unpivoted and symmetric, as the recurrence needs
    assert np.array_equal(lu.perm_r, lu.perm_c), "SuperLU pivoted"
    residual = abs(lu.U - (sparse.diags(lu.U.diagonal()) @ lu.L.T).tocsc()).max()
    scale = abs(solved["gram"]).max()
    print(f"  |U - D L^T|_max / |G|_max = {residual / scale:.2e}")

    print("\n  -- ground truth ---------------------------------------")
    t0 = time.perf_counter()
    truth = exact_blocks_by_solves(lu, n, width)
    truth_ms = (time.perf_counter() - t0) * 1e3
    price_true = prices_from_blocks(truth, coeff)
    print(f"    {dim} explicit column solves        {truth_ms:9.0f} ms")

    print("\n  -- selected inversion (Takahashi) ---------------------")
    selected_inverse_blocks(lu, n, width)          # jit warm-up
    sel_t, (price_sel, blocks) = bench(
        "Takahashi, all cells priced", repeats=5,
        fn=lambda: deletion_costs_selected(solved, ctx))
    gap = np.max(_relative(blocks, truth))
    price_gap = np.max(_relative(price_sel, price_true))
    print(f"    max relative block error  {gap:.3e}")
    print(f"    max relative price error  {price_gap:.3e}")
    assert gap < 1e-8, gap

    print("\n  -- baseline -------------------------------------------")
    base_t, (base_exact, base_rough) = bench(
        "48 probes + 3*(n/12) solves", repeats=3,
        fn=lambda: model.deletion_costs(solved, ctx))
    print(f"    speedup {base_t / sel_t:.1f}x, "
          f"and {n} cells priced instead of "
          f"{int(np.sum(np.isfinite(base_exact)))}")
    listed = np.flatnonzero(np.isfinite(base_exact))
    shortlist_gap = np.max(_relative(base_exact[listed], price_true[listed]))
    print(f"    baseline prices on its own shortlist are exact: "
          f"max relative gap {shortlist_gap:.2e}")

    print("\n  -- how wrong is the 48-probe diagonal? ----------------")
    diag_true = np.einsum("ikk->ik", truth).copy()
    t0 = time.perf_counter()
    rade = rademacher_diag(lu, n, width)
    rade_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    jl = jl_blocks(lu, ctx["design"].tocsr(), ctx["reg"], n, width, k=48)
    jl_ms = (time.perf_counter() - t0) * 1e3
    jl_diag = np.einsum("ikk->ik", jl).copy()
    print(f"    relative error of diag(G^-1), 48 solves each"
          f"   [Rademacher {rade_ms:.0f} ms, JL {jl_ms:.0f} ms]")
    _stats("Rademacher z*(G^-1 z)", rade, diag_true)
    _stats("Johnson-Lindenstrauss", jl_diag, diag_true)

    off_true = truth[:, 0, 1]
    off_rel = _relative(jl[:, 0, 1], off_true)
    print(f"    JL also gets the off-diagonals the probe cannot see at all: "
          f"median {np.median(off_rel):.3f}")

    print("\n  -- what the estimate is used for: the shortlist -------")
    size = max(8, n // 12)
    keep = set(np.argsort(price_true)[:size].tolist())

    def recall(rank):
        return len(keep & set(np.argsort(rank)[:size].tolist())) / size

    def diag_price(est):
        return np.sum(np.sum(
            LAB_WEIGHTS[None, None, :] * coeff * coeff, axis=2) /
            np.maximum(est, 1e-12), axis=1)

    def block_price(bl):
        return prices_from_blocks(bl, coeff)

    print(f"    shortlist of {size} cheapest of {n}, recall against exact:")
    print(f"      baseline `rough` (Rademacher diag, diagonal reading)  "
          f"{recall(base_rough):.3f}")
    print(f"      JL diagonal, same diagonal reading                    "
          f"{recall(diag_price(jl_diag)):.3f}")
    print(f"      JL full 3x3 block, exact reading                      "
          f"{recall(block_price(jl)):.3f}")
    print(f"      exact diagonal, diagonal reading (isolates the two)   "
          f"{recall(diag_price(diag_true)):.3f}")
    print(f"      selected inversion                                    "
          f"{recall(price_sel):.3f}")

    cheapest = np.argsort(price_true)[:size]
    missed = [int(c) for c in cheapest
              if c not in set(np.argsort(base_rough)[:size].tolist())]
    if missed:
        ranks = np.empty(n, dtype=np.int64)
        ranks[np.argsort(price_true)] = np.arange(n)
        worst = min(missed, key=lambda c: ranks[c])
        print(f"    cheapest cell the baseline misses: rank "
              f"{int(ranks[worst])} of {n}, price {price_true[worst]:.4e} "
              f"(baseline shortlist floor {price_true[cheapest[-1]]:.4e})")

    # What `leverage_exchange` actually does with the ranking: retire the
    # `n // 24` cheapest.  That set, not the shortlist, is the decision.
    budget = min(96, max(1, n // 24))
    retired_true = set(np.argsort(price_true)[:budget].tolist())
    retired_base = set(np.argsort(base_exact)[:budget].tolist())
    print(f"    the {budget} cells `leverage_exchange` would retire: "
          f"{len(retired_true & retired_base)} of {budget} agree with exact")

    # Slack meter, in the same units the alpha round used: the price is the
    # rise in the LAB-weighted fitted objective, so compare it to that.
    delta = (model.base_lab - solved["field"]).reshape(-1, 3)
    objective = float(np.sum(LAB_WEIGHTS * np.sum(delta * delta, axis=0)))
    frac = price_true / max(objective, 1e-30)
    print(f"\n    removable within 1% of the fitted objective: "
          f"{int(np.sum(frac < 0.01))} of {n} cells; cheapest cell costs "
          f"{100 * frac.min():.4f}% "
          f"(baseline sees only its {int(np.sum(np.isfinite(base_exact)))}"
          f"-cell shortlist)")

    print("\n  -- where the 32 ms goes -------------------------------")
    ext_t, _ = bench("  extract L, U from SuperLU", repeats=5,
                     fn=lambda: (lu.L, lu.U.diagonal()))
    rec_t, _ = bench("  + Takahashi recurrence + gather", repeats=5,
                     fn=lambda: selected_inverse_blocks(lu, n, width))
    print(f"      pinv + contraction over {n} cells   "
          f"{(sel_t - rec_t) * 1e3:9.2f} ms")
    print(f"      recurrence itself                   "
          f"{(rec_t - ext_t) * 1e3:9.2f} ms  "
          f"({(rec_t - ext_t) * 1e3 / model.solve_stats['factor_ms']:.1f}x "
          f"one factorization)")

    return {
        "image": image, "cells": cells, "unknowns": dim,
        "selected_ms": sel_t * 1e3, "baseline_ms": base_t * 1e3,
        "truth_ms": truth_ms, "block_gap": gap,
        "recall_base": recall(base_rough), "recall_jl": recall(block_price(jl)),
        "retired_agree": len(retired_true & retired_base) / budget,
    }


def check_enriched(image="camera", side=128, cells=700):
    """The width-4 enriched system exercises the same code path."""
    model = fixture(image, side, cells)
    plain = model.assemble(4.0)
    first = model.solve_exact(model.base_lab, plain)
    model.measure_ridge(model.base_lab - first["field"], plain)
    ctx = model.assemble(4.0, enriched=True)
    solved = model.solve_exact(model.base_lab, ctx)
    n, width = ctx["n"], ctx["width"]
    truth = exact_blocks_by_solves(solved["lu"], n, width)
    blocks = selected_inverse_blocks(solved["lu"], n, width)
    gap = np.max(_relative(blocks, truth))
    print(f"\nenriched width={width}: max relative block error {gap:.3e}")
    assert gap < 1e-8, gap


def main():
    rows = []
    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400),
                               ("pikachu", 384, 5400)):
        rows.append(run(image, side, cells))
    check_enriched()
    print("\n  scaling: the baseline spends O(n) solves on its shortlist, the")
    print("  recurrence spends one pass over the factor.")
    print(f"    {'config':<22s}{'unknowns':>10s}{'baseline':>11s}"
          f"{'selected':>11s}{'speedup':>9s}")
    for row in rows:
        print(f"    {row['image'] + ' ' + str(row['cells']):<22s}"
              f"{row['unknowns']:>10d}{row['baseline_ms']:>9.1f} ms"
              f"{row['selected_ms']:>9.1f} ms"
              f"{row['baseline_ms'] / row['selected_ms']:>8.1f}x")


if __name__ == "__main__":
    main()
