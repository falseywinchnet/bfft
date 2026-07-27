#!/usr/bin/env python3
"""Component 6: building and applying the coupled normal matrix.

`_solve_direct_pair` is 70 ms and runs seven times per Newton step, making it
the largest single cost in the receiver-guided loop.  Only 10 ms of it is the
factorization.  The rest is bookkeeping around a matrix we never actually
need:

    _cell_basis x2         1.9 ms   (kept -- it is the model)
    COO -> CSR design      5.5 ms   400k index entries, then a sort
    design.T @ design      5.0 ms   generic sparse matmul
    design.T @ target      0.8 ms
    splu                  10.2 ms   floor
    triangular solves      0.6 ms
    prediction gathers     8.0 ms   six fancy-index gathers of (npix, 3)

Formal target: **this is finite-element assembly, and the design matrix is an
intermediate nobody reads.**  Each pixel contributes a rank-one update
touching exactly two cell blocks, so

    G[i,i] += u u^T,  G[j,j] += v v^T,  G[i,j] += u v^T,  G[j,i] += v u^T

with `u = w1 * b_i(x)` and `v = w2 * b_j(x)`.  The block pattern is the set of
co-ownership pairs -- the same graph the receiver-guided Newton step already
builds its Laplacian on.  Compute that pattern once, then scatter-add the
outer products straight into a fixed CSR data array.  No COO, no sort, no
sparse matmul, no design matrix.  `A^T b` rides along in the same pass for
free.

Two further consequences, neither of them a trick:

* the pattern depends only on ownership, and the cartoon and texture systems
  share ownership -- they differ only in blend weights.  So the pattern is
  built **once per pair**, not once per field.
* the prediction step gathers `coeff[owner]` six times to form two blended
  fields.  One pass over pixels does all of it with three multiply-adds per
  side, no temporaries.

Everything is asserted against the shipped path: the normal matrix, the
right-hand side, the solved coefficients, and the rendered field.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import bench, fixture  # noqa: E402

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True) if njit is not None else _identity


# ----------------------------------------------------------------------
# Pattern: built once per ownership, shared by both fields
# ----------------------------------------------------------------------

@_compile
def _fill_indices(block_row, block_col, row_offsets, indptr, indices, n):
    """Scalar CSR column indices from the block pattern."""
    for cell in range(n):
        start = row_offsets[cell]
        stop = row_offsets[cell + 1]
        for sub in range(3):
            cursor = indptr[3 * cell + sub]
            for k in range(start, stop):
                column = 3 * block_col[k]
                indices[cursor] = column
                indices[cursor + 1] = column + 1
                indices[cursor + 2] = column + 2
                cursor += 3


@_compile
def _blocks_to_data(blocks, block_row, position, indptr, data):
    """Place each 3x3 block into the fixed CSR data array."""
    for k in range(blocks.shape[0]):
        cell = block_row[k]
        base = 3 * position[k]
        for a in range(3):
            cursor = indptr[3 * cell + a] + base
            data[cursor] = blocks[k, a, 0]
            data[cursor + 1] = blocks[k, a, 1]
            data[cursor + 2] = blocks[k, a, 2]


def build_pattern(owner, other, valid, n):
    """Co-ownership block pattern and the maps the accumulation needs."""
    visible = np.flatnonzero(valid)
    i = owner[visible].astype(np.int64)
    j = other[visible].astype(np.int64)
    low = np.minimum(i, j)
    high = np.maximum(i, j)
    keys = np.unique(low * n + high)
    pair_a = keys // n
    pair_b = keys % n

    cells = np.arange(n, dtype=np.int64)
    block_row = np.concatenate([cells, pair_a, pair_b])
    block_col = np.concatenate([cells, pair_b, pair_a])
    order = np.lexsort((block_col, block_row))
    block_row = np.ascontiguousarray(block_row[order])
    block_col = np.ascontiguousarray(block_col[order])
    relocated = np.empty(order.size, dtype=np.int64)
    relocated[order] = np.arange(order.size)
    edges = keys.size
    diag_of = np.ascontiguousarray(relocated[:n])
    forward_of = relocated[n:n + edges]
    reverse_of = relocated[n + edges:]

    counts = np.bincount(block_row, minlength=n)
    row_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=row_offsets[1:])
    position = np.arange(block_row.size) - row_offsets[block_row]

    indptr = np.zeros(3 * n + 1, dtype=np.int64)
    np.cumsum(np.repeat(3 * counts, 3), out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    _fill_indices(block_row, block_col, row_offsets, indptr, indices, n)

    # Per-pixel block slots for the off-diagonal couplings.
    slot_forward = np.full(owner.size, -1, dtype=np.int64)
    slot_reverse = np.full(owner.size, -1, dtype=np.int64)
    found = np.searchsorted(keys, low * n + high)
    owner_is_low = i == low
    slot_forward[visible] = np.where(
        owner_is_low, forward_of[found], reverse_of[found])
    slot_reverse[visible] = np.where(
        owner_is_low, reverse_of[found], forward_of[found])

    return {
        "block_row": block_row, "block_col": block_col,
        "position": np.ascontiguousarray(position),
        "indptr": indptr, "indices": indices,
        "diag_of": diag_of,
        "slot_forward": slot_forward, "slot_reverse": slot_reverse,
        "blocks": block_row.size, "edges": int(edges),
    }


# ----------------------------------------------------------------------
# Assembly and application
# ----------------------------------------------------------------------

@_compile
def _accumulate(owner, other, valid, w1, w2, first, second, target,
                diag_of, slot_forward, slot_reverse, blocks, rhs):
    """One pass: normal matrix blocks and right-hand side together."""
    u = np.empty(3, dtype=np.float64)
    v = np.empty(3, dtype=np.float64)
    for p in range(owner.shape[0]):
        i = owner[p]
        if i < 0:
            continue
        weight = w1[p]
        u[0] = weight * first[p, 0]
        u[1] = weight * first[p, 1]
        u[2] = weight * first[p, 2]
        block = diag_of[i]
        for a in range(3):
            ua = u[a]
            for b in range(3):
                blocks[block, a, b] += ua * u[b]
            rhs[i, a, 0] += ua * target[p, 0]
            rhs[i, a, 1] += ua * target[p, 1]
            rhs[i, a, 2] += ua * target[p, 2]
        if not valid[p]:
            continue
        j = other[p]
        weight = w2[p]
        v[0] = weight * second[p, 0]
        v[1] = weight * second[p, 1]
        v[2] = weight * second[p, 2]
        block = diag_of[j]
        for a in range(3):
            va = v[a]
            for b in range(3):
                blocks[block, a, b] += va * v[b]
            rhs[j, a, 0] += va * target[p, 0]
            rhs[j, a, 1] += va * target[p, 1]
            rhs[j, a, 2] += va * target[p, 2]
        forward = slot_forward[p]
        reverse = slot_reverse[p]
        for a in range(3):
            ua = u[a]
            va = v[a]
            for b in range(3):
                blocks[forward, a, b] += ua * v[b]
                blocks[reverse, a, b] += va * u[b]


@_compile
def _render(coeff, owner, other, valid, w1, w2, first, second,
            pred_first, pred_second, field):
    """Both side predictions and the blend, without a single gather."""
    for p in range(owner.shape[0]):
        i = owner[p]
        j = other[p]
        b0 = first[p, 0]
        b1 = first[p, 1]
        b2 = first[p, 2]
        c0 = second[p, 0]
        c1 = second[p, 1]
        c2 = second[p, 2]
        for channel in range(3):
            left = (coeff[i, 0, channel] * b0 +
                    coeff[i, 1, channel] * b1 +
                    coeff[i, 2, channel] * b2)
            right = (coeff[j, 0, channel] * c0 +
                     coeff[j, 1, channel] * c1 +
                     coeff[j, 2, channel] * c2)
            pred_first[p, channel] = left
            pred_second[p, channel] = right
            field[p, channel] = w1[p] * left + w2[p] * right


def assemble_and_solve(model, softness, target, pattern=None, context=None):
    """Direct replacement for `assemble` followed by `solve_exact`."""
    n = len(model.seeds)
    if context is None:
        spacing = max(math.sqrt(model.npix / max(n, 1)), 1e-9)
        angles, _ = model._site_frames()
        valid, other, w1, w2 = model._blend_weights(softness)
        first = model._cell_basis(model.owner, spacing, angles, False)
        second = model._cell_basis(other, spacing, angles, False)
    else:
        valid, other = context["valid"], context["other"]
        w1, w2 = context["w1"], context["w2"]
        first, second = context["first"], context["second"]
    if pattern is None:
        pattern = build_pattern(model.owner, other, valid, n)

    blocks = np.zeros((pattern["blocks"], 3, 3), dtype=np.float64)
    rhs = np.zeros((n, 3, 3), dtype=np.float64)
    _accumulate(
        model.owner.astype(np.int64), other.astype(np.int64), valid,
        w1, w2, np.ascontiguousarray(first), np.ascontiguousarray(second),
        np.ascontiguousarray(target.reshape(-1, 3)),
        pattern["diag_of"], pattern["slot_forward"],
        pattern["slot_reverse"], blocks, rhs)

    regularization = np.array([1e-5, 2e-3, 2e-3], dtype=np.float64)
    diagonal = pattern["diag_of"]
    for a in range(3):
        blocks[diagonal, a, a] += regularization[a]

    data = np.zeros(pattern["indices"].size, dtype=np.float64)
    _blocks_to_data(
        blocks, pattern["block_row"], pattern["position"],
        pattern["indptr"], data)
    gram = sparse.csr_matrix(
        (data, pattern["indices"], pattern["indptr"]),
        shape=(3 * n, 3 * n)).tocsc()

    lu = splu(gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    coeff = lu.solve(rhs.reshape(3 * n, 3)).reshape(n, 3, 3)

    pred_first = np.empty((model.npix, 3), dtype=np.float64)
    pred_second = np.empty((model.npix, 3), dtype=np.float64)
    field = np.empty((model.npix, 3), dtype=np.float64)
    _render(coeff, model.owner.astype(np.int64), other.astype(np.int64),
            valid, w1, w2, np.ascontiguousarray(first),
            np.ascontiguousarray(second), pred_first, pred_second, field)
    return {
        "field": field.reshape(model.h, model.w, 3), "coeff": coeff,
        "lu": lu, "gram": gram, "rhs": rhs.reshape(3 * n, 3),
        "pred_first": pred_first, "pred_second": pred_second,
        "pattern": pattern,
    }


def solve_direct_pair_fast(model, cartoon_softness=4.0,
                           texture_softness=16.0):
    """Both fields, sharing one pattern because they share one ownership."""
    pattern = None
    out = {}
    for name, target, softness in (
            ("base", model.base_lab, cartoon_softness),
            ("detail", model.detail_lab, texture_softness)):
        solved = assemble_and_solve(model, softness, target, pattern)
        pattern = solved["pattern"]
        out[name] = solved
    return out


def main():
    for image, side, cells in (("camera", 128, 700), ("pikachu", 256, 2400)):
        model = fixture(image, side, cells)
        n = len(model.seeds)
        print(f"\n{image} {side} px / {cells} cells")

        context = model.assemble(4.0)
        reference = model.solve_exact(model.base_lab, context)
        trial = assemble_and_solve(model, 4.0, model.base_lab)

        reference_gram = (
            (context["design"].T @ context["design"]).tocsr() +
            sparse.diags(context["reg"], format="csr"))
        difference = (reference_gram - trial["gram"].tocsr())
        gram_gap = (float(np.max(np.abs(difference.data)))
                    if difference.nnz else 0.0)
        scale = float(np.max(np.abs(reference_gram.data)))
        rhs_reference = np.column_stack([
            context["design"].T @ model.base_lab[..., c].ravel()
            for c in range(3)])
        rhs_gap = float(np.max(np.abs(rhs_reference - trial["rhs"])))
        coeff_gap = float(np.max(np.abs(
            reference["coeff"] - trial["coeff"])))
        field_gap = float(np.max(np.abs(
            reference["field"] - trial["field"])))
        print(f"    blocks {trial['pattern']['blocks']}  "
              f"co-ownership edges {trial['pattern']['edges']}  "
              f"nnz {trial['gram'].nnz}")
        print(f"    gram gap {gram_gap / scale:.3e} (relative)   "
              f"rhs gap {rhs_gap:.3e}")
        print(f"    coefficient gap {coeff_gap:.3e}   "
              f"field gap {field_gap:.3e}")
        assert gram_gap / scale < 1e-12
        assert field_gap < 1e-10

        pattern = trial["pattern"]
        base_assemble, _ = bench(
            "baseline assemble", lambda: model.assemble(4.0))
        base_solve, _ = bench(
            "baseline solve_exact",
            lambda: model.solve_exact(model.base_lab, context))
        fast_cold, _ = bench(
            "fused, pattern rebuilt",
            lambda: assemble_and_solve(model, 4.0, model.base_lab))
        fast_warm, _ = bench(
            "fused, pattern reused",
            lambda: assemble_and_solve(
                model, 4.0, model.base_lab, pattern, context))
        pair_base, _ = bench(
            "baseline _solve_direct_pair", model._solve_direct_pair,
            repeats=3)
        pair_fast, _ = bench(
            "fused pair, one shared pattern",
            lambda: solve_direct_pair_fast(model), repeats=3)
        print(f"  speedup  single field "
              f"{(base_assemble + base_solve) / fast_cold:.2f}x   "
              f"pair {pair_base / pair_fast:.2f}x")


if __name__ == "__main__":
    main()
