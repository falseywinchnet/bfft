"""Trial alpha 1: the coupled fit's own Hessian IS the cell graph.

The spin experiment was rejected because the supplied signed graph was a
geometric guess.  The correction is not a better guess.  The renderer already
determines a graph exactly: each pixel is explained by its owner and its
runner-up, so the normal matrix of the partition-of-unity least squares has a
nonzero 3x3 block for cell pair (i, j) precisely when i and j jointly explain
at least one pixel, weighted by how much they jointly explain it.  Nothing
about geometry, blue noise, or distance enters.  That matrix is the graph.

Two consequences, both non-iterative:

1. `lsmr` is replaced by one sparse factorization of that matrix.  The three
   colour channels share it, and the answer is exact rather than
   tolerance-limited.

2. Once factored, the exact price of removing a cell, and the exact
   complementarity of a pair of cells, are closed forms in the factor.  No
   refit, no re-solve, no re-assignment.  This is the measured
   `a_ij = dJ(i,j) - dJ(i) - dJ(j)` the research log asked for.

Both are consequences of a single fact: constraining a solved least-squares
problem to a subspace raises the objective by a quantity read off the inverse
of the normal matrix restricted to the constrained coordinates.  For the
constraint `C c = 0` on the solved `c*`,

    dJ = (C c*)^T [C H^-1 C^T]^-1 (C c*)

exactly.  Deletion is `C = I` on one cell's three coefficients; a merge is
`C = [T_i, -T_j]` after mapping both cells' local frames to a common global
affine frame.

Nothing here changes allocation.  It changes what a price is.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


# Matches the ridge used by the production lsmr design matrix, where the
# rows carry sqrt(regularization); the normal matrix carries it directly.
RIDGE = np.array([1e-5, 2e-3, 2e-3], dtype=np.float64)


class Parts:
    """Per-pixel two-cell design, shared by assembly and prediction."""

    __slots__ = ("n", "owner", "other", "u", "v", "valid", "spacing",
                 "angles", "seeds")

    def __init__(self, model, softness):
        n = len(model.seeds)
        spacing = math.sqrt(model.npix / max(n, 1))
        angles, _ = model._site_frames()
        valid = model.second >= 0
        other = np.where(valid, model.second, model.owner)
        gap = model.d2 - model.d1
        z = np.clip(0.5 * float(softness) * gap, -50.0, 50.0)
        first_weight = 1.0 / (1.0 + np.exp(-z))
        first_weight[~valid] = 1.0
        other_weight = np.where(valid, 1.0 - first_weight, 0.0)

        def basis(ids):
            sx, sy = model.seeds[ids, 0], model.seeds[ids, 1]
            ct, st = np.cos(angles[ids]), np.sin(angles[ids])
            dx, dy = model.xf - sx, model.yf - sy
            return np.column_stack([
                np.ones(model.npix),
                (dx * ct + dy * st) / max(spacing, 1e-9),
                (-dx * st + dy * ct) / max(spacing, 1e-9),
            ])

        self.n = n
        self.spacing = spacing
        self.angles = angles
        self.seeds = model.seeds
        self.owner = model.owner.astype(np.int64)
        self.other = other.astype(np.int64)
        self.valid = valid
        self.u = basis(self.owner) * first_weight[:, None]
        self.v = basis(self.other) * other_weight[:, None]

    def global_frame(self, i):
        """Map cell i's local affine coefficients to global [1, x, y].

        A merge is only meaningful between cells describing the same affine
        function of the image plane, not between cells whose coefficient
        vectors happen to be numerically close in different frames.
        """
        ct = math.cos(self.angles[i])
        st = math.sin(self.angles[i])
        sx, sy = self.seeds[i, 0], self.seeds[i, 1]
        s = max(self.spacing, 1e-9)
        t = np.zeros((3, 3), dtype=np.float64)
        t[0, 0] = 1.0
        t[1, 1] = ct / s
        t[1, 2] = -st / s
        t[2, 1] = st / s
        t[2, 2] = ct / s
        t[0, 1] = -(t[1, 1] * sx + t[2, 1] * sy)
        t[0, 2] = -(t[1, 2] * sx + t[2, 2] * sy)
        return t.T


def assemble(parts):
    """Accumulate the exact normal matrix H = A^T A + ridge.

    The cross blocks are accumulated over the *unique* (owner, other) pairs
    that actually occur, so the cost is one sort of the pixel array rather
    than anything quadratic in the cell count.  Those unique pairs are the
    edge list of the induced graph.
    """
    n = parts.n
    own, oth, u, v = parts.owner, parts.other, parts.u, parts.v
    idx = np.arange(n, dtype=np.int64)
    rows, cols, data = [], [], []
    diagonal = np.zeros((n, 3, 3), dtype=np.float64)

    for a in range(3):
        for b in range(3):
            block = np.bincount(own, weights=u[:, a] * u[:, b], minlength=n)
            block += np.bincount(oth, weights=v[:, a] * v[:, b], minlength=n)
            if a == b:
                block = block + RIDGE[a]
            diagonal[:, a, b] = block
            rows.append(3 * idx + a)
            cols.append(3 * idx + b)
            data.append(block)

    valid = parts.valid
    key = own[valid] * n + oth[valid]
    unique, inverse = np.unique(key, return_inverse=True)
    pair_i = (unique // n).astype(np.int64)
    pair_j = (unique % n).astype(np.int64)
    uu, vv = u[valid], v[valid]
    m = unique.size
    for a in range(3):
        for b in range(3):
            block = np.bincount(
                inverse, weights=uu[:, a] * vv[:, b], minlength=m)
            rows.append(3 * pair_i + a)
            cols.append(3 * pair_j + b)
            data.append(block)
            rows.append(3 * pair_j + b)
            cols.append(3 * pair_i + a)
            data.append(block)

    h = sparse.coo_matrix(
        (np.concatenate(data),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n, 3 * n)).tocsc()
    return h, pair_i, pair_j, diagonal


def rhs(target, parts):
    n = parts.n
    flat = target.reshape(-1, 3)
    out = np.zeros((3 * n, 3), dtype=np.float64)
    idx = np.arange(n, dtype=np.int64)
    for channel in range(3):
        y = flat[:, channel]
        for a in range(3):
            acc = np.bincount(
                parts.owner, weights=parts.u[:, a] * y, minlength=n)
            acc += np.bincount(
                parts.other, weights=parts.v[:, a] * y, minlength=n)
            out[3 * idx + a, channel] = acc
    return out


class ExactCoupled:
    """One factorization; exact fields, exact deletion and merge prices."""

    def __init__(self, model, softness):
        self.model = model
        self.softness = float(softness)
        self.parts = Parts(model, softness)
        t0 = time.perf_counter()
        self.h, self.pair_i, self.pair_j, self.diagonal = assemble(self.parts)
        self.assemble_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self.lu = splu(self.h, permc_spec="MMD_AT_PLUS_A",
                       diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        self.factor_ms = (time.perf_counter() - t0) * 1000.0
        self.coeff = None

    # -- graph -------------------------------------------------------
    @property
    def edges(self):
        """Directed pairs that jointly explain a pixel, deduplicated."""
        lo = np.minimum(self.pair_i, self.pair_j)
        hi = np.maximum(self.pair_i, self.pair_j)
        keep = lo != hi
        pairs = np.unique(
            np.column_stack([lo[keep], hi[keep]]), axis=0)
        return pairs

    def graph_report(self):
        pairs = self.edges
        n = self.parts.n
        degree = np.bincount(pairs.ravel(), minlength=n)
        return {
            "cells": int(n),
            "edges": int(len(pairs)),
            "mean_degree": float(degree.mean()),
            "max_degree": int(degree.max()) if n else 0,
            "isolated": int(np.sum(degree == 0)),
            "normal_nnz": int(self.h.nnz),
            "assemble_ms": self.assemble_ms,
            "factor_ms": self.factor_ms,
        }

    # -- fields ------------------------------------------------------
    def field(self, target):
        b = rhs(target, self.parts)
        coeff = self.lu.solve(b)
        self.coeff = coeff
        n = self.parts.n
        cube = coeff.reshape(n, 3, 3)
        own, oth = self.parts.owner, self.parts.other
        fitted = (
            np.einsum("pa,pac->pc", self.parts.u, cube[own]) +
            np.einsum("pa,pac->pc", self.parts.v, cube[oth]))
        return fitted.reshape(self.model.h, self.model.w, 3)

    def residual_objective(self, target):
        """The value the prices are increments of."""
        fitted = self.field(target)
        return float(np.sum((target - fitted) ** 2))

    # -- prices ------------------------------------------------------
    def _inverse_columns(self, cells):
        n3 = 3 * self.parts.n
        cells = list(cells)
        e = np.zeros((n3, 3 * len(cells)), dtype=np.float64)
        for k, i in enumerate(cells):
            for a in range(3):
                e[3 * i + a, 3 * k + a] = 1.0
        x = self.lu.solve(e)
        return {i: x[:, 3 * k:3 * k + 3] for k, i in enumerate(cells)}

    def deletion_bounds(self):
        """Exact upper bound on every cell's deletion price, with no solves.

        The exact price is `c_i^T S_i c_i` where `S_i` is the Schur complement
        `H_ii - H_ij H_jj^-1 H_ji`: the cell's own curvature minus everything
        its neighbours could have explained for it.  Dropping the coupling
        term leaves `c_i^T H_ii c_i`, and because the subtracted term is
        positive semidefinite this is an upper bound on the true price for
        every cell at once, in O(n).

        So the bound is the price a cell would carry if the graph did not
        exist, and the gap between bound and exact price is precisely the
        graph's contribution.  Cells whose bound is already small cannot be
        expensive, which makes it a sound filter: shortlist by the bound,
        then pay for exact prices only on the shortlist.
        """
        if self.coeff is None:
            raise RuntimeError("call field() before pricing")
        n = self.parts.n
        cube = self.coeff.reshape(n, 3, 3)
        return np.einsum("iac,iab,ibc->i", cube, self.diagonal, cube)

    def deletion_prices(self, cells):
        """Exact rise in the fitted objective if each cell is switched off.

        This is the death price the allocation market has never had.  It is
        not an integral of residual under the cell: a cell sitting on a large
        but already-explained residual is cheap to remove, and a cell doing
        quiet structural work for its neighbours is expensive.
        """
        if self.coeff is None:
            raise RuntimeError("call field() before pricing")
        columns = self._inverse_columns(cells)
        prices = {}
        for i in cells:
            block = columns[i][3 * i:3 * i + 3, :]
            block = 0.5 * (block + block.T)
            ci = self.coeff[3 * i:3 * i + 3, :]
            try:
                prices[i] = float(np.sum(ci * np.linalg.solve(block, ci)))
            except np.linalg.LinAlgError:
                prices[i] = float("inf")
        return prices, columns

    def merge_prices(self, pairs, columns):
        """Exact rise if the pair is constrained to one global affine plane."""
        out = {}
        for i, j in pairs:
            if i not in columns or j not in columns:
                continue
            sii = columns[i][3 * i:3 * i + 3, :]
            sjj = columns[j][3 * j:3 * j + 3, :]
            sij = columns[j][3 * i:3 * i + 3, :]
            ti = self.parts.global_frame(i)
            tj = self.parts.global_frame(j)
            m = (ti @ sii @ ti.T - ti @ sij @ tj.T -
                 tj @ sij.T @ ti.T + tj @ sjj @ tj.T)
            m = 0.5 * (m + m.T)
            d = (ti @ self.coeff[3 * i:3 * i + 3, :] -
                 tj @ self.coeff[3 * j:3 * j + 3, :])
            try:
                out[(i, j)] = float(np.sum(d * np.linalg.solve(m, d)))
            except np.linalg.LinAlgError:
                out[(i, j)] = float("inf")
        return out

    def complementarity(self, pairs, columns, deletions):
        """a_ij = dJ(merge i,j) - dJ(delete i) - dJ(delete j).

        Negative means the two cells are redundant: tying them costs less
        than the sum of their independent contributions, so one of them is
        nearly free.  Positive means they are genuinely complementary and
        the pair is worth its two units of budget.  This is the entry the
        rejected spin experiment needed and could not afford.
        """
        merges = self.merge_prices(pairs, columns)
        return {
            key: merges[key] - deletions[key[0]] - deletions[key[1]]
            for key in merges
        }


def solve_coupled_exact(model, cartoon_softness=4.0, texture_softness=16.0,
                        multiscale=True):
    """Drop-in exact replacement for TransportVoronoi.solve_coupled."""
    from bfft.effects import lab_to_srgb

    t0 = time.perf_counter()
    if not multiscale:
        cartoon_softness = texture_softness = model.cfg.softness
    base_solver = ExactCoupled(model, cartoon_softness)
    base = base_solver.field(model.base_lab)
    if abs(texture_softness - cartoon_softness) < 1e-12:
        detail_solver = base_solver
    else:
        detail_solver = ExactCoupled(model, texture_softness)
    detail = detail_solver.field(model.detail_lab)

    precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
    model.cartoon_reconstruction = base
    model.texture_reconstruction = detail
    model.reconstruction = base + precision * detail
    base_delta = model.base_lab - base
    detail_delta = model.detail_lab - detail
    delta = model.lab - model.reconstruction
    model.cartoon_error = np.sqrt(
        base_delta[..., 0] ** 2 +
        1.5 * np.sum(base_delta[..., 1:] ** 2, axis=2))
    model.texture_error = np.sqrt(
        detail_delta[..., 0] ** 2 +
        1.5 * np.sum(detail_delta[..., 1:] ** 2, axis=2))
    model.error = np.sqrt(
        delta[..., 0] ** 2 + 1.5 * np.sum(delta[..., 1:] ** 2, axis=2))
    model._update_allocation_pressure()
    rgb = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
    model.rgb_mse = float(np.mean((model.rgb - rgb) ** 2))
    model.psnr = -10.0 * math.log10(max(model.rgb_mse, 1e-12))
    model.last_ms = (time.perf_counter() - t0) * 1000.0
    model.last_action = "exact coupled solve"
    return base_solver, detail_solver
