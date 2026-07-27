"""Sigma round: exact algebra on the coupled system, and descent on the graph.

Nothing here edits the validated model.  Every trial subclasses
`TransportVoronoi` and overrides only what it needs.

The premise of this round is a single observation about the coupled solve.
The design matrix it assembles has exactly two nonzero cell blocks per pixel,
the owner and the runner-up, weighted by the partition of unity.  Its normal
matrix

    G = A^T A + Lambda

is therefore the *renderer's own adjacency graph*, with entries equal to the
measured inner products of the atoms.  The spin experiment failed because it
supplied a distance graph.  `G` is the graph that experiment needed, and the
coupled solve already builds it as a byproduct.  Once `G` is factored rather
than iterated on, the questions that the allocator asks heuristically all
acquire closed forms:

* what is a cell worth      -> exact constrained-deletion cost (Schur block);
* where should a cell go    -> Schur-complement gain of a candidate column;
* how sure is a prediction  -> statistical leverage of the atom.

The second premise is that the geodesic metric is also a decision variable.
A shortest-path distance is a minimum of linear functions of the edge
weights, so by the envelope theorem its derivative is the indicator of the
achieving path.  Accumulating the loss sensitivity backwards over the
Dijkstra predecessor forest therefore yields the *exact* gradient of the
reconstruction error with respect to every edge of the metric, in one linear
pass, with no perturbation, no adjoint solve, and no iteration inside the
gradient itself.  That is descent on the graph rather than on the flow.

Trials
------
`direct`      Replace LSMR with a direct factorization of the normal matrix.
`enriched`    Add one bounded ridge column per cell (partition-of-unity
              enrichment).  Direction and offset are measured from the
              residual's own sign statistics, never from a contour.
`graph`       Learn a scalar barrier field on the pixel graph by exact
              shortest-path-tree sensitivity.
`leverage`    Price every existing cell by exact deletion cost and every
              candidate site by exact Schur gain, then exchange.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import splu

from transport_voronoi import Config, TransportVoronoi, _normalize
from bfft.vision import (assemble_normal, coownership_graph,
                         deletion_prices as exact_deletion_prices,
                         measure_residual_ridges, render_partition)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

LAB_WEIGHTS = np.array([1.0, 1.5, 1.5], dtype=np.float64)


# ----------------------------------------------------------------------
# Compiled two-best walk that also returns its predecessor forest
# ----------------------------------------------------------------------

def _identity(fn):  # pragma: no cover - only used without numba
    return fn


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _dijkstra_two_best_pred(seed_p, reach, base_costs, s_field, h, w):
    """Two-label geodesic walk returning distances and both parent forests.

    `base_costs[d, y, x]` is the unmodulated metric cost of the step from
    `(y, x)` in direction `d`.  `s_field` scales the metric locally; an edge
    costs the base cost times the mean of its endpoints' scales, so the
    modulated metric stays symmetric and every edge has two well-defined
    partial derivatives.
    """
    npix = h * w
    inf = 1e300
    d1 = np.full(npix, inf)
    d2 = np.full(npix, inf)
    own = np.full(npix, -1, dtype=np.int32)
    run = np.full(npix, -1, dtype=np.int32)
    pr1 = np.full(npix, -1, dtype=np.int32)
    pr2 = np.full(npix, -1, dtype=np.int32)
    pl1 = np.zeros(npix, dtype=np.int8)
    pl2 = np.zeros(npix, dtype=np.int8)

    cap = 4 * npix + 256
    hk = np.empty(cap, dtype=np.float64)
    hp = np.empty(cap, dtype=np.int32)
    hs = np.empty(cap, dtype=np.int32)
    size = 0

    for site in range(len(seed_p)):
        p = seed_p[site]
        distance = -reach[site]
        if distance < d1[p]:
            d2[p], run[p], pr2[p], pl2[p] = d1[p], own[p], pr1[p], pl1[p]
            d1[p], own[p], pr1[p], pl1[p] = distance, site, -1, 0
        elif site != own[p] and distance < d2[p]:
            d2[p], run[p], pr2[p], pl2[p] = distance, site, -1, 0
        # push
        if size >= cap:
            cap *= 2
            nk = np.empty(cap, dtype=np.float64)
            npp = np.empty(cap, dtype=np.int32)
            ns = np.empty(cap, dtype=np.int32)
            nk[:size] = hk[:size]
            npp[:size] = hp[:size]
            ns[:size] = hs[:size]
            hk, hp, hs = nk, npp, ns
        hk[size], hp[size], hs[size] = distance, p, site
        child = size
        size += 1
        while child > 0:
            parent = (child - 1) // 2
            if hk[parent] <= hk[child]:
                break
            hk[parent], hk[child] = hk[child], hk[parent]
            hp[parent], hp[child] = hp[child], hp[parent]
            hs[parent], hs[child] = hs[child], hs[parent]
            child = parent

    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    tolerance = 1e-12

    while size > 0:
        distance = hk[0]
        p = hp[0]
        site = hs[0]
        size -= 1
        hk[0], hp[0], hs[0] = hk[size], hp[size], hs[size]
        node = 0
        while True:
            left = 2 * node + 1
            right = left + 1
            smallest = node
            if left < size and hk[left] < hk[smallest]:
                smallest = left
            if right < size and hk[right] < hk[smallest]:
                smallest = right
            if smallest == node:
                break
            hk[node], hk[smallest] = hk[smallest], hk[node]
            hp[node], hp[smallest] = hp[smallest], hp[node]
            hs[node], hs[smallest] = hs[smallest], hs[node]
            node = smallest

        if own[p] == site and distance <= d1[p] + tolerance:
            label = 0
        elif run[p] == site and distance <= d2[p] + tolerance:
            label = 1
        else:
            continue

        y = p // w
        x = p - y * w
        sp = s_field[p]
        for direction in range(8):
            ny = y + dys[direction]
            nx = x + dxs[direction]
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            q = ny * w + nx
            step = base_costs[direction, y, x] * 0.5 * (sp + s_field[q])
            candidate = distance + step
            touched = False
            if own[q] == site:
                if candidate + tolerance < d1[q]:
                    d1[q] = candidate
                    pr1[q] = p
                    pl1[q] = label
                    touched = True
            elif run[q] == site:
                if candidate + tolerance < d2[q]:
                    d2[q] = candidate
                    pr2[q] = p
                    pl2[q] = label
                    if d2[q] < d1[q]:
                        d1[q], d2[q] = d2[q], d1[q]
                        own[q], run[q] = run[q], own[q]
                        pr1[q], pr2[q] = pr2[q], pr1[q]
                        pl1[q], pl2[q] = pl2[q], pl1[q]
                    touched = True
            elif candidate + tolerance < d1[q]:
                d2[q], run[q], pr2[q], pl2[q] = d1[q], own[q], pr1[q], pl1[q]
                d1[q], own[q], pr1[q], pl1[q] = candidate, site, p, label
                touched = True
            elif candidate + tolerance < d2[q]:
                d2[q], run[q], pr2[q], pl2[q] = candidate, site, p, label
                touched = True
            if not touched:
                continue
            if size >= cap:
                cap *= 2
                nk = np.empty(cap, dtype=np.float64)
                npp = np.empty(cap, dtype=np.int32)
                ns = np.empty(cap, dtype=np.int32)
                nk[:size] = hk[:size]
                npp[:size] = hp[:size]
                ns[:size] = hs[:size]
                hk, hp, hs = nk, npp, ns
            hk[size], hp[size], hs[size] = candidate, q, site
            child = size
            size += 1
            while child > 0:
                parent = (child - 1) // 2
                if hk[parent] <= hk[child]:
                    break
                hk[parent], hk[child] = hk[child], hk[parent]
                hp[parent], hp[child] = hp[child], hp[parent]
                hs[parent], hs[child] = hs[child], hs[parent]
                child = parent

    return own, run, d1, d2, pr1, pr2, pl1, pl2


@_compile
def _accumulate_tree(order, seed_sensitivity, pr1, pr2, pl1, pl2,
                     base_costs, h, w):
    """Push loss sensitivity down both shortest-path forests.

    `order` lists the `2 * npix` (label, pixel) states sorted by decreasing
    distance, which is a reverse topological order of both forests because
    every edge cost is positive.  One pass therefore turns per-pixel distance
    sensitivities into per-edge path multiplicities, and each edge deposits
    half its share on each endpoint of the scale field.
    """
    npix = h * w
    acc = seed_sensitivity.copy()
    grad = np.zeros(npix, dtype=np.float64)
    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    for k in range(len(order)):
        state = order[k]
        label = state // npix
        p = state - label * npix
        if label == 0:
            parent = pr1[p]
            parent_label = pl1[p]
        else:
            parent = pr2[p]
            parent_label = pl2[p]
        if parent < 0:
            continue
        share = acc[state]
        if share == 0.0:
            continue
        acc[parent_label * npix + parent] += share
        py = parent // w
        px = parent - py * w
        dy = (p // w) - py
        dx = (p - (p // w) * w) - px
        direction = -1
        for d in range(8):
            if dys[d] == dy and dxs[d] == dx:
                direction = d
                break
        if direction < 0:
            continue
        base = base_costs[direction, py, px]
        grad[parent] += 0.5 * share * base
        grad[p] += 0.5 * share * base
    return grad


# ----------------------------------------------------------------------
# The shared model
# ----------------------------------------------------------------------

class SigmaVoronoi(TransportVoronoi):
    """Adds exact solves, enrichment, and metric sensitivity."""

    def __init__(self, image, config=None, ridge_kappa=4.0,
                 ridge_angles=16, ridge_bins=41):
        self.ridge_kappa = float(ridge_kappa)
        self.ridge_angles = int(ridge_angles)
        self.ridge_bins = int(ridge_bins)
        self.ridge_axis = None
        self.ridge_offset = None
        self.scale_field = None
        self.solve_stats = {}
        super().__init__(image, config)
        self.scale_field = np.ones(self.npix, dtype=np.float64)

    # -- assembly ------------------------------------------------------

    def _blend_weights(self, softness):
        valid = self.second >= 0
        other = np.where(valid, self.second, self.owner)
        z = np.clip(0.5 * float(softness) * (self.d2 - self.d1), -50.0, 50.0)
        w1 = 1.0 / (1.0 + np.exp(-z))
        w1[~valid] = 1.0
        return valid, other, w1, 1.0 - w1

    def _cell_basis(self, site_ids, spacing, angles, enriched):
        sx, sy = self.seeds[site_ids, 0], self.seeds[site_ids, 1]
        ct, st = np.cos(angles[site_ids]), np.sin(angles[site_ids])
        dx, dy = self.xf - sx, self.yf - sy
        q = (dx * ct + dy * st) / spacing
        r = (-dx * st + dy * ct) / spacing
        columns = [np.ones(self.npix), q, r]
        if enriched:
            axis = self.ridge_axis[site_ids]
            offset = self.ridge_offset[site_ids]
            proj = (dx * np.cos(axis) + dy * np.sin(axis)) / spacing
            # A bounded enrichment.  Unlike a quadratic, |tanh| <= 1, so the
            # partition of unity cannot manufacture an overshoot where two
            # cells both extrapolate.  That boundedness, not the shape, is
            # why this composes and the quadratic patch did not.
            columns.append(np.tanh(self.ridge_kappa * (proj - offset)))
        return np.column_stack(columns)

    def assemble(self, softness, enriched=False):
        """Build the coupled design matrix and its normal matrix."""
        n = len(self.seeds)
        spacing = max(math.sqrt(self.npix / max(n, 1)), 1e-9)
        angles, _ = self._site_frames()
        valid, other, w1, w2 = self._blend_weights(softness)
        width = 4 if enriched else 3
        first = self._cell_basis(self.owner, spacing, angles, enriched)
        second = self._cell_basis(other, spacing, angles, enriched)

        rows = np.repeat(np.arange(self.npix, dtype=np.int32), width)
        parts = np.tile(np.arange(width, dtype=np.int32), self.npix)
        first_cols = width * np.repeat(self.owner, width) + parts
        first_data = (first * w1[:, None]).ravel()
        visible = np.flatnonzero(valid)
        other_rows = np.repeat(visible, width)
        other_parts = np.tile(np.arange(width, dtype=np.int32), len(visible))
        other_cols = width * np.repeat(other[valid], width) + other_parts
        other_data = (second[valid] * w2[valid, None]).ravel()

        design = sparse.coo_matrix((
            np.concatenate([first_data, other_data]),
            (np.concatenate([rows, other_rows]),
             np.concatenate([first_cols, other_cols]))),
            shape=(self.npix, width * n)).tocsr()
        base_reg = [1e-5, 2e-3, 2e-3, 2e-3][:width]
        regularization = np.tile(np.array(base_reg, dtype=np.float64), n)
        return {
            "design": design, "width": width, "n": n, "spacing": spacing,
            "first": first, "second": second, "valid": valid, "other": other,
            "w1": w1, "w2": w2, "reg": regularization, "softness": softness,
        }

    def solve_exact(self, target, ctx):
        """Direct factorization of the normal matrix.

        LSMR converges to this answer; factoring reaches it.  The point is
        not only speed: the factorization is what makes the deletion and
        addition algebra below exact rather than estimated.
        """
        design = ctx["design"]
        gram = (design.T @ design).tocsc()
        gram = gram + sparse.diags(ctx["reg"], format="csc")
        t0 = time.perf_counter()
        lu = splu(gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
        factor_ms = (time.perf_counter() - t0) * 1000.0
        n, width = ctx["n"], ctx["width"]
        coeff = np.zeros((n, width, 3), dtype=np.float64)
        fitted = np.zeros((self.npix, 3), dtype=np.float64)
        pred_first = np.zeros((self.npix, 3), dtype=np.float64)
        pred_second = np.zeros((self.npix, 3), dtype=np.float64)
        for channel in range(3):
            rhs = design.T @ target[..., channel].ravel()
            solution = lu.solve(rhs)
            coeff[:, :, channel] = solution.reshape(n, width)
            block = solution.reshape(n, width)
            pred_first[:, channel] = np.sum(
                block[self.owner] * ctx["first"], axis=1)
            pred_second[:, channel] = np.sum(
                block[ctx["other"]] * ctx["second"], axis=1)
            fitted[:, channel] = (
                ctx["w1"] * pred_first[:, channel] +
                ctx["w2"] * pred_second[:, channel])
        self.solve_stats = {
            "factor_ms": factor_ms,
            "unknowns": int(width * n),
            "gram_nnz": int(gram.nnz),
        }
        return {
            "field": fitted.reshape(self.h, self.w, 3),
            "coeff": coeff, "lu": lu, "gram": gram,
            "pred_first": pred_first, "pred_second": pred_second,
        }

    def _solve_fused(self, target, softness, enriched=False, graph=None):
        """Exact solve assembled directly on the measured co-ownership graph."""
        n = len(self.seeds)
        spacing = max(math.sqrt(self.npix / max(n, 1)), 1e-9)
        angles, _ = self._site_frames()
        valid, other, w1, w2 = self._blend_weights(softness)
        width = 4 if enriched else 3
        first = self._cell_basis(self.owner, spacing, angles, enriched)
        second = self._cell_basis(other, spacing, angles, enriched)
        if graph is None or graph.width != width:
            graph = coownership_graph(
                self.owner, other, valid, n, width=width)
        regularization = np.tile(
            np.asarray([1e-5, 2e-3, 2e-3, 2e-3][:width],
                       dtype=np.float64), n)
        gram, rhs, _ = assemble_normal(
            self.owner, other, valid, w1, w2, first, second,
            target.reshape(-1, 3), graph, regularization)
        t0 = time.perf_counter()
        lu = splu(gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
        factor_ms = (time.perf_counter() - t0) * 1000.0
        coeff = lu.solve(rhs).reshape(n, width, 3)
        field, pred_first, pred_second = render_partition(
            coeff, self.owner, other, valid, w1, w2, first, second)
        self.solve_stats = {
            "factor_ms": factor_ms,
            "unknowns": int(width * n),
            "gram_nnz": int(gram.nnz),
            "assembly": "bfft fused measured graph",
        }
        context = {
            "width": width, "n": n, "spacing": spacing,
            "first": first, "second": second, "valid": valid, "other": other,
            "w1": w1, "w2": w2, "reg": regularization,
            "softness": float(softness), "graph": graph,
        }
        solved = {
            "field": field.reshape(self.h, self.w, 3),
            "coeff": coeff, "lu": lu, "gram": gram, "rhs": rhs,
            "pred_first": pred_first, "pred_second": pred_second,
        }
        return context, solved

    def _apply_direct_fields(self, base, detail, action):
        """Install a directly solved cartoon/detail pair in the live model."""
        from bfft.effects import lab_to_srgb

        precision = float(np.clip(self.cfg.detail_precision, 0.0, 1.5))
        self.cartoon_reconstruction = base
        self.texture_reconstruction = detail
        self.reconstruction = base + precision * detail
        base_delta = self.base_lab - base
        detail_delta = self.detail_lab - detail
        delta = self.lab - self.reconstruction
        self.cartoon_error = np.sqrt(
            base_delta[..., 0] ** 2 +
            1.5 * np.sum(base_delta[..., 1:] ** 2, axis=2))
        self.texture_error = np.sqrt(
            detail_delta[..., 0] ** 2 +
            1.5 * np.sum(detail_delta[..., 1:] ** 2, axis=2))
        self.error = np.sqrt(
            delta[..., 0] ** 2 +
            1.5 * np.sum(delta[..., 1:] ** 2, axis=2))
        self._update_allocation_pressure()
        rgb = np.clip(lab_to_srgb(self.reconstruction), 0.0, 1.0)
        self.rgb_mse = float(np.mean((self.rgb - rgb) ** 2))
        self.psnr = -10.0 * math.log10(max(self.rgb_mse, 1e-12))
        self.last_action = str(action)

    def _solve_direct_pair(self, cartoon_softness=4.0,
                           texture_softness=16.0, enriched=False):
        """Factor and solve both BFFT fields under the current ownership."""
        fields = {}
        graph = None
        for name, target, softness in (
                ("base", self.base_lab, cartoon_softness),
                ("detail", self.detail_lab, texture_softness)):
            plain, plain_solution = self._solve_fused(
                target, softness, enriched=False, graph=graph)
            graph = plain["graph"]
            if enriched:
                self.measure_ridge(
                    target - plain_solution["field"], plain)
                context, solution = self._solve_fused(
                    target, softness, enriched=True, graph=None)
            else:
                context = plain
                solution = plain_solution
            fields[name] = (context, solution)
        return fields

    def solve_direct_coupled(self, cartoon_softness=4.0,
                             texture_softness=16.0, enriched=False):
        """Public exact replacement for the iterative coupled viewer solve."""
        t0 = time.perf_counter()
        fields = self._solve_direct_pair(
            cartoon_softness=cartoon_softness,
            texture_softness=texture_softness,
            enriched=enriched)
        self._apply_direct_fields(
            fields["base"][1]["field"],
            fields["detail"][1]["field"],
            "exact coupled + measured ridge" if enriched
            else "exact coupled solve")
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return fields

    def step_direct(self, split=True, cartoon_softness=4.0,
                    texture_softness=16.0):
        """Subdivide once and fit that new geometry with the exact core path."""
        for name in ("fusion_labels", "fusion_groups", "fusion_trace"):
            if hasattr(self, name):
                delattr(self, name)
        started = time.perf_counter()
        if not split:
            self._assign()
            self.solve_direct_coupled(
                cartoon_softness, texture_softness, enriched=False)
            self.last_ms = (time.perf_counter() - started) * 1000.0
            return self.reconstruction

        old_seeds = self.seeds.copy()
        old_marks = self.marks.copy()
        old_parents = self.parents.copy()
        old_generations = self.generations.copy()
        old_psnr = float(self.psnr)
        self._lloyd_update()
        action = self._subdivide()
        self._assign()
        fields = self._solve_direct_pair(
            cartoon_softness, texture_softness, enriched=False)
        self._apply_direct_fields(
            fields["base"][1]["field"], fields["detail"][1]["field"],
            f"{action} + exact core fit")
        gain = self.psnr - old_psnr
        if action == "swap" and gain <= 1e-4:
            self.seeds = old_seeds
            self.marks = old_marks
            self.parents = old_parents
            self.generations = old_generations
            self._assign()
            fields = self._solve_direct_pair(
                cartoon_softness, texture_softness, enriched=False)
            self._apply_direct_fields(
                fields["base"][1]["field"], fields["detail"][1]["field"],
                "rejected exact-core swap")
            self.last_action = "rejected exact-core swap"
            self.last_gain = 0.0
            self.stagnation += 1
        else:
            self.last_action = f"{action} + exact core fit"
            self.last_gain = gain
            self.stagnation = 0 if gain > 1e-4 else self.stagnation + 1
            self.iteration += 1
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return self.reconstruction

    def optimize_site_reach(self, steps=18, learning_rate=0.35,
                            cartoon_softness=4.0,
                            texture_softness=16.0,
                            enriched=False):
        """Descend one additive reach value per site with exact gradients.

        This is Sigma's validated geodesic power-diagram experiment made
        available to the interactive viewer.  Geometry, site count, and the
        BFFT metric remain fixed.  Every proposed reach update is evaluated
        with a direct coupled solve and backtracking; the best measured state
        is retained.
        """
        t0 = time.perf_counter()
        n = len(self.seeds)
        current = getattr(self, "reach_offset", None)
        offset = (
            np.zeros(n, dtype=np.float64)
            if current is None or len(current) != n
            else np.asarray(current, dtype=np.float64).copy())
        best_offset = offset.copy()
        best_mse = np.inf
        previous_mse = np.inf
        previous = offset.copy()
        rate = float(learning_rate)
        history = []
        precision = float(np.clip(self.cfg.detail_precision, 0.0, 1.5))

        for sweep in range(max(0, int(steps)) + 1):
            forest = self.walk_with_predecessors(
                self.scale_field, reach_offset=offset)
            fields = self._solve_direct_pair(
                cartoon_softness=cartoon_softness,
                texture_softness=texture_softness,
                enriched=False)
            self._apply_direct_fields(
                fields["base"][1]["field"],
                fields["detail"][1]["field"],
                "site-reach descent")
            history.append(float(self.psnr))
            if self.rgb_mse < best_mse:
                best_mse = self.rgb_mse
                best_offset = offset.copy()
            if self.rgb_mse > previous_mse:
                offset = previous.copy()
                rate *= 0.5
                if rate < 0.02:
                    break
            else:
                previous_mse = self.rgb_mse
                previous = offset.copy()
            if sweep >= int(steps):
                break

            packed = []
            for name, scale, softness in (
                    ("base", 1.0, cartoon_softness),
                    ("detail", precision, texture_softness)):
                context, solved = fields[name]
                packed.append({
                    "w1": context["w1"],
                    "w2": context["w2"],
                    "softness": softness,
                    "scale": scale,
                    "pred_first": solved["pred_first"],
                    "pred_second": solved["pred_second"],
                })
            _, sensitivity = self.metric_gradient(packed, forest)
            gradient = self.weight_gradient(sensitivity)
            magnitude = float(np.percentile(np.abs(gradient), 95.0))
            if magnitude <= 0.0:
                break
            spacing = math.sqrt(self.npix / max(n, 1))
            offset = offset - rate * spacing * np.clip(
                gradient / magnitude, -4.0, 4.0)
            offset -= float(np.mean(offset))

        self.reach_offset = best_offset
        self.walk_with_predecessors(
            self.scale_field, reach_offset=best_offset)
        fields = self._solve_direct_pair(
            cartoon_softness=cartoon_softness,
            texture_softness=texture_softness,
            enriched=enriched)
        self._apply_direct_fields(
            fields["base"][1]["field"],
            fields["detail"][1]["field"],
            "site reach + measured ridge" if enriched
            else "site-reach descent")
        self.reach_history = history
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "history": history,
            "offset_span": (
                float(best_offset.min()), float(best_offset.max())),
            "steps": max(0, len(history) - 1),
            "enriched": bool(enriched),
        }

    # -- ridge enrichment ---------------------------------------------

    def measure_ridge(self, residual, ctx):
        """Closed-form best bounded ridge per cell, from sign statistics.

        For a fixed direction the optimal step threshold is where the
        cumulative weighted residual is extremal, because the sign basis has
        constant norm inside a cell.  Scanning a fixed angle set and taking
        the cumulative extremum therefore solves direction and offset
        together with two passes of counting and no search.
        """
        n, spacing = ctx["n"], ctx["spacing"]
        sx, sy = self.seeds[self.owner, 0], self.seeds[self.owner, 1]
        score, axis, offset = measure_residual_ridges(
            self.owner, ctx["w1"], residual.reshape(-1, 3),
            self.xf - sx, self.yf - sy, spacing, n,
            angles=self.ridge_angles, bins=self.ridge_bins,
            channel_weights=LAB_WEIGHTS)
        self.ridge_axis = axis
        self.ridge_offset = offset
        self.ridge_score = score
        return score

    # -- exact prices --------------------------------------------------

    def deletion_costs(self, solved, ctx, shortlist=None, probes=48,
                       seed=20260726):
        """Exact objective price for every current cell, with no shortlist."""
        exact, _ = exact_deletion_prices(
            solved["lu"], solved["coeff"], LAB_WEIGHTS)
        return exact, exact.copy()

    def addition_gains(self, solved, ctx, candidates, residual):
        """Exact Schur-complement gain of introducing a candidate cell.

        The local gain field prices a new affine patch as if the existing
        model could not absorb it.  The Schur complement prices what is left
        after the incumbent cells have taken what they can, which is the
        quantity the allocator actually wants.
        """
        lu, spacing = solved["lu"], ctx["spacing"]
        width = ctx["width"]
        design = ctx.get("design")
        if design is None:
            design = self.assemble(
                ctx["softness"], enriched=(width == 4))["design"]
        flat = residual.reshape(-1, 3)
        gains = np.zeros(len(candidates), dtype=np.float64)
        reach = 1.15 * spacing
        # The incumbent model's own pull on the residual.  It is nearly zero
        # because the normal equations hold, but carrying it keeps the price
        # exact rather than nearly exact.
        incumbent = np.column_stack([
            lu.solve(design.T @ flat[:, c]) for c in range(3)])
        for slot, pixel in enumerate(candidates):
            cy, cx = divmod(int(pixel), self.w)
            dx = self.xf - cx
            dy = self.yf - cy
            near = np.flatnonzero(dx * dx + dy * dy <= reach * reach)
            if near.size < 8:
                continue
            window = np.exp(
                -0.5 * (dx[near] ** 2 + dy[near] ** 2) / (0.55 * reach) ** 2)
            column = np.column_stack([
                window,
                window * dx[near] / spacing,
                window * dy[near] / spacing])
            block = sparse.csr_matrix(
                (column.T.ravel(),
                 (np.repeat(np.arange(3), near.size), np.tile(near, 3))),
                shape=(3, self.npix)).T
            cross = (design.T @ block).toarray()
            projected = np.column_stack([
                lu.solve(cross[:, k]) for k in range(3)])
            schur = (block.T @ block).toarray() - cross.T @ projected
            schur += 1e-9 * np.eye(3)
            inverse = np.linalg.pinv(schur)
            for c in range(3):
                effective = (
                    block.T @ flat[:, c] - cross[:, :].T @ incumbent[:, c])
                gains[slot] += LAB_WEIGHTS[c] * float(
                    effective @ inverse @ effective)
        return gains

    # -- metric sensitivity -------------------------------------------

    def weight_gradient(self, sensitivity):
        """Exact d(loss)/d(site reach), in two counts.

        An additive site weight shifts every distance in that site's whole
        subtree by the same amount, so the subtree sum the forest pass
        computes for a metric edge degenerates here into a sum over the
        cell.  This is the same envelope argument as `metric_gradient`, and
        it lands on the additive weights of a geodesic power diagram: the
        object semi-discrete optimal transport solves for prescribed masses,
        driven instead by measured reconstruction error.
        """
        n = len(self.seeds)
        owned = np.bincount(self.owner, weights=sensitivity, minlength=n)
        runner = np.zeros(n, dtype=np.float64)
        visible = self.second >= 0
        if np.any(visible):
            runner = np.bincount(
                self.second[visible], weights=sensitivity[visible],
                minlength=n)
        return owned - runner

    def walk_with_predecessors(self, scale_field=None, reach_offset=None):
        """Re-run assignment, keeping the forest the gradient needs."""
        if scale_field is None:
            scale_field = self.scale_field
        if reach_offset is None:
            reach_offset = getattr(self, "reach_offset", None)
        density_flat = self.density.ravel()
        density_scale = max(float(np.median(density_flat)), 1e-9)
        seed_x = np.clip(
            np.rint(self.seeds[:, 0]).astype(np.int64), 0, self.w - 1)
        seed_y = np.clip(
            np.rint(self.seeds[:, 1]).astype(np.int64), 0, self.h - 1)
        seed_p = (seed_y * self.w + seed_x).astype(np.int64)
        reach = self.cfg.site_reach * np.sqrt(
            density_flat[seed_p] / density_scale)
        if reach_offset is not None:
            reach = reach + np.asarray(reach_offset, dtype=np.float64)
        result = _dijkstra_two_best_pred(
            seed_p, reach.astype(np.float64), self._edge_cost_volume,
            np.ascontiguousarray(scale_field), self.h, self.w)
        own, run, d1, d2, pr1, pr2, pl1, pl2 = result
        self.owner, self.second = own, run
        self.d1, self.d2 = d1, d2
        self.site_regions = np.zeros(len(self.seeds), dtype=np.int32)
        return pr1, pr2, pl1, pl2

    def metric_gradient(self, fields, forest):
        """Exact d(loss)/d(scale) by envelope theorem on the walk.

        A geodesic distance is the minimum over paths of a sum of edge
        weights, hence concave and piecewise linear in those weights.  Its
        derivative is the indicator of the achieving path, so the whole
        gradient is a subtree sum over the predecessor forest.  No adjoint
        system is solved and no finite difference is taken.
        """
        pr1, pr2, pl1, pl2 = forest
        recon = np.zeros((self.npix, 3), dtype=np.float64)
        for field in fields:
            recon += field["scale"] * (
                field["w1"][:, None] * field["pred_first"] +
                field["w2"][:, None] * field["pred_second"])
        target = self.lab.reshape(-1, 3)
        delta = recon - target
        sensitivity = np.zeros(self.npix, dtype=np.float64)
        for field in fields:
            beta = 0.5 * float(field["softness"])
            gap = field["w1"] * field["w2"]
            difference = field["pred_first"] - field["pred_second"]
            partial = 2.0 * np.sum(
                LAB_WEIGHTS[None, :] * delta * difference,
                axis=1) * field["scale"]
            sensitivity += beta * gap * partial
        sensitivity[self.second < 0] = 0.0

        states = np.concatenate([-sensitivity, sensitivity])
        distances = np.concatenate([self.d1, self.d2])
        finite = np.isfinite(distances)
        order = np.argsort(-np.where(finite, distances, -np.inf),
                           kind="stable").astype(np.int64)
        order = order[finite[order]]
        grad = _accumulate_tree(
            order, states, pr1, pr2, pl1, pl2,
            self._edge_cost_volume, self.h, self.w)
        return grad, sensitivity


# ----------------------------------------------------------------------
# Diagnostics shared by every trial
# ----------------------------------------------------------------------

def residual_structure(model):
    """Fraction of residual energy an affine patch could still remove.

    The user's own reading of the cell view is that diffuse error is the
    best error.  This makes that reading a number: it is the removable share
    of what is left, so lower means the residual has stopped being shaped
    like anything the model can express.
    """
    spacing = math.sqrt(model.npix / max(len(model.seeds), 1))
    sigma = max(0.55 * spacing, 0.8)
    residual = model.lab - model.reconstruction
    removable = 0.0
    total = 0.0
    for channel, weight in enumerate(LAB_WEIGHTS):
        plane = residual[..., channel]
        mean = ndi.gaussian_filter(plane, sigma, mode="reflect")
        gx = ndi.gaussian_filter(plane, sigma, order=(0, 1), mode="reflect")
        gy = ndi.gaussian_filter(plane, sigma, order=(1, 0), mode="reflect")
        removable += weight * float(np.sum(
            mean * mean + sigma * sigma * (gx * gx + gy * gy)))
        total += weight * float(np.sum(plane * plane))
    return removable / max(total, 1e-15)
