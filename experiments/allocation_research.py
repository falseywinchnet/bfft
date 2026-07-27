#!/usr/bin/env python3
"""Independent nucleation experiments for the robust Voronoi checkpoint.

This module never changes the production viewer.  It prints a complete JSON
record so a result can be archived only after it has actually run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import lsmr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from transport_voronoi import CARTOON, Config, TransportVoronoi  # noqa: E402


IDEAS = {
    "uniform_control": (
        "Recovered robust allocator: uniform blue-noise nucleation."),
    "residual_continuous": (
        "Every initial blue-noise decision is continuously weighted by the "
        "one-stage residual focus map."),
    "residual_quota": (
        "Keep 75% uniform coverage; spend 25% of initial sites on residual "
        "focus with repulsion from the foundation."),
    "residual_alternating": (
        "Every fourth initial decision sees residual focus; intervening "
        "decisions repair spatial coverage."),
    "spin_rounding": (
        "Keep 75% uniform coverage, lift a redundant/complementary focus "
        "candidate graph to unit vectors, Gaussian-round it, and retain the "
        "best fixed-size subset."),
    "residual_mild": (
        "Retain blue-noise coverage with only a weak continuous residual "
        "density bias."),
    "residual_quota10": (
        "Keep 90% uniform coverage and reserve only 10% for focus anchors."),
    "residual_local_yield": (
        "Start fully uniform, then let each seed move at most half a spacing "
        "toward residual mass inside its own Euclidean cell."),
    "decomposition_pressure_mild": (
        "Keep initialization uniform and give current one-stage composition "
        "discrepancy a very small refinement-pressure weight."),
    "quadratic_postfit": (
        "Keep the recovered allocator and sites exactly fixed, but replace "
        "each affine color plane with a bounded quadratic local surface."),
    "hard_ownership": (
        "Use the recovered allocator and affine cells but remove all "
        "cross-cell blending."),
    "wide_overlap": (
        "Use gentler cross-cell blending to widen the transition band."),
    "sharp_overlap": (
        "Use sharper cross-cell blending to narrow the transition band."),
    "unbounded_affine": (
        "Remove the per-cell affine prediction bounds while retaining the "
        "same fitted slopes and sites."),
    "medium_overlap": (
        "Use a moderately wider global overlap as a midpoint check."),
    "adaptive_edge_overlap": (
        "Blend broadly in smooth regions and sharpen ownership continuously "
        "near strong cartoon/flow edges."),
    "objective_overlap_search": (
        "Run overlap branches 4, 6, and 10 and retain the branch minimizing "
        "RGB MSE + one-stage cartoon MSE + one-stage texture MSE."),
    "expected_affine_gain": (
        "Allocate refinement from the closed-form Gaussian-windowed affine "
        "matching-pursuit gain rather than residual magnitude."),
    "reliability_blend": (
        "Blend neighboring affine predictions by inverse fitted variance "
        "including support-geometry leverage."),
    "gain_plus_reliability": (
        "Combine measured affine-gain allocation with reliability blending."),
    "decomposition_expected_gain": (
        "Spend the sum of expected affine reduction in RGB/OKLab residual, "
        "signed cartoon discrepancy, and signed texture discrepancy."),
    "scheduled_expected_gain": (
        "Begin with RGB/OKLab expected gain, then continuously introduce "
        "cartoon/texture expected gain as the cell budget fills."),
    "hessian_expected_gain": (
        "Use the absolute Hessian of the BFFT cartoon/flow support as the "
        "cell metric, paired with expected affine-gain allocation."),
    "reducibility_expected_gain": (
        "Gate each cell's expected affine gain by the fraction of its local "
        "residual energy represented by a constant-plus-gradient model."),
    "dipole_add_expected_gain": (
        "At a contour-crossing residual, add two geometry-nominated children "
        "on opposite sides while retaining the parent."),
    "dipole_replace_expected_gain": (
        "At a contour-crossing residual, replace the straddling parent by "
        "two geometry-nominated children, a net one-cell split."),
    "global_partition_fit": (
        "Keep expected-gain geometry, then solve every affine cell jointly "
        "under the actual two-cell partition-of-unity renderer."),
    "global_decomposition_fit": (
        "Jointly solve the cartoon and texture affine fields under the actual "
        "partition of unity, then recompose them."),
    "multiscale_global_fit": (
        "Jointly solve cartoon with broad overlap and texture with sharp "
        "overlap, then recompose the two genuinely distinct fields."),
    "staged_gain_25": (
        "Use robust integrated error for the first quarter of refinement, "
        "then switch to expected affine gain."),
    "staged_gain_50": (
        "Use robust integrated error for the first half of refinement, then "
        "switch to expected affine gain."),
}


class NucleationExperiment(TransportVoronoi):
    def __init__(self, image, cfg, idea):
        self.idea = idea
        self.spin_diagnostics = {}
        if idea == "decomposition_pressure_mild":
            cfg.composition_discrepancy_weight = 0.06
        if idea == "hard_ownership":
            cfg.softness = 0.0
        elif idea == "wide_overlap":
            cfg.softness = 4.0
        elif idea == "sharp_overlap":
            cfg.softness = 20.0
        elif idea == "unbounded_affine":
            cfg.bounded_gradients = False
        elif idea == "medium_overlap":
            cfg.softness = 6.0
        super().__init__(image, cfg)

    def _farthest_with_weight(self, count, weight, existing=None):
        count = max(0, min(int(count), self.npix))
        if count == 0:
            return np.empty((0, 2), dtype=np.float64)
        weight = np.maximum(np.asarray(weight).ravel(), 1e-6)
        if existing is None or len(existing) == 0:
            first = int(np.argmax(weight))
            chosen = [first]
            min_d2 = (
                (self.xf - self.xf[first]) ** 2 +
                (self.yf - self.yf[first]) ** 2)
        else:
            existing = np.asarray(existing)
            min_d2 = np.full(self.npix, np.inf)
            for x, y in existing:
                np.minimum(
                    min_d2,
                    (self.xf - x) ** 2 + (self.yf - y) ** 2,
                    out=min_d2)
            chosen = []
        blocked = np.zeros(self.npix, dtype=bool)
        for _ in range(count - len(chosen)):
            score = min_d2 * weight
            score[blocked] = -1
            index = int(np.argmax(score))
            chosen.append(index)
            blocked[index] = True
            np.minimum(
                min_d2,
                (self.xf - self.xf[index]) ** 2 +
                (self.yf - self.yf[index]) ** 2,
                out=min_d2)
        return np.column_stack([
            self.xf[chosen], self.yf[chosen],
        ]).astype(np.float64)

    def _weighted_farthest(self, count):
        if self.idea in (
                "uniform_control", "decomposition_pressure_mild",
                "quadratic_postfit", "hard_ownership", "wide_overlap",
                "sharp_overlap", "unbounded_affine", "medium_overlap",
                "adaptive_edge_overlap", "expected_affine_gain",
                "reliability_blend", "gain_plus_reliability",
                "decomposition_expected_gain", "scheduled_expected_gain",
                "hessian_expected_gain", "reducibility_expected_gain",
                "dipole_add_expected_gain", "dipole_replace_expected_gain",
                "global_partition_fit", "global_decomposition_fit",
                "multiscale_global_fit", "staged_gain_25",
                "staged_gain_50"):
            return super()._weighted_farthest(count)
        focus = np.asarray(self.residual_memory, dtype=np.float64)
        if self.idea == "residual_continuous":
            return self._farthest_with_weight(
                count, 1.0 + 4.0 * focus)
        if self.idea == "residual_mild":
            return self._farthest_with_weight(
                count, 1.0 + 0.75 * focus)
        if self.idea == "residual_alternating":
            chosen = []
            min_d2 = np.full(self.npix, np.inf)
            blocked = np.zeros(self.npix, dtype=bool)
            for index_in_order in range(int(count)):
                weight = (
                    1.0 + 5.0 * focus.ravel()
                    if index_in_order % 4 == 3 else
                    np.ones(self.npix))
                if not chosen:
                    index = int(np.argmax(weight))
                else:
                    score = min_d2 * weight
                    score[blocked] = -1
                    index = int(np.argmax(score))
                chosen.append(index)
                blocked[index] = True
                np.minimum(
                    min_d2,
                    (self.xf - self.xf[index]) ** 2 +
                    (self.yf - self.yf[index]) ** 2,
                    out=min_d2)
            return np.column_stack([
                self.xf[chosen], self.yf[chosen],
            ]).astype(np.float64)

        if self.idea == "residual_local_yield":
            return self._local_residual_yield(count, focus)
        fraction = 0.10 if self.idea == "residual_quota10" else 0.25
        uniform_count = int(round((1.0 - fraction) * count))
        foundation = self._uniform_farthest(uniform_count)
        focus_count = int(count) - uniform_count
        if self.idea in ("residual_quota", "residual_quota10"):
            focused = self._farthest_with_weight(
                focus_count, 0.05 + focus ** 2, foundation)
            return np.vstack([foundation, focused])
        if self.idea == "spin_rounding":
            return self._spin_nucleation(
                foundation, focus_count, focus)
        raise ValueError(self.idea)

    def _geometry(self):
        geometry = super()._geometry()
        if self.idea != "hessian_expected_gain":
            return geometry
        tangent, coherence, density, _, _, _ = geometry
        # The fitted patches are affine, so first derivative magnitude is not
        # itself approximation demand.  Curvature of the BFFT support is.
        support = (
            self.cartoon / 255.0 +
            0.70 * self.flow / 255.0)
        sigma = 1.1
        hxx = ndi.gaussian_filter(
            support, sigma, order=(0, 2), mode="reflect")
        hyy = ndi.gaussian_filter(
            support, sigma, order=(2, 0), mode="reflect")
        hxy = ndi.gaussian_filter(
            support, sigma, order=(1, 1), mode="reflect")
        trace = hxx + hyy
        discriminant = np.sqrt(
            np.maximum((hxx - hyy) ** 2 + 4.0 * hxy * hxy, 0.0))
        lambda_1 = 0.5 * (trace + discriminant)
        lambda_2 = 0.5 * (trace - discriminant)
        theta = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
        ct, st = np.cos(theta), np.sin(theta)
        a1, a2 = np.abs(lambda_1), np.abs(lambda_2)
        abs_xx = a1 * ct * ct + a2 * st * st
        abs_xy = (a1 - a2) * ct * st
        abs_yy = a1 * st * st + a2 * ct * ct
        curvature = a1 + a2
        scale = max(float(np.percentile(curvature, 98.0)), 1e-9)
        strength = 2.5 * float(self.cfg.anisotropy) / scale
        # Preserve the BFFT cartoon/flow barrier, but replace the texture
        # structure-tensor cost with the model-matched curvature metric.
        mxx = self.cartoon_metric_xx + strength * abs_xx
        mxy = self.cartoon_metric_xy + strength * abs_xy
        myy = self.cartoon_metric_yy + strength * abs_yy
        return tangent, coherence, density, mxx, mxy, myy

    def _render(self):
        super()._render()
        if self.idea in ("reliability_blend", "gain_plus_reliability"):
            self._render_reliability()
            return
        if self.idea != "adaptive_edge_overlap":
            return
        base_first = self._predict_for(self.owner, "base")
        detail_first = self._predict_for(self.owner, "detail")
        valid = self.second >= 0
        safe_second = np.where(valid, self.second, self.owner)
        base_second = self._predict_for(safe_second, "base")
        detail_second = self._predict_for(safe_second, "detail")
        local_softness = (
            4.0 + 10.0 * self.edge_strength.ravel())
        z = np.clip(
            0.5 * local_softness * (self.d2 - self.d1), -50.0, 50.0)
        weight = 1.0 / (1.0 + np.exp(-z))
        weight[~valid] = 1.0
        base = (
            base_first * weight[:, None] +
            base_second * (1.0 - weight[:, None]))
        detail = (
            detail_first * weight[:, None] +
            detail_second * (1.0 - weight[:, None]))
        self.cartoon_reconstruction = base.reshape(self.h, self.w, 3)
        self.texture_reconstruction = detail.reshape(self.h, self.w, 3)
        self.reconstruction = (
            self.cartoon_reconstruction +
            self.cfg.detail_precision * self.texture_reconstruction)
        base_delta = self.base_lab - self.cartoon_reconstruction
        detail_delta = self.detail_lab - self.texture_reconstruction
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
        rgb = np.clip(bfft.lab_to_srgb(self.reconstruction), 0.0, 1.0)
        mse = float(np.mean((self.rgb - rgb) ** 2))
        self.psnr = -10.0 * math.log10(max(mse, 1e-12))

    def _fit_models(self):
        super()._fit_models()
        if self.idea not in ("reliability_blend", "gain_plus_reliability"):
            return
        n = len(self.seeds)
        ids = self.owner
        angles, _ = self._site_frames()
        sx, sy = self.seeds[ids, 0], self.seeds[ids, 1]
        ct, st = np.cos(angles[ids]), np.sin(angles[ids])
        dx, dy = self.xf - sx, self.yf - sy
        q, r = dx * ct + dy * st, -dx * st + dy * ct
        count = np.bincount(ids, minlength=n).astype(np.float64)
        normal = np.zeros((n, 3, 3), dtype=np.float64)
        normal[:, 0, 0] = np.maximum(count, 1.0)
        normal[:, 0, 1] = normal[:, 1, 0] = np.bincount(
            ids, weights=q, minlength=n)
        normal[:, 0, 2] = normal[:, 2, 0] = np.bincount(
            ids, weights=r, minlength=n)
        normal[:, 1, 1] = np.bincount(
            ids, weights=q * q, minlength=n) + 1e-4
        normal[:, 2, 2] = np.bincount(
            ids, weights=r * r, minlength=n) + 1e-4
        normal[:, 1, 2] = normal[:, 2, 1] = np.bincount(
            ids, weights=q * r, minlength=n)
        self._normal_inverse = np.linalg.pinv(normal, hermitian=True)
        prediction = (
            self._predict_for(ids, "base") +
            self._predict_for(ids, "detail"))
        residual = self.lab.reshape(-1, 3) - prediction
        perceptual_sse = (
            residual[:, 0] ** 2 +
            1.5 * np.sum(residual[:, 1:] ** 2, axis=1))
        sse = np.bincount(ids, weights=perceptual_sse, minlength=n)
        self._fit_variance = sse / np.maximum(count - 3.0, 1.0)
        floor = max(float(np.percentile(
            self._fit_variance[self._fit_variance > 0], 10.0))
            if np.any(self._fit_variance > 0) else 1e-8, 1e-8)
        self._fit_variance = np.maximum(self._fit_variance, floor)

    def _prediction_variance(self, site_ids):
        angles, _ = self._site_frames()
        sx, sy = self.seeds[site_ids, 0], self.seeds[site_ids, 1]
        ct, st = np.cos(angles[site_ids]), np.sin(angles[site_ids])
        dx, dy = self.xf - sx, self.yf - sy
        g = np.column_stack([
            np.ones(self.npix),
            dx * ct + dy * st,
            -dx * st + dy * ct,
        ])
        leverage = np.einsum(
            "pi,pij,pj->p", g,
            self._normal_inverse[site_ids], g)
        return self._fit_variance[site_ids] * (
            1.0 + np.maximum(leverage, 0.0))

    def _render_reliability(self):
        first_base = self._predict_for(self.owner, "base")
        first_detail = self._predict_for(self.owner, "detail")
        valid = self.second >= 0
        other_ids = np.where(valid, self.second, self.owner)
        other_base = self._predict_for(other_ids, "base")
        other_detail = self._predict_for(other_ids, "detail")
        first_var = self._prediction_variance(self.owner)
        other_var = self._prediction_variance(other_ids)
        weight = other_var / np.maximum(first_var + other_var, 1e-12)
        weight[~valid] = 1.0
        base = (
            first_base * weight[:, None] +
            other_base * (1.0 - weight[:, None]))
        detail = (
            first_detail * weight[:, None] +
            other_detail * (1.0 - weight[:, None]))
        self.cartoon_reconstruction = base.reshape(self.h, self.w, 3)
        self.texture_reconstruction = detail.reshape(self.h, self.w, 3)
        self.reconstruction = (
            self.cartoon_reconstruction +
            self.cfg.detail_precision * self.texture_reconstruction)
        base_delta = self.base_lab - self.cartoon_reconstruction
        detail_delta = self.detail_lab - self.texture_reconstruction
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
        rgb = np.clip(bfft.lab_to_srgb(self.reconstruction), 0.0, 1.0)
        self.rgb_mse = float(np.mean((self.rgb - rgb) ** 2))
        self.psnr = -10.0 * math.log10(max(self.rgb_mse, 1e-12))

    def _update_allocation_pressure(self):
        super()._update_allocation_pressure()
        if self.idea not in (
                "expected_affine_gain", "gain_plus_reliability",
                "decomposition_expected_gain", "scheduled_expected_gain",
                "hessian_expected_gain", "reducibility_expected_gain",
                "dipole_add_expected_gain", "dipole_replace_expected_gain",
                "global_partition_fit", "global_decomposition_fit",
                "multiscale_global_fit", "staged_gain_25",
                "staged_gain_50"):
            return
        if self.idea in ("staged_gain_25", "staged_gain_50"):
            fraction = 0.25 if self.idea == "staged_gain_25" else 0.50
            switch_at = (
                self.cfg.initial_cells +
                fraction * (
                    self.cfg.max_cells - self.cfg.initial_cells))
            if len(self.seeds) < switch_at:
                return
        spacing = math.sqrt(self.npix / max(len(self.seeds), 1))
        sigma = max(0.55 * spacing, 0.8)
        residual = self.lab - self.reconstruction
        gain = np.zeros((self.h, self.w), dtype=np.float64)
        channel_weights = (1.0, 1.5, 1.5)
        for channel, channel_weight in enumerate(channel_weights):
            plane = residual[..., channel]
            mean = ndi.gaussian_filter(
                plane, sigma, mode="reflect")
            gx = ndi.gaussian_filter(
                plane, sigma, order=(0, 1), mode="reflect")
            gy = ndi.gaussian_filter(
                plane, sigma, order=(1, 0), mode="reflect")
            gain += channel_weight * (
                mean * mean + sigma * sigma * (gx * gx + gy * gy))
        if self.idea == "reducibility_expected_gain":
            residual_energy = np.zeros_like(gain)
            for channel, channel_weight in enumerate(channel_weights):
                residual_energy += channel_weight * ndi.gaussian_filter(
                    residual[..., channel] ** 2, sigma, mode="reflect")
            # Evaluate reducibility at cell level.  This avoids declaring an
            # isolated zero crossing irreducible merely because its pointwise
            # ratio is small, and it matches the allocator's budget unit.
            owners = self.owner
            n = len(self.seeds)
            gain_mass = np.bincount(
                owners, weights=gain.ravel(), minlength=n)
            energy_mass = np.bincount(
                owners, weights=residual_energy.ravel(), minlength=n)
            ratio = gain_mass / np.maximum(energy_mass, 1e-12)
            positive = ratio[ratio > 0]
            reference = max(
                float(np.median(positive)) if positive.size else 0.0,
                1e-12)
            cell_gate = np.sqrt(np.clip(ratio / reference, 0.08, 2.5))
            gain *= cell_gate[owners].reshape(self.h, self.w)
        if self.idea in (
                "decomposition_expected_gain", "scheduled_expected_gain"):
            if self.idea == "scheduled_expected_gain":
                progress = np.clip(
                    (len(self.seeds) - self.cfg.initial_cells) /
                    max(self.cfg.max_cells - self.cfg.initial_cells, 1),
                    0.0, 1.0)
                decomposition_weight = 0.70 * progress * progress
            else:
                decomposition_weight = 0.35
            for plane in (
                    self.cartoon_discrepancy_signed,
                    self.texture_discrepancy_signed):
                mean = ndi.gaussian_filter(
                    plane, sigma, mode="reflect")
                gx = ndi.gaussian_filter(
                    plane, sigma, order=(0, 1), mode="reflect")
                gy = ndi.gaussian_filter(
                    plane, sigma, order=(1, 0), mode="reflect")
                component_gain = (
                    mean * mean +
                    sigma * sigma * (gx * gx + gy * gy))
                # Normalize each account by its robust median so no one
                # channel wins merely because its units are larger.
                positive = component_gain[component_gain > 0]
                account_scale = max(
                    float(np.median(positive))
                    if positive.size else 0.0, 1e-12)
                gain += (
                    decomposition_weight * component_gain / account_scale *
                    max(
                    float(np.median(gain[gain > 0]))
                    if np.any(gain > 0) else 1e-12, 1e-12))
        # Retain the proven clearance and edge-crowding terms from the
        # control; replace only its residual-magnitude currency.
        clearance = np.maximum(
            self.d1.reshape(self.h, self.w), 0.0)
        scale = max(float(np.percentile(clearance, 90.0)), 1e-6)
        coverage = 1.0 + 0.75 * np.sqrt(
            np.clip(clearance / scale, 0.0, 4.0))
        edge_penalty = 1.0 / (1.0 + 2.5 * self.edge_strength)
        self.allocation_pressure = gain * coverage * edge_penalty
        self.expected_gain = gain

    def _subdivide(self):
        if self.idea not in (
                "dipole_add_expected_gain", "dipole_replace_expected_gain"):
            return super()._subdivide()
        room = self.cfg.max_cells - len(self.seeds)
        budget = min(max(0, int(self.cfg.split_batch)), room)
        if budget <= 0:
            return self._rebalance()
        n = len(self.seeds)
        pressure = self.allocation_pressure.ravel()
        score = np.bincount(self.owner, weights=pressure, minlength=n)
        cells = np.argsort(score)[::-1]
        residual_l = (self.lab - self.reconstruction)[..., 0]
        residual_scale = max(
            float(np.percentile(np.abs(residual_l), 70.0)), 1e-5)
        spacing = math.sqrt(self.npix / max(n, 1))
        delta = float(np.clip(0.32 * spacing, 1.4, 3.2))
        additions = []
        parents = []
        generations = []
        relocated = set()
        blocked = np.zeros(self.npix, dtype=bool)

        def valid_birth(x, y):
            if x < 0 or x >= self.w or y < 0 or y >= self.h:
                return False
            if self.seeds.size:
                d2 = (
                    (self.seeds[:, 0] - x) ** 2 +
                    (self.seeds[:, 1] - y) ** 2)
                if float(d2.min()) < 1.0:
                    return False
            if additions:
                proposed = np.asarray(additions)
                d2 = (
                    (proposed[:, 0] - x) ** 2 +
                    (proposed[:, 1] - y) ** 2)
                if float(d2.min()) < 1.0:
                    return False
            return True

        for cell in cells:
            pix = np.flatnonzero((self.owner == cell) & ~blocked)
            if pix.size == 0:
                continue
            idx = int(pix[np.argmax(pressure[pix])])
            y0, x0 = divmod(idx, self.w)
            # self.angle is the tangent supplied by BFFT geometry.
            normal = float(self.angle[y0, x0] - 0.5 * math.pi)
            nx, ny = math.cos(normal), math.sin(normal)
            xm, ym = x0 - delta * nx, y0 - delta * ny
            xp, yp = x0 + delta * nx, y0 + delta * ny
            imx = int(np.clip(round(xm), 0, self.w - 1))
            imy = int(np.clip(round(ym), 0, self.h - 1))
            ipx = int(np.clip(round(xp), 0, self.w - 1))
            ipy = int(np.clip(round(yp), 0, self.h - 1))
            rm, rp = residual_l[imy, imx], residual_l[ipy, ipx]
            is_dipole = (
                self.edge_strength[y0, x0] > 0.20 and
                rm * rp < 0.0 and
                abs(rm - rp) > 0.8 * residual_scale and
                valid_birth(xm, ym) and valid_birth(xp, yp))
            needed = 2 if (
                is_dipole and self.idea == "dipole_add_expected_gain") else 1
            if len(additions) + needed > budget:
                continue
            parent = int(self.owner[idx])
            generation = int(self.generations[parent]) + 1
            if is_dipole:
                if self.idea == "dipole_replace_expected_gain":
                    self.seeds[parent] = (xm, ym)
                    relocated.add(parent)
                    additions.append((xp, yp))
                    parents.append(parent)
                    generations.append(generation)
                else:
                    additions.extend([(xm, ym), (xp, yp)])
                    parents.extend([parent, parent])
                    generations.extend([generation, generation])
            else:
                x, y = float(x0), float(y0)
                if not valid_birth(x, y):
                    continue
                additions.append((x, y))
                parents.append(parent)
                generations.append(generation)
            radius = 3
            xa, xb = max(0, x0 - radius), min(self.w, x0 + radius + 1)
            ya, yb = max(0, y0 - radius), min(self.h, y0 + radius + 1)
            blocked.reshape(self.h, self.w)[ya:yb, xa:xb] = True
            if len(additions) >= budget:
                break
        if not additions:
            return "none"
        count = len(additions)
        self.seeds = np.vstack([self.seeds, np.asarray(additions)])
        self.marks = np.concatenate([
            self.marks, np.full(count, CARTOON, dtype=np.uint8)])
        self.parents = np.concatenate([
            self.parents, np.asarray(parents, dtype=np.int32)])
        self.generations = np.concatenate([
            self.generations, np.asarray(generations, dtype=np.int16)])
        self.support_map_indices = np.concatenate([
            self.support_map_indices,
            np.full(count, -1, dtype=np.int32)])
        self.dipole_relocations = len(relocated)
        return "dipole add" if self.idea == "dipole_add_expected_gain" else (
            "dipole split")

    def _local_residual_yield(self, count, focus):
        seeds = self._uniform_farthest(count)
        points = np.column_stack([self.xf, self.yf])
        distance = np.sum(
            (points[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
        owner = np.argmin(distance, axis=1)
        weights = 0.02 + focus.ravel() ** 2
        mass = np.bincount(owner, weights=weights, minlength=count)
        cx = np.bincount(
            owner, weights=weights * self.xf,
            minlength=count) / np.maximum(mass, 1e-12)
        cy = np.bincount(
            owner, weights=weights * self.yf,
            minlength=count) / np.maximum(mass, 1e-12)
        target = np.column_stack([cx, cy])
        displacement = target - seeds
        length = np.linalg.norm(displacement, axis=1)
        cap = 0.5 * math.sqrt(self.npix / max(count, 1))
        displacement *= np.minimum(
            1.0, cap / np.maximum(length, 1e-12))[:, None]
        return seeds + 0.45 * displacement

    def _spin_nucleation(self, foundation, focus_count, focus):
        """Vector lift plus Gaussian rounding for a fixed-size focus subset."""
        pool_size = min(max(4 * focus_count, focus_count), 160)
        pool = self._farthest_with_weight(
            pool_size, 0.01 + focus ** 2, foundation)
        m = len(pool)
        if focus_count == 0 or m <= focus_count:
            return np.vstack([foundation, pool[:focus_count]])
        px = np.clip(np.rint(pool[:, 0]).astype(int), 0, self.w - 1)
        py = np.clip(np.rint(pool[:, 1]).astype(int), 0, self.h - 1)
        unary = focus[py, px]
        unary = unary / max(float(unary.max()), 1e-12)
        delta = pool[:, None, :] - pool[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=2))
        spacing = math.sqrt(self.npix / max(len(foundation) + focus_count, 1))
        # +1 means complementary; -1 means spatially redundant.
        A = np.where(distance >= 1.65 * spacing, 1.0, -1.0)
        np.fill_diagonal(A, 0.0)
        d = max(m - 1, 1)
        identity = np.eye(m)
        rng = np.random.default_rng(20260726)

        def subset_utility(indices):
            selected = np.zeros(m, dtype=bool)
            selected[indices] = True
            redundancy = np.maximum(
                0.0, 1.0 - distance / max(1.65 * spacing, 1e-9))
            pair_cost = float(
                np.sum(redundancy[np.ix_(selected, selected)]) / 2.0)
            return float(2.5 * unary[selected].sum() - pair_cost)

        candidates = []
        # Greedy unary selection is included as a control within the idea.
        candidates.append(np.argsort(unary)[-focus_count:])
        for sigma in (-1.0, 1.0):
            U = (identity + sigma * A / math.sqrt(d)) / math.sqrt(2.0)
            norms = np.linalg.norm(U, axis=1, keepdims=True)
            U /= np.maximum(norms, 1e-12)
            for _ in range(64):
                margin = U @ rng.standard_normal(m)
                # Fixed cardinality is the rounded sign decision with its
                # threshold chosen to retain exactly the requested budget.
                candidates.append(np.argsort(margin)[-focus_count:])
        utilities = np.asarray([subset_utility(c) for c in candidates])
        winner = int(np.argmax(utilities))
        chosen = pool[candidates[winner]]
        self.spin_diagnostics = {
            "pool": m,
            "roundings": len(candidates) - 1,
            "best_utility": float(utilities[winner]),
            "greedy_utility": float(utilities[0]),
            "rounding_won": bool(winner != 0),
        }
        return np.vstack([foundation, chosen])


def native_components(image, cfg):
    split = bfft.meyer_channels(
        image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    return split.cartoon / scale, split.texture / scale


def quadratic_render(model):
    """Bounded moving quadratic patches over the unchanged cell geometry."""
    n = len(model.seeds)
    ids = model.owner
    angles, _ = model._site_frames()
    sx, sy = model.seeds[ids, 0], model.seeds[ids, 1]
    ct, st = np.cos(angles[ids]), np.sin(angles[ids])
    dx, dy = model.xf - sx, model.yf - sy
    spacing = math.sqrt(model.npix / max(n, 1))
    q = (dx * ct + dy * st) / max(spacing, 1e-9)
    r = (-dx * st + dy * ct) / max(spacing, 1e-9)
    basis = np.column_stack([
        np.ones(model.npix), q, r, q * q, q * r, r * r,
    ])
    coeff = np.zeros((n, 3, 6), dtype=np.float64)
    lo = np.zeros((n, 3), dtype=np.float64)
    hi = np.zeros((n, 3), dtype=np.float64)
    target = model.lab.reshape(-1, 3)
    for site in range(n):
        mask = ids == site
        if not np.any(mask):
            continue
        design = basis[mask]
        values = target[mask]
        regularizer = np.diag(
            [1e-7, 2e-4, 2e-4, 2e-3, 2e-3, 2e-3])
        normal = design.T @ design + regularizer
        try:
            fitted = np.linalg.solve(normal, design.T @ values)
        except np.linalg.LinAlgError:
            fitted = np.zeros((6, 3))
            fitted[0] = values.mean(axis=0)
        fitted[1:3] = np.clip(fitted[1:3], -0.12, 0.12)
        fitted[3:] = np.clip(fitted[3:], -0.08, 0.08)
        coeff[site] = fitted.T
        mean, std = values.mean(axis=0), values.std(axis=0)
        margin = 2.7 * std + np.array([0.01, 0.004, 0.004])
        lo[site], hi[site] = mean - margin, mean + margin

    def predict(site_ids):
        ssx, ssy = model.seeds[site_ids, 0], model.seeds[site_ids, 1]
        cct, sst = np.cos(angles[site_ids]), np.sin(angles[site_ids])
        ddx, ddy = model.xf - ssx, model.yf - ssy
        qq = (ddx * cct + ddy * sst) / max(spacing, 1e-9)
        rr = (-ddx * sst + ddy * cct) / max(spacing, 1e-9)
        bb = np.column_stack([
            np.ones(model.npix), qq, rr, qq * qq, qq * rr, rr * rr,
        ])
        pred = np.einsum("pck,pk->pc", coeff[site_ids], bb)
        return np.minimum(
            np.maximum(pred, lo[site_ids]), hi[site_ids])

    first = predict(model.owner)
    valid = model.second >= 0
    safe_second = np.where(valid, model.second, model.owner)
    second = predict(safe_second)
    gap = model.d2 - model.d1
    z = np.clip(0.5 * model.cfg.softness * gap, -50.0, 50.0)
    weight = 1.0 / (1.0 + np.exp(-z))
    weight[~valid] = 1.0
    lab = (
        first * weight[:, None] +
        second * (1.0 - weight[:, None])).reshape(
            model.h, model.w, 3)
    return np.clip(bfft.lab_to_srgb(lab), 0.0, 1.0)


def global_partition_render(
        model, decomposed=False, target_override=None, softness=None,
        return_lab=False):
    """Solve all affine coefficients against the renderer's real basis."""
    n = len(model.seeds)
    npix = model.npix
    spacing = math.sqrt(npix / max(n, 1))
    angles, _ = model._site_frames()
    valid = model.second >= 0
    second_ids = np.where(valid, model.second, model.owner)
    gap = model.d2 - model.d1
    active_softness = (
        model.cfg.softness if softness is None else float(softness))
    z = np.clip(0.5 * active_softness * gap, -50.0, 50.0)
    first_weight = 1.0 / (1.0 + np.exp(-z))
    first_weight[~valid] = 1.0
    second_weight = 1.0 - first_weight

    def basis_for(site_ids):
        sx, sy = model.seeds[site_ids, 0], model.seeds[site_ids, 1]
        ct, st = np.cos(angles[site_ids]), np.sin(angles[site_ids])
        dx, dy = model.xf - sx, model.yf - sy
        return np.column_stack([
            np.ones(npix),
            (dx * ct + dy * st) / max(spacing, 1e-9),
            (-dx * st + dy * ct) / max(spacing, 1e-9),
        ])

    first_basis = basis_for(model.owner)
    second_basis = basis_for(second_ids)
    rows = np.repeat(np.arange(npix, dtype=np.int32), 3)
    components = np.tile(np.arange(3, dtype=np.int32), npix)
    first_cols = (
        3 * np.repeat(model.owner, 3) + components)
    first_data = (
        first_basis * first_weight[:, None]).ravel()
    valid_rows = np.flatnonzero(valid)
    second_rows = np.repeat(valid_rows, 3)
    second_components = np.tile(
        np.arange(3, dtype=np.int32), len(valid_rows))
    second_cols = (
        3 * np.repeat(second_ids[valid], 3) + second_components)
    second_data = (
        second_basis[valid] * second_weight[valid, None]).ravel()

    # Tiny slope regularization removes null directions in cells with little
    # visible support without materially changing the least-squares target.
    regularization = np.tile(
        np.array([1e-5, 2e-3, 2e-3], dtype=np.float64), n)
    reg_rows = np.arange(npix, npix + 3 * n, dtype=np.int32)
    reg_cols = np.arange(3 * n, dtype=np.int32)
    design = sparse.coo_matrix((
        np.concatenate([
            first_data, second_data, np.sqrt(regularization)]),
        (
            np.concatenate([rows, second_rows, reg_rows]),
            np.concatenate([first_cols, second_cols, reg_cols]),
        )), shape=(npix + 3 * n, 3 * n)).tocsr()

    def solve_target(target):
        fitted = np.zeros((npix, 3), dtype=np.float64)
        padded = np.zeros(npix + 3 * n, dtype=np.float64)
        for channel in range(3):
            padded[:npix] = target[..., channel].ravel()
            coeff = lsmr(
                design, padded, atol=2e-6, btol=2e-6,
                maxiter=160)[0].reshape(n, 3)
            first_prediction = np.sum(
                coeff[model.owner] * first_basis, axis=1)
            second_prediction = np.sum(
                coeff[second_ids] * second_basis, axis=1)
            fitted[:, channel] = (
                first_weight * first_prediction +
                second_weight * second_prediction)
        return fitted.reshape(model.h, model.w, 3)

    if target_override is not None:
        lab = solve_target(target_override)
    elif decomposed:
        lab = (
            solve_target(model.base_lab) +
            model.cfg.detail_precision * solve_target(model.detail_lab))
    else:
        lab = solve_target(model.lab)
    if return_lab:
        return lab
    return np.clip(bfft.lab_to_srgb(lab), 0.0, 1.0)


def multiscale_global_render(model):
    base = global_partition_render(
        model, target_override=model.base_lab, softness=4.0,
        return_lab=True)
    detail = global_partition_render(
        model, target_override=model.detail_lab, softness=16.0,
        return_lab=True)
    return np.clip(bfft.lab_to_srgb(
        base + model.cfg.detail_precision * detail), 0.0, 1.0)


def evaluate(image, cfg, idea):
    if idea == "objective_overlap_search":
        branches = []
        for softness in (4.0, 6.0, 10.0):
            branch_idea = {
                4.0: "wide_overlap",
                6.0: "medium_overlap",
                10.0: "uniform_control",
            }[softness]
            result = evaluate(
                image, replace(cfg, softness=softness), branch_idea)
            result["objective"] = (
                result["rgb_mse"] +
                result["cartoon_mse"] +
                result["texture_mse"])
            result["softness"] = softness
            branches.append(result)
        best = min(branches, key=lambda item: item["objective"])
        best = dict(best)
        best["idea"] = idea
        best["description"] = IDEAS[idea]
        best["branch_search"] = [{
            "softness": item["softness"],
            "objective": item["objective"],
            "rgb_mse": item["rgb_mse"],
            "cartoon_mse": item["cartoon_mse"],
            "texture_mse": item["texture_mse"],
        } for item in branches]
        return best
    started = time.perf_counter()
    model = NucleationExperiment(image, cfg, idea)
    while len(model.seeds) < cfg.max_cells:
        model.step()
    # Give the recovered weak-site exchange mechanism three opportunities.
    for _ in range(3):
        model.step()
    if idea == "quadratic_postfit":
        reconstruction = quadratic_render(model)
    elif idea == "global_partition_fit":
        reconstruction = global_partition_render(model, decomposed=False)
    elif idea == "global_decomposition_fit":
        reconstruction = global_partition_render(model, decomposed=True)
    elif idea == "multiscale_global_fit":
        reconstruction = multiscale_global_render(model)
    else:
        reconstruction = np.clip(
            bfft.lab_to_srgb(model.reconstruction), 0.0, 1.0)
    target_cartoon, target_texture = native_components(model.rgb, cfg)
    recon_cartoon, recon_texture = native_components(reconstruction, cfg)
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    cartoon_mse = float(np.mean(
        (target_cartoon - recon_cartoon) ** 2))
    texture_mse = float(np.mean(
        (target_texture - recon_texture) ** 2))
    return {
        "idea": idea,
        "description": IDEAS[idea],
        "cells": len(model.seeds),
        "rgb_mse": rgb_mse,
        "psnr": float(-10.0 * math.log10(max(rgb_mse, 1e-12))),
        "cartoon_mse": cartoon_mse,
        "texture_mse": texture_mse,
        "elapsed_s": time.perf_counter() - started,
        "spin": model.spin_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=96)
    parser.add_argument("--cells", type=int, default=420)
    parser.add_argument(
        "--images", nargs="+",
        default=["camera", "coins", "grass", "chelsea"])
    parser.add_argument(
        "--ideas", nargs="+", choices=list(IDEAS),
        default=list(IDEAS))
    args = parser.parse_args()
    ideas = args.ideas
    record = {
        "protocol": {
            "images": args.images,
            "max_side": args.max_side,
            "initial_cells": 96,
            "maximum_cells": args.cells,
            "split_batch": 36,
            "post_budget_exchanges": 3,
            "recursive_residual_stages": 1,
            "selection_objective": (
                "report RGB, one-stage cartoon, and one-stage texture MSE; "
                "lower is better"),
        },
        "results": [],
    }
    for key in args.images:
        image = gallery.load(key)
        for idea in ideas:
            cfg = Config(
                max_side=args.max_side, passes=6, flow_sweeps=24,
                initial_cells=96, max_cells=args.cells, split_batch=36,
                recursive_memory_stages=1,
                residual_memory_weight=0.0,
                composition_discrepancy_weight=0.0)
            result = evaluate(image, cfg, idea)
            result["image"] = key
            record["results"].append(result)
            print(
                f"{key:8s} {idea:20s} "
                f"{result['psnr']:6.2f} dB "
                f"C {result['cartoon_mse']:.3e} "
                f"T {result['texture_mse']:.3e}",
                file=sys.stderr, flush=True)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
