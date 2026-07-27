#!/usr/bin/env python3
"""Local resource-transport cells: an owner-free segmentation experiment.

This is intentionally not another Voronoi allocator.  There are no pixel
owners, runner-up labels, candidate searches, rankings, top-k selections,
site deletions, or global sparse solves.

Each site emits a compact, smooth elliptical support ``phi_i``.  Supports
overlap, and the rendered field is their partition of unity.  Reconstruction
error is treated as a shared nutrient:

    available(x) = error(x) / (occupancy_floor + sum_i phi_i(x))

A cell absorbs ``phi_i * available`` locally.  Existing support therefore
depletes the same resource that a neighbour would need in order to expand.
Sites move toward absorbed resource, redistribute area according to uptake,
and acquire anisotropy from the direction of absorbed flux.

The BFFT glass/transport state is a conductivity, not an ownership metric:

    J = (I + glass_strength * v v^T) grad(error)

where ``v`` is the normalized gradient of the BFFT flow state.  Flow can only
elongate a cell when reconstruction demand actually travels through it.

The implementation uses two local splat passes per round.  With bounded
overlap its work is proportional to image pixels, not pixels times sites.
It is an isolated research control and does not modify the canonical viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import lab_to_srgb  # noqa: E402
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


@dataclass
class ResourceConfig:
    max_side: int = 192
    cells: int = 180
    passes: int = 12
    flow_sweeps: int = 32
    lam: float = 0.054
    mu: float = 0.02
    initial_overlap: float = 5.0
    occupancy_floor: float = 0.002
    kernel_family: str = "mixture"
    kernel_power: float = 4.0
    adaptive_hardness: bool = True
    hardness_rate: float = 0.10
    min_hardness: float = 1.5
    max_hardness: float = 24.0
    adaptive_crystallinity: bool = True
    initial_crystallinity: float = 0.05
    crystallinity_rate: float = 0.18
    adaptive_activity: bool = False
    birth_activity: float = 0.05
    activity_rate: float = 0.30
    min_activity: float = 0.005
    max_activity: float = 4.0
    lock_initial_activity: bool = True
    birth_target_hypothesis: bool = False
    glass_strength: float = 0.0
    glass_shape_strength: float = 0.5
    glass_mode: str = "fixed"
    glass_guide_side: int = 128
    flux_mix: float = 0.45
    color_rate: float = 0.70
    shared_residual_credit: bool = True
    shared_geometry_credit: bool = False
    center_rate: float = 0.22
    area_rate: float = 0.12
    shape_rate: float = 0.24
    shape_memory_gain: float = 0.0
    shape_memory_decay: float = 0.72
    max_ratio: float = 14.0
    min_area_fraction: float = 0.005
    max_area_multiple: float = 12.0
    germination: bool = True
    germination_threshold: float = 0.25
    germination_decay: float = 0.72
    germination_diffusion_fraction: float = 0.09
    germination_diffusion_min: float = 0.65
    germination_diffusion_max: float = 2.5
    germination_inhibition: float = 0.0
    resolution_selective_germination: bool = False
    fine_germination_inhibition: float = 8.0
    germination_separation: float = 0.34
    germination_initial_scale: float = 0.15
    conserved_cell_scale: bool = True


def _r2_sites(count: int, width: int, height: int) -> np.ndarray:
    """Deterministic low-discrepancy germs with no search or ranking."""
    plastic = 1.324717957244746
    index = np.arange(1, count + 1, dtype=np.float64)
    x = np.mod(0.5 + index / plastic, 1.0)
    y = np.mod(0.5 + index / (plastic * plastic), 1.0)
    margin_x = min(0.5 * width, 0.015 * max(width, height))
    margin_y = min(0.5 * height, 0.015 * max(width, height))
    x = margin_x + x * max(width - 2.0 * margin_x, 1.0)
    y = margin_y + y * max(height - 2.0 * margin_y, 1.0)
    return np.column_stack([x, y])


def _principal_tensor(
        xx: float, xy: float, yy: float,
        fallback_angle: float,
) -> tuple[float, float, float]:
    """Closed-form principal axis; no general eigensolver or sorting."""
    trace = max(xx + yy, 1e-12)
    disc = math.hypot(xx - yy, 2.0 * xy)
    high = max(0.5 * (trace + disc), 1e-12)
    low = max(0.5 * (trace - disc), 1e-12)
    if disc <= 1e-12 * trace:
        angle = fallback_angle
    else:
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    return angle, high, low


class ResourceTransportCells:
    """Smooth overlapping cells driven by locally depleted residual energy."""

    def __init__(self, image: np.ndarray, cfg: ResourceConfig | None = None):
        self.cfg = cfg or ResourceConfig()
        self.rgb = _fit_rgb(image, self.cfg.max_side)
        self.h, self.w = self.rgb.shape[:2]
        self.npix = self.h * self.w
        self.lab = srgb_to_lab(self.rgb)
        self.spacing = math.sqrt(self.npix / max(self.cfg.cells, 1))
        self.reference_area = (
            self.cfg.initial_overlap * self.npix /
            max(self.cfg.cells, 1))

        light = self.lab[..., 0] * 255.0
        guide_scale = min(
            1.0, self.cfg.glass_guide_side / max(self.h, self.w))
        self.guide_h = max(8, int(round(self.h * guide_scale)))
        self.guide_w = max(8, int(round(self.w * guide_scale)))
        if (self.guide_h, self.guide_w) == (self.h, self.w):
            guide_light = light
        else:
            from skimage.transform import resize
            guide_light = resize(
                light, (self.guide_h, self.guide_w),
                order=1, mode="reflect", anti_aliasing=True,
                preserve_range=True)
        self.target_glass = self._compute_glass(guide_light)
        tgx = ndi.sobel(
            self.target_glass, axis=1, mode="reflect") / 8.0
        tgy = ndi.sobel(
            self.target_glass, axis=0, mode="reflect") / 8.0
        self.glass_scale = max(
            float(np.percentile(np.hypot(tgx, tgy), 95.0)), 1e-12)
        self._set_glass_guide(self.target_glass)

        self.glass = self.target_glass

        self.centers = _r2_sites(self.cfg.cells, self.w, self.h)
        radius = math.sqrt(
            self.cfg.initial_overlap * self.npix /
            (math.pi * max(self.cfg.cells, 1)))
        self.major = np.full(self.cfg.cells, radius, dtype=np.float64)
        self.minor = np.full(self.cfg.cells, radius, dtype=np.float64)
        self.angle = np.zeros(self.cfg.cells, dtype=np.float64)
        self.hardness = np.full(
            self.cfg.cells, self.cfg.kernel_power, dtype=np.float64)
        initial_crystallinity = float(np.clip(
            self.cfg.initial_crystallinity, 1e-4, 1.0 - 1e-4))
        self.crystallinity_logit = np.full(
            self.cfg.cells,
            math.log(initial_crystallinity / (1.0 - initial_crystallinity)),
            dtype=np.float64)
        self.initial_cell_count = int(self.cfg.cells)
        self.activity = np.ones(self.cfg.cells, dtype=np.float64)
        self.axis_memory = np.zeros(
            (self.cfg.cells, 2), dtype=np.float64)
        self.uptake_reference = np.zeros(
            self.cfg.cells, dtype=np.float64)
        self.coeff = np.zeros((self.cfg.cells, 3, 3), dtype=np.float64)
        xi = np.clip(
            np.rint(self.centers[:, 0]).astype(np.int64), 0, self.w - 1)
        yi = np.clip(
            np.rint(self.centers[:, 1]).astype(np.int64), 0, self.h - 1)
        self.coeff[:, :, 0] = self.lab[yi, xi]
        self.iteration = 0
        self.reconstruction = np.zeros_like(self.lab)
        self.occupancy = np.zeros((self.h, self.w), dtype=np.float64)
        self.support_square_sum = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.fine_occupancy = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.error = np.zeros((self.h, self.w), dtype=np.float64)
        self.germination_field = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.germination_scale = 0.0
        self.last_births = 0
        self.last_ms = 0.0
        self.last_visits = 0
        self.trace: list[dict[str, float]] = []
        self._decomposition_objective = None
        self._render()

    def _compute_glass(self, light: np.ndarray) -> np.ndarray:
        cartoon, texture = bfft.meyer_split(
            light, lam=self.cfg.lam, mu=self.cfg.mu,
            passes=min(self.cfg.passes, 8), threads=4)
        projected = bfft.rof(
            light - texture, c=self.cfg.lam,
            eta=2.0 * self.cfg.lam,
            sweeps=min(self.cfg.flow_sweeps, 24),
            tol=0.0, threads=4)
        return (cartoon - projected) / 255.0

    def _set_glass_guide(self, field: np.ndarray):
        gx = ndi.sobel(field, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(field, axis=0, mode="reflect") / 8.0
        raw_magnitude = np.hypot(gx, gy)
        magnitude = np.clip(
            raw_magnitude / max(self.glass_scale, 1e-12), 0.0, 1.0)
        reciprocal = 1.0 / np.maximum(raw_magnitude, 1e-12)
        fx = gx * reciprocal
        fy = gy * reciprocal
        fx[magnitude <= 1e-8] = 0.0
        fy[magnitude <= 1e-8] = 0.0
        if field.shape != (self.h, self.w):
            from skimage.transform import resize
            fx = resize(
                fx, (self.h, self.w), order=1, mode="reflect",
                anti_aliasing=False, preserve_range=True)
            fy = resize(
                fy, (self.h, self.w), order=1, mode="reflect",
                anti_aliasing=False, preserve_range=True)
            magnitude = resize(
                magnitude, (self.h, self.w), order=1, mode="reflect",
                anti_aliasing=False, preserve_range=True)
            renorm = 1.0 / np.maximum(np.hypot(fx, fy), 1e-12)
            fx *= renorm
            fy *= renorm
        self.flow_x = fx
        self.flow_y = fy
        self.flow_coherence = magnitude

    def _refresh_glass_guide(self):
        mode = str(self.cfg.glass_mode).lower()
        if (
            mode == "off" or
            max(
                self.cfg.glass_strength,
                self.cfg.glass_shape_strength) <= 0.0
        ):
            self.flow_x.fill(0.0)
            self.flow_y.fill(0.0)
            self.flow_coherence.fill(0.0)
            return
        if mode == "fixed":
            self._set_glass_guide(self.target_glass)
            return
        light = self.reconstruction[..., 0] * 255.0
        if (self.guide_h, self.guide_w) != (self.h, self.w):
            from skimage.transform import resize
            light = resize(
                light, (self.guide_h, self.guide_w),
                order=1, mode="reflect", anti_aliasing=True,
                preserve_range=True)
        current_glass = self._compute_glass(light)
        self._set_glass_guide(self.target_glass - current_glass)

    def _germination_peaks(self) -> np.ndarray:
        """Advance the local germination field and return simultaneous germs.

        Every operation is a local stencil or pointwise reaction.  There is
        no candidate list, ordering, score sort, top-k, or fixed birth batch.
        The maximum filter represents lateral inhibition: a germ can cross
        threshold only when its activation exceeds the surrounding reaction
        field over one fraction of the current cell spacing.
        """
        if not self.cfg.germination:
            return np.empty((0, 2), dtype=np.float64)

        current_spacing = math.sqrt(
            self.npix / max(len(self.centers), 1))
        diffusion_sigma = max(
            float(self.cfg.germination_diffusion_min),
            min(
                float(self.cfg.germination_diffusion_max),
                float(self.cfg.germination_diffusion_fraction) *
                current_spacing))
        resource = self.error / np.maximum(
            self.occupancy, self.cfg.occupancy_floor)
        if self.cfg.resolution_selective_germination:
            # Coarse support still consumes reconstruction error through the
            # denominator above, but only later/fine support inhibits another
            # fine germ.  Once a peak has a fine coat, nearby residual becomes
            # the stronger unconsumed resource and the reaction can propagate
            # outward without a ranked allocator.
            resource /= (
                1.0 +
                float(self.cfg.fine_germination_inhibition) *
                self.fine_occupancy)
        drive = ndi.gaussian_filter(
            resource, diffusion_sigma, mode="reflect")
        if self.germination_scale <= 0.0:
            self.germination_scale = max(
                float(np.mean(drive)), 1e-12)
        normalized_drive = drive / self.germination_scale

        # Existing germs secrete a short-range inhibitor.  It does not assign
        # territory; it simply makes resource harder to accumulate close to
        # already active centers.
        impulses = np.zeros((self.h, self.w), dtype=np.float64)
        ix = np.clip(
            np.rint(self.centers[:, 0]).astype(np.int64), 0, self.w - 1)
        iy = np.clip(
            np.rint(self.centers[:, 1]).astype(np.int64), 0, self.h - 1)
        np.add.at(impulses, (iy, ix), 1.0)
        inhibit_sigma = max(0.8, 0.18 * current_spacing)
        inhibitor = ndi.gaussian_filter(
            impulses, inhibit_sigma, mode="reflect")
        inhibitor *= 2.0 * math.pi * inhibit_sigma * inhibit_sigma

        self.germination_field = (
            self.cfg.germination_decay * self.germination_field +
            normalized_drive /
            (1.0 + self.cfg.germination_inhibition * inhibitor))

        separation = max(
            3, int(round(
                self.cfg.germination_separation * current_spacing)))
        if separation % 2 == 0:
            separation += 1
        local_envelope = ndi.maximum_filter(
            self.germination_field, size=separation, mode="reflect")
        active = (
            (self.germination_field >= self.cfg.germination_threshold) &
            (self.germination_field >= local_envelope - 1e-12))
        # Do not germinate in effectively exhausted or unsupported numerical
        # background.  This is a local energy condition, not a cell label.
        active &= normalized_drive > 0.35
        yy, xx = np.nonzero(active)
        if xx.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Simultaneous firing consumes the accumulated activator in a
        # diffused neighbourhood, preventing the same locus from firing again
        # on the next round without renewed residual supply.
        fired = np.zeros_like(self.germination_field)
        fired[yy, xx] = 1.0
        refractory = ndi.gaussian_filter(
            fired, max(0.8, 0.22 * current_spacing), mode="reflect")
        peak = max(float(np.max(refractory)), 1e-12)
        self.germination_field *= np.clip(
            1.0 - refractory / peak, 0.0, 1.0)
        return np.column_stack([
            xx.astype(np.float64), yy.astype(np.float64)])

    def _germinate(self) -> int:
        births = self._germination_peaks()
        count = int(len(births))
        if count == 0:
            self.last_births = 0
            return 0

        old_count = len(self.centers)
        new_total = old_count + count
        if self.cfg.conserved_cell_scale:
            germ_area = self.reference_area
        else:
            germ_area = (
                self.cfg.initial_overlap * self.npix /
                max(new_total, 1))
        germ_radius = (
            self.cfg.germination_initial_scale *
            math.sqrt(germ_area / math.pi))
        germ_radius = max(germ_radius, 1.5)
        self.centers = np.vstack([self.centers, births])
        self.major = np.concatenate([
            self.major, np.full(count, germ_radius, dtype=np.float64)])
        self.minor = np.concatenate([
            self.minor, np.full(count, germ_radius, dtype=np.float64)])
        self.angle = np.concatenate([
            self.angle, np.zeros(count, dtype=np.float64)])
        self.hardness = np.concatenate([
            self.hardness,
            np.full(count, self.cfg.kernel_power, dtype=np.float64)])
        initial_crystallinity = float(np.clip(
            self.cfg.initial_crystallinity, 1e-4, 1.0 - 1e-4))
        self.crystallinity_logit = np.concatenate([
            self.crystallinity_logit,
            np.full(
                count,
                math.log(
                    initial_crystallinity /
                    (1.0 - initial_crystallinity)),
                dtype=np.float64)])
        birth_activity = (
            float(self.cfg.birth_activity)
            if self.cfg.adaptive_activity else 1.0)
        self.activity = np.concatenate([
            self.activity,
            np.full(count, birth_activity, dtype=np.float64)])
        self.axis_memory = np.vstack([
            self.axis_memory, np.zeros((count, 2), dtype=np.float64)])
        self.uptake_reference = np.concatenate([
            self.uptake_reference, np.zeros(count, dtype=np.float64)])
        born_coeff = np.zeros((count, 3, 3), dtype=np.float64)
        xi = np.clip(
            np.rint(births[:, 0]).astype(np.int64), 0, self.w - 1)
        yi = np.clip(
            np.rint(births[:, 1]).astype(np.int64), 0, self.h - 1)
        # Ordinarily a germ is born underneath the current mixture.  The
        # optional activity experiment instead gives a low-activity germ a
        # target-directed hypothesis.  Its small learned activity, rather
        # than a copied atom, keeps birth visually continuous while making
        # the activity derivative identifiable immediately.
        hypothesis = (
            self.lab
            if (
                self.cfg.adaptive_activity and
                self.cfg.birth_target_hypothesis)
            else self.reconstruction)
        xl = np.maximum(xi - 1, 0)
        xr = np.minimum(xi + 1, self.w - 1)
        yu = np.maximum(yi - 1, 0)
        yd = np.minimum(yi + 1, self.h - 1)
        grad_x = 0.5 * (
            hypothesis[yi, xr] -
            hypothesis[yi, xl])
        grad_y = 0.5 * (
            hypothesis[yd, xi] -
            hypothesis[yu, xi])
        born_coeff[:, :, 0] = hypothesis[yi, xi]
        born_coeff[:, :, 1] = (
            self.spacing * grad_x)
        born_coeff[:, :, 2] = (
            self.spacing * grad_y)
        born_coeff[:, :, 1:] = np.clip(
            born_coeff[:, :, 1:], -0.55, 0.55)
        self.coeff = np.concatenate([self.coeff, born_coeff], axis=0)
        self.last_births = count
        return count

    def _patch(self, site: int):
        cx, cy = self.centers[site]
        a = max(float(self.major[site]), 1.5)
        b = max(float(self.minor[site]), 1.5)
        theta = float(self.angle[site])
        ct, st = math.cos(theta), math.sin(theta)
        site_hardness = float(self.hardness[site])
        if self.cfg.kernel_family == "logistic":
            tail_widths = 8.0
            support_q = 1.0 + tail_widths / max(site_hardness, 1e-6)
        elif self.cfg.kernel_family == "mixture":
            crystallinity = 1.0 / (
                1.0 + math.exp(-float(
                    self.crystallinity_logit[site])))
            # Do not raster an eight-width logistic tail when its mixture
            # mass is already negligible.  This is an amplitude cutoff, not
            # a neighbour or candidate search.
            tail_widths = float(np.clip(
                math.log(max(crystallinity, 1e-12) / 1e-3),
                0.0, 8.0))
            support_q = (
                1.0 + tail_widths / max(site_hardness, 1e-6))
        else:
            support_q = 1.0
        extent_scale = math.sqrt(support_q)
        extent_x = extent_scale * math.sqrt(
            (a * ct) ** 2 + (b * st) ** 2)
        extent_y = extent_scale * math.sqrt(
            (a * st) ** 2 + (b * ct) ** 2)
        x0 = max(0, int(math.floor(cx - extent_x)))
        x1 = min(self.w, int(math.ceil(cx + extent_x)) + 1)
        y0 = max(0, int(math.floor(cy - extent_y)))
        y1 = min(self.h, int(math.ceil(cy + extent_y)) + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = xx - cx
        dy = yy - cy
        along = dx * ct + dy * st
        across = -dx * st + dy * ct
        q = (along / a) ** 2 + (across / b) ** 2
        inside = q < support_q
        if not np.any(inside):
            return None
        if self.cfg.kernel_family in ("logistic", "mixture"):
            # A true boundary-temperature family.  Its unbounded radial
            # integral is softplus(k)/k; normalization holds that mass equal
            # to the reference power kernel's 1/(p0+1).  The compact cutoff
            # is eight logistic widths beyond q=1.
            softplus = float(np.logaddexp(0.0, site_hardness))
            radial_mass = softplus / max(site_hardness, 1e-12)
            amplitude = (
                1.0 /
                ((self.cfg.kernel_power + 1.0) * radial_mass))
            sigmoid = 1.0 / (
                1.0 + np.exp(np.clip(
                    site_hardness * (q - 1.0), -60.0, 60.0)))
            logistic_phi = amplitude * sigmoid
            if self.cfg.kernel_family == "mixture":
                crystallinity = 1.0 / (
                    1.0 + math.exp(-float(
                        self.crystallinity_logit[site])))
                power_amplitude = (
                    (site_hardness + 1.0) /
                    (self.cfg.kernel_power + 1.0))
                power_phi = power_amplitude * np.power(
                    np.maximum(1.0 - q, 0.0), site_hardness)
                power_phi[q >= 1.0] = 0.0
                phi = (
                    (1.0 - crystallinity) * power_phi +
                    crystallinity * logistic_phi)
            else:
                phi = logistic_phi
        else:
            # Power concentration control.  This is smooth at the compact
            # boundary but p controls concentration, not edge temperature.
            amplitude = (
                (site_hardness + 1.0) /
                (self.cfg.kernel_power + 1.0))
            phi = amplitude * np.power(
                np.maximum(1.0 - q, 0.0), site_hardness)
        phi[~inside] = 0.0
        phi *= float(self.activity[site])
        return y0, y1, x0, x1, dx, dy, q, phi

    def _cell_prediction(
        self, site: int, dx: np.ndarray, dy: np.ndarray,
    ) -> np.ndarray:
        basis_x = dx / self.spacing
        basis_y = dy / self.spacing
        c = self.coeff[site]
        prediction = (
            c[:, 0][None, None, :] +
            basis_x[..., None] * c[:, 1][None, None, :] +
            basis_y[..., None] * c[:, 2][None, None, :])
        prediction[..., 0] = np.clip(prediction[..., 0], 0.0, 1.0)
        prediction[..., 1:] = np.clip(prediction[..., 1:], -0.45, 0.45)
        return prediction

    def _render(self):
        occupancy = np.full(
            (self.h, self.w), self.cfg.occupancy_floor,
            dtype=np.float64)
        support_square_sum = np.full(
            (self.h, self.w), self.cfg.occupancy_floor ** 2,
            dtype=np.float64)
        fine_occupancy = np.zeros(
            (self.h, self.w), dtype=np.float64)
        background = np.mean(self.lab, axis=(0, 1))
        numerator = (
            self.cfg.occupancy_floor *
            np.broadcast_to(background, self.lab.shape).copy())
        visits = 0
        for site in range(len(self.centers)):
            patch = self._patch(site)
            if patch is None:
                continue
            y0, y1, x0, x1, dx, dy, _, phi = patch
            pred = self._cell_prediction(site, dx, dy)
            occupancy[y0:y1, x0:x1] += phi
            support_square_sum[y0:y1, x0:x1] += phi * phi
            if site >= self.initial_cell_count:
                fine_occupancy[y0:y1, x0:x1] += phi
            numerator[y0:y1, x0:x1] += phi[..., None] * pred
            visits += phi.size
        self.occupancy = occupancy
        self.support_square_sum = support_square_sum
        self.fine_occupancy = fine_occupancy
        self.reconstruction = numerator / occupancy[..., None]
        residual = self.lab - self.reconstruction
        self.error = np.sum(
            residual * residual *
            np.array([1.0, 0.55, 0.55])[None, None, :], axis=2)
        self.last_visits = visits

    @property
    def rgb_reconstruction(self):
        return np.clip(lab_to_srgb(self.reconstruction), 0.0, 1.0)

    @property
    def psnr(self):
        mse = float(np.mean((self.rgb - self.rgb_reconstruction) ** 2))
        return -10.0 * math.log10(max(mse, 1e-12))

    def decomposition_metrics(self):
        """Score RGB and one-stage BFFT components with a cached target."""
        if self._decomposition_objective is None:
            from bfft.vision import SingleStageDecompositionObjective
            self._decomposition_objective = (
                SingleStageDecompositionObjective(self.rgb))
        return self._decomposition_objective.evaluate(
            self.rgb_reconstruction)

    def step(self):
        started = time.perf_counter()
        self._refresh_glass_guide()
        residual = self.lab - self.reconstruction
        energy = self.error

        n = len(self.centers)
        delta_coeff = np.zeros_like(self.coeff)
        uptake_mass = np.zeros(n, dtype=np.float64)
        uptake_density = np.zeros(n, dtype=np.float64)
        center_delta = np.zeros((n, 2), dtype=np.float64)
        log_axis_delta = np.zeros((n, 2), dtype=np.float64)
        angle_delta = np.zeros(n, dtype=np.float64)
        hardness_delta = np.zeros(n, dtype=np.float64)
        crystallinity_delta = np.zeros(n, dtype=np.float64)
        activity_delta = np.zeros(n, dtype=np.float64)
        visits = 0

        for site in range(n):
            patch = self._patch(site)
            if patch is None:
                continue
            y0, y1, x0, x1, dx, dy, q, phi = patch
            z = self.occupancy[y0:y1, x0:x1]
            weight = phi / np.maximum(z, 1e-12)
            local_residual = residual[y0:y1, x0:x1]
            local_render = self.reconstruction[y0:y1, x0:x1]
            prediction = self._cell_prediction(site, dx, dy)
            atom_gap = prediction - local_render
            channel_weight = np.array([1.0, 0.55, 0.55])
            usefulness = np.sum(
                channel_weight[None, None, :] *
                local_residual * atom_gap, axis=2)
            atom_energy = np.sum(
                channel_weight[None, None, :] *
                atom_gap * atom_gap, axis=2)
            local_square_sum = self.support_square_sum[
                y0:y1, x0:x1]
            residual_credit = (
                phi * phi / np.maximum(local_square_sum, 1e-12))
            if self.cfg.shared_geometry_credit:
                geometry_usefulness = usefulness * residual_credit
            else:
                geometry_usefulness = usefulness

            bx = dx / self.spacing
            by = dy / self.spacing
            basis = np.stack([
                np.ones_like(phi), np.broadcast_to(bx, phi.shape),
                np.broadcast_to(by, phi.shape)], axis=-1)
            design_weight = weight * weight
            normal = np.einsum(
                "...,...i,...j->ij", design_weight, basis, basis,
                optimize=True)
            normal += np.eye(3) * (
                1e-5 * max(float(np.trace(normal)), 1.0))
            if self.cfg.shared_residual_credit:
                color_credit = residual_credit
            else:
                color_credit = 1.0
            rhs = np.einsum(
                "...,...i,...c->ci",
                weight * color_credit, basis, local_residual,
                optimize=True)
            try:
                change = np.linalg.solve(normal, rhs.T).T
            except np.linalg.LinAlgError:
                change = np.zeros((3, 3), dtype=np.float64)
            change[:, 0] = np.clip(change[:, 0], -0.12, 0.12)
            change[:, 1:] = np.clip(change[:, 1:], -0.18, 0.18)
            delta_coeff[site] = change

            # Error is a conserved local resource.  Overlap raises occupancy
            # and divides the resource among all present cells.
            available = (
                energy[y0:y1, x0:x1] /
                np.maximum(z, self.cfg.occupancy_floor))
            if self.cfg.kernel_family in ("logistic", "mixture"):
                support_sigmoid = 1.0 / (
                    1.0 + np.exp(np.clip(
                        float(self.hardness[site]) * (q - 1.0),
                        -60.0, 60.0)))
                logistic_shell = (
                    4.0 * support_sigmoid *
                    (1.0 - support_sigmoid))
                if self.cfg.kernel_family == "mixture":
                    crystallinity = 1.0 / (
                        1.0 + math.exp(-float(
                            self.crystallinity_logit[site])))
                    power_shell = np.clip(
                        4.0 * q * (1.0 - q), 0.0, 1.0)
                    shell = (
                        (1.0 - crystallinity) * power_shell +
                        crystallinity * logistic_shell)
                else:
                    shell = logistic_shell
            else:
                shell = np.clip(4.0 * q * (1.0 - q), 0.0, 1.0)
            uptake = phi * available * (0.20 + 0.80 * shell)
            mass = float(np.sum(uptake))
            if mass <= 1e-18:
                continue
            uptake_mass[site] = mass
            area = math.pi * self.major[site] * self.minor[site]
            uptake_density[site] = mass / max(area, 1e-12)

            # Target-directed demand supplies grad(error).  The cached BFFT
            # glass state only changes its conductivity.
            fx = self.flow_x[y0:y1, x0:x1]
            fy = self.flow_y[y0:y1, x0:x1]
            coherence = self.flow_coherence[y0:y1, x0:x1]
            # Exact local support derivatives.  Increasing phi changes the
            # normalized field by (prediction - reconstruction) / Z, so this
            # is the true receiver-side gradient without a global graph.
            a = max(float(self.major[site]), 1.5)
            b = max(float(self.minor[site]), 1.5)
            theta = float(self.angle[site])
            ct, st = math.cos(theta), math.sin(theta)
            along = dx * ct + dy * st
            across = -dx * st + dy * ct
            site_hardness = float(self.hardness[site])
            site_activity = float(self.activity[site])
            if self.cfg.kernel_family in ("logistic", "mixture"):
                support_q = 1.0 + 8.0 / max(site_hardness, 1e-6)
                inside = q < support_q
                softplus = float(np.logaddexp(0.0, site_hardness))
                radial_mass = softplus / max(site_hardness, 1e-12)
                support_amplitude = (
                    1.0 /
                    ((self.cfg.kernel_power + 1.0) * radial_mass))
                support_sigmoid = 1.0 / (
                    1.0 + np.exp(np.clip(
                        site_hardness * (q - 1.0), -60.0, 60.0)))
                logistic_phi = support_amplitude * support_sigmoid
                logistic_phi *= site_activity
                logistic_derivative_scale = (
                    support_amplitude * site_hardness *
                    support_sigmoid * (1.0 - support_sigmoid))
                logistic_derivative_scale *= site_activity
                if self.cfg.kernel_family == "mixture":
                    crystallinity = 1.0 / (
                        1.0 + math.exp(-float(
                            self.crystallinity_logit[site])))
                    power_amplitude = (
                        (site_hardness + 1.0) /
                        (self.cfg.kernel_power + 1.0))
                    power_base = np.maximum(1.0 - q, 0.0)
                    power_phi = (
                        power_amplitude *
                        np.power(power_base, site_hardness))
                    power_phi[q >= 1.0] = 0.0
                    power_phi *= site_activity
                    power_derivative_scale = (
                        power_amplitude * site_hardness *
                        np.power(
                            power_base,
                            max(site_hardness - 1.0, 0.0)))
                    power_derivative_scale[q >= 1.0] = 0.0
                    power_derivative_scale *= site_activity
                    derivative_scale = (
                        (1.0 - crystallinity) *
                        power_derivative_scale +
                        crystallinity * logistic_derivative_scale)
                else:
                    derivative_scale = logistic_derivative_scale
            else:
                inside = q < 1.0
                support_amplitude = (
                    (site_hardness + 1.0) /
                    (self.cfg.kernel_power + 1.0))
                edge_base = np.power(
                    np.maximum(1.0 - q, 0.0),
                    max(site_hardness - 1.0, 0.0))
                derivative_scale = (
                    support_amplitude * site_hardness * edge_base)
                derivative_scale *= site_activity
            derivative_scale[~inside] = 0.0
            dq_cx = (
                -2.0 * along * ct / (a * a) +
                2.0 * across * st / (b * b))
            dq_cy = (
                -2.0 * along * st / (a * a) -
                2.0 * across * ct / (b * b))
            dq_angle = (
                2.0 * along * across *
                (1.0 / (a * a) - 1.0 / (b * b)))
            dphi_cx = -derivative_scale * dq_cx
            dphi_cy = -derivative_scale * dq_cy
            dphi_loga = (
                2.0 * derivative_scale * along * along / (a * a))
            dphi_logb = (
                2.0 * derivative_scale * across * across / (b * b))
            dphi_angle = -derivative_scale * dq_angle

            def scalar_gn(dphi):
                jacobian = dphi / np.maximum(z, 1e-12)
                gradient = -2.0 * float(np.sum(
                    geometry_usefulness * jacobian))
                curvature = 2.0 * float(np.sum(
                    atom_energy * jacobian * jacobian))
                ridge = 1e-6 * max(float(np.sum(np.abs(jacobian))), 1.0)
                return -gradient / max(curvature + ridge, 1e-12)

            jx = dphi_cx / np.maximum(z, 1e-12)
            jy = dphi_cy / np.maximum(z, 1e-12)
            gradient_center = -2.0 * np.array([
                float(np.sum(geometry_usefulness * jx)),
                float(np.sum(geometry_usefulness * jy))])
            center_curvature = float(np.sum(
                atom_energy * (jx * jx + jy * jy)))
            center_curvature = max(2.0 * center_curvature, 1e-9)

            # Glass is an SPD preconditioner of a measured descent direction.
            # It cannot create a direction when the target gradient is zero.
            guide_weight = (
                np.maximum(geometry_usefulness, 0.0) *
                weight * coherence)
            guide_mass = float(np.sum(guide_weight))
            if guide_mass > 1e-18:
                txx = float(np.sum(guide_weight * fx * fx) / guide_mass)
                txy = float(np.sum(guide_weight * fx * fy) / guide_mass)
                tyy = float(np.sum(guide_weight * fy * fy) / guide_mass)
            else:
                txx = txy = tyy = 0.0
            conductivity = self.cfg.glass_strength
            gx0, gy0 = gradient_center
            guided_gradient = np.array([
                gx0 + conductivity * (txx * gx0 + txy * gy0),
                gy0 + conductivity * (txy * gx0 + tyy * gy0)])
            center_delta[site] = -guided_gradient / center_curvature

            raw_axis_a = scalar_gn(dphi_loga)
            raw_axis_b = scalar_gn(dphi_logb)
            major_flow = (
                ct * ct * txx + 2.0 * ct * st * txy +
                st * st * tyy)
            minor_flow = (
                st * st * txx - 2.0 * ct * st * txy +
                ct * ct * tyy)
            uptake_reference = float(self.uptake_reference[site])
            if uptake_reference > 1e-18:
                conductivity_gate = math.sqrt(float(np.clip(
                    uptake_density[site] / uptake_reference, 0.0, 1.0)))
            else:
                conductivity_gate = 1.0
            shape_conductivity = (
                self.cfg.glass_shape_strength * conductivity_gate)
            log_axis_delta[site] = (
                np.clip(
                    raw_axis_a *
                    (1.0 + shape_conductivity * major_flow),
                    -0.18, 0.18),
                np.clip(
                    raw_axis_b *
                    (1.0 + shape_conductivity * minor_flow),
                    -0.18, 0.18),
            )
            angle_delta[site] = np.clip(
                scalar_gn(dphi_angle), -0.30, 0.30)
            if self.cfg.adaptive_hardness:
                if self.cfg.kernel_family in ("logistic", "mixture"):
                    sigmoid_at_hardness = 1.0 / (
                        1.0 + math.exp(-site_hardness))
                    normalization_derivative = (
                        1.0 -
                        site_hardness * sigmoid_at_hardness /
                        max(
                            float(np.logaddexp(
                                0.0, site_hardness)),
                            1e-12))
                    logistic_hardness_derivative = logistic_phi * (
                        normalization_derivative +
                        site_hardness * (1.0 - q) *
                        (1.0 - support_sigmoid))
                    if self.cfg.kernel_family == "mixture":
                        power_hardness_derivative = power_phi * (
                            site_hardness /
                            (site_hardness + 1.0) +
                            site_hardness *
                            np.log(np.maximum(1.0 - q, 1e-12)))
                        power_hardness_derivative[q >= 1.0] = 0.0
                        dphi_log_hardness = (
                            (1.0 - crystallinity) *
                            power_hardness_derivative +
                            crystallinity *
                            logistic_hardness_derivative)
                    else:
                        dphi_log_hardness = (
                            logistic_hardness_derivative)
                else:
                    support_base = np.maximum(1.0 - q, 1e-12)
                    dphi_log_hardness = phi * (
                        site_hardness / (site_hardness + 1.0) +
                        site_hardness * np.log(support_base))
                dphi_log_hardness[~inside] = 0.0
                hardness_delta[site] = np.clip(
                    scalar_gn(dphi_log_hardness), -0.24, 0.24)
            if (
                self.cfg.kernel_family == "mixture" and
                self.cfg.adaptive_crystallinity
            ):
                dphi_logit_crystallinity = (
                    crystallinity * (1.0 - crystallinity) *
                    (logistic_phi - power_phi))
                dphi_logit_crystallinity[~inside] = 0.0
                crystallinity_delta[site] = np.clip(
                    scalar_gn(dphi_logit_crystallinity), -0.35, 0.35)
            if (
                self.cfg.adaptive_activity and
                not (
                    self.cfg.lock_initial_activity and
                    site < self.initial_cell_count)
            ):
                # Since phi = activity * kernel,
                # d phi / d log(activity) = phi exactly.
                activity_delta[site] = np.clip(
                    scalar_gn(phi), -0.60, 0.60)

            visits += phi.size

        self.coeff += self.cfg.color_rate * delta_coeff
        self.coeff[:, 0, 0] = np.clip(self.coeff[:, 0, 0], 0.0, 1.0)
        self.coeff[:, 1:, 0] = np.clip(self.coeff[:, 1:, 0], -0.65, 0.65)
        self.coeff[:, :, 1:] = np.clip(
            self.coeff[:, :, 1:], -0.55, 0.55)
        memory_decay = float(np.clip(
            self.cfg.shape_memory_decay, 0.0, 0.999))
        self.axis_memory = (
            memory_decay * self.axis_memory +
            (1.0 - memory_decay) * log_axis_delta)
        if self.cfg.adaptive_hardness:
            self.hardness *= np.exp(
                self.cfg.hardness_rate * hardness_delta)
            self.hardness = np.clip(
                self.hardness, self.cfg.min_hardness,
                self.cfg.max_hardness)
        if (
            self.cfg.kernel_family == "mixture" and
            self.cfg.adaptive_crystallinity
        ):
            self.crystallinity_logit += (
                self.cfg.crystallinity_rate * crystallinity_delta)
            self.crystallinity_logit = np.clip(
                self.crystallinity_logit, -8.0, 8.0)
        if self.cfg.adaptive_activity:
            self.activity *= np.exp(
                self.cfg.activity_rate * activity_delta)
            self.activity = np.clip(
                self.activity,
                self.cfg.min_activity,
                self.cfg.max_activity)
        new_reference = (
            (self.uptake_reference <= 1e-18) &
            (uptake_density > 1e-18))
        self.uptake_reference[new_reference] = uptake_density[new_reference]
        memory_resource_gate = np.sqrt(np.clip(
            uptake_density /
            np.maximum(self.uptake_reference, 1e-18),
            0.0, 1.0))

        positive = uptake_density > 1e-18
        if self.cfg.conserved_cell_scale:
            base_area = self.reference_area
        else:
            base_area = (
                self.cfg.initial_overlap * self.npix /
                max(len(self.centers), 1))
        for site in range(n):
            if not positive[site]:
                continue
            cap = 0.22 * max(float(self.minor[site]), 1.5)
            move = self.cfg.center_rate * center_delta[site]
            norm = float(np.hypot(move[0], move[1]))
            if norm > cap:
                move *= cap / norm
            self.centers[site] += move
            self.centers[site, 0] = np.clip(
                self.centers[site, 0], 0.0, self.w - 1.0)
            self.centers[site, 1] = np.clip(
                self.centers[site, 1], 0.0, self.h - 1.0)

            area = math.pi * self.major[site] * self.minor[site]
            log_area_step = 0.5 * (
                log_axis_delta[site, 0] + log_axis_delta[site, 1])
            log_ratio_step = 0.5 * (
                log_axis_delta[site, 0] - log_axis_delta[site, 1])
            remembered_ratio_step = 0.5 * (
                self.axis_memory[site, 0] -
                self.axis_memory[site, 1])
            area *= math.exp(2.0 * self.cfg.area_rate * log_area_step)
            current_log_ratio = math.log(
                max(self.major[site] / self.minor[site], 1e-12))
            log_ratio = (
                current_log_ratio +
                2.0 * self.cfg.shape_rate *
                (log_ratio_step +
                 self.cfg.shape_memory_gain *
                 memory_resource_gate[site] *
                 remembered_ratio_step))
            log_ratio = float(np.clip(
                log_ratio, -math.log(self.cfg.max_ratio),
                math.log(self.cfg.max_ratio)))
            if log_ratio < 0.0:
                self.angle[site] += math.pi * 0.5
                log_ratio = -log_ratio
                self.axis_memory[site] = self.axis_memory[site, ::-1]
            self.angle[site] += (
                self.cfg.shape_rate * angle_delta[site])
            area = float(np.clip(
                area,
                self.cfg.min_area_fraction * base_area,
                self.cfg.max_area_multiple * base_area))
            root = math.sqrt(area / math.pi)
            ratio_root = math.exp(0.5 * log_ratio)
            self.major[site] = np.clip(
                root * ratio_root, 1.5, 2.0 * max(self.w, self.h))
            self.minor[site] = np.clip(
                root / ratio_root, 1.5, 2.0 * max(self.w, self.h))

        self._render()
        births = self._germinate()
        if births:
            self._render()
        self.iteration += 1
        self.last_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "iteration": self.iteration,
            "psnr": self.psnr,
            "elapsed_ms": self.last_ms,
            "splat_visits": int(visits + self.last_visits),
            "visits_per_pixel": (
                float(visits + self.last_visits) / self.npix),
            "mean_overlap": float(np.mean(
                self.occupancy - self.cfg.occupancy_floor)),
            "axis_ratio_mean": float(np.mean(self.major / self.minor)),
            "axis_ratio_max": float(np.max(self.major / self.minor)),
            "hardness_mean": float(np.mean(self.hardness)),
            "hardness_min": float(np.min(self.hardness)),
            "hardness_max": float(np.max(self.hardness)),
            "crystallinity_mean": float(np.mean(
                1.0 / (1.0 + np.exp(-self.crystallinity_logit)))),
            "activity_mean": float(np.mean(self.activity)),
            "activity_min": float(np.min(self.activity)),
            "activity_max": float(np.max(self.activity)),
            "area_cv": float(
                np.std(math.pi * self.major * self.minor) /
                max(np.mean(math.pi * self.major * self.minor), 1e-12)),
            "cells": int(len(self.centers)),
            "births": int(births),
            "germination_max": float(np.max(self.germination_field)),
        }
        self.trace.append(report)
        return report


def _load_image(path: str | None, gallery_key: str):
    if path:
        from skimage.io import imread
        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _save_panel(model: ResourceTransportCells, path: Path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    error = model.error
    error_view = np.clip(
        error / max(float(np.percentile(error, 99.0)), 1e-12), 0.0, 1.0)
    occupancy = model.occupancy - model.cfg.occupancy_floor
    occupancy_view = np.clip(
        occupancy /
        max(float(np.percentile(occupancy, 99.0)), 1e-12), 0.0, 1.0)
    for axis, image, title in zip(
        axes,
        (model.rgb, model.rgb_reconstruction, error_view, occupancy_view),
        ("target", f"resource cells ({model.psnr:.2f} dB)",
         "remaining resource", "smooth occupancy"),
    ):
        axis.imshow(image, cmap=None if image.ndim == 3 else "magma")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="camera")
    parser.add_argument("--max-side", type=int, default=192)
    parser.add_argument("--cells", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--flow-sweeps", type=int, default=32)
    parser.add_argument("--glass-strength", type=float, default=0.0)
    parser.add_argument(
        "--glass-shape-strength", type=float, default=0.5)
    parser.add_argument(
        "--glass-mode", choices=["off", "fixed", "discrepancy"],
        default="fixed")
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--occupancy-floor", type=float, default=0.002)
    parser.add_argument(
        "--kernel-family", choices=["power", "logistic", "mixture"],
        default="mixture")
    parser.add_argument("--kernel-power", type=float, default=4.0)
    parser.add_argument(
        "--adaptive-hardness",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--adaptive-crystallinity",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--initial-crystallinity", type=float, default=0.05)
    parser.add_argument(
        "--crystallinity-rate", type=float, default=0.18)
    parser.add_argument(
        "--shared-residual-credit",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shared-geometry-credit", action="store_true")
    parser.add_argument(
        "--shape-memory-gain", type=float, default=0.0)
    parser.add_argument(
        "--min-area-fraction", type=float, default=0.005)
    parser.add_argument(
        "--germinate", action=argparse.BooleanOptionalAction, default=True,
        help="enable local reaction/diffusion germination")
    parser.add_argument(
        "--germination-threshold", type=float, default=0.25)
    parser.add_argument(
        "--germination-inhibition", type=float, default=0.0)
    parser.add_argument(
        "--germination-initial-scale", type=float, default=0.15)
    parser.add_argument(
        "--conserved-cell-scale",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save", type=Path,
        default=ROOT / "experiments/out/resource_transport_cells.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/resource_transport_cells.json")
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    cfg = ResourceConfig(
        max_side=args.max_side, cells=args.cells, passes=args.passes,
        flow_sweeps=args.flow_sweeps,
        glass_strength=args.glass_strength,
        glass_shape_strength=args.glass_shape_strength,
        glass_mode=args.glass_mode,
        initial_overlap=args.overlap,
        occupancy_floor=args.occupancy_floor,
        kernel_family=args.kernel_family,
        kernel_power=args.kernel_power,
        adaptive_hardness=args.adaptive_hardness,
        adaptive_crystallinity=args.adaptive_crystallinity,
        initial_crystallinity=args.initial_crystallinity,
        crystallinity_rate=args.crystallinity_rate,
        shared_residual_credit=args.shared_residual_credit,
        shared_geometry_credit=args.shared_geometry_credit,
        shape_memory_gain=args.shape_memory_gain,
        min_area_fraction=args.min_area_fraction,
        germination=args.germinate,
        germination_threshold=args.germination_threshold,
        germination_inhibition=args.germination_inhibition,
        germination_initial_scale=args.germination_initial_scale,
        conserved_cell_scale=args.conserved_cell_scale)
    started = time.perf_counter()
    model = ResourceTransportCells(image, cfg)
    init_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"{source} | {model.w}x{model.h} | {len(model.centers)} cells | "
        f"initial {model.psnr:.3f} dB | {init_ms:.1f} ms")
    for _ in range(args.iterations):
        report = model.step()
        print(
            f"round {report['iteration']:02d} | "
            f"{report['psnr']:.3f} dB | "
            f"{report['elapsed_ms']:.1f} ms | "
            f"{report['cells']} cells (+{report['births']}) | "
            f"{report['visits_per_pixel']:.2f} visits/pixel | "
            f"ratio {report['axis_ratio_mean']:.2f}/"
            f"{report['axis_ratio_max']:.2f}")

    _save_panel(model, args.save)
    metrics = model.decomposition_metrics()
    print(
        "objective "
        f"{metrics['objective']:.6e} | "
        f"RGB {metrics['rgb_mse']:.6e} | "
        f"cartoon {metrics['cartoon_mse']:.6e} | "
        f"texture {metrics['texture_mse']:.6e}")
    result = {
        "source": source,
        "shape": [model.h, model.w],
        "config": vars(cfg),
        "initialization_ms": init_ms,
        "final_psnr": model.psnr,
        "decomposition_metrics": metrics,
        "trace": model.trace,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2))
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
