"""Recursive BFFT decomposition and flow-boundary nucleation study."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from bfft.effects import (
    _from_working, lab_to_srgb, meyer_channels, srgb_to_lab,
)


MODES = ("half both", "texture decay", "cartoon decay")


def _normalize(a, percentile=99.0):
    scale = max(float(np.percentile(np.abs(a), percentile)), 1e-12)
    return np.clip(np.abs(a) / scale, 0.0, 1.0)


def _fit_rgb(image, max_side):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        a = a / 255.0 if a.max() > 1.5 else a
        a = np.repeat(a[..., None], 3, axis=2)
    else:
        a = a[..., :3]
        if a.max() > 1.5:
            a = a / 255.0
    h, w = a.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    oh, ow = max(16, round(h * scale)), max(16, round(w * scale))
    if (oh, ow) != (h, w):
        a = resize(a, (oh, ow), anti_aliasing=True, preserve_range=True)
    return np.clip(a, 0.0, 1.0)


def _native_from_working(planes, split):
    native = np.empty_like(planes)
    for channel in range(planes.shape[2]):
        native[..., channel] = (
            split.offset[channel] +
            planes[..., channel] / split.scale[channel])
    rgb = _from_working(
        native, split.space, split.carried,
        split.carried.get("_was_gray", False))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    return np.clip(rgb[..., :3], 0.0, 1.0)


def _layer_displays(split):
    residual = split.residual
    cartoon = _native_from_working(residual + split.cartoon, split)
    texture_scalar = np.mean(split.texture, axis=2)
    texture_scale = max(
        float(np.percentile(np.abs(texture_scalar), 99.0)), 1e-9)
    texture = np.repeat(
        np.clip(0.5 + 0.5 * texture_scalar / texture_scale, 0.0, 1.0)[
            ..., None], 3, axis=2)
    residual_scalar = np.mean(residual, axis=2)
    residual_scale = max(
        float(np.percentile(np.abs(residual_scalar), 99.0)), 1e-9)
    residual_view = np.repeat(
        np.clip(0.5 + 0.5 * residual_scalar / residual_scale, 0.0, 1.0)[
            ..., None], 3, axis=2)
    return cartoon, texture, residual_view


@dataclass
class RecursiveConfig:
    max_side: int = 256
    iterations: int = 6
    alpha: float = 0.5
    mode: str = "half both"
    space: str = "oklab_lc"
    passes: int = 16
    lam: float = 0.05
    mu: float = 40.0
    coarse_cells: int = 180
    detail_cells: int = 320
    density_floor: float = 0.35
    detail_density_floor: float = 0.04
    residual_scale: float = 2.5
    detail_anisotropy: float = 8.0
    residual_gain: float = 8.0
    boundary_gain: float = 3.0
    stage_decay: float = 0.72
    cell_irregularity: float = 0.28
    cell_softness: float = 8.0
    descent_steps: int = 8
    descent_rate: float = 0.35
    blue_noise_repulsion: float = 0.18


class RecursiveDecomposition:
    """Cached recursive stack plus an isotropic coarse-cell preview."""

    def __init__(self, image, config=None):
        self.cfg = config or RecursiveConfig()
        self.rgb = _fit_rgb(image, self.cfg.max_side)
        self.h, self.w = self.rgb.shape[:2]
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self.stages = [self.rgb]
        self.splits = []
        self.cartoons = []
        self.textures = []
        self.residuals = []
        self.removed = []
        self.boundaries = []
        self.collective_residual_signed = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.collective_residual_energy = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.boundary_accumulation = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.density = np.ones((self.h, self.w), dtype=np.float64)
        self.detail_density = np.ones((self.h, self.w), dtype=np.float64)
        self.seeds = np.empty((0, 2), dtype=np.float64)
        self.detail_seeds = np.empty((0, 2), dtype=np.float64)
        self.residual_fineness = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.residual_coherence = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.residual_angle = np.zeros(
            (self.h, self.w), dtype=np.float64)
        self.detail_ratios = np.empty(0, dtype=np.float64)
        self.detail_angles = np.empty(0, dtype=np.float64)
        self.coarse_owner = np.zeros((self.h, self.w), dtype=np.int32)
        self.coarse_cells = self.rgb.copy()
        self.descent_loss = []
        self.elapsed_ms = 0.0
        self._compute()

    def _gains(self):
        alpha = float(np.clip(self.cfg.alpha, 0.0, 1.0))
        if self.cfg.mode == "texture decay":
            return 1.0, alpha
        if self.cfg.mode == "cartoon decay":
            return alpha, 1.0
        return alpha, alpha

    def _recur(self, split, cartoon_gain, texture_gain):
        """Recombine layer contrast while retaining the cartoon DC level."""
        anchor = np.mean(
            split.cartoon, axis=(0, 1), keepdims=True)
        working = (
            split.residual + anchor +
            cartoon_gain * (split.cartoon - anchor) +
            texture_gain * split.texture)
        return _native_from_working(working, split)

    def _compute(self):
        t0 = time.perf_counter()
        current = self.rgb.copy()
        gc, gt = self._gains()
        cartoon_lightness = []
        for _ in range(max(1, int(self.cfg.iterations))):
            split = meyer_channels(
                current, space=self.cfg.space, lam=self.cfg.lam,
                mu=self.cfg.mu, passes=self.cfg.passes, threads=4)
            cartoon, texture, residual = _layer_displays(split)
            nxt = self._recur(split, gc, gt)
            if nxt.ndim == 2:
                nxt = np.repeat(nxt[..., None], 3, axis=2)
            nxt = np.clip(nxt[..., :3], 0.0, 1.0)
            self.splits.append(split)
            self.cartoons.append(cartoon)
            self.textures.append(texture)
            self.residuals.append(residual)
            self.removed.append(current - nxt)
            residual_scalar = np.mean(split.residual, axis=2)
            stage_weight = self.cfg.stage_decay ** (len(self.splits) - 1)
            self.collective_residual_signed += (
                stage_weight * residual_scalar)
            self.collective_residual_energy += (
                stage_weight * _normalize(residual_scalar))
            cartoon_lightness.append(srgb_to_lab(cartoon)[..., 0])
            self.stages.append(nxt)
            current = nxt

        for index, lightness in enumerate(cartoon_lightness):
            if index + 1 < len(cartoon_lightness):
                delta = lightness - cartoon_lightness[index + 1]
            else:
                delta = lightness - srgb_to_lab(
                    self.stages[index + 1])[..., 0]
            gx = ndi.sobel(delta, axis=1, mode="reflect") / 8.0
            gy = ndi.sobel(delta, axis=0, mode="reflect") / 8.0
            boundary = np.hypot(gx, gy)
            self.boundaries.append(boundary)
            self.boundary_accumulation += (
                self.cfg.stage_decay ** index) * _normalize(boundary)

        self.boundary_accumulation = _normalize(
            ndi.gaussian_filter(
                self.boundary_accumulation, 0.7, mode="reflect"))
        self.collective_residual_energy = _normalize(
            ndi.gaussian_filter(
                self.collective_residual_energy, 0.7, mode="reflect"))
        self.density = (
            max(float(self.cfg.density_floor), 1e-4) +
            self.cfg.boundary_gain * self.boundary_accumulation)
        self.detail_density = (
            max(float(self.cfg.detail_density_floor), 1e-4) +
            self.cfg.residual_gain * self.collective_residual_energy)
        self.seeds = self._weighted_farthest(
            self.cfg.coarse_cells, self.density)
        self.detail_seeds = self._weighted_farthest(
            self.cfg.detail_cells, self.detail_density)
        self._detail_geometry()
        self._evolve_in_decomposition_space()
        self._render_coarse_cells()
        self.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _weighted_farthest(self, count, density):
        count = max(1, min(int(count), self.h * self.w))
        xf, yf = self.xx.ravel(), self.yy.ravel()
        weight = np.asarray(density, dtype=np.float64).ravel()
        first = int(np.argmax(weight))
        selected = [first]
        blocked = np.zeros(len(xf), dtype=bool)
        blocked[first] = True
        min_d2 = (xf - xf[first]) ** 2 + (yf - yf[first]) ** 2
        for _ in range(1, count):
            score = min_d2 * weight
            score[blocked] = -1.0
            index = int(np.argmax(score))
            if score[index] < 0:
                break
            selected.append(index)
            blocked[index] = True
            d2 = (xf - xf[index]) ** 2 + (yf - yf[index]) ** 2
            np.minimum(min_d2, d2, out=min_d2)
        return np.column_stack(
            [xf[selected], yf[selected]]).astype(np.float64)

    def _detail_geometry(self):
        """Infer continuous scale, direction, and anisotropic propensity."""
        residual = self.collective_residual_signed
        sigma = max(float(self.cfg.residual_scale), 0.35)
        smooth = ndi.gaussian_filter(residual, sigma, mode="reflect")
        fine = np.abs(residual - smooth)
        coarse = ndi.gaussian_filter(
            np.abs(residual), sigma, mode="reflect")
        fine_n = _normalize(fine)
        coarse_n = _normalize(coarse)
        self.residual_fineness = (
            fine_n / np.maximum(fine_n + coarse_n, 1e-9))

        gx = ndi.sobel(residual, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(residual, axis=0, mode="reflect") / 8.0
        jxx = ndi.gaussian_filter(gx * gx, 1.4, mode="reflect")
        jyy = ndi.gaussian_filter(gy * gy, 1.4, mode="reflect")
        jxy = ndi.gaussian_filter(gx * gy, 1.4, mode="reflect")
        disc = np.sqrt(np.maximum(
            (jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
        self.residual_coherence = (
            disc / np.maximum(jxx + jyy, 1e-12))
        normal = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        self.residual_angle = normal + math.pi * 0.5

        xy = np.rint(self.detail_seeds).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, self.w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.h - 1)
        self.detail_angles = self.residual_angle[xy[:, 1], xy[:, 0]]
        propensity = (
            self.residual_fineness[xy[:, 1], xy[:, 0]] *
            self.residual_coherence[xy[:, 1], xy[:, 0]])
        self.detail_ratios = np.clip(
            1.0 + self.cfg.detail_anisotropy * propensity,
            1.0, 16.0)

    def _decomposition_features(self):
        cartoon_l = srgb_to_lab(self.cartoons[0])[..., 0]
        texture = np.mean(self.splits[0].texture, axis=2)
        features = np.stack([
            cartoon_l, self.boundary_accumulation,
        ], axis=-1)
        mean = features.mean(axis=(0, 1), keepdims=True)
        std = np.maximum(
            features.std(axis=(0, 1), keepdims=True), 1e-6)
        return ((features - mean) / std).reshape(
            -1, features.shape[-1])

    def _soft_membership(self, points, spacing2):
        delta = points[:, None, :] - self.seeds[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        logits = (
            -max(float(self.cfg.cell_softness), 0.25) *
            squared / max(spacing2, 1e-9))
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(np.clip(logits, -50.0, 0.0))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-15)
        return weights, delta

    def _evolve_in_decomposition_space(self):
        """Alternating descent on cartoon/texture/boundary reconstruction."""
        steps = max(0, int(self.cfg.descent_steps))
        if steps == 0 or len(self.seeds) < 2:
            return
        points = np.column_stack(
            [self.xx.ravel(), self.yy.ravel()]).astype(np.float64)
        target = self._decomposition_features()
        n = len(self.seeds)
        spacing2 = self.h * self.w / n
        tau2 = (
            2.0 * max(float(self.cfg.cell_softness), 0.25) /
            max(spacing2, 1e-9))
        for _ in range(steps):
            mass = np.zeros(n, dtype=np.float64)
            feature_sum = np.zeros((n, target.shape[1]), dtype=np.float64)
            for start in range(0, len(points), 2048):
                stop = min(start + 2048, len(points))
                phi, _ = self._soft_membership(
                    points[start:stop], spacing2)
                mass += phi.sum(axis=0)
                feature_sum += phi.T @ target[start:stop]
            colors = feature_sum / np.maximum(mass[:, None], 1e-12)

            gradient = np.zeros_like(self.seeds)
            loss = 0.0
            for start in range(0, len(points), 2048):
                stop = min(start + 2048, len(points))
                phi, delta = self._soft_membership(
                    points[start:stop], spacing2)
                predicted = phi @ colors
                residual = predicted - target[start:stop]
                loss += float(np.sum(residual * residual))
                contrast = colors[None, :, :] - predicted[:, None, :]
                scalar = phi * np.sum(
                    residual[:, None, :] * contrast, axis=2)
                gradient += tau2 * np.sum(
                    scalar[..., None] * delta, axis=0)

            normalized = gradient / np.maximum(mass[:, None], 1e-9)
            displacement = -float(self.cfg.descent_rate) * normalized

            # A mild short-range repulsion preserves the blue-noise coat
            # while the decomposition objective moves germs toward structure.
            minimum = 0.55 * math.sqrt(spacing2)
            delta_sites = self.seeds[:, None, :] - self.seeds[None, :, :]
            distance = np.sqrt(
                np.sum(delta_sites * delta_sites, axis=2) + 1e-12)
            close = (distance < minimum) & (
                ~np.eye(n, dtype=bool))
            repel = np.sum(
                np.where(close[..., None],
                         delta_sites / distance[..., None] *
                         (minimum - distance)[..., None] / minimum,
                         0.0), axis=1)
            displacement += (
                float(self.cfg.blue_noise_repulsion) * repel)
            length = np.linalg.norm(displacement, axis=1)
            cap = 1.25
            displacement *= np.minimum(
                1.0, cap / np.maximum(length, 1e-12))[:, None]
            self.seeds += displacement
            self.seeds[:, 0] = np.clip(self.seeds[:, 0], 0, self.w - 1)
            self.seeds[:, 1] = np.clip(self.seeds[:, 1], 0, self.h - 1)
            self.descent_loss.append(loss / len(points))

    def _render_coarse_cells(self):
        points = np.column_stack(
            [self.xx.ravel(), self.yy.ravel()]).astype(np.float64)
        n = len(self.seeds)
        first = np.empty(len(points), dtype=np.int32)
        second = np.empty(len(points), dtype=np.int32)
        d1 = np.empty(len(points), dtype=np.float64)
        d2 = np.empty(len(points), dtype=np.float64)
        spacing2 = self.h * self.w / max(n, 1)
        ids = np.arange(n, dtype=np.uint32)
        hashed = ((ids * 1664525 + 1013904223) & 0xFFFFFFFF) / 2**32
        power = (
            self.cfg.cell_irregularity * spacing2 * (hashed - 0.5))
        for start in range(0, len(points), 4096):
            stop = min(start + 4096, len(points))
            delta = points[start:stop, None, :] - self.seeds[None, :, :]
            distance = np.sum(delta * delta, axis=2) - power[None, :]
            pair = np.argpartition(distance, 1, axis=1)[:, :2]
            pair_distance = np.take_along_axis(distance, pair, axis=1)
            order = np.argsort(pair_distance, axis=1)
            ranked = np.take_along_axis(pair, order, axis=1)
            ranked_distance = np.take_along_axis(
                pair_distance, order, axis=1)
            first[start:stop], second[start:stop] = (
                ranked[:, 0], ranked[:, 1])
            d1[start:stop], d2[start:stop] = (
                ranked_distance[:, 0], ranked_distance[:, 1])

        lab = srgb_to_lab(self.rgb).reshape(-1, 3)
        counts = np.maximum(np.bincount(first, minlength=n), 1)
        colors = np.column_stack([
            np.bincount(first, weights=lab[:, channel], minlength=n) / counts
            for channel in range(3)
        ])
        sharpness = max(float(self.cfg.cell_softness), 0.0)
        if sharpness == 0:
            rendered = colors[first]
        else:
            gap = np.clip((d2 - d1) / max(spacing2, 1e-9), -20.0, 20.0)
            blend = 1.0 / (1.0 + np.exp(-sharpness * gap))
            rendered = (
                colors[first] * blend[:, None] +
                colors[second] * (1.0 - blend[:, None]))
        self.coarse_owner = first.reshape(self.h, self.w)
        self.coarse_cells = np.clip(
            lab_to_srgb(rendered.reshape(self.h, self.w, 3)), 0.0, 1.0)

    def _signed_rgb(self, field):
        scale = max(float(np.percentile(np.abs(field), 99.0)), 1e-9)
        return np.clip(0.5 + field / (2.0 * scale), 0.0, 1.0)

    def view(self, mode, stage=0):
        index = int(np.clip(stage, 0, len(self.splits) - 1))
        if mode == "Original":
            return self.rgb
        if mode == "Recursive image":
            return self.stages[index + 1]
        if mode == "Final recursive image":
            return self.stages[-1]
        if mode == "Cartoon at stage":
            return self.cartoons[index]
        if mode == "Texture at stage":
            return self.textures[index]
        if mode == "Carried residual":
            return self.residuals[index]
        if mode == "Removed shell":
            return self._signed_rgb(self.removed[index])
        if mode == "Accumulated removal":
            difference = self.rgb - self.stages[index + 1]
            return self._signed_rgb(difference)
        if mode == "Stage boundary":
            g = _normalize(self.boundaries[index])
            return np.stack([g, g * g, 0.15 * (1.0 - g)], axis=-1)
        if mode == "Boundary accumulation":
            g = self.boundary_accumulation
            return np.stack([g, 0.35 * g, 1.0 - g], axis=-1)
        if mode == "Collective carried residual":
            scale = max(float(np.percentile(
                np.abs(self.collective_residual_signed), 99.0)), 1e-9)
            g = np.clip(
                0.5 + 0.5 * self.collective_residual_signed / scale,
                0.0, 1.0)
            return np.stack([g, 0.3 + 0.4 * g, 1.0 - g], axis=-1)
        if mode == "Collective residual support":
            g = self.collective_residual_energy
            return np.stack([g, g * g, 0.12 * (1.0 - g)], axis=-1)
        if mode == "Nucleation density":
            g = _normalize(self.density)
            return np.stack([0.1 * g, g, 1.0 - g], axis=-1)
        if mode == "Detail germination density":
            g = _normalize(self.detail_density)
            return np.stack([g, 0.15 * g, 1.0 - g], axis=-1)
        if mode == "Residual scale map":
            fine = self.residual_fineness
            return np.stack([
                fine, 0.25 + 0.5 * (1.0 - fine), 1.0 - fine,
            ], axis=-1)
        if mode == "Residual directional consistency":
            g = self.residual_coherence
            return np.stack([0.1 * g, g, 0.8 * g], axis=-1)
        if mode == "Nucleation seeds":
            out = self.rgb.copy()
            for x, y in self.seeds:
                ix, iy = int(round(x)), int(round(y))
                out[max(0, iy - 1):min(self.h, iy + 2),
                    max(0, ix - 1):min(self.w, ix + 2)] = (1.0, 0.05, 0.0)
            return out
        if mode == "Detail germination order":
            out = np.full_like(self.rgb, 0.08)
            total = max(len(self.detail_seeds) - 1, 1)
            for order, (x, y) in enumerate(self.detail_seeds):
                ix, iy = int(round(x)), int(round(y))
                phase = order / total
                color = (1.0 - 0.55 * phase,
                         0.05 + 0.75 * phase,
                         0.95 * phase)
                out[max(0, iy - 1):min(self.h, iy + 2),
                    max(0, ix - 1):min(self.w, ix + 2)] = color
            return out
        if mode == "Detail anisotropy preview":
            out = np.clip(0.22 + 0.62 * self.rgb, 0.0, 1.0)
            for (x, y), angle, ratio in zip(
                    self.detail_seeds, self.detail_angles,
                    self.detail_ratios):
                ix, iy = int(round(x)), int(round(y))
                color_phase = (ratio - 1.0) / 15.0
                color = (
                    1.0, 0.75 * (1.0 - color_phase),
                    0.15 + 0.85 * color_phase)
                out[max(0, iy - 1):min(self.h, iy + 2),
                    max(0, ix - 1):min(self.w, ix + 2)] = color
                length = 1.5 + min(9.0, ratio)
                for parameter in np.linspace(-length, length, 17):
                    px = int(round(x + math.cos(angle) * parameter))
                    py = int(round(y + math.sin(angle) * parameter))
                    if 0 <= px < self.w and 0 <= py < self.h:
                        out[py, px] = color
            return out
        if mode == "Diffuse coarse cells":
            return self.coarse_cells
        ids = self.coarse_owner.astype(np.uint32)
        r = ((ids * 1664525 + 1013904223) & 255) / 255.0
        g = ((ids * 22695477 + 1) & 255) / 255.0
        b = ((ids * 1103515245 + 12345) & 255) / 255.0
        return np.stack([r, g, b], axis=-1)
