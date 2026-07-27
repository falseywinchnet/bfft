"""Recursive cell improvement driven by BFFT decomposition-space loss."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import ndimage as ndi

from bfft.effects import lab_to_srgb, meyer_channels, srgb_to_lab
from recursive_decomposition import (
    RecursiveConfig, RecursiveDecomposition, _normalize,
)


@dataclass
class ImprovementConfig(RecursiveConfig):
    rounds: int = 8
    spawn_batch: int = 32
    atom_radius: float = 5.0
    correction_gain: float = 1.0
    cartoon_weight: float = 1.0
    texture_weight: float = 1.4
    residual_weight: float = 1.2
    gradient_weight: float = 1.0
    mean_anchor: float = 0.04
    chroma_anchor: float = 0.18


def _split_fields(rgb, cfg):
    split = meyer_channels(
        rgb, space=cfg.space, lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    # The first working plane is perceptual lightness for the default spaces.
    scale = max(float(split.scale[0]), 1e-12)
    cartoon = split.offset[0] + split.cartoon[..., 0] / scale
    texture = split.texture[..., 0] / scale
    residual = split.residual[..., 0] / scale
    return split, (cartoon, texture, residual)


def _component_energy(delta, cfg):
    gx = ndi.sobel(delta, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(delta, axis=0, mode="reflect") / 8.0
    return (
        cfg.gradient_weight * (gx * gx + gy * gy) +
        cfg.mean_anchor * delta * delta)


class RecursiveImprovement:
    """Spawn anisotropic residual atoms until decomposition loss stops."""

    def __init__(self, image, config=None):
        self.cfg = config or ImprovementConfig()
        self.analysis = RecursiveDecomposition(image, self.cfg)
        self.rgb = self.analysis.rgb
        self.h, self.w = self.rgb.shape[:2]
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self.target_lab = srgb_to_lab(self.rgb)
        self.target_split, self.target_fields = _split_fields(
            self.rgb, self.cfg)
        self.reconstructions = [self.analysis.coarse_cells.copy()]
        self.losses = []
        self.psnrs = []
        self.error_fields = []
        self.component_fields = []
        self.spawn_batches = []
        self.spawn_angles = []
        self.spawn_ratios = []
        self.all_seeds = np.empty((0, 2), dtype=np.float64)
        self.elapsed_ms = 0.0
        self.stopped_reason = ""
        self._run()

    def _evaluate(self, rgb):
        split, fields = _split_fields(rgb, self.cfg)
        deltas = [
            target - source
            for target, source in zip(self.target_fields, fields)
        ]
        components = [
            _component_energy(delta, self.cfg) for delta in deltas
        ]
        weights = (
            self.cfg.cartoon_weight,
            self.cfg.texture_weight,
            self.cfg.residual_weight,
        )
        total = sum(
            weight * component
            for weight, component in zip(weights, components))
        loss = float(np.mean(total))
        return loss, total, components, deltas, split, fields

    def _spawn(self, field):
        count = max(1, int(self.cfg.spawn_batch))
        points = np.column_stack(
            [self.xx.ravel(), self.yy.ravel()]).astype(np.float64)
        priority = _normalize(
            ndi.gaussian_filter(field, 0.8, mode="reflect")).ravel()
        if len(self.all_seeds):
            delta = points[:, None, :] - self.all_seeds[None, :, :]
            min_d2 = np.min(np.sum(delta * delta, axis=2), axis=1)
        else:
            min_d2 = np.full(
                len(points), self.h * self.h + self.w * self.w,
                dtype=np.float64)
        selected = []
        for _ in range(count):
            clearance = np.sqrt(np.maximum(min_d2, 0.0))
            clearance /= max(float(np.percentile(clearance, 95.0)), 1e-9)
            score = priority * (0.22 + np.clip(clearance, 0.0, 1.5))
            index = int(np.argmax(score))
            if score[index] <= 0:
                break
            selected.append(index)
            d2 = np.sum((points - points[index]) ** 2, axis=1)
            np.minimum(min_d2, d2, out=min_d2)
            priority[index] = 0.0
        return points[selected]

    def _geometry(self, structural_residual, seeds):
        sigma = max(float(self.cfg.residual_scale), 0.35)
        smooth = ndi.gaussian_filter(
            structural_residual, sigma, mode="reflect")
        fine = _normalize(np.abs(structural_residual - smooth))
        coarse = _normalize(ndi.gaussian_filter(
            np.abs(structural_residual), sigma, mode="reflect"))
        fineness = fine / np.maximum(fine + coarse, 1e-9)
        gx = ndi.sobel(
            structural_residual, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(
            structural_residual, axis=0, mode="reflect") / 8.0
        jxx = ndi.gaussian_filter(gx * gx, 1.35, mode="reflect")
        jyy = ndi.gaussian_filter(gy * gy, 1.35, mode="reflect")
        jxy = ndi.gaussian_filter(gx * gy, 1.35, mode="reflect")
        disc = np.sqrt(np.maximum(
            (jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
        coherence = disc / np.maximum(jxx + jyy, 1e-12)
        angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + math.pi * 0.5
        xy = np.rint(seeds).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, self.w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.h - 1)
        sampled_angle = angle[xy[:, 1], xy[:, 0]]
        ratio = np.clip(
            1.0 + self.cfg.detail_anisotropy *
            fineness[xy[:, 1], xy[:, 0]] *
            coherence[xy[:, 1], xy[:, 0]],
            1.0, 16.0)
        return sampled_angle, ratio

    def _fit_correction(self, reconstruction, deltas, seeds, angles, ratios):
        current_lab = srgb_to_lab(reconstruction)
        pixel_delta = self.target_lab - current_lab
        weight_sum = (
            self.cfg.cartoon_weight + self.cfg.texture_weight +
            self.cfg.residual_weight)
        desired_l = (
            self.cfg.cartoon_weight * deltas[0] +
            self.cfg.texture_weight * deltas[1] +
            self.cfg.residual_weight * deltas[2]) / max(weight_sum, 1e-9)
        desired = np.empty_like(pixel_delta)
        desired[..., 0] = desired_l
        desired[..., 1:] = (
            self.cfg.chroma_anchor * pixel_delta[..., 1:])

        accumulation = np.zeros_like(desired)
        coverage = np.zeros((self.h, self.w), dtype=np.float64)
        base = max(float(self.cfg.atom_radius), 1.0)
        for (x, y), angle, ratio in zip(seeds, angles, ratios):
            major = base * math.sqrt(float(ratio))
            minor = base / math.sqrt(float(ratio))
            radius = int(math.ceil(2.75 * major))
            x0, x1 = max(0, int(x) - radius), min(
                self.w, int(x) + radius + 1)
            y0, y1 = max(0, int(y) - radius), min(
                self.h, int(y) + radius + 1)
            dx = self.xx[y0:y1, x0:x1] - x
            dy = self.yy[y0:y1, x0:x1] - y
            ct, st = math.cos(angle), math.sin(angle)
            tangent = dx * ct + dy * st
            normal = -dx * st + dy * ct
            kernel = np.exp(-0.5 * (
                (tangent / major) ** 2 + (normal / minor) ** 2))
            design = np.column_stack([
                np.ones(kernel.size),
                tangent.ravel() / max(major, 1e-9),
                normal.ravel() / max(minor, 1e-9),
            ])
            weights = kernel.ravel()
            normal_matrix = (
                design.T @ (weights[:, None] * design) +
                np.diag([1e-6, 2e-3, 2e-3]))
            patch = desired[y0:y1, x0:x1].reshape(-1, 3)
            rhs = design.T @ (weights[:, None] * patch)
            try:
                coeff = np.linalg.solve(normal_matrix, rhs)
            except np.linalg.LinAlgError:
                coeff = np.zeros((3, 3), dtype=np.float64)
            prediction = (design @ coeff).reshape(
                y1 - y0, x1 - x0, 3)
            prediction[..., 0] = np.clip(
                prediction[..., 0], -0.22, 0.22)
            prediction[..., 1:] = np.clip(
                prediction[..., 1:], -0.08, 0.08)
            accumulation[y0:y1, x0:x1] += (
                kernel[..., None] * prediction)
            coverage[y0:y1, x0:x1] += kernel
        mean_correction = accumulation / np.maximum(
            coverage[..., None], 1e-9)
        alpha = 1.0 - np.exp(-coverage)
        return mean_correction * alpha[..., None]

    def _run(self):
        t0 = time.perf_counter()
        reconstruction = self.reconstructions[0]
        evaluation = self._evaluate(reconstruction)
        loss, field, components, deltas = evaluation[:4]
        self.losses.append(loss)
        self.error_fields.append(field)
        self.component_fields.append(components)
        self.psnrs.append(self._psnr(reconstruction))

        for _ in range(max(0, int(self.cfg.rounds))):
            seeds = self._spawn(field)
            if len(seeds) == 0:
                self.stopped_reason = "no residual germination pressure"
                break
            structural = deltas[1] + deltas[2]
            angles, ratios = self._geometry(structural, seeds)
            correction = self._fit_correction(
                reconstruction, deltas, seeds, angles, ratios)
            current_lab = srgb_to_lab(reconstruction)

            accepted = None
            gain = float(self.cfg.correction_gain)
            for _attempt in range(5):
                candidate = np.clip(
                    lab_to_srgb(current_lab + gain * correction),
                    0.0, 1.0)
                candidate_evaluation = self._evaluate(candidate)
                if candidate_evaluation[0] < loss - 1e-12:
                    accepted = (candidate, candidate_evaluation)
                    break
                gain *= 0.5
            if accepted is None:
                self.stopped_reason = "line search found no descent"
                break

            reconstruction, evaluation = accepted
            loss, field, components, deltas = evaluation[:4]
            self.spawn_batches.append(seeds)
            self.spawn_angles.append(angles)
            self.spawn_ratios.append(ratios)
            self.all_seeds = np.vstack([self.all_seeds, seeds])
            self.reconstructions.append(reconstruction)
            self.losses.append(loss)
            self.error_fields.append(field)
            self.component_fields.append(components)
            self.psnrs.append(self._psnr(reconstruction))
        if not self.stopped_reason:
            self.stopped_reason = "round budget reached"
        self.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _psnr(self, reconstruction):
        mse = float(np.mean((self.rgb - reconstruction) ** 2))
        return -10.0 * math.log10(max(mse, 1e-12))

    def view(self, mode, round_index=-1):
        index = int(np.clip(
            round_index, 0, len(self.reconstructions) - 1))
        if mode == "Original":
            return self.rgb
        if mode == "Reconstruction":
            return self.reconstructions[index]
        if mode == "Decomposition error":
            g = _normalize(self.error_fields[index], 99.0)
            return np.stack([g, g * g, 0.1 * (1.0 - g)], axis=-1)
        names = {
            "Cartoon gradient error": 0,
            "Texture gradient error": 1,
            "Residual gradient error": 2,
        }
        if mode in names:
            g = _normalize(
                self.component_fields[index][names[mode]], 99.0)
            return np.stack([g, 0.25 * g, 1.0 - g], axis=-1)
        if mode == "Improvement":
            difference = (
                self.reconstructions[index] -
                self.reconstructions[0])
            scale = max(float(np.percentile(
                np.abs(difference), 99.0)), 1e-9)
            return np.clip(0.5 + difference / (2.0 * scale), 0.0, 1.0)
        out = self.reconstructions[index].copy()
        if index > 0:
            seeds = self.spawn_batches[index - 1]
            angles = self.spawn_angles[index - 1]
            ratios = self.spawn_ratios[index - 1]
            for (x, y), angle, ratio in zip(seeds, angles, ratios):
                color = (1.0, 0.15, min(1.0, (ratio - 1.0) / 8.0))
                length = 1.5 + min(9.0, float(ratio))
                for parameter in np.linspace(-length, length, 17):
                    px = int(round(x + math.cos(angle) * parameter))
                    py = int(round(y + math.sin(angle) * parameter))
                    if 0 <= px < self.w and 0 <= py < self.h:
                        out[py, px] = color
        return out
