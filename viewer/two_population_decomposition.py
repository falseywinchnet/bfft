"""Coupled coarse/fine cells disciplined by decomposition of their mixture."""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import math
import time

import numpy as np
from scipy import ndimage as ndi

import bfft
from bfft.effects import lab_to_srgb, meyer_channels, srgb_to_lab
from recursive_decomposition import (
    RecursiveConfig, RecursiveDecomposition, _native_from_working,
    _normalize,
)
from seeded_decomposition import SeededConfig, SeededDecomposition


@dataclass
class TwoPopulationConfig(RecursiveConfig):
    rounds: int = 7
    coarse_batch: int = 16
    fine_batch: int = 36
    coarse_radius: float = 10.0
    fine_radius: float = 4.5
    coarse_gain: float = 1.0
    fine_gain: float = 1.0
    cartoon_weight: float = 1.0
    texture_weight: float = 1.25
    gradient_weight: float = 1.0
    cartoon_value_weight: float = 1.0
    texture_value_weight: float = 1.0
    mean_anchor: float = 0.035
    chroma_anchor: float = 0.15
    recursive_priority: float = 0.8
    flow_weight: float = 0.0
    flow_orientation: float = 0.0
    flow_sweeps: int = 24


def _split_fields(rgb, cfg):
    split = meyer_channels(
        rgb, space=cfg.space, lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = max(float(split.scale[0]), 1e-12)
    cartoon = split.offset[0] + split.cartoon[..., 0] / scale
    texture = split.texture[..., 0] / scale
    return split, cartoon, texture


def _flow_field(split, cfg):
    if cfg.flow_weight <= 0.0 and cfg.flow_orientation <= 0.0:
        return np.zeros(split.planes.shape[:2], dtype=np.float64)
    light = split.planes[..., 0]
    projected = bfft.rof(
        light - split.texture[..., 0], c=cfg.lam,
        eta=2.0 * cfg.lam, sweeps=cfg.flow_sweeps,
        tol=0.0, threads=4)
    return (
        split.cartoon[..., 0] - projected) / max(
            float(split.scale[0]), 1e-12)


def _mismatch_energy(delta, cfg, value_weight):
    gx = ndi.sobel(delta, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(delta, axis=0, mode="reflect") / 8.0
    return (
        float(value_weight) * delta * delta +
        cfg.gradient_weight * (gx * gx + gy * gy))


class TwoPopulationDecomposition:
    """Alternating coarse/fine birth against one shared composition."""

    def __init__(self, image, config=None):
        self.cfg = config or TwoPopulationConfig()
        t0 = time.perf_counter()
        self.analysis = RecursiveDecomposition(image, self.cfg)
        self.rgb = self.analysis.rgb
        self.target_lab = srgb_to_lab(self.rgb)
        self.h, self.w = self.rgb.shape[:2]
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self.points = np.column_stack([
            self.xx.ravel(), self.yy.ravel(),
        ]).astype(np.float64)
        self.target_split, self.target_cartoon, self.target_texture = (
            _split_fields(self.rgb, self.cfg))
        self.target_flow = _flow_field(self.target_split, self.cfg)

        # Recursive analysis controls metadata only. Pixel values begin from
        # a direct, target-colored global fit over the recursively chosen
        # sites; no recursively attenuated image enters the signal path.
        seed_kwargs = {
            field.name: getattr(self.cfg, field.name)
            for field in dataclass_fields(RecursiveConfig)
        }
        seeded = SeededDecomposition(
            image, SeededConfig(**seed_kwargs), analysis=self.analysis)
        initial_rgb = seeded.reconstruction
        initial_split = meyer_channels(
            initial_rgb, space=self.cfg.space, lam=self.cfg.lam,
            mu=self.cfg.mu, passes=self.cfg.passes, threads=4)
        coarse_rgb = _native_from_working(
            initial_split.residual + initial_split.cartoon,
            initial_split)
        initial_lab = srgb_to_lab(initial_rgb)
        self.coarse_lab = srgb_to_lab(coarse_rgb)
        self.fine_lab = initial_lab - self.coarse_lab
        self.coarse_seeds = self.analysis.seeds.copy()
        self.fine_seeds = self.analysis.detail_seeds.copy()
        self.reconstructions = []
        self.coarse_layers = []
        self.fine_layers = []
        self.losses = []
        self.psnrs = []
        self.cartoon_errors = []
        self.texture_errors = []
        self.flow_errors = []
        self.coarse_births = []
        self.fine_births = []
        self.fine_angles = []
        self.fine_ratios = []
        self.stopped_reason = ""
        self._run()
        self.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _compose(self, coarse_lab=None, fine_lab=None):
        coarse = self.coarse_lab if coarse_lab is None else coarse_lab
        fine = self.fine_lab if fine_lab is None else fine_lab
        return np.clip(lab_to_srgb(coarse + fine), 0.0, 1.0)

    def _evaluate(self, composition):
        split, cartoon, texture = _split_fields(composition, self.cfg)
        flow = _flow_field(split, self.cfg)
        cartoon_delta = self.target_cartoon - cartoon
        texture_delta = self.target_texture - texture
        flow_delta = self.target_flow - flow
        cartoon_error = _mismatch_energy(
            cartoon_delta, self.cfg, self.cfg.cartoon_value_weight)
        texture_error = _mismatch_energy(
            texture_delta, self.cfg, self.cfg.texture_value_weight)
        flow_error = _mismatch_energy(flow_delta, self.cfg, 1.0)
        lab_delta = self.target_lab - srgb_to_lab(composition)
        anchor = self.cfg.mean_anchor * np.mean(
            lab_delta * lab_delta, axis=2)
        total = (
            self.cfg.cartoon_weight * cartoon_error +
            self.cfg.texture_weight * texture_error +
            self.cfg.flow_weight * flow_error + anchor)
        return {
            "loss": float(np.mean(total)),
            "total": total,
            "cartoon_error": cartoon_error,
            "texture_error": texture_error,
            "flow_error": flow_error,
            "cartoon_delta": cartoon_delta,
            "texture_delta": texture_delta,
            "flow_delta": flow_delta,
            "lab_delta": lab_delta,
        }

    def _spawn(self, field, existing, count):
        priority = _normalize(
            ndi.gaussian_filter(field, 0.75, mode="reflect")).ravel()
        if len(existing):
            min_d2 = np.full(len(self.points), np.inf)
            for start in range(0, len(existing), 128):
                delta = (
                    self.points[:, None, :] -
                    existing[None, start:start + 128, :])
                local = np.min(np.sum(delta * delta, axis=2), axis=1)
                np.minimum(min_d2, local, out=min_d2)
        else:
            min_d2 = np.full(
                len(self.points), self.h * self.h + self.w * self.w,
                dtype=np.float64)
        selected = []
        for _ in range(max(1, int(count))):
            clearance = np.sqrt(np.maximum(min_d2, 0.0))
            scale = max(float(np.percentile(clearance, 95.0)), 1e-9)
            score = priority * (
                0.18 + np.clip(clearance / scale, 0.0, 1.5))
            index = int(np.argmax(score))
            if score[index] <= 0.0:
                break
            selected.append(index)
            distance = np.sum(
                (self.points - self.points[index]) ** 2, axis=1)
            np.minimum(min_d2, distance, out=min_d2)
            priority[index] = 0.0
        return self.points[selected]

    def _fine_geometry(self, evaluation, seeds):
        structural = (
            evaluation["texture_delta"] +
            self.cfg.flow_orientation * evaluation["flow_delta"])
        sigma = max(float(self.cfg.residual_scale), 0.35)
        smooth = ndi.gaussian_filter(structural, sigma, mode="reflect")
        fine = _normalize(np.abs(structural - smooth))
        coarse = _normalize(ndi.gaussian_filter(
            np.abs(structural), sigma, mode="reflect"))
        fineness = fine / np.maximum(fine + coarse, 1e-9)
        gx = ndi.sobel(structural, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(structural, axis=0, mode="reflect") / 8.0
        jxx = ndi.gaussian_filter(gx * gx, 1.35, mode="reflect")
        jyy = ndi.gaussian_filter(gy * gy, 1.35, mode="reflect")
        jxy = ndi.gaussian_filter(gx * gy, 1.35, mode="reflect")
        disc = np.sqrt(np.maximum(
            (jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
        coherence = disc / np.maximum(jxx + jyy, 1e-12)
        angle = 0.5 * np.arctan2(
            2.0 * jxy, jxx - jyy) + math.pi * 0.5
        xy = np.rint(seeds).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, self.w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.h - 1)
        sampled_angles = angle[xy[:, 1], xy[:, 0]]
        ratios = np.clip(
            1.0 + self.cfg.detail_anisotropy *
            fineness[xy[:, 1], xy[:, 0]] *
            coherence[xy[:, 1], xy[:, 0]],
            1.0, 16.0)
        return sampled_angles, ratios

    def _fit_atoms(self, desired, seeds, angles, ratios, radius):
        accumulation = np.zeros_like(self.coarse_lab)
        coverage = np.zeros((self.h, self.w), dtype=np.float64)
        base = max(float(radius), 1.0)
        for (x, y), angle, ratio in zip(seeds, angles, ratios):
            major = base * math.sqrt(float(ratio))
            minor = base / math.sqrt(float(ratio))
            bound = int(math.ceil(2.75 * major))
            x0, x1 = max(0, int(x) - bound), min(
                self.w, int(x) + bound + 1)
            y0, y1 = max(0, int(y) - bound), min(
                self.h, int(y) + bound + 1)
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
            matrix = (
                design.T @ (weights[:, None] * design) +
                np.diag([1e-6, 2e-3, 2e-3]))
            patch = desired[y0:y1, x0:x1].reshape(-1, 3)
            rhs = design.T @ (weights[:, None] * patch)
            try:
                coeff = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                coeff = np.zeros((3, 3), dtype=np.float64)
            prediction = (design @ coeff).reshape(
                y1 - y0, x1 - x0, 3)
            prediction[..., 0] = np.clip(
                prediction[..., 0], -0.2, 0.2)
            prediction[..., 1:] = np.clip(
                prediction[..., 1:], -0.07, 0.07)
            accumulation[y0:y1, x0:x1] += (
                kernel[..., None] * prediction)
            coverage[y0:y1, x0:x1] += kernel
        correction = accumulation / np.maximum(
            coverage[..., None], 1e-9)
        return correction * (
            1.0 - np.exp(-coverage))[..., None]

    def _coarse_desired(self, evaluation):
        desired = np.zeros_like(self.coarse_lab)
        desired[..., 0] = evaluation["cartoon_delta"]
        desired[..., 1:] = (
            self.cfg.chroma_anchor *
            evaluation["lab_delta"][..., 1:])
        return desired

    def _fine_desired(self, evaluation):
        desired = np.zeros_like(self.fine_lab)
        desired[..., 0] = evaluation["texture_delta"]
        desired[..., 1:] = (
            0.5 * self.cfg.chroma_anchor *
            evaluation["lab_delta"][..., 1:])
        return desired

    def _line_search(self, population, correction, gain, current_loss):
        trial_gain = max(float(gain), 1e-4)
        for _ in range(6):
            if population == "coarse":
                coarse = self.coarse_lab + trial_gain * correction
                composition = self._compose(
                    coarse_lab=coarse, fine_lab=self.fine_lab)
            else:
                fine = self.fine_lab + trial_gain * correction
                composition = self._compose(
                    coarse_lab=self.coarse_lab, fine_lab=fine)
            evaluation = self._evaluate(composition)
            if evaluation["loss"] < current_loss - 1e-12:
                return trial_gain, composition, evaluation
            trial_gain *= 0.5
        return None

    def _record(self, composition, evaluation):
        self.reconstructions.append(composition)
        self.coarse_layers.append(self.coarse_lab.copy())
        self.fine_layers.append(self.fine_lab.copy())
        self.losses.append(evaluation["loss"])
        self.cartoon_errors.append(evaluation["cartoon_error"])
        self.texture_errors.append(evaluation["texture_error"])
        self.flow_errors.append(evaluation["flow_error"])
        mse = float(np.mean((self.rgb - composition) ** 2))
        self.psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))

    def _run(self):
        composition = self._compose()
        evaluation = self._evaluate(composition)
        self._record(composition, evaluation)
        for _ in range(max(0, int(self.cfg.rounds))):
            accepted_any = False

            coarse_seeds = self._spawn(
                evaluation["cartoon_error"], self.coarse_seeds,
                self.cfg.coarse_batch)
            coarse_angles = np.zeros(len(coarse_seeds))
            coarse_ratios = np.ones(len(coarse_seeds))
            coarse_correction = self._fit_atoms(
                self._coarse_desired(evaluation), coarse_seeds,
                coarse_angles, coarse_ratios, self.cfg.coarse_radius)
            result = self._line_search(
                "coarse", coarse_correction, self.cfg.coarse_gain,
                evaluation["loss"])
            if result is not None:
                gain, composition, evaluation = result
                self.coarse_lab += gain * coarse_correction
                self.coarse_seeds = np.vstack([
                    self.coarse_seeds, coarse_seeds])
                self.coarse_births.append(coarse_seeds)
                accepted_any = True
            else:
                self.coarse_births.append(np.empty((0, 2)))

            recursive_prior = self.analysis.collective_residual_energy
            fine_pressure = (
                evaluation["texture_error"] +
                self.cfg.recursive_priority *
                np.mean(evaluation["texture_error"]) *
                recursive_prior / max(
                    float(np.mean(recursive_prior)), 1e-9))
            fine_seeds = self._spawn(
                fine_pressure, self.fine_seeds, self.cfg.fine_batch)
            fine_angles, fine_ratios = self._fine_geometry(
                evaluation, fine_seeds)
            fine_correction = self._fit_atoms(
                self._fine_desired(evaluation), fine_seeds,
                fine_angles, fine_ratios, self.cfg.fine_radius)
            result = self._line_search(
                "fine", fine_correction, self.cfg.fine_gain,
                evaluation["loss"])
            if result is not None:
                gain, composition, evaluation = result
                self.fine_lab += gain * fine_correction
                self.fine_seeds = np.vstack([
                    self.fine_seeds, fine_seeds])
                self.fine_births.append(fine_seeds)
                self.fine_angles.append(fine_angles)
                self.fine_ratios.append(fine_ratios)
                accepted_any = True
            else:
                self.fine_births.append(np.empty((0, 2)))
                self.fine_angles.append(np.empty(0))
                self.fine_ratios.append(np.empty(0))

            if not accepted_any:
                self.stopped_reason = "neither population found descent"
                break
            self._record(composition, evaluation)
        if not self.stopped_reason:
            self.stopped_reason = "round budget reached"

    def view(self, mode, round_index=-1):
        index = int(np.clip(
            round_index, 0, len(self.reconstructions) - 1))
        if mode == "Original":
            return self.rgb
        if mode == "Composition":
            return self.reconstructions[index]
        if mode == "Coarse layer":
            return np.clip(
                lab_to_srgb(self.coarse_layers[index]), 0.0, 1.0)
        if mode == "Fine layer":
            field = self.fine_layers[index]
            scale = max(float(np.percentile(
                np.abs(field), 99.0)), 1e-9)
            return np.clip(0.5 + field / (2.0 * scale), 0.0, 1.0)
        if mode == "Cartoon mismatch":
            g = _normalize(self.cartoon_errors[index], 99.0)
            return np.stack([g, 0.65 * g, 0.05 * g], axis=-1)
        if mode == "Texture mismatch":
            g = _normalize(self.texture_errors[index], 99.0)
            return np.stack([g, 0.1 * g, 1.0 - g], axis=-1)
        if mode == "Flow mismatch":
            g = _normalize(self.flow_errors[index], 99.0)
            return np.stack([0.1 * g, g, g * g], axis=-1)
        if mode == "Recursive detail prior":
            return self.analysis.view("Collective residual support")
        out = self.reconstructions[index].copy()
        if index > 0:
            for x, y in self.coarse_births[index - 1]:
                ix, iy = int(round(x)), int(round(y))
                out[max(0, iy - 2):min(self.h, iy + 3),
                    max(0, ix - 2):min(self.w, ix + 3)] = (1.0, 0.72, 0.0)
            for (x, y), angle, ratio in zip(
                    self.fine_births[index - 1],
                    self.fine_angles[index - 1],
                    self.fine_ratios[index - 1]):
                color = (1.0, 0.0, 0.85)
                length = 1.5 + min(9.0, float(ratio))
                for parameter in np.linspace(-length, length, 17):
                    px = int(round(x + math.cos(angle) * parameter))
                    py = int(round(y + math.sin(angle) * parameter))
                    if 0 <= px < self.w and 0 <= py < self.h:
                        out[py, px] = color
        return out
