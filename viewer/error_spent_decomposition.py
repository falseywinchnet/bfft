"""Blue-noise image cells allocated by spending reconstruction error.

The recursive BFFT stack is used only to describe *where* a cell should be
coarse or fine.  Pixel values are always fitted directly to the original.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import ndimage as ndi

from bfft.effects import srgb_to_lab
from recursive_decomposition import RecursiveConfig, RecursiveDecomposition
from seeded_decomposition import (
    COARSE, DETAIL, SeededConfig, SeededDecomposition,
)


@dataclass
class ErrorSpentConfig(RecursiveConfig):
    total_cells: int = 1200
    foundation_cells: int = 48
    allocation_batch: int = 48
    error_blur: float = 1.1
    blue_noise_strength: float = 1.0
    debit_radius: float = 0.78
    detail_threshold: float = 0.43
    detail_reach: float = 0.82
    detail_anisotropy: float = 8.0
    ownership_softness: float = 9.0
    power_irregularity: float = 0.16
    bounded_gradients: bool = True


class ErrorSpentDecomposition:
    """Allocate a single cell budget with blue noise and measured error."""

    def __init__(self, image, config=None, initialize_only=False):
        self.cfg = config or ErrorSpentConfig()
        t0 = time.perf_counter()
        self.analysis = RecursiveDecomposition(image, self.cfg)
        self.rgb = self.analysis.rgb
        self.lab = srgb_to_lab(self.rgb)
        self.h, self.w = self.rgb.shape[:2]
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self._points = np.column_stack([
            self.xx.ravel(), self.yy.ravel(),
        ]).astype(np.float64)
        self.allocation_psnr = []
        self.allocation_error = []
        self._initialize()
        self._adopt_fit(self._fit_current())
        if not initialize_only:
            while len(self.seeds) < max(2, int(self.cfg.total_cells)):
                self.step(min(
                    max(1, int(self.cfg.allocation_batch)),
                    int(self.cfg.total_cells) - len(self.seeds)))
        self.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _site_geometry(self, seeds, marks):
        xy = np.rint(seeds).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, self.w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.h - 1)
        angles = self.analysis.residual_angle[xy[:, 1], xy[:, 0]]
        propensity = (
            self.analysis.residual_fineness[xy[:, 1], xy[:, 0]] *
            self.analysis.residual_coherence[xy[:, 1], xy[:, 0]])
        ratios = np.where(
            marks == DETAIL,
            np.clip(
                1.0 + float(self.cfg.detail_anisotropy) * propensity,
                1.0, 16.0),
            1.0)
        angles = np.where(marks == DETAIL, angles, 0.0)
        return angles, ratios

    def _fit_current(self):
        """Use the proven direct affine-cell fitter without rerunning BFFT."""
        model = SeededDecomposition.__new__(SeededDecomposition)
        model.cfg = SeededConfig(
            ownership_softness=self.cfg.ownership_softness,
            detail_reach=self.cfg.detail_reach,
            power_irregularity=self.cfg.power_irregularity,
            bounded_gradients=self.cfg.bounded_gradients)
        model.analysis = self.analysis
        model.rgb = self.rgb
        model.lab = self.lab
        model.h, model.w = self.h, self.w
        model.yy, model.xx = self.yy, self.xx
        model.xf = self.xx.ravel().astype(np.float64)
        model.yf = self.yy.ravel().astype(np.float64)
        model.seeds = self.seeds.copy()
        model.marks = self.marks.copy()
        model.angles = self.angles.copy()
        model.ratios = self.ratios.copy()
        model._assign()
        model._fit()
        model._render()
        return model

    def _foundation(self, count):
        """A small, even coat; structure does not get to consume it all."""
        density = np.ones((self.h, self.w), dtype=np.float64)
        seeds = self.analysis._weighted_farthest(count, density)
        marks = np.full(len(seeds), COARSE, dtype=np.uint8)
        return seeds, marks

    def _detail_propensity(self):
        # Fine residuals and coherent directions favor anisotropic cells.
        # Broad recursive boundaries favor diffuse coarse cells.
        fine = (
            0.58 * self.analysis.collective_residual_energy +
            0.27 * self.analysis.residual_fineness +
            0.15 * self.analysis.residual_coherence)
        coarse = (
            0.34 + 0.66 * self.analysis.boundary_accumulation)
        return fine / np.maximum(fine + coarse, 1e-9)

    def _measured_error(self, model):
        delta = self.lab - model.reconstruction_lab
        value = (
            delta[..., 0] ** 2 +
            1.5 * (delta[..., 1] ** 2 + delta[..., 2] ** 2))
        # Local error mass is what a cell can plausibly purchase.  Retaining
        # the value term prevents the old washed-out gradient nullspace.
        sigma = max(float(self.cfg.error_blur), 0.0)
        if sigma:
            value = ndi.gaussian_filter(value, sigma, mode="reflect")
        return np.maximum(value, 1e-12)

    def _allocate_from_account(self, account, number, detail_propensity):
        """Spend an error account while maintaining variable blue noise."""
        flat = account.ravel().copy()
        min_d2 = np.full(len(flat), np.inf, dtype=np.float64)
        if len(self.seeds):
            for point in self.seeds:
                d2 = (
                    (self._points[:, 0] - point[0]) ** 2 +
                    (self._points[:, 1] - point[1]) ** 2)
                np.minimum(min_d2, d2, out=min_d2)

        chosen = []
        chosen_marks = []
        # The footprint follows the density after this manually requested
        # round.  There is deliberately no terminal cell budget.
        spacing = math.sqrt(self.h * self.w / max(
            len(self.seeds) + number, 1))
        debit_sigma = max(float(self.cfg.debit_radius) * spacing, 0.6)
        blue_power = max(float(self.cfg.blue_noise_strength), 0.0)
        for _ in range(number):
            separation = np.maximum(min_d2, 0.25) ** blue_power
            score = flat * separation
            index = int(np.argmax(score))
            if not np.isfinite(score[index]) or score[index] <= 0.0:
                break
            x, y = self._points[index]
            chosen.append((x, y))
            mark = (
                DETAIL if detail_propensity[int(y), int(x)] >=
                float(self.cfg.detail_threshold) else COARSE)
            chosen_marks.append(mark)

            d2 = (
                (self._points[:, 0] - x) ** 2 +
                (self._points[:, 1] - y) ** 2)
            np.minimum(min_d2, d2, out=min_d2)
            # The selected cell debits the error it can cover.  A Gaussian
            # debit leaves partial credit at its edge instead of carving a
            # hard exclusion disk.
            debit = 1.0 - np.exp(-0.5 * d2 / (debit_sigma * debit_sigma))
            flat *= debit

        return (
            np.asarray(chosen, dtype=np.float64).reshape(-1, 2),
            np.asarray(chosen_marks, dtype=np.uint8))

    def _initialize(self):
        foundation = max(
            2, min(int(self.cfg.foundation_cells), self.h * self.w))
        self.seeds, self.marks = self._foundation(foundation)
        self.angles, self.ratios = self._site_geometry(
            self.seeds, self.marks)

    def step(self, number=None):
        """Manually purchase one more blue-noise round from current error."""
        count = max(1, int(
            self.cfg.allocation_batch if number is None else number))
        count = min(count, self.h * self.w - len(self.seeds))
        if count <= 0:
            return 0
        error = self._measured_error(self)
        new_seeds, new_marks = self._allocate_from_account(
            error, count, self._detail_propensity())
        if not len(new_seeds):
            return 0
        self.seeds = np.vstack([self.seeds, new_seeds])
        self.marks = np.concatenate([self.marks, new_marks])
        self.angles, self.ratios = self._site_geometry(
            self.seeds, self.marks)
        self._adopt_fit(self._fit_current())
        return len(new_seeds)

    def _adopt_fit(self, model):
        for name in (
                "owner", "second", "d1", "d2", "coeff", "lo", "hi",
                "reconstruction_lab", "reconstruction", "error", "psnr"):
            setattr(self, name, getattr(model, name))
        self.allocation_psnr.append(float(self.psnr))
        self.allocation_error.append(float(np.mean(self.error ** 2)))

    def view(self, mode):
        if mode == "Original":
            return self.rgb
        if mode == "Reconstruction":
            return self.reconstruction
        if mode == "Error spent":
            g = np.clip(
                self.error / max(float(np.percentile(
                    self.error, 99.0)), 1e-9), 0.0, 1.0)
            return np.stack([g, g * g, 0.08 * (1.0 - g)], axis=-1)
        if mode == "Cell classes":
            marks = self.marks[self.owner.reshape(self.h, self.w)]
            return np.stack([
                np.where(marks == COARSE, 0.95, 0.8),
                np.where(marks == COARSE, 0.72, 0.05),
                np.where(marks == COARSE, 0.08, 0.95),
            ], axis=-1)
        if mode == "Allocation order":
            out = np.clip(0.18 + 0.58 * self.rgb, 0.0, 1.0)
            total = max(len(self.seeds) - 1, 1)
            for order, ((x, y), mark) in enumerate(zip(
                    self.seeds, self.marks)):
                phase = order / total
                color = (
                    (1.0, 0.65 * (1.0 - phase), 0.05)
                    if mark == COARSE else
                    (1.0 - 0.55 * phase, 0.05, 1.0))
                ix, iy = int(round(x)), int(round(y))
                out[max(0, iy - 1):min(self.h, iy + 2),
                    max(0, ix - 1):min(self.w, ix + 2)] = color
            return out
        if mode == "Recursive priority only":
            return self.analysis.view("Collective residual support")
        if mode == "Local cell character":
            return self.analysis.view("Detail anisotropy preview")
        ids = self.owner.reshape(self.h, self.w).astype(np.uint32)
        r = ((ids * 1664525 + 1013904223) & 255) / 255.0
        g = ((ids * 22695477 + 1) & 255) / 255.0
        b = ((ids * 1103515245 + 12345) & 255) / 255.0
        return np.stack([r, g, b], axis=-1)
