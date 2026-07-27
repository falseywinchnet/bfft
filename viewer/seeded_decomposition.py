"""One-shot image decomposition nucleated by recursive BFFT fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from bfft.effects import lab_to_srgb, srgb_to_lab
from recursive_decomposition import RecursiveConfig, RecursiveDecomposition


COARSE = np.uint8(0)
DETAIL = np.uint8(1)


@dataclass
class SeededConfig(RecursiveConfig):
    coarse_cells: int = 180
    detail_cells: int = 1020
    ownership_softness: float = 9.0
    detail_reach: float = 0.8
    power_irregularity: float = 0.16
    bounded_gradients: bool = True


class SeededDecomposition:
    """Direct global OKLab fit over recursively nucleated power cells."""

    def __init__(self, image, config=None, analysis=None):
        self.cfg = config or SeededConfig()
        t0 = time.perf_counter()
        self.analysis = (
            analysis if analysis is not None
            else RecursiveDecomposition(image, self.cfg))
        self.rgb = self.analysis.rgb
        self.lab = srgb_to_lab(self.rgb)
        self.h, self.w = self.rgb.shape[:2]
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self.xf = self.xx.ravel().astype(np.float64)
        self.yf = self.yy.ravel().astype(np.float64)
        self._build_sites()
        self._assign()
        self._fit()
        self._render()
        self.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _build_sites(self):
        coarse = self.analysis.seeds
        detail = self.analysis.detail_seeds
        # Avoid exact duplicate germs while preserving the residual ordering.
        accepted_detail = []
        accepted_indices = []
        existing = coarse.copy()
        for detail_index, point in enumerate(detail):
            if len(existing):
                d2 = np.sum((existing - point) ** 2, axis=1)
                if float(d2.min()) < 0.75:
                    continue
            accepted_detail.append(point)
            accepted_indices.append(detail_index)
            existing = np.vstack([existing, point])
        detail = np.asarray(accepted_detail, dtype=np.float64)
        if detail.size == 0:
            detail = np.empty((0, 2), dtype=np.float64)
        self.seeds = np.vstack([coarse, detail])
        self.marks = np.concatenate([
            np.full(len(coarse), COARSE, dtype=np.uint8),
            np.full(len(detail), DETAIL, dtype=np.uint8),
        ])
        self.angles = np.concatenate([
            np.zeros(len(coarse), dtype=np.float64),
            self.analysis.detail_angles[accepted_indices],
        ])
        self.ratios = np.concatenate([
            np.ones(len(coarse), dtype=np.float64),
            self.analysis.detail_ratios[accepted_indices],
        ])

    def _assign(self):
        points = np.column_stack([self.xf, self.yf])
        n = len(self.seeds)
        npix = len(points)
        self.owner = np.empty(npix, dtype=np.int32)
        self.second = np.empty(npix, dtype=np.int32)
        self.d1 = np.empty(npix, dtype=np.float64)
        self.d2 = np.empty(npix, dtype=np.float64)
        spacing2 = self.h * self.w / max(n, 1)
        ids = np.arange(n, dtype=np.uint32)
        hashed = ((ids * 1664525 + 1013904223) & 0xFFFFFFFF) / 2**32
        power = (
            self.cfg.power_irregularity * spacing2 * (hashed - 0.5))
        reach = np.where(
            self.marks == DETAIL,
            max(float(self.cfg.detail_reach), 0.1), 1.0)
        power += spacing2 * (reach * reach - 1.0)
        ct, st = np.cos(self.angles), np.sin(self.angles)
        ratio = np.maximum(self.ratios, 1.0)

        for start in range(0, npix, 1024):
            stop = min(start + 1024, npix)
            dx = points[start:stop, None, 0] - self.seeds[None, :, 0]
            dy = points[start:stop, None, 1] - self.seeds[None, :, 1]
            tangent = dx * ct[None, :] + dy * st[None, :]
            normal = -dx * st[None, :] + dy * ct[None, :]
            distance = (
                tangent * tangent / ratio[None, :] +
                normal * normal * ratio[None, :] - power[None, :])
            pair = np.argpartition(distance, 1, axis=1)[:, :2]
            pair_distance = np.take_along_axis(distance, pair, axis=1)
            order = np.argsort(pair_distance, axis=1)
            ranked = np.take_along_axis(pair, order, axis=1)
            ranked_distance = np.take_along_axis(
                pair_distance, order, axis=1)
            self.owner[start:stop] = ranked[:, 0]
            self.second[start:stop] = ranked[:, 1]
            self.d1[start:stop] = ranked_distance[:, 0]
            self.d2[start:stop] = ranked_distance[:, 1]

    def _coordinates(self, ids):
        sx, sy = self.seeds[ids, 0], self.seeds[ids, 1]
        dx, dy = self.xf - sx, self.yf - sy
        ct, st = np.cos(self.angles[ids]), np.sin(self.angles[ids])
        return dx * ct + dy * st, -dx * st + dy * ct

    def _fit(self):
        n = len(self.seeds)
        ids = self.owner
        q, r = self._coordinates(ids)
        count = np.bincount(ids, minlength=n).astype(np.float64)
        count_safe = np.maximum(count, 1.0)
        sum_q = np.bincount(ids, weights=q, minlength=n)
        sum_r = np.bincount(ids, weights=r, minlength=n)
        normal = np.zeros((n, 3, 3), dtype=np.float64)
        normal[:, 0, 0] = count_safe
        normal[:, 0, 1] = normal[:, 1, 0] = sum_q
        normal[:, 0, 2] = normal[:, 2, 0] = sum_r
        normal[:, 1, 1] = (
            np.bincount(ids, weights=q * q, minlength=n) + 1e-4)
        normal[:, 2, 2] = (
            np.bincount(ids, weights=r * r, minlength=n) + 1e-4)
        normal[:, 1, 2] = normal[:, 2, 1] = np.bincount(
            ids, weights=q * r, minlength=n)
        flat = self.lab.reshape(-1, 3)
        self.coeff = np.zeros((n, 3, 3), dtype=np.float64)
        self.lo = np.zeros((n, 3), dtype=np.float64)
        self.hi = np.zeros((n, 3), dtype=np.float64)
        for channel in range(3):
            values = flat[:, channel]
            sum_y = np.bincount(ids, weights=values, minlength=n)
            sum_y2 = np.bincount(
                ids, weights=values * values, minlength=n)
            rhs = np.column_stack([
                sum_y,
                np.bincount(ids, weights=q * values, minlength=n),
                np.bincount(ids, weights=r * values, minlength=n),
            ])
            try:
                fitted = np.linalg.solve(normal, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                fitted = np.zeros((n, 3), dtype=np.float64)
                fitted[:, 0] = sum_y / count_safe
            fitted[:, 1:] = np.clip(fitted[:, 1:], -0.09, 0.09)
            self.coeff[:, channel, :] = fitted
            mean = sum_y / count_safe
            std = np.sqrt(np.maximum(
                sum_y2 / count_safe - mean * mean, 0.0))
            margin = 2.8 * std + (0.01 if channel == 0 else 0.004)
            self.lo[:, channel] = mean - margin
            self.hi[:, channel] = mean + margin

    def _predict(self, ids):
        q, r = self._coordinates(ids)
        prediction = (
            self.coeff[ids, :, 0] +
            self.coeff[ids, :, 1] * q[:, None] +
            self.coeff[ids, :, 2] * r[:, None])
        if self.cfg.bounded_gradients:
            prediction = np.minimum(
                np.maximum(prediction, self.lo[ids]), self.hi[ids])
        return prediction

    def _render(self):
        first = self._predict(self.owner)
        softness = max(float(self.cfg.ownership_softness), 0.0)
        if softness == 0.0:
            fitted = first
        else:
            other = self._predict(self.second)
            spacing2 = self.h * self.w / max(len(self.seeds), 1)
            gap = np.clip(
                (self.d2 - self.d1) / max(spacing2, 1e-9),
                -40.0, 40.0)
            blend = 1.0 / (1.0 + np.exp(-softness * gap))
            fitted = (
                first * blend[:, None] +
                other * (1.0 - blend[:, None]))
        self.reconstruction_lab = fitted.reshape(self.h, self.w, 3)
        self.reconstruction = np.clip(
            lab_to_srgb(self.reconstruction_lab), 0.0, 1.0)
        delta = self.lab - self.reconstruction_lab
        self.error = np.sqrt(
            delta[..., 0] ** 2 +
            1.5 * (delta[..., 1] ** 2 + delta[..., 2] ** 2))
        mse = float(np.mean((self.rgb - self.reconstruction) ** 2))
        self.psnr = -10.0 * math.log10(max(mse, 1e-12))

    def view(self, mode):
        if mode == "Original":
            return self.rgb
        if mode == "Reconstruction":
            return self.reconstruction
        if mode == "Error":
            scale = max(float(np.percentile(self.error, 99.0)), 1e-9)
            g = np.clip(self.error / scale, 0.0, 1.0)
            return np.stack([g, g * g, 0.1 * (1.0 - g)], axis=-1)
        if mode == "Cell classes":
            marks = self.marks[
                self.owner.reshape(self.h, self.w)]
            return np.stack([
                np.where(marks == COARSE, 1.0, 0.85),
                np.where(marks == COARSE, 0.7, 0.05),
                np.where(marks == COARSE, 0.05, 0.9),
            ], axis=-1)
        if mode == "Cells":
            ids = self.owner.reshape(self.h, self.w).astype(np.uint32)
            r = ((ids * 1664525 + 1013904223) & 255) / 255.0
            g = ((ids * 22695477 + 1) & 255) / 255.0
            b = ((ids * 1103515245 + 12345) & 255) / 255.0
            return np.stack([r, g, b], axis=-1)
        if mode == "Recursive boundary field":
            return self.analysis.view("Boundary accumulation")
        if mode == "Collective residual":
            return self.analysis.view("Collective carried residual")
        if mode == "Detail germination order":
            return self.analysis.view("Detail germination order")
        out = self.reconstruction.copy()
        for (x, y), mark, angle, ratio in zip(
                self.seeds, self.marks, self.angles, self.ratios):
            color = (1.0, 0.72, 0.0) if mark == COARSE else (1.0, 0.0, 0.8)
            ix, iy = int(round(x)), int(round(y))
            out[max(0, iy - 1):min(self.h, iy + 2),
                max(0, ix - 1):min(self.w, ix + 2)] = color
            if mark == DETAIL:
                length = 1.5 + min(8.0, float(ratio))
                for parameter in np.linspace(-length, length, 13):
                    px = int(round(x + math.cos(angle) * parameter))
                    py = int(round(y + math.sin(angle) * parameter))
                    if 0 <= px < self.w and 0 <= py < self.h:
                        out[py, px] = color
        return out
