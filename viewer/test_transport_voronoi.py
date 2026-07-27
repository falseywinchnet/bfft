#!/usr/bin/env python3
"""Small deterministic regression for the transport-cell research model."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bfft  # noqa: E402
from transport_voronoi import (  # noqa: E402
    CARTOON, TEXTURE, Config, TransportVoronoi,
    _dijkstra_two_best_packed)


def synthetic_image():
    y, x = np.mgrid[0:64, 0:96]
    edge = (x > 44 + 9 * np.sin(y / 12)).astype(float)
    rings = 0.12 * np.sin(np.hypot(x - 68, y - 31) * 1.5)
    rgb = np.stack([
        0.15 + 0.55 * edge + rings,
        0.2 + 0.35 * x / 95 + 0.08 * np.sin(y / 2),
        0.7 - 0.45 * edge + 0.07 * np.cos((x + y) / 3),
    ], axis=-1)
    return np.clip(rgb, 0, 1)


class TransportVoronoiTest(unittest.TestCase):
    @unittest.skipIf(
        _dijkstra_two_best_packed is None, "Numba backend unavailable")
    def test_bucket_assignment_matches_binary_heap_exactly(self):
        cfg = Config(
            max_side=96, passes=3, initial_cells=31,
            max_cells=60, territory_count=1)
        model = TransportVoronoi(synthetic_image(), cfg)
        seed_x = np.rint(model.seeds[:, 0]).astype(np.int64)
        seed_y = np.rint(model.seeds[:, 1]).astype(np.int64)
        density = model.density.ravel()
        scale = max(float(np.median(density)), 1e-9)
        reach = cfg.site_reach * np.sqrt(
            density[seed_y * model.w + seed_x] / scale)
        owner, runner, first, second = _dijkstra_two_best_packed(
            seed_x, seed_y, reach, model._edge_cost_volume,
            model.h, model.w)
        np.testing.assert_array_equal(model.owner, owner)
        np.testing.assert_array_equal(model.second, runner)
        np.testing.assert_allclose(model.d1, first, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(model.d2, second, atol=0.0, rtol=0.0)

    def test_hierarchy_is_finite_bounded_and_improves(self):
        cfg = Config(
            max_side=96, passes=4, initial_cells=28, max_cells=60,
            split_batch=16, lloyd=0.1)
        model = TransportVoronoi(synthetic_image(), cfg)
        initial_psnr = model.psnr
        self.assertTrue(np.all(model.marks == CARTOON))
        self.assertEqual(len(model.marks), len(model.seeds))
        light = model.lab[..., 0] * 255.0
        projected = bfft.rof(
            light - model.texture, c=cfg.lam, eta=2 * cfg.lam,
            sweeps=cfg.flow_sweeps, tol=0.0, threads=4)
        np.testing.assert_allclose(model.flow, model.cartoon - projected)
        determinant = (model.metric_xx * model.metric_yy -
                       model.metric_xy * model.metric_xy)
        self.assertGreater(float(determinant.min()), 0.0)
        model.step()
        model.step()

        self.assertGreater(model.psnr, initial_psnr)
        self.assertTrue(np.all(model.marks == CARTOON))
        self.assertEqual(len(model.marks), len(model.seeds))
        self.assertEqual(len(model.parents), len(model.seeds))
        self.assertEqual(len(model.generations), len(model.seeds))
        self.assertTrue(np.all(model.support_map_indices < 0))
        self.assertLessEqual(len(model.seeds), cfg.max_cells)
        at_budget_psnr = model.psnr
        model.step()
        self.assertEqual(len(model.seeds), cfg.max_cells)
        self.assertGreaterEqual(model.psnr, at_budget_psnr - 1e-10)
        self.assertTrue(np.isfinite(model.coeff).all())
        self.assertTrue(np.isfinite(model.reconstruction).all())
        self.assertTrue(np.isfinite(model.allocation_pressure).all())
        self.assertTrue(np.isfinite(model.rgb_mse))
        # These metrics require another full decomposition.  HD operation
        # keeps them out of the ordinary render path and evaluates them only
        # when the user requests the composite objective or this measurement.
        self.assertFalse(model.decomp_metrics_fresh)
        model.refresh_decomposition_metrics()
        self.assertTrue(np.isfinite(model.cartoon_decomp_mse))
        self.assertTrue(np.isfinite(model.texture_decomp_mse))
        before_coupling = model.rgb_mse
        model.solve_coupled(multiscale=True)
        self.assertLessEqual(model.rgb_mse, before_coupling + 1e-10)
        self.assertTrue(np.isfinite(model.reconstruction).all())
        np.testing.assert_allclose(
            model.reconstruction,
            model.cartoon_reconstruction +
            model.cfg.detail_precision * model.texture_reconstruction)
        self.assertGreaterEqual(int(model.owner.min()), 0)
        self.assertLess(int(model.owner.max()), len(model.seeds))
        legacy_identity = model.view("Legacy BFFT recomposition")
        self.assertLess(float(np.max(np.abs(legacy_identity - model.rgb))),
                        1e-9)
        model.cfg.legacy_texture_gain = 2.0
        legacy_boosted = model.view("Legacy BFFT recomposition")
        self.assertGreater(float(np.mean(np.abs(
            legacy_boosted - legacy_identity))), 1e-5)
        for mode in ("Cartoon", "Texture", "TV flow", "Flow territories",
                     "Texture activity", "Texture entropy", "Texture demand",
                     "Seed density", "Error", "Allocation pressure",
                     "Expected affine gain",
                     "Cartoon pressure", "Texture pressure", "Cells",
                     "Recursive residual memory",
                     "Composition discrepancy",
                     "Marked cells", "Cartoon-cell reconstruction",
                     "Texture-cell correction",
                     "Legacy BFFT recomposition",
                     "Reconstruction + sites"):
            self.assertEqual(model.view(mode).shape, model.rgb.shape)


if __name__ == "__main__":
    unittest.main()
