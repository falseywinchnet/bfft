#!/usr/bin/env python3
"""Regression for the recursively nucleated one-shot cell fit."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seeded_decomposition import SeededConfig, SeededDecomposition  # noqa: E402


def synthetic():
    y, x = np.mgrid[0:48, 0:64]
    edge = (x > 31 + 4 * np.sin(y / 7)).astype(float)
    detail = 0.1 * np.sin(1.5 * x + 0.3 * y)
    return np.clip(np.stack([
        0.18 + 0.58 * edge + detail,
        0.22 + 0.35 * x / 63,
        0.7 - 0.42 * edge - detail,
    ], axis=-1), 0.0, 1.0)


class SeededDecompositionTest(unittest.TestCase):
    def test_direct_fit_is_finite_and_structured(self):
        model = SeededDecomposition(
            synthetic(), SeededConfig(
                max_side=64, iterations=2, passes=3,
                coarse_cells=24, detail_cells=56, descent_steps=2))
        self.assertGreater(len(model.seeds), 70)
        self.assertEqual(len(model.seeds), len(model.marks))
        self.assertTrue(np.isfinite(model.reconstruction).all())
        self.assertTrue(np.isfinite(model.coeff).all())
        self.assertGreater(model.psnr, 18.0)
        for mode in (
                "Reconstruction + sites", "Reconstruction", "Error",
                "Cell classes", "Cells", "Recursive boundary field",
                "Collective residual", "Detail germination order"):
            self.assertEqual(model.view(mode).shape, model.rgb.shape)


if __name__ == "__main__":
    unittest.main()
