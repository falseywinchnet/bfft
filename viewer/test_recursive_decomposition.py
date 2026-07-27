#!/usr/bin/env python3
"""Regression tests for the recursive BFFT adaptation study."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recursive_decomposition import (  # noqa: E402
    RecursiveConfig, RecursiveDecomposition,
)


def synthetic():
    y, x = np.mgrid[0:48, 0:64]
    edge = (x > 30 + 5 * np.sin(y / 8)).astype(float)
    weave = 0.12 * np.sin(x * 1.7) * np.cos(y * 1.3)
    return np.clip(np.stack([
        0.2 + 0.55 * edge + weave,
        0.25 + 0.3 * x / 63 + weave,
        0.7 - 0.4 * edge - weave,
    ], axis=-1), 0.0, 1.0)


class RecursiveDecompositionTest(unittest.TestCase):
    def test_stack_identity_boundaries_and_cells(self):
        identity = RecursiveDecomposition(
            synthetic(), RecursiveConfig(
                max_side=64, iterations=3, passes=3, alpha=1.0,
                coarse_cells=24))
        np.testing.assert_allclose(
            identity.stages[-1], identity.rgb, atol=1e-12)

        model = RecursiveDecomposition(
            synthetic(), RecursiveConfig(
                max_side=64, iterations=4, passes=3, alpha=0.5,
                coarse_cells=32))
        self.assertEqual(len(model.stages), 5)
        self.assertEqual(len(model.boundaries), 4)
        self.assertEqual(len(model.seeds), 32)
        self.assertGreater(float(model.boundary_accumulation.max()), 0.0)
        self.assertGreater(float(np.mean(np.abs(
            model.stages[-1] - model.stages[0]))), 1e-3)
        self.assertTrue(np.isfinite(model.coarse_cells).all())
        for mode in (
                "Recursive image", "Removed shell", "Accumulated removal",
                "Stage boundary", "Boundary accumulation",
                "Collective carried residual",
                "Collective residual support",
                "Detail germination density",
                "Detail germination order",
                "Residual scale map",
                "Residual directional consistency",
                "Detail anisotropy preview",
                "Nucleation density", "Nucleation seeds",
                "Diffuse coarse cells", "Coarse cell regions"):
            self.assertEqual(model.view(mode, 2).shape, model.rgb.shape)


if __name__ == "__main__":
    unittest.main()
