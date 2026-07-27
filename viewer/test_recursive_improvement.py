#!/usr/bin/env python3
"""Regression for decomposition-space recursive improvement."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recursive_improvement import (  # noqa: E402
    ImprovementConfig, RecursiveImprovement,
)


def synthetic():
    y, x = np.mgrid[0:48, 0:64]
    body = np.exp(-((x - 34) ** 2 / 260 + (y - 25) ** 2 / 150))
    detail = 0.1 * np.sin(1.8 * x) * np.cos(1.35 * y)
    return np.clip(np.stack([
        0.15 + 0.65 * body + detail,
        0.18 + 0.45 * body,
        0.65 - 0.4 * body - detail,
    ], axis=-1), 0.0, 1.0)


class RecursiveImprovementTest(unittest.TestCase):
    def test_decomposition_loss_descends(self):
        model = RecursiveImprovement(
            synthetic(), ImprovementConfig(
                max_side=64, iterations=2, passes=3,
                coarse_cells=28, detail_cells=36, descent_steps=2,
                rounds=4, spawn_batch=8, atom_radius=3.5))
        losses = np.asarray(model.losses)
        self.assertGreaterEqual(len(losses), 2)
        self.assertTrue(np.all(np.diff(losses) < 0.0))
        self.assertEqual(len(model.reconstructions), len(losses))
        self.assertEqual(len(model.error_fields), len(losses))
        self.assertTrue(np.isfinite(model.reconstructions[-1]).all())
        self.assertGreater(len(model.all_seeds), 0)
        for mode in (
                "Reconstruction", "Decomposition error",
                "Cartoon gradient error", "Texture gradient error",
                "Residual gradient error", "Improvement",
                "Spawned atoms"):
            self.assertEqual(
                model.view(mode, len(losses) - 1).shape,
                model.rgb.shape)


if __name__ == "__main__":
    unittest.main()
