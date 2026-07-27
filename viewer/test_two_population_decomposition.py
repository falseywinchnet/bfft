#!/usr/bin/env python3
"""Regression for coupled coarse/fine decomposition populations."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_population_decomposition import (  # noqa: E402
    TwoPopulationConfig, TwoPopulationDecomposition,
)


def synthetic():
    y, x = np.mgrid[0:48, 0:64]
    edge = (x > 31 + 5 * np.sin(y / 8)).astype(float)
    detail = 0.11 * np.sin(1.6 * x) * np.cos(1.2 * y)
    return np.clip(np.stack([
        0.16 + 0.62 * edge + detail,
        0.2 + 0.32 * x / 63,
        0.72 - 0.44 * edge - detail,
    ], axis=-1), 0.0, 1.0)


class TwoPopulationTest(unittest.TestCase):
    def test_shared_objective_descends(self):
        model = TwoPopulationDecomposition(
            synthetic(), TwoPopulationConfig(
                max_side=64, iterations=2, passes=3,
                coarse_cells=24, detail_cells=36, descent_steps=2,
                rounds=4, coarse_batch=5, fine_batch=9,
                coarse_radius=6.0, fine_radius=3.0))
        losses = np.asarray(model.losses)
        self.assertGreaterEqual(len(losses), 2)
        self.assertTrue(np.all(np.diff(losses) < 0.0))
        self.assertGreater(len(model.coarse_seeds), 24)
        self.assertGreater(len(model.fine_seeds), 0)
        self.assertEqual(len(model.reconstructions), len(losses))
        self.assertTrue(np.isfinite(model.reconstructions[-1]).all())
        for mode in (
                "Composition", "Coarse layer", "Fine layer",
                "Cartoon mismatch", "Texture mismatch", "Flow mismatch",
                "Recursive detail prior", "Births"):
            self.assertEqual(
                model.view(mode, len(losses) - 1).shape,
                model.rgb.shape)


if __name__ == "__main__":
    unittest.main()
