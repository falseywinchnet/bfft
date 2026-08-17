from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.certified_relaxation import (
    RelaxationConfig,
    _global_rate_weight,
    relax_rgb,
    solve_coefficients,
)


class CertifiedRelaxationTests(unittest.TestCase):
    def test_rate_only_matches_closed_form_unique_optimum(self):
        rng = np.random.default_rng(12)
        source = rng.normal(0.0, 8.0, size=(4, 64, 3))
        labels = np.arange(4, dtype=np.int32).reshape(2, 2)
        config = RelaxationConfig(
            rate_lambda=1.7,
            connection_lambda=0.0,
            frame_mode="identity",
            iterations=1200,
            relative_gap_tolerance=1e-9,
        )
        result, diagnostics = solve_coefficients(source, labels, config)
        threshold = config.rate_lambda * _global_rate_weight()[None]
        expected = np.sign(source) * np.maximum(np.abs(source) - threshold, 0.0)
        self.assertLess(float(np.max(np.abs(result - expected))), 2e-6)
        self.assertTrue(diagnostics["converged"])
        self.assertLess(float(diagnostics["relative_gap"]), 1e-9)
        self.assertLessEqual(
            float(diagnostics["dual_lower_bound"]),
            float(diagnostics["primal"]) + 1e-8,
        )

    def test_connection_problem_has_valid_certificate(self):
        rng = np.random.default_rng(13)
        source = rng.normal(0.0, 5.0, size=(6, 64, 3))
        labels = np.zeros((2, 3), dtype=np.int32)
        config = RelaxationConfig(
            rate_lambda=0.8,
            connection_lambda=1.2,
            frame_mode="chroma",
            iterations=2000,
            relative_gap_tolerance=2e-7,
        )
        _, diagnostics = solve_coefficients(source, labels, config)
        self.assertTrue(diagnostics["converged"])
        self.assertLess(float(diagnostics["relative_gap"]), 2e-7)
        self.assertLessEqual(
            float(diagnostics["connection_residual_after"]),
            float(diagnostics["connection_residual_before"]),
        )

    def test_rgb_relaxation_preserves_shape_and_finite_values(self):
        y, x = np.mgrid[:24, :32]
        rgb = np.stack(((x * 8) % 256, (y * 10) % 256, ((x + y) * 6) % 256), axis=-1)
        result = relax_rgb(
            rgb,
            RelaxationConfig(
                rate_lambda=0.3,
                connection_lambda=0.2,
                iterations=300,
                relative_gap_tolerance=1e-5,
            ),
        )
        self.assertEqual(result.rgb.shape, rgb.shape)
        self.assertTrue(np.all(np.isfinite(result.rgb)))
        self.assertGreaterEqual(result.primal + 1e-8, result.dual_lower_bound)


if __name__ == "__main__":
    unittest.main()
