"""Falsification controls for the 1-D affine-relation simmer."""

from __future__ import annotations

import unittest

import numpy as np

from .affine_relation_transport import RelationResolution, denoise_affine_relations
from .sample_series import PRESETS, compose_series, corrupt


class AffineRelationTransportTests(unittest.TestCase):
    def test_affine_characteristic_is_exact(self):
        x = np.linspace(0.0, 1.0, 96)
        observation = 0.21 + 0.43 * x
        output, diagnostics = denoise_affine_relations(
            observation, RelationResolution(maximum_lag=12, scale_quadrature=7))
        np.testing.assert_allclose(output, observation, atol=2e-15, rtol=0.0)
        self.assertTrue(diagnostics["pointwise_j_invariant"])

    def test_constant_is_fixed(self):
        observation = np.full(64, 0.37)
        output, _ = denoise_affine_relations(observation)
        np.testing.assert_allclose(output, observation, atol=1e-15, rtol=0.0)

    def test_mixed_replacement_uniform_improves_stress_composite(self):
        truth = compose_series(256, PRESETS["mixed transport stress"])[1]
        observation = corrupt(
            truth, "mixed replacement + uniform",
            amount=0.15, density=0.25, seed=7)
        output, diagnostics = denoise_affine_relations(
            observation, RelationResolution(maximum_lag=16, scale_quadrature=8))
        observed_mse = float(np.mean((observation - truth) ** 2))
        output_mse = float(np.mean((output - truth) ** 2))
        self.assertLess(output_mse, 0.2 * observed_mse)
        self.assertIn("relation-horizon survival is not yet transported",
                      diagnostics["unresolved"])


if __name__ == "__main__":
    unittest.main()
