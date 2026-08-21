"""Invariants for the continuous predictive Hopf--Lax atlas probe."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_tangent_atlas_2d import continuous_tangent_causal_atlas_2d


class CausalTangentAtlas2DTests(unittest.TestCase):
    def test_affine_chart_is_exact_across_population_phase(self):
        yy, xx = np.mgrid[:12, :15]
        field = 0.2 + 0.011 * xx + 0.014 * yy
        results = []
        for phase in (0.0, 0.25, 0.5, 0.75):
            result, diagnostic = continuous_tangent_causal_atlas_2d(
                field,
                angular_count=4,
                quantile_count=8,
                population_phase=phase,
            )
            np.testing.assert_allclose(result, field, atol=8e-13, rtol=0.0)
            self.assertEqual(
                diagnostic["observation_graph_maximum_error"], 0.0)
            self.assertFalse(diagnostic["population"]["safety_limit_hit"])
            results.append(result)
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], atol=8e-13, rtol=0.0)

    def test_constant_chart_has_one_horizontal_and_causal_support_unit(self):
        field = np.full((11, 13), 0.37)
        result, diagnostic = continuous_tangent_causal_atlas_2d(
            field,
            angular_count=4,
            quantile_count=8,
            population_phase=0.375,
        )
        np.testing.assert_allclose(result, field, atol=3e-15, rtol=0.0)
        self.assertAlmostEqual(
            diagnostic["horizontal_implied_support"], 1.0, places=12)
        self.assertAlmostEqual(
            diagnostic["causal_jet_implied_support"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
