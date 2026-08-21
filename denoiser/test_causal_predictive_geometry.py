"""Invariants for ancestry-transported predictive value laws."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_predictive_geometry import (
    causal_predictive_fixed_point,
    causal_parity_predictive_geometry,
    weighted_quantile_particles,
)


class CausalPredictiveGeometryTests(unittest.TestCase):
    def test_weighted_quantiles_resolve_atomic_law(self):
        weights = np.array(((((0.25, 0.75),),)))
        particles = weighted_quantile_particles(weights, (0.1, 0.9), 4)
        np.testing.assert_allclose(particles.ravel(), (0.1, 0.9, 0.9, 0.9))

    def test_particle_order_is_value_order_not_root_order(self):
        weights = np.array(((((0.25, 0.75),),)))
        first = weighted_quantile_particles(weights, (0.1, 0.9), 8)
        second = weighted_quantile_particles(
            weights[..., ::-1], (0.9, 0.1), 8)
        np.testing.assert_array_equal(first, second)

    def test_constant_law_has_one_support_unit(self):
        particles, diagnostic = causal_parity_predictive_geometry(
            np.full((10, 12), 0.37), quantile_count=8)
        np.testing.assert_allclose(particles, 0.37)
        geometry = diagnostic["predictive_geometry"]
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=13)
        np.testing.assert_allclose(geometry["metric_determinant"], 1.0)

    def test_constant_fixed_point_reaches_equilibrium(self):
        particles, diagnostic = causal_predictive_fixed_point(
            np.full((10, 12), 0.37),
            quantile_count=8,
            maximum_continuations=4,
        )
        np.testing.assert_allclose(particles, 0.37)
        self.assertTrue(diagnostic["equilibrium"])
        self.assertFalse(diagnostic["continuation_ceiling_hit"])

    def test_fixed_point_accepts_only_self_consistency_descent(self):
        yy, xx = np.mgrid[:10, :12]
        field = 0.4 + 0.15 * np.sin(xx / 2.0) + 0.05 * (yy > 5)
        _particles, diagnostic = causal_predictive_fixed_point(
            field,
            quantile_count=8,
            maximum_continuations=4,
        )
        actions = [
            record["self_consistency_action"]
            for record in diagnostic["continuations"]
        ]
        self.assertTrue(all(
            after < before for before, after in zip(actions, actions[1:])))


if __name__ == "__main__":
    unittest.main()
