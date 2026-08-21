"""Invariants for causal joint-information branch transport."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_information_lineage_2d import (
    _determinant_one_precision,
    causal_information_lineage_law_2d,
    causal_information_lineage_readouts_2d,
    causal_information_phase_integrated_law_2d,
    causal_information_phase_integrated_readouts_2d,
    causal_information_phase_refinement_readouts_2d,
    nested_population_phases,
)


class CausalInformationLineage2DTests(unittest.TestCase):
    def test_complete_residual_moment_retains_zero_noise_reference(self):
        coordinates = np.array([
            [0.2, 0.0, 0.0, 0.4],
            [0.3, 0.1, 0.0, 0.6],
            [0.4, 0.0, 0.1, 0.5],
            [0.5, 0.1, 0.1, 0.7],
        ])
        weights = np.full(4, 0.25)
        central, _ = _determinant_one_precision(
            coordinates, weights, 1e-12)
        complete, _ = _determinant_one_precision(
            coordinates, weights, 1e-12, complete_residual_moment=True)
        self.assertAlmostEqual(float(np.linalg.det(central)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.det(complete)), 1.0, places=5)
        self.assertFalse(np.allclose(central, complete))

    def test_constant_is_exact_for_every_readout(self):
        image = np.full((8, 8), 0.37)
        forms, diagnostic = causal_information_lineage_readouts_2d(
            image, angular_count=4, quantile_count=8)
        for value in forms.values():
            np.testing.assert_allclose(value, image, atol=2e-15, rtol=0.0)
        self.assertLess(diagnostic["mass_maximum_error"], 2e-15)
        refinement = causal_information_phase_refinement_readouts_2d(
            image, angular_count=4, quantile_count=8, phase_counts=(2, 4))
        for count in (2, 4):
            for value in refinement[count][0].values():
                np.testing.assert_allclose(value, image, atol=2e-15, rtol=0.0)
        self.assertAlmostEqual(
            diagnostic["causal_implied_support"],
            diagnostic["initial_implied_support"], places=14)
        self.assertLess(diagnostic["causal_measure_relative_rms"], 2e-15)

    def test_causal_branch_mass_is_conserved(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.2 + 0.4 * xx / 7.0 + 0.1 * np.sin(yy)
        law, diagnostic = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=-1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["root_mass"], axis=(-2, -1)),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["root_mass"], axis=-2),
            law["mass"], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_collision_mass"], axis=-1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_ancestry_collision_mass"], axis=-1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_simplex_collision_mass"], axis=-1),
            1.0, atol=2e-15, rtol=0.0)
        self.assertTrue(np.all(np.isfinite(law["hj_path_score"])))
        self.assertGreaterEqual(
            float(np.min(law["hj_simplex_collision_order"])), 1.0)
        self.assertLessEqual(
            float(np.max(law["hj_simplex_collision_order"])), 3.0)
        self.assertTrue(np.all(np.isfinite(
            law["causal_hj_collision_section"])))
        self.assertTrue(np.all(np.isfinite(
            law["causal_hj_collision_barycenter"])))
        self.assertTrue(np.all(np.isfinite(
            law["causal_hj_collision_w1_barycenter"])))
        self.assertGreaterEqual(
            diagnostic["minimum_hj_ancestry_collision_order"], 2.0 - 1e-14)
        self.assertGreaterEqual(
            diagnostic["minimum_hj_simplex_collision_order"], 1.0)
        self.assertLessEqual(
            diagnostic["maximum_hj_simplex_collision_order"], 3.0)
        self.assertGreaterEqual(diagnostic["mean_hj_collision_score_gap"], 0.0)
        self.assertGreaterEqual(diagnostic["continuous_root_count"], 1)
        self.assertTrue(diagnostic["joint_population"]["target_identity_excluded"])

    def test_root_resolved_readout_is_finite_before_lineages_meet(self):
        image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
        forms, diagnostic = causal_information_lineage_readouts_2d(
            image, angular_count=4, quantile_count=8)
        self.assertTrue(np.all(np.isfinite(
            forms["causal_cross_lineage_median"])))
        self.assertLess(diagnostic["marginal_maximum_error"], 2e-15)

    def test_phase_quadrature_is_nested_and_constant_exact(self):
        self.assertEqual(
            nested_population_phases(8),
            (0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875),
        )
        image = np.full((8, 8), 0.29)
        law, diagnostic = causal_information_phase_integrated_law_2d(
            image, angular_count=4, quantile_count=8, phase_count=4)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=-1), 1.0, atol=2e-15, rtol=0.0)
        forms, _ = causal_information_phase_integrated_readouts_2d(
            image, angular_count=4, quantile_count=8, phase_count=4)
        for value in forms.values():
            np.testing.assert_allclose(value, image, atol=2e-15, rtol=0.0)
        self.assertLess(diagnostic["mass_maximum_error"], 2e-15)


if __name__ == "__main__":
    unittest.main()
