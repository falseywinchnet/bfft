"""Invariants for direct support-derived compressed observation."""

from __future__ import annotations

import unittest

import numpy as np

from .compressed_eikonal_observer_2d import (
    _direct_affine_projection,
    _transport_cross_gain,
    _transport_signed_phase,
    compressed_eikonal_observation_2d,
    cross_measured_eikonal_observation_2d,
    interlaced_scene_views_2d,
    phase_ordered_cross_observation_2d,
    phase_resolved_eikonal_observation_2d,
    phase_union_eikonal_observation_2d,
    pursue_compressed_eikonal_scene_2d,
    screened_selling_posterior_observation_2d,
)


class CompressedEikonalObserver2DTests(unittest.TestCase):
    def test_direct_projection_is_orthogonal_and_exactly_bookkept(self):
        rng = np.random.default_rng(7)
        field = rng.normal(size=(8, 8))
        labels = np.repeat(np.arange(4, dtype=np.int32), 16).reshape(8, 8)
        explanation, diagnostic = _direct_affine_projection(field, labels)
        residual = field - explanation
        self.assertLess(diagnostic["maximum_normal_error"], 2e-13)
        np.testing.assert_allclose(
            explanation + residual, field, atol=3e-16, rtol=0.0)

    def test_affine_scene_is_fully_measured_on_any_cell_partition(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.03 * xx - 0.017 * yy
        labels = np.repeat(np.arange(4, dtype=np.int32), 16).reshape(8, 8)
        explanation, diagnostic = _direct_affine_projection(field, labels)
        np.testing.assert_allclose(explanation, field, atol=2e-15, rtol=0.0)
        self.assertLess(diagnostic["maximum_normal_error"], 2e-14)

    def test_constant_scene_is_exactly_observable(self):
        field = np.full((8, 8), 0.37)
        explanation, residual, diagnostic = (
            compressed_eikonal_observation_2d(field))
        np.testing.assert_array_equal(explanation, field)
        np.testing.assert_array_equal(residual, 0.0)
        self.assertEqual(diagnostic["sensor_rank"], 1)

    def test_interlaced_views_have_disjoint_observation_sources(self):
        rng = np.random.default_rng(31)
        field = rng.normal(size=(8, 8))
        yy, xx = np.mgrid[:8, :8]
        first, second = interlaced_scene_views_2d(field)
        changed = field.copy()
        changed[((xx + yy) & 1) == 1] += rng.normal(
            size=np.count_nonzero(((xx + yy) & 1) == 1))
        changed_first, changed_second = interlaced_scene_views_2d(changed)
        np.testing.assert_array_equal(changed_first, first)
        self.assertFalse(np.array_equal(changed_second, second))

    def test_interlaced_views_preserve_constants(self):
        field = np.full((8, 8), 0.41)
        for covector in ((1, 0), (0, 1), (1, 1)):
            first, second = interlaced_scene_views_2d(
                field, parity_covector=covector)
            np.testing.assert_array_equal(first, field)
            np.testing.assert_array_equal(second, field)

    def test_complete_parity_covectors_each_have_disjoint_sources(self):
        rng = np.random.default_rng(37)
        field = rng.normal(size=(8, 8))
        yy, xx = np.mgrid[:8, :8]
        for cy, cx in ((1, 0), (0, 1), (1, 1)):
            first, second = interlaced_scene_views_2d(
                field, parity_covector=(cy, cx))
            changed = field.copy()
            second_owner = ((cy * yy + cx * xx) & 1) == 1
            changed[second_owner] += rng.normal(
                size=np.count_nonzero(second_owner))
            changed_first, changed_second = interlaced_scene_views_2d(
                changed, parity_covector=(cy, cx))
            np.testing.assert_array_equal(changed_first, first)
            self.assertFalse(np.array_equal(changed_second, second))

    def test_cross_measure_retains_exact_residual_bookkeeping(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.03 * xx - 0.017 * yy
        prior, residual, diagnostic = cross_measured_eikonal_observation_2d(
            field)
        np.testing.assert_allclose(
            prior + residual, field, atol=3e-16, rtol=0.0)
        self.assertLess(
            diagnostic["exact_bookkeeping_maximum_error"], 3e-16)
        self.assertEqual(len(diagnostic["charts"]), 2)

    def test_screened_cross_gain_obeys_positive_maximum_principle(self):
        labels = np.asarray((
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (2, 2, 3, 3),
            (2, 2, 3, 3),
        ), dtype=np.int32)
        measured = np.asarray((2.0, 3.0, 4.0, 5.0))
        cross = np.asarray((-1.0, 1.5, 4.0, 1.0))
        gain, diagnostic = _transport_cross_gain(
            labels, cross, measured)
        self.assertGreaterEqual(float(np.min(gain)), 0.0)
        self.assertLessEqual(float(np.max(gain)), 1.0)
        self.assertLess(
            diagnostic["screened_transport_residual_maximum"], 2e-15)

    def test_signed_phase_transport_obeys_two_sided_maximum_principle(self):
        labels = np.asarray((
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (2, 2, 3, 3),
            (2, 2, 3, 3),
        ), dtype=np.int32)
        total = np.asarray((2.0, 3.0, 4.0, 5.0))
        cross = np.asarray((-2.0, 1.5, 4.0, -1.0))
        phase, diagnostic = _transport_signed_phase(
            labels, cross, total)
        self.assertGreaterEqual(float(np.min(phase)), -1.0)
        self.assertLessEqual(float(np.max(phase)), 1.0)
        self.assertLess(
            diagnostic["signed_screened_transport_residual_maximum"], 2e-15)

    def test_phase_resolved_prior_is_exactly_bookkept(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.03 * xx - 0.017 * yy
        prior, residual, diagnostic = phase_resolved_eikonal_observation_2d(
            field)
        np.testing.assert_allclose(
            prior + residual, field, atol=3e-16, rtol=0.0)
        self.assertEqual(len(diagnostic["charts"]), 2)
        self.assertGreaterEqual(diagnostic["mean_phase_certainty"], 0.0)
        self.assertLessEqual(diagnostic["mean_phase_certainty"], 1.0)

    def test_phase_resolved_constant_is_exact(self):
        field = np.full((8, 8), 0.41)
        prior, residual, diagnostic = phase_resolved_eikonal_observation_2d(
            field)
        np.testing.assert_array_equal(prior, field)
        np.testing.assert_array_equal(residual, 0.0)
        self.assertEqual(diagnostic["theory_status"], "exact constant fixed point")

    def test_phase_ordered_cross_prior_is_a_contraction(self):
        rng = np.random.default_rng(113)
        field = rng.normal(size=(8, 8))
        prior, residual, diagnostic = phase_ordered_cross_observation_2d(field)
        np.testing.assert_allclose(
            prior + residual, field, atol=3e-16, rtol=0.0)
        self.assertGreaterEqual(diagnostic["phase_order"], 0.0)
        self.assertLessEqual(diagnostic["phase_order"], 1.0)
        cross = diagnostic["readouts"]["transported_cross_prior"]
        mean = diagnostic["common_mean"]
        self.assertLessEqual(
            np.linalg.norm(prior - mean),
            np.linalg.norm(cross - mean) + 1e-15,
        )

    def test_phase_union_is_smooth_and_exactly_bookkept(self):
        rng = np.random.default_rng(119)
        field = rng.normal(size=(8, 8))
        prior, residual, diagnostic = phase_union_eikonal_observation_2d(field)
        np.testing.assert_allclose(
            prior + residual, field, atol=3e-16, rtol=0.0)
        self.assertEqual(
            diagnostic["covectors"], ((1, 0), (0, 1), (1, 1)))
        self.assertGreaterEqual(diagnostic["phase_union_order"], 0.0)
        self.assertLessEqual(diagnostic["phase_union_order"], 1.0)
        self.assertAlmostEqual(
            sum(diagnostic["covector_barycentric_mass"]), 1.0)

    def test_screened_selling_posterior_is_exactly_bookkept_and_contracting(self):
        rng = np.random.default_rng(127)
        field = rng.normal(size=(8, 8))
        posterior, residual, diagnostic = (
            screened_selling_posterior_observation_2d(field))
        np.testing.assert_allclose(
            posterior + residual, field, atol=3e-16, rtol=0.0)
        self.assertLessEqual(diagnostic["resolvent_contraction_ratio"], 1.0)
        self.assertLess(
            diagnostic["screened_system_residual_maximum"], 2e-15)
        self.assertGreaterEqual(diagnostic["dirichlet_action"], -1e-15)

    def test_screened_selling_posterior_preserves_constant(self):
        field = np.full((8, 8), 0.37)
        posterior, residual, diagnostic = (
            screened_selling_posterior_observation_2d(field))
        np.testing.assert_array_equal(posterior, field)
        np.testing.assert_array_equal(residual, 0.0)
        self.assertEqual(diagnostic["phase_union_order"], 1.0)

    def test_pursuit_retains_exact_scene_residual_bookkeeping(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.03 * xx - 0.017 * yy
        explanation, diagnostic = pursue_compressed_eikonal_scene_2d(
            field, maximum_observations=2)
        residual = diagnostic["unexplained_scene"]
        np.testing.assert_allclose(
            explanation + residual, field, atol=3e-15, rtol=0.0)
        self.assertLess(
            diagnostic["exact_bookkeeping_maximum_error"], 3e-15)


if __name__ == "__main__":
    unittest.main()
