"""Invariants for continual eikonal radiance/noise/statistic transport."""

from __future__ import annotations

import unittest

import numpy as np

from .continual_eikonal_noise_transport_2d import (
    ContinualEikonalResolution,
    _mixture_moment_fusion,
    continual_anisotropic_noise_metric,
    continual_transport_metric,
    denoise_continual_eikonal_noise_transport_2d,
    directional_noise_witnesses,
    eikonal_noise_witnesses,
    phase_covector_noise_authority,
    phase_covector_sufficient_statistics,
)


FAST = ContinualEikonalResolution(maximum_iterations=12)


class ContinualEikonalNoiseTransportTests(unittest.TestCase):
    def test_constant_is_exact_fixed_point(self):
        image = np.full((24, 29), 0.37)
        result, diagnostic = denoise_continual_eikonal_noise_transport_2d(
            image, FAST)
        np.testing.assert_array_equal(result, image)
        self.assertLessEqual(diagnostic["accepted_iterations"], 2)
        self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)

    def test_directional_witnesses_exclude_target(self):
        image = np.zeros((20, 20))
        image[10, 10] = 1.0
        centre, variance, radius, _ = directional_noise_witnesses(image, image)
        self.assertEqual(centre[10, 10], 1.0)
        self.assertEqual(variance[10, 10], 0.0)
        self.assertEqual(radius[10, 10], 0.0)

    def test_metric_is_positive_definite(self):
        rng = np.random.default_rng(71)
        image = rng.random((21, 27))
        variance = 0.01 * rng.random(image.shape)
        metric = continual_transport_metric(image, variance)
        determinant = (
            metric["metric_xx"] * metric["metric_yy"]
            - metric["metric_xy"] * metric["metric_xy"]
        )
        self.assertGreater(float(np.min(metric["metric_xx"])), 0.0)
        self.assertGreater(float(np.min(determinant)), 0.0)

    def test_anisotropic_noise_metric_is_positive_definite(self):
        rng = np.random.default_rng(72)
        image = rng.random((21, 27))
        centre = rng.normal(0.0, 0.1, image.shape)
        variance = 0.01 * rng.random(image.shape)
        authority = rng.random(image.shape)
        metric = continual_anisotropic_noise_metric(
            image, centre, variance, authority)
        determinant = (
            metric["metric_xx"] * metric["metric_yy"]
            - metric["metric_xy"] * metric["metric_xy"])
        self.assertGreater(float(np.min(metric["metric_xx"])), 0.0)
        self.assertGreater(float(np.min(determinant)), 0.0)

    def test_eikonal_witness_is_finite_bounded_law(self):
        rng = np.random.default_rng(81)
        image = rng.random((19, 23))
        _c, variance, _r, _ = directional_noise_witnesses(image, image)
        metric = continual_transport_metric(image, variance)
        centre, variance, radius, diagnostic = eikonal_noise_witnesses(
            image, image, metric)
        self.assertTrue(np.all(np.isfinite(centre)))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertTrue(np.all(radius >= 0.0))
        self.assertGreaterEqual(diagnostic["selling_reach_p90"], 1.0)

    def test_mixture_fusion_does_not_invent_precision(self):
        centre = np.full((8, 9), 0.2)
        variance = np.full((8, 9), 0.07)
        radius = np.full((8, 9), 0.4)
        fused = _mixture_moment_fusion(
            centre, variance, radius, centre, variance, radius, 0.37)
        np.testing.assert_allclose(fused[0], centre, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(fused[1], variance, atol=1e-16, rtol=0.0)
        np.testing.assert_allclose(fused[2], radius, atol=1e-16, rtol=0.0)

    def test_exact_plane_has_integrable_phase_covector(self):
        yy, xx = np.mgrid[:24, :28]
        plane = 0.2 + 0.01 * xx + 0.015 * yy
        numerator, denominator = phase_covector_sufficient_statistics(plane)
        authority, diagnostic = phase_covector_noise_authority(
            numerator[:, 2:-2, 2:-2], denominator[:, 2:-2, 2:-2])
        self.assertLess(float(np.max(authority)), 1e-10)
        self.assertLess(diagnostic["mean_phase_covector_defect"], 1e-20)

    def test_isotropic_impulses_are_not_integrable_wave_phase(self):
        image = np.zeros((25, 25))
        image[4::5, 3::6] = 1.0
        numerator, denominator = phase_covector_sufficient_statistics(image)
        authority, _ = phase_covector_noise_authority(numerator, denominator)
        active = np.sum(denominator, axis=0) > 0.0
        self.assertGreater(float(np.mean(authority[active])), 0.25)

    def test_every_iteration_descends_its_frozen_action(self):
        yy, xx = np.mgrid[-1:1:28j, -1:1:32j]
        truth = 0.25 + 0.5 * (xx > 0.05) + 0.08 * np.sin(5.0 * np.pi * yy)
        rng = np.random.default_rng(13)
        observed = np.clip(truth + rng.uniform(-0.18, 0.18, truth.shape), 0, 1)
        result, diagnostic = denoise_continual_eikonal_noise_transport_2d(
            observed, FAST)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreaterEqual(float(np.min(result)), float(np.min(observed)))
        self.assertLessEqual(float(np.max(result)), float(np.max(observed)))
        self.assertGreater(len(diagnostic["iterations"]), 0)
        for record in diagnostic["iterations"]:
            self.assertLessEqual(
                record["frozen_action_after"],
                record["frozen_action_before"]
                + 1e-11 * max(record["frozen_action_before"], 1.0),
            )
            self.assertLess(record["laplacian_row_sum_error"], 1e-12)
            self.assertLess(record["transport_row_sum_error"], 1e-12)
            self.assertLess(record["transport_column_sum_error"], 1e-12)
            self.assertGreaterEqual(record["selling_minimum_coefficient"], 0.0)
        accepted = [
            record for record in diagnostic["iterations"]
            if record["accepted"]
        ]
        for record in accepted:
            self.assertLess(
                record["noise_contractor_action_after"],
                record["noise_contractor_action_before"],
            )
        self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
