"""Invariants for the pure-averaging FABADA-order eikonal experiment."""

from __future__ import annotations

import unittest

import numpy as np

from .continual_eikonal_noise_transport_2d import ContinualEikonalResolution
from .continual_fabada_eikonal_2d import (
    _zero_noise_mixture,
    denoise_complete_moment_residual_posterior_2d,
    denoise_continual_fabada_eikonal_2d,
    denoise_continual_residual_posterior_2d,
)


FAST = ContinualEikonalResolution(maximum_iterations=12)


class ContinualFabadaEikonalTests(unittest.TestCase):
    def test_zero_noise_branch_uses_complete_mixture_variance(self):
        centre = np.array([[2.0, -1.0]])
        variance = np.array([[3.0, 0.5]])
        radius = np.array([[0.4, 0.2]])
        probability = np.array([[0.25, 0.75]])
        mean, mixed_variance, mixed_radius = _zero_noise_mixture(
            centre, variance, radius, probability)
        np.testing.assert_allclose(mean, probability * centre)
        np.testing.assert_allclose(
            mixed_variance,
            probability * variance
            + probability * (1.0 - probability) * centre * centre,
        )
        self.assertTrue(np.all(mixed_radius >= np.abs(mean)))

    def test_constant_is_exact_fixed_point(self):
        image = np.full((23, 27), 0.41)
        result, diagnostic = denoise_continual_fabada_eikonal_2d(image, FAST)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)

    def test_every_accepted_path_step_is_positive_conservative_descent(self):
        yy, xx = np.mgrid[-1:1:28j, -1:1:31j]
        truth = 0.24 + 0.52 * (xx > 0.08) + 0.07 * np.sin(7.0 * np.pi * yy)
        rng = np.random.default_rng(81)
        observed = np.clip(truth + rng.uniform(-0.2, 0.2, truth.shape), 0, 1)
        result, diagnostic = denoise_continual_fabada_eikonal_2d(observed, FAST)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreaterEqual(float(np.min(result)), float(np.min(observed)))
        self.assertLessEqual(float(np.max(result)), float(np.max(observed)))
        for record in diagnostic["iterations"]:
            self.assertLessEqual(
                record["path_dirichlet_after"],
                record["path_dirichlet_before"]
                + 1e-11 * max(record["path_dirichlet_before"], 1.0))
            self.assertGreaterEqual(record["averaging_minimum_diagonal"], -1e-14)
            self.assertLess(record["averaging_row_sum_error"], 1e-12)
            self.assertLess(record["averaging_column_sum_error"], 1e-12)
        for record in diagnostic["iterations"]:
            if record["accepted"]:
                self.assertLess(
                    record["noise_contractor_action_after"],
                    record["noise_contractor_action_before"])

    def test_trajectory_readout_is_not_screened_endpoint(self):
        image = np.zeros((25, 25))
        image[12, 12] = 1.0
        result, diagnostic = denoise_continual_fabada_eikonal_2d(image, FAST)
        self.assertGreaterEqual(diagnostic["evaluated_iterations"], 1)
        # Any accepted readout is an average of the identity endpoint and at
        # least one positive-smoothed endpoint, never a direct screened solve.
        if diagnostic["accepted_iterations"]:
            self.assertGreater(float(result[12, 12]), 0.0)
            self.assertLess(float(result[12, 12]), 1.0)

    def test_residual_posterior_constant_identity_and_mixture_invariants(self):
        constant = np.full((21, 24), 0.37)
        result, diagnostic = denoise_continual_residual_posterior_2d(
            constant, FAST)
        np.testing.assert_array_equal(result, constant)
        self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)
        for record in diagnostic["iterations"]:
            self.assertGreaterEqual(record["mean_noise_probability"], 0.0)
            self.assertLessEqual(record["mean_noise_probability"], 1.0)
            self.assertGreaterEqual(
                record["minimum_posterior_noise_variance"], 0.0)

    def test_residual_posterior_accepted_steps_contract(self):
        yy, xx = np.mgrid[-1:1:27j, -1:1:29j]
        truth = 0.25 + 0.45 * (xx > 0.1) + 0.08 * np.cos(5 * np.pi * yy)
        rng = np.random.default_rng(117)
        observed = np.clip(truth + rng.normal(0.0, 0.12, truth.shape), 0, 1)
        result, diagnostic = denoise_continual_residual_posterior_2d(
            observed, FAST)
        self.assertTrue(np.all(np.isfinite(result)))
        for record in diagnostic["iterations"]:
            self.assertLessEqual(
                record["path_dirichlet_after"],
                record["path_dirichlet_before"]
                + 1e-11 * max(record["path_dirichlet_before"], 1.0),
            )
            self.assertGreaterEqual(record["averaging_minimum_diagonal"], -1e-14)
            self.assertLess(record["averaging_row_sum_error"], 1e-12)
            self.assertLess(record["averaging_column_sum_error"], 1e-12)
            if record["accepted"]:
                self.assertLess(
                    record["noise_contractor_action_after"],
                    record["noise_contractor_action_before"],
                )

    def test_complete_moment_metric_is_identity_safe_and_dominates_central(self):
        constant = np.full((16, 18), 0.29)
        result, _diagnostic = denoise_complete_moment_residual_posterior_2d(
            constant, FAST)
        np.testing.assert_array_equal(result, constant)

        rng = np.random.default_rng(52)
        observed = rng.uniform(size=(16, 18))
        _central, central_diagnostic = denoise_continual_residual_posterior_2d(
            observed, FAST, metric_noise_moment="central")
        _complete, complete_diagnostic = denoise_continual_residual_posterior_2d(
            observed, FAST, metric_noise_moment="complete")
        for central, complete in zip(
            central_diagnostic["iterations"], complete_diagnostic["iterations"]
        ):
            self.assertGreaterEqual(
                complete["mean_metric_noise_second_moment"],
                central["mean_metric_noise_second_moment"],
            )


if __name__ == "__main__":
    unittest.main()
