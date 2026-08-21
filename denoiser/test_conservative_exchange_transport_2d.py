"""Invariants for the conservative posterior/residual exchange orbit."""

import unittest

import numpy as np

from .conservative_exchange_transport_2d import (
    conservative_exchange_cycle_2d,
    denoise_conservative_exchange_transport_2d,
)


class ConservativeExchangeTransport2DTests(unittest.TestCase):
    def test_every_substep_preserves_observation_pointwise(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.12 * np.sin(0.8 * xx) + 0.07 * np.cos(0.5 * yy)
        posterior = 0.4 + 0.08 * np.sin(0.8 * xx)
        residual = image - posterior
        next_posterior, next_residual, diagnostic = (
            conservative_exchange_cycle_2d(image, posterior, residual))
        for name in (
            "initial_conservation_error",
            "posterior_transfer_conservation_error",
            "residual_transfer_conservation_error",
            "joint_transfer_conservation_error",
        ):
            self.assertLess(diagnostic[name], 2e-15)
        np.testing.assert_allclose(
            next_posterior + next_residual, image, atol=2e-15, rtol=0.0)
        self.assertLessEqual(
            diagnostic["residual_donation_action"],
            diagnostic["smoothed_residual_action"] + 2e-15,
        )
        self.assertLessEqual(
            diagnostic["joint_reassignment_action"],
            diagnostic["joint_candidate_action"] + 2e-15,
        )

    def test_constant_scene_is_an_exact_fixed_point(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_conservative_exchange_transport_2d(
            image, numerical_cycle_ceiling=3)
        np.testing.assert_allclose(estimate, image, atol=2e-14, rtol=0.0)
        self.assertTrue(diagnostic["equilibrium"])
        self.assertEqual(diagnostic["completed_cycles"], 1)

    def test_trajectory_retains_exact_signed_residual(self):
        rng = np.random.default_rng(841)
        image = rng.uniform(0.0, 1.0, size=(8, 8))
        estimate, diagnostic = denoise_conservative_exchange_transport_2d(
            image, numerical_cycle_ceiling=2)
        posterior = diagnostic["posterior_trajectory"]
        residual = diagnostic["residual_trajectory"]
        self.assertEqual(posterior.shape[0], 3)
        np.testing.assert_allclose(
            posterior + residual,
            np.broadcast_to(image, posterior.shape),
            atol=2e-15,
            rtol=0.0,
        )
        np.testing.assert_array_equal(estimate, posterior[-1])

    def test_rejects_a_nonconservative_input_decomposition(self):
        image = np.zeros((8, 8))
        with self.assertRaises(ValueError):
            conservative_exchange_cycle_2d(
                image,
                np.ones_like(image),
                np.ones_like(image),
            )


if __name__ == "__main__":
    unittest.main()
