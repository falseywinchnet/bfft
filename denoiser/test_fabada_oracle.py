"""Invariants for the repaired, oracle-noise PFABADA comparison."""

from __future__ import annotations

import unittest

import numpy as np

from .fabada_oracle import (
    denoise_oracle_fabada_from_corruption_1d,
    oracle_corruption_moments_1d,
    reflected_mean_operator_1d,
)
from .sample_series import PRESETS, compose_series, corrupt


class OracleFabadaTests(unittest.TestCase):
    def test_reflected_mean_is_symmetric_conservative_and_positive(self):
        operator = reflected_mean_operator_1d(32)
        np.testing.assert_allclose(operator, operator.T, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(operator, axis=0), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(operator, axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertTrue(np.all(operator >= 0.0))

    def test_zero_noise_is_exact_identity(self):
        truth = compose_series(64, PRESETS["mixed transport stress"])[1]
        forms, diagnostic = denoise_oracle_fabada_from_corruption_1d(
            truth, truth, "none", amount=0.0, density=0.0)
        np.testing.assert_allclose(forms["global"], truth, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(forms["local"], truth, atol=0.0, rtol=0.0)
        self.assertTrue(diagnostic["zero_noise_identity"])

    def test_replacement_moments_match_theoretical_affine_law(self):
        truth = np.linspace(0.05, 0.95, 40)
        probability = 0.25
        mean, variance, diagnostic = oracle_corruption_moments_1d(
            truth,
            "random-value replacement",
            amount=0.2,
            density=probability,
        )
        expected_mean = (1.0 - probability) * truth + 0.5 * probability
        expected_second = (
            (1.0 - probability) * truth * truth + probability / 3.0)
        np.testing.assert_allclose(mean, expected_mean, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            variance,
            expected_second - expected_mean * expected_mean,
            atol=2e-15,
            rtol=0.0,
        )
        self.assertEqual(diagnostic["observation_gain"], 0.75)
        self.assertEqual(diagnostic["observation_offset"], 0.125)

    def test_known_gaussian_risk_reduces_mixed_signal_error(self):
        truth = compose_series(96, PRESETS["mixed transport stress"])[1]
        observation = corrupt(
            truth,
            "Gaussian additive",
            amount=0.15,
            density=0.25,
            seed=20100,
        )
        forms, diagnostic = denoise_oracle_fabada_from_corruption_1d(
            observation,
            truth,
            "Gaussian additive",
            amount=0.15,
            density=0.25,
        )
        observed_mse = float(np.mean((observation - truth) ** 2))
        output_mse = float(np.mean((forms["global"] - truth) ** 2))
        self.assertLess(output_mse, observed_mse)
        self.assertEqual(diagnostic["candidate_count"], truth.size)
        self.assertTrue(diagnostic["oracle_noise_statistics"])

    def test_outputs_are_finite_for_sparse_oracle_laws(self):
        truth = compose_series(64, PRESETS["pulses + drift"])[1]
        for kind, density in (
            ("random-value replacement", 0.25),
            ("salt and pepper", 0.10),
            ("mixed replacement + uniform", 0.25),
        ):
            observation = corrupt(
                truth, kind, amount=0.15, density=density, seed=20100)
            forms, _ = denoise_oracle_fabada_from_corruption_1d(
                observation,
                truth,
                kind,
                amount=0.15,
                density=density,
            )
            self.assertTrue(np.all(np.isfinite(forms["global"])))
            self.assertTrue(np.all(np.isfinite(forms["local"])))
            self.assertTrue(np.all((forms["global"] >= 0.0)))
            self.assertTrue(np.all((forms["global"] <= 1.0)))

    def test_full_replacement_does_not_leak_hidden_truth(self):
        truth = compose_series(64, PRESETS["pulses + drift"])[1]
        observation = np.linspace(0.0, 1.0, truth.size)
        forms, diagnostic = denoise_oracle_fabada_from_corruption_1d(
            observation,
            truth,
            "random-value replacement",
            amount=0.0,
            density=1.0,
        )
        np.testing.assert_allclose(forms["global"], 0.5)
        np.testing.assert_allclose(forms["local"], 0.5)
        self.assertTrue(diagnostic["informationless_observation"])


if __name__ == "__main__":
    unittest.main()
