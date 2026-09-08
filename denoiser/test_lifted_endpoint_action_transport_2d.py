"""Invariants for compact endpoint action competition."""

import unittest

import numpy as np

from .lifted_endpoint_action_transport_2d import (
    _causal_temporal_action_fusion,
    _temporal_stopping_certificate,
    denoise_lifted_endpoint_action_transport_2d,
)


class LiftedEndpointActionTransport2DTests(unittest.TestCase):
    def test_stopping_certificate_is_dimensional_continuation(self):
        endpoint_variance = np.ones((8, 8))
        no_disagreement = np.zeros((4, 8, 8))
        continuation, _diagnostic = _temporal_stopping_certificate(
            endpoint_variance, no_disagreement)
        np.testing.assert_allclose(continuation, 1.0, rtol=0.0, atol=0.0)

        disagreement = np.ones((4, 8, 8))
        balanced, _diagnostic = _temporal_stopping_certificate(
            endpoint_variance, disagreement)
        np.testing.assert_allclose(balanced, 0.5, rtol=0.0, atol=0.0)

        stopped, _diagnostic = _temporal_stopping_certificate(
            np.zeros((8, 8)), disagreement)
        np.testing.assert_allclose(stopped, 0.0, rtol=0.0, atol=0.0)

    def test_temporal_fusion_rejects_unsupported_and_explained_innovation(self):
        first = np.zeros((4, 8, 8))
        first[0] = 1.0
        unsupported = np.zeros_like(first)
        unsupported[1] = 1.0
        zeros = np.zeros((8, 8))
        ones = np.ones((8, 8))
        fused, variance, diagnostic = _causal_temporal_action_fusion(
            first, unsupported, zeros, 0.0, zeros)
        np.testing.assert_allclose(fused, first, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(variance >= 0.0))
        self.assertEqual(diagnostic["mean_second_action_authority"], 0.0)

        explained, _variance, explained_diagnostic = (
            _causal_temporal_action_fusion(
                first, first, ones, 1.0, ones))
        np.testing.assert_allclose(explained, first, rtol=0.0, atol=0.0)
        self.assertEqual(
            explained_diagnostic["mean_second_action_authority"], 0.0)

        admitted, _variance, admitted_diagnostic = (
            _causal_temporal_action_fusion(
                first, first, zeros, 0.0, zeros))
        np.testing.assert_allclose(admitted, 2.0 * first, rtol=0.0, atol=0.0)
        self.assertEqual(
            admitted_diagnostic["maximum_second_action_authority"], 1.0)

    def test_endpoint_estimator_is_bounded_and_conservative(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.1 * np.sin(0.7 * xx) + 0.04 * np.cos(0.5 * yy)
        estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(image)
        self.assertTrue(np.all(np.isfinite(estimate)))
        self.assertTrue(np.all(diagnostic["fine_endpoint"] >= 0.0))
        self.assertTrue(np.all(diagnostic["fine_endpoint"] <= 1.0))
        self.assertTrue(np.all(diagnostic["coarse_endpoint"] >= 0.0))
        self.assertTrue(np.all(diagnostic["coarse_endpoint"] <= 1.0))
        self.assertLess(diagnostic["observation_recomposition_error"], 2e-15)

    def test_second_cycle_retains_four_positive_actions_and_variance(self):
        yy, xx = np.mgrid[:8, :8]
        image = (
            0.45 + 0.12 * np.sin(0.8 * xx)
            + 0.07 * np.cos(0.6 * yy)
        )
        _estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(
            image)
        action = diagnostic["retained_raw_endpoint_action"]
        variance = diagnostic["between_cycle_action_variance"]
        self.assertEqual(action.shape, (4,) + image.shape)
        self.assertEqual(variance.shape, (4,) + image.shape)
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(action >= 0.0))
        self.assertTrue(np.all(np.isfinite(variance)))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertLess(
            diagnostic["endpoint_action_evidence"]["first_cycle"]
            ["carried_action_mass_conservation_error"],
            2e-14,
        )
        self.assertEqual(diagnostic["two_cycle_persistent_dimension"], 22)

    def test_both_conservative_exchange_cycles_recompose_observation(self):
        yy, xx = np.mgrid[:8, :8]
        clean = 0.4 + 0.1 * np.sin(0.7 * xx) + 0.04 * np.cos(0.5 * yy)
        noise = 0.06 * np.sin(2.1 * xx + 1.7 * yy)
        image = clean + noise
        _estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(
            image)
        self.assertLess(diagnostic["provisional_recomposition_error"], 2e-15)
        self.assertLess(
            diagnostic["first_cycle_observation_recomposition_error"], 2e-15)
        self.assertLess(diagnostic["observation_recomposition_error"], 2e-15)

    def test_constant_scene_remains_exact(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(image)
        self.assertLess(np.max(np.abs(estimate - image)), 2e-15)
        self.assertLess(diagnostic["observation_recomposition_error"], 2e-15)


if __name__ == "__main__":
    unittest.main()
