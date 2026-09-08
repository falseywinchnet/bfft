"""Invariants for continuous cross-validated transport stopping."""

from __future__ import annotations

import unittest

import numpy as np

from .cross_validated_transport_stopping import (
    _continuous_action_endpoint,
    _continuous_population_action_endpoint,
    _continuous_phase_balance,
    denoise_cross_validated_transport_chambolle,
)
from .transport_chambolle import TransportChambolleResolution


class CrossValidatedTransportStoppingTests(unittest.TestCase):
    def test_piecewise_quadratic_endpoint_is_continuous(self) -> None:
        source = np.zeros((2, 2), dtype=np.float64)
        estimates = [np.zeros_like(source), np.ones_like(source)]
        observers = [np.ones_like(source), np.zeros_like(source)]
        estimate, diagnostic = _continuous_action_endpoint(
            source,
            estimates,
            observers,
            np.ones_like(source),
            1.0,
        )
        self.assertAlmostEqual(
            diagnostic["continuous_observer_coordinate"], 0.5)
        np.testing.assert_allclose(estimate, 0.5)

    def test_phase_balance_interpolates_first_crossing(self) -> None:
        self.assertEqual(_continuous_phase_balance([-1.0, -2.0]), 0.0)
        self.assertAlmostEqual(
            _continuous_phase_balance([3.0, 1.0, -1.0]), 1.5)

    def test_complete_population_endpoint_retains_lane_variance(self) -> None:
        source = np.zeros((2, 2), dtype=np.float64)
        estimates = [np.zeros_like(source), np.ones_like(source)]
        populations = [
            np.ones((3, 2, 2), dtype=np.float64),
            np.zeros((3, 2, 2), dtype=np.float64),
        ]
        estimate, diagnostic = _continuous_population_action_endpoint(
            source,
            estimates,
            populations,
            np.ones_like(populations[0], dtype=bool),
            np.ones_like(source),
            1.0,
        )
        self.assertAlmostEqual(
            diagnostic["continuous_observer_coordinate"], 0.5)
        np.testing.assert_allclose(estimate, 0.5)

    def test_constant_image_is_exact_identity(self) -> None:
        image = np.full((12, 12), 0.375, dtype=np.float64)
        estimate, diagnostic = denoise_cross_validated_transport_chambolle(
            image,
            TransportChambolleResolution(
                maximum_iterations=24,
                minimum_iterations=2,
                maximum_observer_cycles=1,
            ),
        )
        np.testing.assert_allclose(estimate, image, atol=1.0e-13)
        self.assertLessEqual(diagnostic["recomposition_error"], 1.0e-15)
        self.assertEqual(diagnostic["observer_count"], 3.0)
        self.assertGreaterEqual(
            diagnostic["continuous_observer_coordinate"], 0.0)
        self.assertLessEqual(
            diagnostic["continuous_observer_coordinate"], 1.0)


if __name__ == "__main__":
    unittest.main()
