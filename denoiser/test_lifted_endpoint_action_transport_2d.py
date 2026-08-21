"""Invariants for compact endpoint action competition."""

import unittest

import numpy as np

from .lifted_endpoint_action_transport_2d import (
    denoise_lifted_endpoint_action_transport_2d,
)


class LiftedEndpointActionTransport2DTests(unittest.TestCase):
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

    def test_constant_scene_remains_exact(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(image)
        self.assertLess(np.max(np.abs(estimate - image)), 2e-15)
        self.assertLess(diagnostic["observation_recomposition_error"], 2e-15)


if __name__ == "__main__":
    unittest.main()
