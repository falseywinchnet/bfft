from __future__ import annotations

import unittest

import numpy as np

from experiments.eikonal_path_oscillation import eikonal_path_oscillation


class EikonalPathOscillationTests(unittest.TestCase):
    def test_identity_has_no_oscillation(self):
        field = np.linspace(0.0, 1.0, 65)[None, :] * np.ones((65, 1))
        result = eikonal_path_oscillation(
            field, field, angles=np.array((0.0, 45.0, 90.0)), maximum_lag=6
        )
        self.assertEqual(result.maximum_negative_product_moment, 0.0)

    def test_oriented_ripple_is_found_at_its_tangent(self):
        y, x = np.indices((81, 81), dtype=np.float64)
        angle = 35.0
        radians = np.deg2rad(angle)
        tangent = np.cos(radians) * (x - 40.0) + np.sin(radians) * (y - 40.0)
        normal = -np.sin(radians) * (x - 40.0) + np.cos(radians) * (y - 40.0)
        ripple = (
            0.04 * np.sin(2.0 * np.pi * tangent / 6.0)
            * np.exp(-0.5 * (normal / 1.5) ** 2)
        )
        result = eikonal_path_oscillation(
            ripple, np.zeros_like(ripple),
            angles=np.array((0.0, angle, 90.0, (-angle) % 180.0)), maximum_lag=6,
        )
        self.assertGreater(result.maximum_negative_product_moment, 0.0)
        self.assertGreater(result.maximum_ringing_product_moment, 0.0)
        self.assertGreater(result.maximum_excess_backtracking, 0.0)
        self.assertEqual(result.orientation_degrees, (-angle) % 180.0)
        self.assertIn(result.lag, (2, 3, 4))

    def test_single_point_does_not_qualify_as_repeated_oscillation(self):
        impulse = np.zeros((65, 65), dtype=np.float64)
        impulse[32, 32] = 1.0
        result = eikonal_path_oscillation(
            impulse, np.zeros_like(impulse),
            angles=np.array((0.0, 45.0, 90.0)), maximum_lag=6,
        )
        self.assertEqual(result.maximum_negative_product_moment, 0.0)
        self.assertEqual(result.maximum_ringing_product_moment, 0.0)


if __name__ == "__main__":
    unittest.main()
