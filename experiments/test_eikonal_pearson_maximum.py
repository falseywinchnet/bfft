from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.eikonal_pearson_maximum import eikonal_pearson_maximum


class EikonalPearsonMaximumTests(unittest.TestCase):
    @staticmethod
    def wave(angle_degrees: float, amplitude: float = 0.1) -> np.ndarray:
        y, x = np.indices((129, 129), dtype=np.float64)
        angle = math.radians(angle_degrees)
        coordinate = math.cos(angle) * x + math.sin(angle) * y
        return amplitude * np.sin(2.0 * math.pi * coordinate / 17.0)

    def test_identity_has_zero_maximum(self):
        truth = self.wave(31.0)
        result = eikonal_pearson_maximum(truth, truth)
        self.assertEqual(result.maximum_mean_square_error, 0.0)
        self.assertEqual(result.maximum_rms_error, 0.0)

    def test_diagonal_error_is_not_reduced_to_x_or_y(self):
        truth = np.zeros((129, 129), dtype=np.float64)
        result = eikonal_pearson_maximum(self.wave(37.5), truth)
        self.assertGreater(
            result.maximum_mean_square_error,
            max(result.xx_product_moment, result.yy_product_moment),
        )
        error = min(
            abs(result.principal_angle_degrees - 37.5),
            abs(result.principal_angle_degrees - 37.5 + 180.0),
            abs(result.principal_angle_degrees - 37.5 - 180.0),
        )
        self.assertLess(error, 0.5)

    def test_rotation_preserves_maximum_for_axis_and_diagonal_waves(self):
        truth = np.zeros((129, 129), dtype=np.float64)
        horizontal = eikonal_pearson_maximum(self.wave(0.0), truth)
        vertical = eikonal_pearson_maximum(self.wave(90.0), truth)
        self.assertAlmostEqual(
            horizontal.maximum_mean_square_error,
            vertical.maximum_mean_square_error,
            delta=2e-14,
        )

    def test_one_point_is_diluted_but_a_supported_error_is_not(self):
        truth = np.zeros((129, 129), dtype=np.float64)
        isolated = np.zeros_like(truth)
        isolated[64, 64] = 0.1
        supported = self.wave(37.5, amplitude=0.1)
        one = eikonal_pearson_maximum(isolated, truth)
        many = eikonal_pearson_maximum(supported, truth)
        self.assertGreater(
            many.maximum_mean_square_error,
            10.0 * one.maximum_mean_square_error,
        )


if __name__ == "__main__":
    unittest.main()
