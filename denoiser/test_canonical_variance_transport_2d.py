"""Invariants for observation-law coordinate transport."""

import unittest

import numpy as np

from .canonical_variance_transport_2d import (
    crossfit_variance_law,
    denoise_canonical_variance_transport_2d,
    variance_coordinate,
)


class CanonicalVarianceTransportTests(unittest.TestCase):
    def test_variance_law_is_nonnegative(self):
        image = np.arange(100, dtype=np.float64).reshape(10, 10) / 99.0
        coefficients, diagnostic = crossfit_variance_law(image)
        self.assertTrue(np.all(coefficients >= 0.0))
        self.assertGreater(diagnostic["minimum_fitted_variance"], 0.0)

    def test_coordinates_are_monotone_and_roundtrip(self):
        image = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        coefficients = np.array([0.01, 0.05, 0.02])
        for coordinate in ("fisher", "canonical"):
            mapped, knots, transformed, diagnostic = variance_coordinate(
                image, coefficients, coordinate)
            self.assertTrue(np.all(np.diff(transformed) > 0.0))
            np.testing.assert_allclose(
                np.interp(mapped, transformed, knots), image, atol=1e-14)
            self.assertLess(diagnostic["maximum_roundtrip_error"], 1e-14)

    def test_constant_is_exact_fixed_point(self):
        image = np.full((10, 12), 0.37)
        for coordinate in ("fisher", "canonical"):
            estimate, diagnostic = denoise_canonical_variance_transport_2d(
                image, coordinate)
            np.testing.assert_array_equal(estimate, image)
            self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
