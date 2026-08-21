"""Invariants for residual-reflection consistency."""

from __future__ import annotations

import unittest

import numpy as np

from .reflection_consistent_posterior_2d import (
    denoise_reflection_consistent_posterior_2d,
    reflection_consistency_authority,
)


class ReflectionConsistencyTests(unittest.TestCase):
    def test_authority_is_bounded_and_zero_at_identity(self):
        y = np.arange(64, dtype=float).reshape(8, 8) / 63.0
        authority = reflection_consistency_authority(y, y, y)
        np.testing.assert_array_equal(authority, np.zeros_like(y))
        shifted = reflection_consistency_authority(y, 0.8 * y, 0.8 * y)
        self.assertGreaterEqual(float(np.min(shifted)), 0.0)
        self.assertLessEqual(float(np.max(shifted)), 1.0)

    def test_perfect_reflection_consistency_retains_base_estimate(self):
        y = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        x = 0.75 * y
        authority = reflection_consistency_authority(y, x, x)
        expected = y - authority * (y - x)
        active = np.abs(y - x) > 1e-7
        np.testing.assert_allclose(expected[active], x[active], atol=1e-13)

    def test_constant_is_exact_fixed_point(self):
        image = np.full((19, 23), 0.42)
        result, diagnostic = denoise_reflection_consistent_posterior_2d(image)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(diagnostic["maximum_reflection_identity_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
