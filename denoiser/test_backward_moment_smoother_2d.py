"""Invariants for the backward moment-smoothing ablation."""

import unittest

import numpy as np
from scipy import sparse

from .backward_moment_smoother_2d import (
    denoise_backward_moment_smoother_2d,
    diagonal_backward_gain,
)


class BackwardMomentSmootherTests(unittest.TestCase):
    def test_gain_is_physical(self):
        variance = np.linspace(0.01, 0.2, 64).reshape(8, 8)
        averaging = sparse.eye(variance.size, format="csr")
        gain, diagnostic = diagonal_backward_gain(variance, averaging)
        self.assertTrue(np.all(gain >= 0.0))
        self.assertTrue(np.all(gain <= 1.0))
        self.assertLessEqual(diagnostic["maximum_backward_gain"], 1.0)

    def test_constant_is_exact_fixed_point(self):
        image = np.full((12, 15), 0.43)
        result, diagnostic = denoise_backward_moment_smoother_2d(image)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(diagnostic["maximum_observation_identity_error"], 0.0)

    def test_result_stays_in_observation_range(self):
        rng = np.random.default_rng(83)
        image = rng.random((16, 18))
        result, _diagnostic = denoise_backward_moment_smoother_2d(image)
        self.assertGreaterEqual(float(result.min()), float(image.min()))
        self.assertLessEqual(float(result.max()), float(image.max()))


if __name__ == "__main__":
    unittest.main()
