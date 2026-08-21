"""Invariants for the nuisance-geometry diagnostic generators."""

import unittest

import numpy as np

from .probe_nuisance_geometry_2d import (
    poisson_observation,
    row_correlated_signal_dependent_observation,
)


class NuisanceGeometryProbeTests(unittest.TestCase):
    def test_poisson_observation_is_bounded_and_reproducible(self):
        truth = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        first = poisson_observation(truth, 16.0, np.random.default_rng(7))
        second = poisson_observation(truth, 16.0, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_row_correlated_observation_is_bounded_and_nontrivial(self):
        truth = np.full((16, 16), 0.5)
        observed = row_correlated_signal_dependent_observation(
            truth, 0.1, np.random.default_rng(11))
        self.assertEqual(observed.shape, truth.shape)
        self.assertGreater(float(np.std(observed - truth)), 0.0)
        self.assertGreaterEqual(float(observed.min()), 0.0)
        self.assertLessEqual(float(observed.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
