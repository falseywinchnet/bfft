from __future__ import annotations

import unittest

import numpy as np

from .population_interval_alignment import estimate_population_row_multiplier


class PopulationIntervalAlignmentTests(unittest.TestCase):
    def test_recovers_global_row_dilation_without_labels(self) -> None:
        rng = np.random.default_rng(4)
        base = np.cumsum(rng.lognormal(-3.0, 0.5, size=(20, 16, 8)), axis=2)
        result = estimate_population_row_multiplier(base, 1.2 * base)
        self.assertAlmostEqual(result.row_multiplier, 1.0 / 1.2, places=12)
        self.assertLess(result.quantile_rms_after, 1e-12)

    def test_rejects_non_surface_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "count x time x quantile"):
            estimate_population_row_multiplier(np.ones((8, 8)), np.ones((8, 8)))


if __name__ == "__main__":
    unittest.main()
