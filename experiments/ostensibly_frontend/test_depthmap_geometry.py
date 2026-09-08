import unittest

import numpy as np

from experiments.ostensibly_frontend.depthmap_geometry import (
    DepthmapGeometryConfig,
    backfill_extrema_geometry,
    lift_random_seeds_to_extrema,
    multiscale_morphological_residual,
)


class DepthmapGeometryTests(unittest.TestCase):
    def test_continuous_morphology_suppresses_broad_mass_but_retains_ridge(self):
        field = np.zeros((64, 64), dtype=np.float64)
        field[8:28, 8:28] = 1.0
        field[44, 8:56] = 1.0
        residual, scales = multiscale_morphological_residual(field)
        self.assertEqual(scales.shape, (3, 64, 64))
        self.assertAlmostEqual(float(residual[18, 18]), 0.0)
        self.assertGreater(float(np.mean(residual[44, 12:52])), 0.9)

    def test_random_seed_lift_is_reproducible_and_lands_on_maxima(self):
        yy, xx = np.mgrid[:41, :41]
        field = np.exp(-((yy - 12) ** 2 + (xx - 10) ** 2) / 18.0)
        field += 0.8 * np.exp(-((yy - 30) ** 2 + (xx - 31) ** 2) / 12.0)
        config = DepthmapGeometryConfig(seed_count=512, random_seed=17)
        left, left_hits = lift_random_seeds_to_extrema(field, config)
        right, right_hits = lift_random_seeds_to_extrema(field, config)
        self.assertTrue(np.array_equal(left, right))
        self.assertTrue(np.array_equal(left_hits, right_hits))
        self.assertIn((12, 10), {tuple(item) for item in left[:, :2].astype(int)})
        self.assertIn((30, 31), {tuple(item) for item in left[:, :2].astype(int)})

    def test_backfill_adds_supported_points_between_near_extrema(self):
        field = np.zeros((32, 32), dtype=np.float64)
        field[12, 8:17] = 1.0
        extrema = np.array([[12.0, 8.0, 1.0], [12.0, 16.0, 1.0]])
        config = DepthmapGeometryConfig(
            connection_radius=10.0,
            backfill_spacing=0.5,
        )
        geometry = backfill_extrema_geometry(
            field, extrema, np.array([5, 7]), config
        )
        self.assertEqual(geometry.segments.shape, (1, 4))
        self.assertGreater(geometry.points.shape[0], extrema.shape[0])


if __name__ == "__main__":
    unittest.main()
