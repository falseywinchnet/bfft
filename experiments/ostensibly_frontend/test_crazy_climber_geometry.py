import unittest

import numpy as np

from experiments.ostensibly_frontend.crazy_climber_geometry import (
    CrazyClimberConfig,
    crazy_climber_occupation,
    multiscale_hessian_ridge_surface,
)


class CrazyClimberGeometryTests(unittest.TestCase):
    def test_hessian_surface_prefers_a_line_to_a_broad_blob(self):
        field = np.zeros((72, 72), dtype=np.float64)
        field[18:42, 8:32] = 1.0
        field[56, 8:64] = 1.0
        surface = multiscale_hessian_ridge_surface(field)
        self.assertGreater(
            float(np.mean(surface.saliency[56, 12:60])),
            3.0 * float(np.mean(surface.saliency[24:36, 14:26])),
        )

    def test_crazy_climbers_concentrate_on_recurrent_ridges(self):
        rows, frames = 64, 48
        yy, tt = np.mgrid[:rows, :frames]
        center = 24.0 + 4.0 * np.sin(tt / 7.0)
        surface = np.exp(-0.5 * ((yy - center) / 1.2) ** 2)
        surface += 0.08 * np.random.default_rng(8).random(surface.shape)
        config = CrazyClimberConfig(
            climbers=256,
            steps=1200,
            burn_in=200,
            sample_stride=4,
            random_seed=19,
        )
        occupation = crazy_climber_occupation(surface, config)
        ridge_rows = np.rint(center[0]).astype(int)
        ridge_values = occupation.weighted[ridge_rows, np.arange(frames)]
        background = occupation.weighted[:8]
        self.assertGreater(float(np.mean(ridge_values)), 8.0 * float(np.mean(background)))
        frame_mass = occupation.unweighted.sum(axis=0)
        self.assertLess(float(np.std(frame_mass) / np.mean(frame_mass)), 0.2)

    def test_crazy_climbers_are_reproducible(self):
        field = np.eye(24, dtype=np.float64)
        config = CrazyClimberConfig(
            climbers=64,
            steps=200,
            burn_in=40,
            sample_stride=4,
            random_seed=3,
        )
        left = crazy_climber_occupation(field, config)
        right = crazy_climber_occupation(field, config)
        self.assertTrue(np.array_equal(left.weighted, right.weighted))
        self.assertEqual(left.accepted_vertical_fraction, right.accepted_vertical_fraction)


if __name__ == "__main__":
    unittest.main()
