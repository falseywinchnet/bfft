from __future__ import annotations

import unittest
import numpy as np

from .conditional_ridge_atlas import (
    compile_conditional_ridge_atlas,
    conditional_ridge_distance,
    conditional_ridge_signature,
)


class ConditionalRidgeAtlasTests(unittest.TestCase):
    def test_prefers_matching_ordered_trajectory(self) -> None:
        frame = np.linspace(0.0, 1.0, 2000)
        rising = np.column_stack((frame, frame, np.ones_like(frame)))
        falling = np.column_stack((1.0 - frame, frame, np.ones_like(frame)))
        atlas = compile_conditional_ridge_atlas((("UP", "u", rising), ("DOWN", "d", falling)), time_bins=16, row_quantiles=8)
        self.assertEqual(atlas.rank(rising)[0]["phone"], "UP")

    def test_signature_distance_is_zero_only_for_matching_trace(self) -> None:
        frame = np.linspace(0.0, 1.0, 2000)
        rising = np.column_stack((frame, frame, np.ones_like(frame)))
        falling = np.column_stack((1.0 - frame, frame, np.ones_like(frame)))
        up_surface, up_mass = conditional_ridge_signature(rising, 16, 8)
        down_surface, down_mass = conditional_ridge_signature(falling, 16, 8)
        self.assertAlmostEqual(
            conditional_ridge_distance(up_surface, up_mass, up_surface, up_mass),
            0.0,
        )
        self.assertGreater(
            conditional_ridge_distance(up_surface, up_mass, down_surface, down_mass),
            0.0,
        )

    def test_saliency_weighting_changes_mass_and_ridge_quantiles(self) -> None:
        frame = np.linspace(0.0, 1.0, 2000)
        rows = np.where(frame < 0.5, frame, 1.0 - frame)
        heights = np.where(frame < 0.5, 0.01, 1.0)
        cloud = np.column_stack((rows, frame, heights))
        uniform = conditional_ridge_signature(cloud, 16, 8)
        weighted = conditional_ridge_signature(
            cloud, 16, 8, saliency_power=0.5
        )
        self.assertFalse(np.allclose(uniform[0], weighted[0]))
        self.assertFalse(np.allclose(uniform[1], weighted[1]))
        with self.assertRaises(ValueError):
            conditional_ridge_signature(cloud, 16, 8, saliency_power=-0.5)


if __name__ == "__main__": unittest.main()
