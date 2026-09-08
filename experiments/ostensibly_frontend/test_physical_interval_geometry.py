from __future__ import annotations

import unittest

import numpy as np

from .physical_interval_geometry import (
    compile_physical_interval_atlas,
    physical_interval_distance,
    physical_interval_signature,
)


class PhysicalIntervalGeometryTests(unittest.TestCase):
    def test_global_pitch_translation_preserves_interval_geometry(self) -> None:
        time = np.repeat(np.arange(16), 8)
        rows = np.tile(np.asarray((10, 12, 20, 23, 35, 39, 52, 57)), 16)
        height = np.ones_like(rows)
        first = np.column_stack((rows, time, height))
        shifted = first.copy()
        shifted[:, 0] += 17
        left = physical_interval_signature(first, 8, 4)
        right = physical_interval_signature(shifted, 8, 4)
        self.assertAlmostEqual(
            physical_interval_distance(*left, *right, maximum_shift=0), 0.0
        )

    def test_changed_band_spacing_has_positive_distance(self) -> None:
        time = np.repeat(np.arange(16), 8)
        rows = np.tile(np.asarray((10, 12, 20, 23, 35, 39, 52, 57)), 16)
        first = np.column_stack((rows, time, np.ones_like(rows)))
        changed = first.copy()
        changed[:, 0] = np.where(changed[:, 0] > 30, changed[:, 0] * 1.4, changed[:, 0])
        left = physical_interval_signature(first, 8, 4)
        right = physical_interval_signature(changed, 8, 4)
        self.assertGreater(
            physical_interval_distance(*left, *right, maximum_shift=0), 0.01
        )

    def test_vectorized_atlas_ranking_recovers_spacing_class(self) -> None:
        time = np.repeat(np.arange(16), 8)
        rows = np.tile(np.asarray((10, 12, 20, 23, 35, 39, 52, 57)), 16)
        first = np.column_stack((rows, time, np.ones_like(rows)))
        changed = first.copy()
        changed[:, 0] = np.where(changed[:, 0] > 30, changed[:, 0] * 1.4, changed[:, 0])
        atlas = compile_physical_interval_atlas(
            (("A", "a", first), ("B", "b", changed)),
            time_bins=8,
            row_quantiles=4,
            maximum_shift=0,
        )
        shifted = first.copy()
        shifted[:, 0] += 11
        ranking = atlas.rank(shifted)
        self.assertEqual(ranking[0]["phone"], "A")
        self.assertAlmostEqual(float(ranking[0]["distance"]), 0.0)

    def test_component_rankings_isolate_trajectory_and_mass(self) -> None:
        time = np.repeat(np.arange(16), 8)
        base_rows = np.tile(np.asarray((10, 12, 20, 23, 35, 39, 52, 57)), 16)
        static = np.column_stack((base_rows, time, np.ones_like(base_rows)))
        moving = static.copy()
        moving[:, 0] += np.repeat(np.arange(16), 8)
        atlas = compile_physical_interval_atlas(
            (("STATIC", "s", static), ("MOVING", "m", moving)),
            time_bins=8,
            row_quantiles=4,
            maximum_shift=0,
        )
        trajectory = atlas.rank_with_weights(
            moving, interval_weight=0.0, trajectory_weight=1.0, mass_weight=0.0
        )
        self.assertEqual(trajectory[0]["phone"], "MOVING")
        with self.assertRaises(ValueError):
            atlas.rank_with_weights(
                moving,
                interval_weight=0.0,
                trajectory_weight=0.0,
                mass_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
