from __future__ import annotations

import unittest
import numpy as np

from experiments.conv_fermi_owner_demo import (
    DISTANCE_RESPONSES,
    build_distance_atlas,
    build_direct_distance_atlas,
    build_owner_atlas,
    v3_germ_direct_resize,
    eikonal_basin_conv_resize,
    synthesize_distance_responses,
    synthesize_owner_charts,
)


class ConvFermiOwnerDemoTests(unittest.TestCase):
    def test_v3_germ_reduction_is_basin_analysis(self):
        y, x = np.indices((17, 19), dtype=np.float64)
        source = (0.2 + 0.6 / (1.0 + np.exp(-(x - 0.5 * y)))).astype(
            np.float32
        )
        expected = v3_germ_direct_resize(source, (9, 11))
        from backend import conv_basin_average
        np.testing.assert_array_equal(expected, conv_basin_average(source, (9, 11)))

    def test_v3_germ_enlargement_is_deterministic_and_bounded(self):
        source = np.full((9, 11), 0.375, dtype=np.float32)
        first = v3_germ_direct_resize(source, (17, 21))
        second = v3_germ_direct_resize(source, (17, 21))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.full((17, 21), 0.375))

    def test_deployable_form_is_deterministic(self):
        y, x = np.indices((17, 19), dtype=np.float64)
        source = (0.5 + 0.4 * np.tanh((x - 0.7 * y) / 2.0)).astype(np.float32)
        first = eikonal_basin_conv_resize(source, (33, 37))
        second = eikonal_basin_conv_resize(source, (33, 37))
        np.testing.assert_array_equal(first, second)

    def test_constant_is_exact_fixed_point(self):
        source = np.full((17, 19), 0.375, dtype=np.float32)
        atlas = build_owner_atlas(source)
        result = synthesize_owner_charts(source, (33, 37), atlas)
        np.testing.assert_array_equal(result, np.full((33, 37), 0.375))

    def test_chart_synthesis_stays_in_global_source_range(self):
        y, x = np.indices((25, 29), dtype=np.float64)
        source = (0.15 + 0.7 / (1.0 + np.exp(-(x - 0.6 * y - 3.0)))).astype(
            np.float32
        )
        atlas = build_owner_atlas(source)
        result = synthesize_owner_charts(source, (49, 57), atlas)
        self.assertGreaterEqual(float(np.min(result)), float(np.min(source)))
        self.assertLessEqual(float(np.max(result)), float(np.max(source)))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_all_distance_responses_are_deterministic_and_bounded(self):
        y, x = np.indices((13, 15), dtype=np.float64)
        source = (0.2 + 0.6 / (1.0 + np.exp(-(x - 0.5 * y)))).astype(
            np.float32
        )
        atlas = build_distance_atlas(source)
        first = synthesize_distance_responses(
            source, (25, 29), atlas, DISTANCE_RESPONSES
        )
        second = synthesize_distance_responses(
            source, (25, 29), atlas, DISTANCE_RESPONSES
        )
        for name in DISTANCE_RESPONSES:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertGreaterEqual(float(np.min(first[name])), float(np.min(source)))
            self.assertLessEqual(float(np.max(first[name])), float(np.max(source)))

    def test_direct_distance_frames_require_no_owner_labels(self):
        y, x = np.indices((13, 15), dtype=np.float64)
        source = (0.5 + 0.35 * np.tanh((x + y - 11.0) / 2.0)).astype(
            np.float32
        )
        atlas = build_direct_distance_atlas(source)
        self.assertEqual(atlas.distances.shape[0], len(atlas.frames))
        self.assertEqual(atlas.distances.shape[1:], source.shape)
        self.assertTrue(np.all(np.isfinite(atlas.distances)))

if __name__ == "__main__":
    unittest.main()
