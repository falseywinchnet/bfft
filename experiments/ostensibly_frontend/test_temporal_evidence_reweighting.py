from __future__ import annotations

import unittest

import numpy as np

from .temporal_evidence_reweighting import (
    cleanup_shark_extraction_support,
    temporal_reweight_cloud,
)


class TemporalEvidenceReweightingTests(unittest.TestCase):
    def test_support_is_a_bounded_union_not_a_product(self) -> None:
        support = cleanup_shark_extraction_support(
            np.asarray((0.8, 0.1)),
            np.asarray((0.0, 1.5)),
            np.asarray((0.2, 0.7)),
        )
        self.assertTrue(np.all((0.0 <= support) & (support <= 1.0)))
        self.assertGreaterEqual(support[0], 0.8)
        self.assertGreaterEqual(support[1], 0.7)

    def test_floor_one_is_the_exact_cloud_baseline(self) -> None:
        cloud = np.asarray(
            ((0.0, 0.0, 0.2), (1.0, 1.0, 0.5), (2.0, 1.0, 0.7))
        )
        result = temporal_reweight_cloud(
            cloud, np.asarray((0.0, 1.0)), support_floor=1.0
        )
        np.testing.assert_array_equal(result, cloud)

    def test_zero_floor_moves_mass_to_supported_frames_without_moving_points(self) -> None:
        cloud = np.asarray(
            tuple((float(index), float(frame), 0.5) for frame in (0, 1) for index in range(8))
        )
        result = temporal_reweight_cloud(
            cloud, np.asarray((0.0, 1.0)), support_floor=0.0
        )
        self.assertTrue(np.all(result[:, 1] == 1.0))
        self.assertTrue(set(map(tuple, result)).issubset(set(map(tuple, cloud))))


if __name__ == "__main__":
    unittest.main()
