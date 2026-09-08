from __future__ import annotations

import unittest

import numpy as np

from experiments.conv_four_child_atlas_census import (
    _identity_cell_overlap,
    _overlap_matrix,
    _polygon_halfplane_fraction,
    run_census,
)


class FourChildAtlasCensusTests(unittest.TestCase):
    def test_overlap_matrix_is_positive_normalized_and_identity_exact(self) -> None:
        for source, target in ((7, 7), (17, 9), (25, 13), (13, 25)):
            matrix = _overlap_matrix(source, target)
            self.assertGreaterEqual(float(np.min(matrix)), 0.0)
            np.testing.assert_allclose(
                np.sum(matrix, axis=1), 1.0, atol=2.0e-15, rtol=0.0
            )
        rng = np.random.default_rng(4)
        image = rng.normal(size=(9, 11, 3))
        np.testing.assert_array_equal(
            _identity_cell_overlap(image, image.shape[:2]), image
        )

    def test_axis_halfplane_cell_coverage_is_exact(self) -> None:
        coverage = _polygon_halfplane_fraction(
            (8, 8), np.array((1.0, 0.0)), 0.5
        )
        np.testing.assert_array_equal(coverage[:, :4], 0.0)
        np.testing.assert_array_equal(coverage[:, 4:], 1.0)

    def test_quick_census_preserves_factorwise_certificate(self) -> None:
        result = run_census(quick=True, include_affine=False)
        self.assertEqual(result["record_count"], 100)
        self.assertEqual(
            result["summary"]["all"]["maximum_horizontal_sign_surplus"], 0
        )
        self.assertEqual(
            result["summary"]["all"]["maximum_vertical_sign_surplus"], 0
        )
        self.assertGreater(
            result["summary"]["plane_wave"]["atlas_4x_material_win_count"], 0
        )
        self.assertGreater(
            result["summary"]["thin_strip"]["atlas_4x_material_loss_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
