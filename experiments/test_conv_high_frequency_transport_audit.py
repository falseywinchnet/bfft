from __future__ import annotations

import unittest

import numpy as np

from experiments.conv_conservative_multiresolution import (
    conv_moments_2d,
    face_sign_certificate_2d,
    synthesize_block_moments_2d,
)
from experiments.conv_high_frequency_transport_audit import (
    _exact_affine_cell_overlap,
    _nodal_stage_record,
)
from experiments.conv_warp.geometry import ProjectiveMap


class ConvHighFrequencyTransportAuditTests(unittest.TestCase):
    def test_identity_cell_overlap_is_exact(self) -> None:
        source = np.random.default_rng(7).normal(size=(7, 9, 3))
        result = _exact_affine_cell_overlap(
            source, ProjectiveMap(np.eye(3)), source.shape[:2]
        )
        np.testing.assert_allclose(result, source, atol=2.0e-15, rtol=0.0)

    def test_moment_refinement_conserves_means_without_sign_surplus(self) -> None:
        y, x = np.mgrid[:9, :11]
        source = np.sin(0.31 * x + 0.23 * y) + 0.2 * np.cos(0.7 * x)
        moments = conv_moments_2d(source)
        self.assertEqual(
            face_sign_certificate_2d(source, moments),
            {"horizontal_sign_surplus": 0, "vertical_sign_surplus": 0},
        )
        fine = synthesize_block_moments_2d(moments)
        recovered = fine.reshape(9, 2, 11, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(recovered, source, atol=8.0e-16, rtol=0.0)

    def test_double_precision_removes_false_axis_constraint(self) -> None:
        record = _nodal_stage_record((9, 9), (2.0, 0.0), 0.37)
        margin = record["minimum_support_gradient_margin"]
        self.assertLess(margin["float32_sampled"], -1.0e-9)
        self.assertGreater(margin["float64_sampled"], -1.0e-11)


if __name__ == "__main__":
    unittest.main()
