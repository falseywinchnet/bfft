"""Invariants for the one-pass post-lineage scalar readout."""

from __future__ import annotations

import unittest

import numpy as np

from .post_lineage_prolongation_2d import (
    denoise_post_lineage_prolongation_2d,
    denoise_post_lineage_residual_2d,
)


class PostLineageProlongation2DTests(unittest.TestCase):
    def test_constant_is_reproduced_and_target_identity_is_excluded(self):
        field = np.full((9, 11), 0.37)
        estimate, diagnostic = denoise_post_lineage_prolongation_2d(
            field, angular_count=4)
        np.testing.assert_allclose(estimate, field, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)
        self.assertLess(
            diagnostic["lineage_row_mass_maximum_error"], 2e-15)
        self.assertEqual(diagnostic["observation_graph_maximum_error"], 0.0)

    def test_constant_residual_loop_is_exact_and_target_excluded(self):
        field = np.full((9, 11), 0.37)
        estimate, diagnostic = denoise_post_lineage_residual_2d(
            field, angular_count=4)
        np.testing.assert_allclose(estimate, field, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)
        self.assertEqual(diagnostic["observation_graph_maximum_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
