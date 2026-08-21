"""Invariants for the fixed-dimensional continuous-scale moment lift."""

import unittest

import numpy as np

from .lifted_scale_moment_transport_2d import (
    _canonical_scale_coordinate,
    _moment_readouts,
    _raw_scale_moment_lift,
    affine_scale_action,
    lifted_scale_moment_transport_state_2d,
)


class LiftedScaleMomentTransport2DTests(unittest.TestCase):
    def test_raw_moments_recompose_signed_lineage_exactly(self):
        rng = np.random.default_rng(808)
        lineage = rng.normal(size=(31, 7))
        labels = tuple(
            {
                "kind": "coarse_endpoint" if index == 0 else "heat_increment",
                "transport_time_coarse": float(2 ** (8 - index)),
                "transport_time_fine": float(2 ** (7 - index)),
            }
            for index in range(7)
        )
        lift, diagnostic = _raw_scale_moment_lift(lineage, labels)
        self.assertLess(diagnostic["signed_recomposition_error"], 2e-15)
        self.assertTrue(np.all(lift[2] >= np.abs(lift[0]) - 2e-15))
        coordinate = _canonical_scale_coordinate(labels)
        self.assertTrue(np.all((coordinate >= 0.0) & (coordinate <= 1.0)))
        expected = lineage @ (0.3 + 0.4 * coordinate)
        self.assertLess(np.max(np.abs(
            affine_scale_action(lift, 0.3, 0.4) - expected
        )), 3e-15)

    def test_readouts_are_bounded_without_transporting_ratios(self):
        raw = np.asarray((
            (0.0, 1.0),
            (0.1, 0.25),
            (2.0, 1.0),
            (1.0, 0.25),
            (1.0, 0.0625),
        ))
        readout = _moment_readouts(raw)
        self.assertTrue(np.all(readout["scale_variance"] >= 0.0))
        self.assertTrue(np.all(
            (readout["sign_coherence"] >= 0.0)
            & (readout["sign_coherence"] <= 1.0)
        ))
        self.assertTrue(np.all(readout["transport_uncertainty"] >= 0.0))

    def test_image_state_has_fixed_dimension_and_exact_conservation(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.1 * np.sin(0.7 * xx) + 0.04 * np.cos(0.5 * yy)
        state = lifted_scale_moment_transport_state_2d(image)
        self.assertEqual(state["persistent_dimension"], 14)
        self.assertEqual(state["vertex_lift"].shape, (10, 8, 8))
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["lifted_residual_recomposition_error"], 4e-13)
        self.assertLess(state["pushed_signed_commutation_error"], 4e-13)
        self.assertGreaterEqual(
            state["joint_normal_audit"][
                "full_action_constraint_violation_fraction"], 0.0)

    def test_refinement_changes_lineages_not_persistent_dimension(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.35 + 0.12 * np.sin(0.6 * xx - 0.2 * yy)
        coarse = lifted_scale_moment_transport_state_2d(
            image, trace_refinement=0)
        refined = lifted_scale_moment_transport_state_2d(
            image, trace_refinement=1)
        self.assertEqual(coarse["persistent_dimension"], 14)
        self.assertEqual(refined["persistent_dimension"], 14)
        self.assertGreater(
            refined["lineage_count_used_to_form_moments"],
            coarse["lineage_count_used_to_form_moments"],
        )


if __name__ == "__main__":
    unittest.main()
