"""Invariants for the cross-fitted joint value/first-jet contractor."""

import unittest

import numpy as np

from .joint_value_jet_zonotope_contractor_2d import (
    _covariance_normal_constraints,
    _crossfit_value_jet_constraints,
    contract_joint_value_jet_scale_edge_state_2d,
)


class JointValueJetZonotopeContractor2DTests(unittest.TestCase):
    def test_parallel_witnesses_exclude_every_target_endpoint(self):
        yy, xx = np.mgrid[:8, :9]
        residual = np.sin(0.4 * xx + 0.2 * yy)
        operator, lower, upper, diagnostic = (
            _crossfit_value_jet_constraints(residual))
        self.assertEqual(diagnostic["target_exclusion_error"], 0)
        self.assertGreater(diagnostic["fold_zero_edge_count"], 0)
        self.assertGreater(diagnostic["fold_one_edge_count"], 0)
        self.assertTrue(all(
            count > 0
            for count in diagnostic["orientation_edge_counts"].values()
        ))
        self.assertEqual(operator.shape[0], 2 * diagnostic["target_edge_count"])
        self.assertTrue(np.all(lower <= 0.0))
        self.assertTrue(np.all(upper >= 0.0))

    def test_covariance_normal_annihilates_witnessed_tangent(self):
        yy, xx = np.mgrid[:8, :9]
        field = 0.3 + 0.1 * np.sin(0.5 * xx + 0.2 * yy)
        _operator, _lower, _upper, rectangle = (
            _crossfit_value_jet_constraints(field))
        _normal_operator, lower, upper, diagnostic = (
            _covariance_normal_constraints(
                field, rectangle, additive_transfer=False))
        values = field.reshape(-1)
        witness = rectangle["witness_vertices"]
        minus = np.column_stack((
            values[witness[:, 0]],
            values[witness[:, 1]] - values[witness[:, 0]],
        ))
        plus = np.column_stack((
            values[witness[:, 2]],
            values[witness[:, 3]] - values[witness[:, 2]],
        ))
        tangent_pairing = np.sum(
            diagnostic["normal"] * (plus - minus), axis=1)
        self.assertLess(np.max(np.abs(tangent_pairing)), 2e-15)
        self.assertTrue(np.all(lower <= 0.0))
        self.assertTrue(np.all(upper >= 0.0))

    def test_joint_state_is_feasible_and_never_widens_outer_box(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.1 * np.sin(0.7 * xx) + 0.05 * np.cos(0.5 * yy)
        state = contract_joint_value_jet_scale_edge_state_2d(image)
        constrained = state["constrained_zonotope"]
        self.assertTrue(state["joint_contraction"]["feasible_outer_component"])
        self.assertTrue(constrained["zero_transfer_feasible"])
        self.assertTrue(np.all(
            state["coefficient_lower"]
            >= state["coefficient_lower_before_joint"] - 2e-15))
        self.assertTrue(np.all(
            state["coefficient_upper"]
            <= state["coefficient_upper_before_joint"] + 2e-15))
        self.assertLessEqual(
            constrained["mean_constraint_support_width_ratio"], 1.0 + 2e-15)
        self.assertEqual(
            set(state["branches"]),
            {
                "uncontracted_parent_identity_lineage",
                "uncontracted_parent_positive_push_lineage",
                "joint_noise_identity_lineage",
                "joint_noise_positive_push_lineage",
            },
        )
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["full_lineage_recomposition_error"], 3e-13)

    def test_constant_scene_remains_admissible(self):
        image = np.full((8, 8), 0.37)
        state = contract_joint_value_jet_scale_edge_state_2d(image)
        constrained = state["constrained_zonotope"]
        self.assertTrue(state["joint_contraction"]["feasible_outer_component"])
        self.assertTrue(constrained["zero_transfer_feasible"])
        self.assertLess(state["observation_recomposition_error"], 2e-15)


if __name__ == "__main__":
    unittest.main()
