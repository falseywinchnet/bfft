"""Invariants for CRPS-stopped continuous tangent source geometry."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from .continuous_tangent_source_geometry_2d import (
    _analytic_local_joint_transport,
    _causal_support_joint_transport,
    _joint_bundle_graph_gradient_flow,
    _weighted_support_crps,
    _projective_jet_law_crps,
    _joint_bundle_energy_score,
    continuous_tangent_source_geometry_2d,
)


class ContinuousTangentSourceGeometry2DTests(unittest.TestCase):
    def test_joint_graph_gradient_flow_conserves_and_descends_pointwise(self):
        coordinate = np.array([[0.0], [0.4], [1.0]])
        source_distance = np.abs(coordinate - coordinate.T)
        cross_cost = np.array([
            [0.9, 0.4, 0.0],
            [0.1, 0.2, 0.8],
            [0.6, 0.1, 0.5],
        ])
        lineage = np.array([
            [0.0, 0.4, 0.6],
            [0.6, 0.0, 0.4],
            [0.4, 0.6, 0.0],
        ])
        operator = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        candidate, diagnostic = _joint_bundle_graph_gradient_flow(
            lineage,
            sparse.csr_matrix(operator),
            np.full(3, 1.0 / 3.0),
            source_distance,
            cross_cost,
        )
        np.testing.assert_allclose(np.sum(candidate, axis=1), 1.0)
        np.testing.assert_allclose(np.diag(candidate), 0.0)
        self.assertGreaterEqual(float(np.min(candidate)), 0.0)
        self.assertLessEqual(
            diagnostic["maximum_gradient_flow_energy_increase"], 2e-16)

    def test_causal_support_joint_transport_is_supportwise_nonworsening(self):
        coordinate = np.array([[0.0], [0.4], [1.0]])
        source_distance = np.abs(coordinate - coordinate.T)
        cross_cost = np.array([
            [0.9, 0.4, 0.0],
            [0.1, 0.2, 0.8],
            [0.6, 0.1, 0.5],
        ])
        lineage = np.eye(3)
        transported = np.array([
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ])
        candidate, step, diagnostic = _causal_support_joint_transport(
            lineage,
            transported,
            np.array([[0, 0, 1]]),
            source_distance,
            cross_cost,
        )
        self.assertTrue(np.all((step >= 0.0) & (step <= 1.0)))
        self.assertEqual(step[0], step[1])
        np.testing.assert_allclose(np.sum(candidate, axis=1), 1.0)
        self.assertLessEqual(
            diagnostic["maximum_support_energy_increase"], 2e-16)

    def test_analytic_local_joint_transport_is_pointwise_nonworsening(self):
        coordinate = np.array([[0.0], [0.4], [1.0]])
        source_distance = np.abs(coordinate - coordinate.T)
        cross_cost = np.array([
            [0.9, 0.4, 0.0],
            [0.1, 0.2, 0.8],
            [0.6, 0.1, 0.5],
        ])
        lineage = np.eye(3)
        transported = np.array([
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ])
        candidate, step, diagnostic = _analytic_local_joint_transport(
            lineage, transported, source_distance, cross_cost)
        self.assertTrue(np.all((step >= 0.0) & (step <= 1.0)))
        np.testing.assert_allclose(np.sum(candidate, axis=1), 1.0)
        self.assertLessEqual(
            diagnostic["maximum_local_energy_increase"], 2e-16)
        self.assertGreater(diagnostic["mean_local_energy_decrease"], 0.0)

    def test_weighted_support_crps_is_zero_for_exact_point_laws(self):
        weights = np.eye(4)
        support = np.array([-0.4, 0.1, 0.6, 1.2])
        np.testing.assert_allclose(
            _weighted_support_crps(weights, support, support), 0.0)

    def test_projective_jet_law_score_is_zero_for_constant_laws(self):
        shape = (4, 5)
        pixels = np.prod(shape)
        lineage = np.full((pixels, pixels), 1.0 / pixels)
        gradient = np.zeros(shape)
        tangent = np.array([[0.0, 1.0], [1.0, 0.0]])
        mass = np.full(shape + (2,), 0.5)
        derivative = np.zeros_like(mass)
        action, _ = _projective_jet_law_crps(
            lineage, gradient, gradient, mass, derivative, tangent)
        self.assertEqual(action, 0.0)

    def test_joint_bundle_energy_is_zero_for_exact_source_particles(self):
        lineage = np.eye(2)
        residual = np.array([[0.1, -0.2]])
        gx = np.array([[0.03, -0.01]])
        gy = np.array([[0.02, 0.04]])
        query_mass = np.ones((1, 2, 1))
        action, _ = _joint_bundle_energy_score(
            lineage,
            residual,
            gx,
            gy,
            residual[..., None],
            gx[..., None],
            gy[..., None],
            query_mass,
        )
        self.assertEqual(action, 0.0)

    def test_constant_has_one_fused_support_unit_and_excludes_target(self):
        field = np.full((10, 12), 0.43)
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field, angular_count=4, quantile_count=8)
        self.assertAlmostEqual(
            geometry["horizontal"]["implied_support"], 1.0, places=12)
        self.assertAlmostEqual(
            geometry["vertical"]["implied_support"], 1.0, places=12)
        self.assertAlmostEqual(
            geometry["fused"]["implied_support"], 1.0, places=12)
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)
        self.assertFalse(diagnostic["source_transport_ceiling_hit"])

    def test_affine_has_no_vertical_jet_population(self):
        yy, xx = np.mgrid[:10, :13]
        field = 0.2 + 0.013 * xx + 0.009 * yy
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field, angular_count=4, quantile_count=8)
        self.assertAlmostEqual(
            geometry["vertical"]["implied_support"], 1.0, places=11)
        self.assertLess(
            diagnostic["lineage_row_mass_maximum_error"], 2e-15)
        for record in diagnostic["source_transports"]:
            if record["accepted"]:
                self.assertLess(
                    record["held_out_residual_crps_after"],
                    record["held_out_residual_crps_before"],
                )

    def test_dynamic_fused_remetricization_preserves_constant_invariants(self):
        field = np.full((10, 12), 0.43)
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field,
            angular_count=4,
            quantile_count=8,
            remetricize=True,
            joint_jet_action=True,
        )
        self.assertAlmostEqual(geometry["fused"]["implied_support"], 1.0, 12)
        self.assertTrue(diagnostic["remetricize"])
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)

    def test_strict_joint_bundle_preserves_constant_invariants(self):
        field = np.full((10, 12), 0.43)
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field,
            angular_count=4,
            quantile_count=8,
            remetricize=True,
            strict_joint_bundle_action=True,
            line_search=True,
        )
        self.assertAlmostEqual(geometry["fused"]["implied_support"], 1.0, 12)
        self.assertTrue(diagnostic["strict_joint_bundle_action"])
        for record in diagnostic["source_transports"]:
            if record["accepted"]:
                self.assertLess(
                    record["transport_action_after"],
                    record["transport_action_before"],
                )

    def test_local_strict_joint_bundle_preserves_constant_invariants(self):
        field = np.full((8, 9), 0.43)
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field,
            angular_count=4,
            quantile_count=8,
            remetricize=True,
            strict_joint_bundle_action=True,
            local_joint_transport=True,
        )
        self.assertAlmostEqual(geometry["fused"]["implied_support"], 1.0, 12)
        self.assertTrue(diagnostic["local_joint_transport"])
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)
        for record in diagnostic["source_transports"]:
            self.assertLessEqual(
                record["maximum_local_energy_increase"], 2e-15)

    def test_causal_support_strict_bundle_preserves_constant_invariants(self):
        field = np.full((8, 9), 0.43)
        geometry, diagnostic = continuous_tangent_source_geometry_2d(
            field,
            angular_count=4,
            quantile_count=8,
            remetricize=True,
            strict_joint_bundle_action=True,
            causal_support_joint_transport=True,
        )
        self.assertAlmostEqual(geometry["fused"]["implied_support"], 1.0, 12)
        self.assertTrue(diagnostic["causal_support_joint_transport"])
        self.assertEqual(diagnostic["causal_support"]["center_count"], 1)
        self.assertEqual(diagnostic["maximum_target_self_lineage"], 0.0)


if __name__ == "__main__":
    unittest.main()
