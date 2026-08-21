"""Invariants and falsification controls for cross-predictive transport."""

from __future__ import annotations

import unittest

import numpy as np

from .cross_predictive_transport import (
    _bernoulli_geodesic_residual_1d,
    _connection_defect_ownership_1d,
    _connection_geodesic_acceleration_1d,
    _transported_connection_tangent_acceleration_1d,
    _transported_connection_spherical_phase_1d,
    _transported_gaussian_connection_contrast_1d,
    action_contracting_connection_readout_forms,
    action_contracting_connection_transport_1d,
    CrossPredictiveResolution,
    causal_collision_particle_law,
    causal_crossfit_particle_law,
    ancestry_connection_lineage_transport_1d,
    complete_participation_lineage_transport_1d,
    connection_ownership_lineage_transport_1d,
    connection_ownership_readout_forms,
    continuous_curvature_lineage_readout_forms,
    continuous_curvature_particle_law_1d,
    curvature_consensus_lineage_readout_forms,
    curvature_consensus_lineage_transport_1d,
    denoise_causal_collision_transport,
    denoise_cross_predictive_transport,
    denoise_cross_predictive_particle_transport,
    denoise_information_lineage_transport,
    denoise_lineage_branch_transport,
    distinct_ancestry_lineage_transport_1d,
    distinct_ancestry_particle_law_1d,
    independent_side_collision_readout_forms,
    independent_side_lineage_transport_1d,
    independent_side_particle_law_1d,
    gaussian_connection_potential_lineage_transport_1d,
    gaussian_connection_potential_readout_forms,
    lineage_branch_transport_1d,
    nested_midpoint_lineage_readout_forms,
    nested_midpoint_lineage_transport_1d,
    nested_midpoint_particle_law_1d,
    paired_side_collision_lineage_readout_forms,
    paired_side_collision_lineage_transport_1d,
    paired_side_collision_particle_law_1d,
    relation_scale_particle_law,
    relation_scale_transport,
    root_context_collision_lineage_readout_forms,
    root_context_collision_lineage_transport_1d,
    root_context_collision_particle_law_1d,
    symmetric_second_jet_lineage_readout_forms,
    symmetric_second_jet_lineage_transport_1d,
    symmetric_second_jet_curvature_readout_forms,
    symmetric_second_jet_curvature_transport_1d,
    symmetric_second_jet_particle_law_1d,
    transport_distribution_lineage_transport_1d,
)
from .sample_series import PRESETS, compose_series, corrupt


class CrossPredictiveTransportTests(unittest.TestCase):
    def test_ancestry_connection_is_conservative_and_determinant_one(self):
        x = np.linspace(0.0, 1.0, 40, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = ancestry_connection_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.linalg.det(law["edge_precision"]),
            1.0,
            atol=2e-11,
            rtol=2e-11,
        )
        self.assertGreaterEqual(
            diagnostic["mean_connection_family_disagreement"], 0.0)

    def test_distinct_ancestry_action_excludes_interior_target(self):
        x = np.linspace(0.0, 1.0, 96, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        changed = line.copy()
        changed[48] += 0.4
        law, diagnostic = distinct_ancestry_particle_law_1d(line)
        altered, _ = distinct_ancestry_particle_law_1d(changed)
        valid = law["reference_mass"][48] > 0.0
        np.testing.assert_array_equal(
            law["prediction"][48, valid], altered["prediction"][48, valid])
        np.testing.assert_allclose(
            law["total_action"][48, valid],
            altered["total_action"][48, valid],
            atol=2e-15,
            rtol=0.0,
        )
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_distinct_ancestry_transport_is_constant_exact(self):
        line = np.full(32, 0.37)
        law, diagnostic = distinct_ancestry_lineage_transport_1d(line)
        np.testing.assert_array_equal(law["prediction"], 0.37)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertFalse(diagnostic["target_value_enters_local_action"])

    def test_connection_ownership_is_constant_exact(self):
        line = np.full(32, 0.37)
        for ownership_measure in (
            "root_context",
            "connection_hotelling",
            "connection_hellinger",
            "transported_hellinger_contrast",
            "transported_covariance_contrast",
            "transported_gaussian_law_contrast",
        ):
            forms, diagnostic = connection_ownership_readout_forms(
                line, ownership_measure=ownership_measure)
            for value in forms.values():
                np.testing.assert_array_equal(value, line)
            self.assertEqual(diagnostic["physical_parameters"], "none")

    def test_connection_hellinger_defect_is_coordinate_invariant(self):
        line = np.linspace(0.2, 0.8, 8)
        edge = np.arange(7, dtype=np.float64)
        mean = np.column_stack((
            0.1 * edge,
            np.sin(0.3 * edge),
            0.02 * edge * edge,
        ))
        covariance = np.stack([
            np.diag((0.3 + 0.02 * index, 0.7, 1.1 - 0.03 * index))
            for index in range(7)
        ])
        transform = np.array([
            [1.2, 0.1, -0.2],
            [0.3, 0.8, 0.1],
            [-0.1, 0.2, 1.4],
        ])
        transformed_mean = mean @ transform.T
        transformed_covariance = np.einsum(
            "ab,ebc,dc->ead", transform, covariance, transform)
        survival, action = _connection_defect_ownership_1d(
            line, mean, covariance, include_covariance=True)
        transformed_survival, transformed_action = (
            _connection_defect_ownership_1d(
                line,
                transformed_mean,
                transformed_covariance,
                include_covariance=True,
            )
        )
        np.testing.assert_allclose(
            transformed_survival, survival, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(
            transformed_action, action, atol=2e-14, rtol=2e-14)

    def test_transported_gaussian_connection_law_is_coordinate_invariant(self):
        line = np.linspace(0.2, 0.8, 8)
        edge = np.arange(7, dtype=np.float64)
        mean = np.column_stack((
            0.1 * edge,
            np.sin(0.3 * edge),
            0.02 * edge * edge,
        ))
        covariance = np.stack([
            np.diag((0.3 + 0.02 * index, 0.7, 1.1 - 0.03 * index))
            for index in range(7)
        ])
        source_identity = np.empty((8, 3, 2), dtype=np.int64)
        index = np.arange(8)
        source_identity[:, 0] = np.column_stack((
            np.maximum(index - 1, 0), np.minimum(index + 1, 7)))
        source_identity[:, 1] = np.column_stack((
            np.maximum(index - 2, 0), np.maximum(index - 1, 0)))
        source_identity[:, 2] = np.column_stack((
            np.minimum(index + 1, 7), np.minimum(index + 2, 7)))
        branch_mass = np.broadcast_to(
            np.array((0.4, 0.3, 0.3)), (8, 3)).copy()
        transform = np.array([
            [1.2, 0.1, -0.2],
            [0.3, 0.8, 0.1],
            [-0.1, 0.2, 1.4],
        ])
        transformed_mean = mean @ transform.T
        transformed_covariance = np.einsum(
            "ab,ebc,dc->ead", transform, covariance, transform)
        survival, action, branch_survival = (
            _transported_gaussian_connection_contrast_1d(
                line, mean, covariance, source_identity, branch_mass)
        )
        (
            transformed_survival,
            transformed_action,
            transformed_branch_survival,
        ) = (
            _transported_gaussian_connection_contrast_1d(
                line,
                transformed_mean,
                transformed_covariance,
                source_identity,
                branch_mass,
            )
        )
        np.testing.assert_allclose(
            transformed_survival, survival, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(
            transformed_action, action, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(
            transformed_branch_survival,
            branch_survival,
            atol=3e-13,
            rtol=3e-13,
        )
        acceleration, acceleration_action = (
            _connection_geodesic_acceleration_1d(
                line, mean, covariance)
        )
        transformed_acceleration, transformed_acceleration_action = (
            _connection_geodesic_acceleration_1d(
                line, transformed_mean, transformed_covariance)
        )
        np.testing.assert_allclose(
            transformed_acceleration,
            acceleration,
            atol=3e-13,
            rtol=3e-13,
        )
        np.testing.assert_allclose(
            transformed_acceleration_action,
            acceleration_action,
            atol=3e-13,
            rtol=3e-13,
        )
        tangent, tangent_diagnostic = (
            _transported_connection_tangent_acceleration_1d(
                line, mean, covariance, source_identity, branch_mass)
        )
        transformed_tangent, transformed_tangent_diagnostic = (
            _transported_connection_tangent_acceleration_1d(
                line,
                transformed_mean,
                transformed_covariance,
                source_identity,
                branch_mass,
            )
        )
        np.testing.assert_allclose(
            transformed_tangent, tangent, atol=3e-12, rtol=3e-12)
        self.assertEqual(
            tangent_diagnostic["physical_parameters"], "none")
        self.assertEqual(
            transformed_tangent_diagnostic["physical_parameters"], "none")
        spherical_phase, spherical_diagnostic = (
            _transported_connection_spherical_phase_1d(
                line, mean, covariance, source_identity, branch_mass)
        )
        transformed_spherical_phase, transformed_spherical_diagnostic = (
            _transported_connection_spherical_phase_1d(
                line,
                transformed_mean,
                transformed_covariance,
                source_identity,
                branch_mass,
            )
        )
        np.testing.assert_allclose(
            transformed_spherical_phase,
            spherical_phase,
            atol=3e-12,
            rtol=3e-12,
        )
        self.assertEqual(spherical_diagnostic["physical_parameters"], "none")
        self.assertEqual(
            transformed_spherical_diagnostic["physical_parameters"], "none")

    def test_action_contracting_connection_is_constant_exact(self):
        line = np.full(32, 0.37)
        forms, diagnostic = action_contracting_connection_readout_forms(line)
        for value in forms.values():
            np.testing.assert_array_equal(value, line)
        self.assertEqual(diagnostic["physical_parameters"], "none")

    def test_action_contracting_connection_conserves_and_contracts(self):
        x = np.linspace(0.0, 1.0, 40, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        for keyword in (
            {},
            {"require_population_phase_collision": True},
            {"fuse_population_phase_odds": True},
            {"fuse_connection_acceleration_odds": True},
            {"fuse_connection_jerk_odds": True},
            {"fuse_connection_tangent_odds": True},
            {"fuse_connection_spherical_phase_odds": True},
            {"fuse_connection_spherical_phase_union": True},
            {"suppress_connection_on_spherical_phase": True},
            {"fuse_phase_defect_spherical_odds": True},
            {
                "fuse_population_phase_odds": True,
                "newton_optimize_connection": True,
            },
            {
                "fuse_population_phase_odds": True,
                "marginalize_connection_action": True,
            },
            {"marginalize_gaussian_connection": True},
            {"phase_coherent_connection_posterior": True},
        ):
            law, diagnostic = action_contracting_connection_transport_1d(
                line, **keyword)
            np.testing.assert_allclose(
                np.sum(law["mass"], axis=1),
                1.0,
                atol=2e-15,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                np.sum(law["path_collision_mass"], axis=1),
                1.0,
                atol=2e-15,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                np.sum(law["effective_kernels"], axis=2),
                1.0,
                atol=2e-15,
                rtol=0.0,
            )
            self.assertLessEqual(
                diagnostic["maximum_harmonic_action_violation"], 2e-12)
            self.assertGreater(
                diagnostic["valid_distinct_ancestry_fraction"], 0.9)
            if keyword.get("marginalize_connection_action", False):
                self.assertGreaterEqual(
                    diagnostic[
                        "mean_connection_action_posterior_variance"],
                    0.0,
                )
                self.assertLessEqual(
                    diagnostic[
                        "mean_connection_action_posterior_variance"],
                    1.0 / 12.0 + 2e-15,
                )
                self.assertGreaterEqual(
                    diagnostic["mean_connection_action_posterior"], 0.0)
                self.assertLessEqual(
                    diagnostic["mean_connection_action_posterior"], 1.0)
            if keyword.get("phase_coherent_connection_posterior", False):
                self.assertGreaterEqual(
                    diagnostic["mean_observation_cavity_surprise"], 0.0)
                self.assertLessEqual(
                    diagnostic["mean_observation_cavity_surprise"], 1.0)
                self.assertGreaterEqual(
                    diagnostic["mean_transport_support_fidelity"], 0.0)
                self.assertLessEqual(
                    diagnostic["mean_transport_support_fidelity"], 1.0)
                self.assertTrue(np.all(
                    law["transport_support_fidelity"] >= 0.0))
                self.assertTrue(np.all(
                    law["transport_support_fidelity"] <= 1.0))

    def test_action_connection_rejects_multiple_population_combinations(self):
        line = np.linspace(0.2, 0.8, 24)
        with self.assertRaises(ValueError):
            action_contracting_connection_transport_1d(
                line,
                require_population_phase_collision=True,
                fuse_population_phase_odds=True,
            )
        with self.assertRaises(ValueError):
            action_contracting_connection_transport_1d(
                line,
                newton_optimize_connection=True,
                marginalize_connection_action=True,
            )
        with self.assertRaises(ValueError):
            action_contracting_connection_transport_1d(
                line,
                marginalize_connection_action=True,
                marginalize_gaussian_connection=True,
            )
        with self.assertRaises(ValueError):
            action_contracting_connection_transport_1d(
                line,
                newton_optimize_connection=True,
                phase_coherent_connection_posterior=True,
            )

    def test_gaussian_connection_potential_is_constant_exact(self):
        line = np.full(32, 0.37)
        forms, diagnostic = gaussian_connection_potential_readout_forms(line)
        for value in forms.values():
            np.testing.assert_array_equal(value, line)
        self.assertEqual(
            diagnostic["transport_distribution"]["physical_parameters"],
            "none",
        )

    def test_bernoulli_geodesic_residual_annihilates_affine_phase(self):
        probability = np.linspace(0.1, 0.8, 32)
        survival, action = _bernoulli_geodesic_residual_1d(probability)
        np.testing.assert_allclose(survival, 0.0, atol=3e-16, rtol=0.0)
        np.testing.assert_allclose(action, 0.0, atol=3e-16, rtol=0.0)

    def test_gaussian_connection_potential_conserves_probability(self):
        x = np.linspace(0.0, 1.0, 40, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = gaussian_connection_potential_lineage_transport_1d(
            line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["path_collision_mass"], axis=1),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        self.assertEqual(
            diagnostic["transport_distribution"]["conductance"],
            "erf(r / sqrt(2)) / r",
        )

    def test_connection_ownership_joint_path_conserves_probability(self):
        x = np.linspace(0.0, 1.0, 40, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = connection_ownership_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=(1, 2)),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.sum(law["path_collision_mass"], axis=(1, 2)),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.sum(law["mode_transition"], axis=2),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        transported_mode_reference = np.einsum(
            "em,emn->en",
            law["mode_reference"][:-1],
            law["mode_transition"],
        )
        np.testing.assert_allclose(
            transported_mode_reference,
            law["mode_reference"][1:],
            atol=2e-15,
            rtol=0.0,
        )
        self.assertTrue(np.all(law["ownership_emission"] >= 0.0))
        self.assertTrue(np.all(law["ownership_emission"] <= 1.0))
        np.testing.assert_allclose(
            np.sum(law["ownership_emission"], axis=1),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        self.assertGreaterEqual(diagnostic["minimum_drift_ownership"], 0.0)
        self.assertLessEqual(diagnostic["maximum_drift_ownership"], 1.0)

    def test_complete_participation_transport_is_positive_and_constant_exact(self):
        line = np.full(48, 0.37)
        law, diagnostic = complete_participation_lineage_transport_1d(line)
        np.testing.assert_array_equal(law["prediction"], 0.37)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(
            diagnostic["participation_algebra"]["selected_coordinates"],
            "none",
        )

    def test_complete_participation_precision_is_determinant_one(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, _diagnostic = complete_participation_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.linalg.det(law["edge_precision"]),
            1.0,
            atol=2e-12,
            rtol=2e-12,
        )

    def test_transport_distribution_is_conservative_and_determinant_one(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = transport_distribution_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.linalg.det(law["edge_precision"]),
            1.0,
            atol=2e-12,
            rtol=2e-12,
        )
        self.assertEqual(
            diagnostic["transport_distribution"]["selected_noise_model"],
            "none",
        )

    def test_information_lineage_denoiser_is_exact_on_constant(self):
        line = np.full(48, 0.37)
        result, diagnostic = denoise_information_lineage_transport(line)
        np.testing.assert_array_equal(result, line)
        self.assertEqual(diagnostic["continuation"], "none")

    def test_lineage_continuation_is_exact_on_constant(self):
        line = np.full(48, 0.37)
        result, diagnostic = denoise_lineage_branch_transport(line)
        np.testing.assert_array_equal(result, line)
        self.assertFalse(diagnostic["continuation_ceiling_hit"])

    def test_lineage_continuation_descends_observation_action(self):
        truth = compose_series(64, PRESETS["oscillatory composite"])[1]
        _result, diagnostic = denoise_lineage_branch_transport(truth)
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"],
                )
    def test_lineage_branch_transport_conserves_probability(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = lineage_branch_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_joint_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_joint_collision_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_coupled_phase_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_coupled_phase_collision_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_coupled_phase_coverage_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["hj_coupled_phase_bundle_coverage_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["path_collision_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["path_affinity_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["path_fidelity_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["transport_fidelity_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        for name in (
            "transport_plan_history_mass",
            "self_consistent_transport_mass",
            "distributed_transport_mass",
            "action_contracting_transport_mass",
            "two_history_action_transport_mass",
        ):
            np.testing.assert_allclose(
                np.sum(law[name], axis=1),
                1.0, atol=2e-15, rtol=0.0)
        self.assertTrue(np.all(law["transport_edge_fidelity"] >= 0.0))
        self.assertTrue(np.all(law["transport_edge_fidelity"] <= 1.0))
        self.assertTrue(np.all(law["transport_vertex_fidelity"] >= 0.0))
        self.assertTrue(np.all(law["transport_vertex_fidelity"] <= 1.0))
        np.testing.assert_allclose(
            np.sum(law["symmetric_parent_mass"], axis=1),
            1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(law["prediction"].shape, (48, 72))
        self.assertGreater(diagnostic["mean_transition_population"], 1.0)

    def test_lineage_collision_readouts_are_exact_on_constant(self):
        from .cross_predictive_transport import lineage_branch_readout_forms
        line = np.full(48, 0.37)
        forms, _diagnostic = lineage_branch_readout_forms(
            line, include_experimental=True)
        np.testing.assert_array_equal(forms["collision_mean"], line)
        np.testing.assert_array_equal(forms["collision_median"], line)
        np.testing.assert_array_equal(forms["hj_joint_mean"], line)
        np.testing.assert_array_equal(forms["hj_joint_median"], line)
        np.testing.assert_array_equal(forms["hj_joint_collision_mean"], line)
        np.testing.assert_array_equal(forms["hj_joint_collision_median"], line)
        np.testing.assert_array_equal(forms["hj_coupled_phase_mean"], line)
        np.testing.assert_array_equal(forms["hj_coupled_phase_median"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_collision_mean"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_collision_median"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_coverage_mean"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_coverage_median"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_bundle_coverage_mean"], line)
        np.testing.assert_array_equal(
            forms["hj_coupled_phase_bundle_coverage_median"], line)
        np.testing.assert_array_equal(
            forms["hj_global_characteristic_section"], line)
        np.testing.assert_array_equal(
            forms["posterior_characteristic_section"], line)
        np.testing.assert_array_equal(forms["path_collision_mean"], line)
        np.testing.assert_array_equal(forms["path_collision_median"], line)
        np.testing.assert_array_equal(forms["path_affinity_mean"], line)
        np.testing.assert_array_equal(forms["path_affinity_median"], line)
        np.testing.assert_array_equal(forms["path_fidelity_mean"], line)
        np.testing.assert_array_equal(forms["path_fidelity_median"], line)
        np.testing.assert_array_equal(forms["transport_fidelity_mean"], line)
        np.testing.assert_array_equal(forms["transport_fidelity_median"], line)
        np.testing.assert_array_equal(
            forms["transport_plan_history_mean"], line)
        np.testing.assert_array_equal(
            forms["self_consistent_transport_mean"], line)
        np.testing.assert_array_equal(
            forms["distributed_transport_mean"], line)
        np.testing.assert_array_equal(
            forms["action_contracting_transport_mean"], line)
        np.testing.assert_array_equal(
            forms["two_history_action_transport_mean"], line)
        np.testing.assert_array_equal(
            forms["path_fidelity_participation_section"], line)
        np.testing.assert_array_equal(
            forms["transport_history_participation_mean"], line)
        np.testing.assert_array_equal(
            forms["transport_history_participation_median"], line)
        np.testing.assert_array_equal(forms["joint_w1_value_jet"], line)
        np.testing.assert_array_equal(forms["joint_information_field"], line)
        np.testing.assert_array_equal(forms["symmetric_parent_mean"], line)
        np.testing.assert_array_equal(forms["symmetric_parent_median"], line)
        np.testing.assert_array_equal(forms["energy_root_mean"], line)
        np.testing.assert_array_equal(forms["energy_root_median"], line)
        np.testing.assert_array_equal(forms["energy_root_collision_mean"], line)

    def test_lineage_branch_transport_is_exact_on_constant(self):
        line = np.full(48, 0.37)
        law, _diagnostic = lineage_branch_transport_1d(line)
        np.testing.assert_array_equal(law["prediction"], 0.37)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)

    def test_paired_side_lineage_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, _diagnostic = paired_side_collision_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        forms, _ = paired_side_collision_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)

    def test_paired_side_interior_action_excludes_target(self):
        x = np.linspace(0.0, 1.0, 128, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = paired_side_collision_particle_law_1d(line)
        altered, _ = paired_side_collision_particle_law_1d(changed)
        np.testing.assert_array_equal(
            law["prediction"][64, :16], altered["prediction"][64, :16])
        np.testing.assert_array_equal(
            law["total_action"][64, :16], altered["total_action"][64, :16])
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_nested_midpoint_lineage_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, _diagnostic = nested_midpoint_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        forms, _ = nested_midpoint_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)

    def test_nested_midpoint_interior_is_affine_exact_and_excludes_target(self):
        line = np.linspace(0.2, 0.8, 128)
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = nested_midpoint_particle_law_1d(line)
        altered, _ = nested_midpoint_particle_law_1d(changed)
        np.testing.assert_allclose(
            law["prediction"][64, :16], line[64], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            law["total_action"][64, :16], 0.0, atol=2e-15, rtol=0.0)
        np.testing.assert_array_equal(
            law["prediction"][64, :16], altered["prediction"][64, :16])
        np.testing.assert_array_equal(
            law["total_action"][64, :16], altered["total_action"][64, :16])
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_coupled_phase_transport_is_affine_exact_and_contracts_action(self):
        line = np.linspace(0.2, 0.8, 48)
        forms, diagnostic = nested_midpoint_lineage_readout_forms(line)
        for name in (
            "coupled_phase_mean",
            "coupled_phase_median",
            "coupled_phase_collision_mean",
            "coupled_phase_collision_median",
            "global_characteristic_section",
            "posterior_characteristic_section",
            "path_collision_mean",
            "path_collision_median",
            "path_affinity_mean",
            "path_affinity_median",
            "path_fidelity_mean",
            "path_fidelity_median",
            "transport_fidelity_mean",
            "transport_fidelity_median",
            "transport_plan_history_mean",
            "self_consistent_transport_mean",
            "distributed_transport_mean",
            "action_contracting_transport_mean",
            "two_history_action_transport_mean",
        ):
            # Reflection is only a numerical boundary closure.  On the
            # physical interior the paired transport section remains affine
            # exact; no claim is made for the reflected boundary chart.
            np.testing.assert_allclose(
                forms[name][8:-8], line[8:-8], atol=6e-12, rtol=0.0)

        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        observation = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        _forms, diagnostic = nested_midpoint_lineage_readout_forms(observation)
        coupling = diagnostic["coupled_transport_posterior"]
        self.assertLessEqual(
            coupling["mean_contracted_phase_action"],
            coupling["mean_baseline_phase_action"] + 2e-14,
        )
        self.assertLessEqual(coupling["maximum_action_increase"], 2e-14)
        coverage = diagnostic["coupled_transport_coverage"]
        self.assertLessEqual(coverage["maximum_coverage_deficit"], 2e-14)
        phase_coverage = diagnostic["coupled_transport_phase_coverage"]
        self.assertLessEqual(
            phase_coverage["maximum_coverage_deficit"], 2e-14)

    def test_global_characteristic_is_one_valid_traceback(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        observation = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = nested_midpoint_lineage_transport_1d(observation)
        branch = law["hj_viterbi_branch"]
        predecessor = law["hj_viterbi_predecessor"]
        for index in range(1, observation.size):
            self.assertEqual(
                branch[index - 1], predecessor[index, branch[index]])
        np.testing.assert_array_equal(
            law["hj_viterbi_section"],
            law["prediction"][np.arange(observation.size), branch],
        )
        self.assertEqual(
            diagnostic["global_characteristic_section"]["physical_parameters"],
            "none",
        )
        posterior_branch = law["posterior_characteristic_branch"]
        posterior_predecessor = law["posterior_characteristic_predecessor"]
        for index in range(1, observation.size):
            self.assertEqual(
                posterior_branch[index - 1],
                posterior_predecessor[index, posterior_branch[index]],
            )
        np.testing.assert_array_equal(
            law["posterior_characteristic_section"],
            law["prediction"][np.arange(observation.size), posterior_branch],
        )

    def test_nested_energy_root_authority_is_a_probability(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        forms, diagnostic = nested_midpoint_lineage_readout_forms(line)
        authority = diagnostic["energy_root_participation"]
        self.assertGreaterEqual(authority["minimum_authority"], 0.0)
        self.assertLessEqual(authority["maximum_authority"], 1.0)
        self.assertFalse(authority["root_enters_context_action"])
        for name in (
            "energy_root_mean",
            "energy_root_median",
            "energy_root_collision_mean",
            "energy_root_quantile",
            "transport_energy_root_quantile",
            "hilbert_value_jet_mean",
            "hilbert_value_jet_collision",
            "phase_sasaki_mean",
            "phase_sasaki_collision",
            "phase_sasaki_energy_root_mean",
            "phase_sasaki_energy_root_collision",
            "phase_collision_mean",
            "phase_collision_median",
        ):
            self.assertTrue(np.all(np.isfinite(forms[name])))
        phase_order = diagnostic["phase_collision_order"]
        self.assertGreaterEqual(phase_order["minimum"], 1.0)
        self.assertLessEqual(phase_order["maximum"], 2.0)
        for key in ("hilbert_value_jet_mean", "hilbert_value_jet_collision"):
            self.assertLessEqual(
                diagnostic[key]["section_energy"],
                diagnostic[key]["initial_mean_energy"] + 2e-15,
            )
        for key in ("phase_sasaki_mean", "phase_sasaki_collision"):
            self.assertLessEqual(
                diagnostic[key]["section_action"],
                diagnostic[key]["initial_action"] + 2e-13,
            )
            self.assertEqual(diagnostic[key]["continuation"], "none")
            self.assertGreaterEqual(
                diagnostic[key]["minimum_root_authority"], 0.0)
            self.assertLessEqual(
                diagnostic[key]["maximum_root_authority"], 1.0)

    def test_root_context_collision_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, diagnostic = root_context_collision_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(law["reference_mass"], axis=1),
            1.0,
            atol=2e-15,
            rtol=0.0,
        )
        forms, _ = root_context_collision_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)
        self.assertIn("two-source", diagnostic["particle_law"])

    def test_root_context_simplex_measure_follows_source_multiplicity(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, _diagnostic = root_context_collision_particle_law_1d(line)
        scales = line.size // 2
        np.testing.assert_allclose(
            np.sum(law["reference_mass"][:, :scales], axis=1),
            2.0 / 3.0,
            atol=2e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.sum(law["reference_mass"][:, scales:], axis=1),
            1.0 / 3.0,
            atol=2e-15,
            rtol=0.0,
        )
        self.assertTrue(np.all(
            law["total_action"][:, scales:]
            >= law["total_action"][:, :scales]))

    def test_effective_ancestry_geodesic_is_continuous_probability_law(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        forms, diagnostic = root_context_collision_lineage_readout_forms(line)
        ancestry = diagnostic["effective_ancestry"]
        self.assertGreaterEqual(ancestry["minimum_overlap"], 0.0)
        self.assertLessEqual(ancestry["maximum_overlap"], 1.0 + 2e-15)
        self.assertLessEqual(ancestry["maximum_mass_error"], 2e-15)
        self.assertFalse(ancestry["root_enters_scalar_particle"])
        for name in (
            "ancestry_geodesic_mean",
            "ancestry_geodesic_median",
            "ancestry_geodesic_simplex_mean",
        ):
            self.assertTrue(np.all(np.isfinite(forms[name])))

    def test_independent_side_collision_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, diagnostic = independent_side_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        forms, readout_diagnostic = independent_side_collision_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)
        self.assertTrue(diagnostic["preserve_branch_role"])
        self.assertLessEqual(
            readout_diagnostic["terminal_collision"]["maximum_mass_error"],
            2e-15,
        )

    def test_independent_side_action_is_affine_exact_and_target_free(self):
        line = np.linspace(0.2, 0.8, 128)
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = independent_side_particle_law_1d(line)
        altered, _ = independent_side_particle_law_1d(changed)
        scales = line.size // 4
        branches = np.r_[np.arange(8), scales + np.arange(8)]
        np.testing.assert_allclose(
            law["prediction"][64, branches], line[64], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            law["total_action"][64, branches], 0.0, atol=2e-15, rtol=0.0)
        np.testing.assert_array_equal(
            law["prediction"][64, branches],
            altered["prediction"][64, branches],
        )
        np.testing.assert_array_equal(
            law["total_action"][64, branches],
            altered["total_action"][64, branches],
        )
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_symmetric_second_jet_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, _diagnostic = symmetric_second_jet_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        forms, _ = symmetric_second_jet_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)

    def test_symmetric_second_jet_is_quadratic_exact_and_target_free(self):
        x = np.linspace(-1.0, 1.0, 128)
        line = 0.4 + 0.1 * x + 0.15 * x * x
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = symmetric_second_jet_particle_law_1d(line)
        altered, _ = symmetric_second_jet_particle_law_1d(changed)
        np.testing.assert_allclose(
            law["prediction"][64, :8], line[64], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            law["total_action"][64, :8], 0.0, atol=2e-15, rtol=0.0)
        np.testing.assert_array_equal(
            law["prediction"][64, :8], altered["prediction"][64, :8])
        np.testing.assert_array_equal(
            law["total_action"][64, :8], altered["total_action"][64, :8])
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_second_order_bundle_is_conservative_and_determinant_one(self):
        x = np.linspace(0.0, 1.0, 48, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = symmetric_second_jet_curvature_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(law["edge_precision"].shape, (47, 4, 4))
        np.testing.assert_allclose(
            np.linalg.det(law["edge_precision"]),
            1.0,
            atol=5e-4,
            rtol=5e-4,
        )
        self.assertEqual(
            diagnostic["bundle_metric"], "joint_information_curvature")

    def test_second_order_bundle_readouts_are_constant_exact(self):
        line = np.full(48, 0.37)
        forms, _ = symmetric_second_jet_curvature_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)

    def test_curvature_consensus_is_constant_exact_and_conservative(self):
        line = np.full(48, 0.37)
        law, _ = curvature_consensus_lineage_transport_1d(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        forms, _ = curvature_consensus_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)

    def test_continuous_curvature_simplex_contains_quadratic_section(self):
        x = np.linspace(-1.0, 1.0, 128)
        line = 0.4 + 0.1 * x + 0.15 * x * x
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = continuous_curvature_particle_law_1d(
            line, curvature_intervals=4)
        altered, _ = continuous_curvature_particle_law_1d(
            changed, curvature_intervals=4)
        exact = np.flatnonzero(law["curvature_coordinate"] == 1.0)[:8]
        np.testing.assert_allclose(
            law["prediction"][64, exact], line[64], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            law["local_action"][64, exact], 0.0, atol=2e-15, rtol=0.0)
        np.testing.assert_array_equal(
            law["prediction"][64, exact], altered["prediction"][64, exact])
        np.testing.assert_array_equal(
            law["local_action"][64, exact], altered["local_action"][64, exact])
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_continuous_curvature_readouts_are_constant_exact(self):
        line = np.full(48, 0.37)
        forms, diagnostic = continuous_curvature_lineage_readout_forms(line)
        for value in forms.values():
            np.testing.assert_allclose(value, line, atol=2e-15, rtol=0.0)
        self.assertEqual(
            diagnostic["transported_curvature_coordinate"][
                "representation_intervals"],
            4,
        )

    def test_crossfit_action_excludes_its_interior_target(self):
        x = np.linspace(0.0, 1.0, 128, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        changed = line.copy()
        changed[64] += 0.4
        law, diagnostic = causal_crossfit_particle_law(line)
        altered, _ = causal_crossfit_particle_law(changed)
        lag = 7
        branches = slice(3 * (lag - 1), 3 * lag)
        np.testing.assert_array_equal(
            law["prediction"][64, branches],
            altered["prediction"][64, branches],
        )
        np.testing.assert_allclose(
            law["total_action"][64, branches],
            altered["total_action"][64, branches],
            atol=2e-15,
            rtol=0.0,
        )
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_causal_collision_law_conserves_probability(self):
        x = np.linspace(0.0, 1.0, 70, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = causal_collision_particle_law(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(law["prediction"].shape, (70, 105))
        self.assertFalse(diagnostic["target_value_enters_interior_action"])

    def test_causal_collision_is_exact_on_constant(self):
        line = np.full(64, 0.37)
        result, diagnostic = denoise_causal_collision_transport(line)
        np.testing.assert_array_equal(result, line)
        self.assertFalse(
            diagnostic["collision_law"]["target_value_enters_interior_action"])

    def test_particle_law_conserves_probability(self):
        x = np.linspace(0.0, 1.0, 70, endpoint=False)
        line = 0.3 + 0.1 * x + 0.03 * np.sin(14.0 * np.pi * x)
        law, diagnostic = relation_scale_particle_law(line)
        np.testing.assert_allclose(
            np.sum(law["mass"], axis=1), 1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(law["prediction"].shape, (70, 105))
        self.assertEqual(diagnostic["characteristic_count"], 105)

    def test_particle_continuation_is_exact_on_constant(self):
        line = np.full(64, 0.37)
        result, diagnostic = denoise_cross_predictive_particle_transport(line)
        np.testing.assert_array_equal(result, line)
        self.assertFalse(diagnostic["continuation_ceiling_hit"])
    def test_constant_line_is_an_exact_fixed_point(self):
        line = np.full(64, 0.37)
        result, diagnostics = denoise_cross_predictive_transport(line)
        np.testing.assert_allclose(result, line, atol=0.0, rtol=0.0)
        self.assertFalse(diagnostics["continuation_ceiling_hit"])
        self.assertEqual(diagnostics["accepted_continuations"], 0)

    def test_scale_state_uses_the_complete_topological_interval(self):
        line = np.linspace(0.1, 0.9, 70)
        _result, diagnostics = relation_scale_transport(line)
        self.assertEqual(diagnostics["minimum_lag"], 1)
        self.assertEqual(diagnostics["maximum_lag"], 35)
        self.assertEqual(diagnostics["scale_count"], 35)
        self.assertEqual(diagnostics["characteristics_per_scale"], 3)
        self.assertEqual(diagnostics["characteristic_count"], 105)

    def test_every_accepted_continuation_decreases_residual_action(self):
        truth = compose_series(192, PRESETS["oscillatory composite"])[1]
        _result, diagnostics = denoise_cross_predictive_transport(truth)
        accepted = [
            record for record in diagnostics["continuations"]
            if record["accepted"]
        ]
        self.assertGreater(len(accepted), 0)
        for record in accepted:
            self.assertLess(
                record["residual_action_after"],
                record["residual_action_before"],
            )
        self.assertFalse(diagnostics["continuation_ceiling_hit"])

    def test_mixed_replacement_error_falls_without_truth_runtime_input(self):
        truth = compose_series(256, PRESETS["mixed transport stress"])[1]
        observation = corrupt(
            truth,
            "mixed replacement + uniform",
            amount=0.15,
            density=0.25,
            seed=7401,
        )
        result, diagnostics = denoise_cross_predictive_transport(observation)
        observed_mse = float(np.mean((observation - truth) ** 2))
        result_mse = float(np.mean((result - truth) ** 2))
        self.assertLess(result_mse, 0.2 * observed_mse)
        self.assertFalse(diagnostics["continuation_ceiling_hit"])

    def test_clean_oscillatory_structure_survives_equilibrium(self):
        truth = compose_series(256, PRESETS["oscillatory composite"])[1]
        result, diagnostics = denoise_cross_predictive_transport(truth)
        truth_tv = float(np.sum(np.abs(np.diff(truth))))
        result_tv = float(np.sum(np.abs(np.diff(result))))
        self.assertLess(float(np.mean((result - truth) ** 2)), 2.0e-4)
        self.assertLess(
            float(np.mean((np.diff(result) - np.diff(truth)) ** 2)), 3.0e-4)
        self.assertGreater(result_tv / truth_tv, 0.80)
        self.assertFalse(diagnostics["continuation_ceiling_hit"])

    def test_continuation_ceiling_is_exposed_as_unresolved(self):
        truth = compose_series(128, PRESETS["chirp packet"])[1]
        _result, diagnostics = denoise_cross_predictive_transport(
            truth, CrossPredictiveResolution(maximum_continuations=1))
        self.assertTrue(diagnostics["continuation_ceiling_hit"])
        self.assertIn("unresolved", diagnostics["status"])


if __name__ == "__main__":
    unittest.main()
