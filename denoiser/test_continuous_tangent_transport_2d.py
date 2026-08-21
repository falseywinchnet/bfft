"""Invariants for common-scale target-free tangent transport."""

from __future__ import annotations

import unittest

import numpy as np

from .continuous_tangent_transport_2d import (
    _target_free_affine_sample,
    continuous_tangent_joint_measure_2d,
    continuous_tangent_joint_population_2d,
    continuous_tangent_jet_particles_2d,
    continuous_tangent_jet_projection_2d,
    continuous_tangent_proposals_2d,
    continuous_tangent_signal_population_2d,
    denoise_continuous_tangent_lineage_covariance_2d,
    uniform_projective_tangents,
)


class ContinuousTangentTransport2DTests(unittest.TestCase):
    def test_angular_nodes_are_nested_and_have_projective_mass(self):
        first = uniform_projective_tangents(4)
        second = uniform_projective_tangents(8)
        third = uniform_projective_tangents(16)
        self.assertTrue(set(first) < set(second) < set(third))
        self.assertAlmostEqual(np.pi / len(third) * len(third), np.pi)

    def test_offgrid_sample_is_affine_exact_and_target_free(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        value, valid, source, coefficient = _target_free_affine_sample(
            field, 0.7, -1.3)
        truth = 0.2 + 0.01 * (xx - 1.3) + 0.015 * (yy + 0.7)
        np.testing.assert_allclose(value[valid], truth[valid], atol=2e-15, rtol=0.0)
        target = np.arange(field.size).reshape(field.shape + (1,))
        np.testing.assert_array_equal(
            np.where(source == target, coefficient, 0.0), 0.0)
        np.testing.assert_allclose(np.sum(coefficient, axis=-1), 1.0)

    def test_proposals_and_joint_law_exclude_target_and_reproduce_affine(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        for angular_count in (4, 8):
            proposal, diagnostic = continuous_tangent_proposals_2d(
                field, angular_count=angular_count)
            self.assertEqual(diagnostic["maximum_target_self_coefficient"], 0.0)
            np.testing.assert_allclose(
                np.sum(proposal["source_coefficient"], axis=-1), 1.0,
                atol=5e-13, rtol=0.0)
            population, joint_diagnostic = continuous_tangent_joint_population_2d(
                field, angular_count=angular_count)
            np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
            self.assertTrue(joint_diagnostic["target_identity_excluded"])
            for barycenter in ("mean", "median"):
                result, _ = continuous_tangent_joint_measure_2d(
                    field,
                    barycenter=barycenter,
                    angular_count=angular_count,
                )
                np.testing.assert_allclose(result, field, atol=5e-14, rtol=0.0)

    def test_target_change_does_not_change_its_own_proposal_law(self):
        rng = np.random.default_rng(821)
        field = rng.random((18, 22))
        changed = field.copy()
        target = (9, 11)
        changed[target] = 1.0 - changed[target]
        first, _ = continuous_tangent_proposals_2d(field, angular_count=8)
        second, _ = continuous_tangent_proposals_2d(changed, angular_count=8)
        for key in ("prediction", "variation", "scale_conductance"):
            np.testing.assert_allclose(
                first[key][target], second[key][target], atol=0.0, rtol=0.0)

    def test_integrable_jet_projection_reproduces_affine_field(self):
        yy, xx = np.mgrid[:14, :16]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        result, diagnostic = continuous_tangent_jet_projection_2d(
            field, angular_count=8)
        np.testing.assert_allclose(result, field, atol=2e-7, rtol=0.0)
        self.assertLessEqual(abs(diagnostic["jet_projection_mean_error"]), 2e-16)

    def test_full_jet_particles_reproduce_affine_value_and_gradient(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        population, _ = continuous_tangent_signal_population_2d(
            field, angular_count=8)
        particles, diagnostic = continuous_tangent_jet_particles_2d(population)
        valid = particles["mass"] > 0.0
        expected = np.broadcast_to(field[..., None], particles["signal"].shape)
        np.testing.assert_allclose(
            particles["signal"][valid], expected[valid],
            atol=7e-13, rtol=0.0)
        np.testing.assert_allclose(
            particles["gradient_x"][valid], 0.01,
            atol=7e-13, rtol=0.0)
        np.testing.assert_allclose(
            particles["gradient_y"][valid], 0.015,
            atol=7e-13, rtol=0.0)
        self.assertLess(diagnostic["maximum_mass_error"], 2e-15)

    def test_lineage_covariance_continuations_descend_and_exclude_self(self):
        result, diagnostic = denoise_continuous_tangent_lineage_covariance_2d(
            np.random.default_rng(822).random((14, 16)),
            angular_count=8,
            maximum_continuations=3,
        )
        self.assertEqual(result.shape, (14, 16))
        for record in diagnostic["continuations"]:
            self.assertEqual(record["maximum_target_self_lineage"], 0.0)
            self.assertLessEqual(
                record["lineage_row_mass_maximum_error"], 2e-13)
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])


if __name__ == "__main__":
    unittest.main()
