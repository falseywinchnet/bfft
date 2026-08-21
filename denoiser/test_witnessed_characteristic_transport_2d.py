"""Invariants for witnessed dense characteristic transport."""

from __future__ import annotations

import unittest

import numpy as np

from .witnessed_characteristic_transport_2d import (
    _four_colour_leave_one_out_residual_crps,
    _lineage_overlap_uncertainty,
    _lineage_covariance_authority,
    _source_influence_and_lineage,
    dense_characteristic_proposals_2d,
    denoise_joint_characteristic_transport_2d,
    denoise_joint_authority_transport_2d,
    denoise_joint_source_authority_transport_2d,
    denoise_joint_overlap_covariance_transport_2d,
    denoise_joint_lineage_covariance_transport_2d,
    joint_characteristic_measure_2d,
    joint_characteristic_population_2d,
    lineage_joint_characteristic_measure_2d,
    lineage_joint_characteristic_population_2d,
    transported_lineage_joint_characteristic_measure_2d,
    transported_lineage_joint_characteristic_population_2d,
    witnessed_characteristic_measure_2d,
    witnessed_characteristic_population_2d,
)


class WitnessedCharacteristicTransport2DTests(unittest.TestCase):
    def test_constant_and_affine_fields_are_exact(self):
        yy, xx = np.mgrid[:18, :22]
        for field in (
            np.full((18, 22), 0.43),
            0.2 + 0.01 * xx + 0.015 * yy,
        ):
            for barycenter in ("mean", "median"):
                result, _diagnostic = witnessed_characteristic_measure_2d(
                    field, barycenter=barycenter)
                np.testing.assert_allclose(result, field, atol=3e-14, rtol=0.0)
        for angular_order in (2, 3):
            result, diagnostic = joint_characteristic_measure_2d(
                field, barycenter="mean", angular_order=angular_order)
            np.testing.assert_allclose(result, field, atol=4e-14, rtol=0.0)
            self.assertEqual(
                diagnostic["signal_law"]["proposal"][
                    "angular_quadrature_order"],
                angular_order,
            )

    def test_target_change_cannot_change_proposals_or_witnessed_mass(self):
        rng = np.random.default_rng(814)
        field = rng.random((18, 22))
        changed = field.copy()
        target = (9, 11)
        changed[target] = 1.0 - changed[target]
        first_proposal, _ = dense_characteristic_proposals_2d(field)
        second_proposal, _ = dense_characteristic_proposals_2d(changed)
        np.testing.assert_allclose(
            first_proposal["prediction"][target],
            second_proposal["prediction"][target], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            first_proposal["variation"][target],
            second_proposal["variation"][target], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            first_proposal["scale_conductance"][target],
            second_proposal["scale_conductance"][target], atol=0.0, rtol=0.0)
        first, diagnostic = witnessed_characteristic_population_2d(field)
        second, _ = witnessed_characteristic_population_2d(changed)
        np.testing.assert_allclose(
            first["mass"][target], second["mass"][target],
            atol=0.0, rtol=0.0)
        self.assertTrue(diagnostic["target_identity_excluded"])

    def test_population_mass_is_exact(self):
        rng = np.random.default_rng(815)
        population, _ = witnessed_characteristic_population_2d(
            rng.random((18, 22)))
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        self.assertTrue(np.all(population["mass"] >= 0.0))
        self.assertTrue(np.all(population["crps"] >= 0.0))
        self.assertEqual(population["source_identity"].shape[-1], 2)
        np.testing.assert_allclose(
            np.sum(population["mass"][..., None]
                   * population["source_coefficient"], axis=(-2, -1)),
            1.0,
        )

    def test_joint_observation_graph_and_affine_exactness(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        population, diagnostic = joint_characteristic_population_2d(field)
        np.testing.assert_allclose(
            population["signal"] + population["residual"],
            np.broadcast_to(field[..., None], population["signal"].shape),
            atol=2e-16, rtol=0.0)
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        self.assertLessEqual(
            diagnostic["observation_graph_maximum_error"], 2e-16)
        for barycenter in ("mean", "median"):
            result, _ = joint_characteristic_measure_2d(
                field, barycenter=barycenter)
            np.testing.assert_allclose(result, field, atol=3e-14, rtol=0.0)
        population, diagnostic = lineage_joint_characteristic_population_2d(field)
        np.testing.assert_allclose(
            population["signal"] + population["residual"],
            np.broadcast_to(field[..., None], population["signal"].shape),
            atol=2e-16, rtol=0.0)
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        self.assertTrue(diagnostic["target_identity_excluded_from_prior"])
        result, _ = lineage_joint_characteristic_measure_2d(
            field, barycenter="mean")
        np.testing.assert_allclose(result, field, atol=3e-14, rtol=0.0)
        population, diagnostic = (
            transported_lineage_joint_characteristic_population_2d(
                field, maximum_lineage_transports=4)
        )
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        self.assertTrue(diagnostic["target_identity_excluded_from_prior"])
        self.assertFalse(diagnostic["lineage_transport_ceiling_hit"])
        result, _ = transported_lineage_joint_characteristic_measure_2d(
            field, barycenter="mean", maximum_lineage_transports=4)
        np.testing.assert_allclose(result, field, atol=3e-14, rtol=0.0)

    def test_four_colour_residual_prior_excludes_target_identity(self):
        rng = np.random.default_rng(816)
        field = rng.random((18, 22))
        changed = field.copy()
        target = (8, 10)
        changed[target] = 1.0 - changed[target]
        first_witness, _ = witnessed_characteristic_population_2d(field)
        second_witness, _ = witnessed_characteristic_population_2d(changed)
        # The residual-prior implementation consumes only the strict witness;
        # obtain it indirectly from the same constructor inputs.
        from .crossfit_characteristic_transport_2d import (
            crossfit_characteristic_population_2d,
        )
        strict_first, _ = crossfit_characteristic_population_2d(field)
        strict_second, _ = crossfit_characteristic_population_2d(changed)
        fixed_query = np.linspace(-0.7, 0.7, 9)
        query = np.broadcast_to(fixed_query, field.shape + (9,)).copy()
        first_score = _four_colour_leave_one_out_residual_crps(
            field, strict_first, query)
        second_score = _four_colour_leave_one_out_residual_crps(
            changed, strict_second, query)
        np.testing.assert_allclose(
            first_score[target], second_score[target], atol=5e-16, rtol=0.0)
        # Keep these constructed laws live so the test also catches accidental
        # shape drift in the signal proposal path.
        self.assertEqual(first_witness["mass"].shape, second_witness["mass"].shape)

    def test_joint_continuations_strictly_descend_residual_action(self):
        rng = np.random.default_rng(817)
        _result, diagnostic = denoise_joint_characteristic_transport_2d(
            rng.random((16, 18)), maximum_continuations=3)
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])
        _result, diagnostic = denoise_joint_overlap_covariance_transport_2d(
            rng.random((16, 18)), maximum_continuations=3)
        for record in diagnostic["continuations"]:
            self.assertEqual(record["maximum_target_self_influence"], 0.0)
            self.assertEqual(record["maximum_target_self_lineage"], 0.0)
            self.assertLessEqual(
                record["influence_row_mass_maximum_error"], 5e-15)
            self.assertLessEqual(
                record["lineage_row_mass_maximum_error"], 5e-15)
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])

    def test_lineage_overlap_covariance_is_psd_and_counts_overlap(self):
        mass = np.array([
            [[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]],
        ])
        source = np.array([
            [[[0, 1], [0, 2]]],
            [[[0, 1], [1, 2]]],
            [[[0, 2], [1, 2]]],
        ])
        coefficient = np.full(source.shape, 0.5)
        influence, lineage = _source_influence_and_lineage(
            mass, source, coefficient)
        uncertainty, diagnostic = _lineage_overlap_uncertainty(
            influence, lineage, np.array([1.0, -0.5, 0.75]))
        self.assertTrue(np.all(uncertainty >= 0.0))
        np.testing.assert_allclose(np.sum(influence, axis=1), 1.0)
        np.testing.assert_allclose(np.sum(lineage, axis=1), 1.0)
        self.assertGreater(
            diagnostic["mean_overlap_source_uncertainty"], 0.0)
        rng = np.random.default_rng(818)
        _result, diagnostic = denoise_joint_source_authority_transport_2d(
            rng.random((16, 18)), maximum_continuations=3)
        for record in diagnostic["continuations"]:
            self.assertLessEqual(record["maximum_target_self_influence"], 0.0)
            self.assertLessEqual(
                record["source_ancestry_row_mass_maximum_error"], 5e-15)
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])

    def test_lineage_covariance_authority_rejects_chance_and_excludes_self(self):
        lineage = np.array((
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ))
        authority, diagnostic = _lineage_covariance_authority(
            lineage,
            np.array((1.0, -1.0, 1.0)),
            np.array((1.0, 1.0, 1.0)),
        )
        self.assertTrue(np.all((0.0 <= authority) & (authority <= 1.0)))
        self.assertGreaterEqual(diagnostic["mean_lineage_covariance_variance"], 0.0)
        _result, run = denoise_joint_lineage_covariance_transport_2d(
            np.random.default_rng(819).random((16, 18)),
            maximum_continuations=3,
        )
        for record in run["continuations"]:
            self.assertEqual(record["maximum_target_self_lineage"], 0.0)
            self.assertLessEqual(
                record["lineage_row_mass_maximum_error"], 5e-15)
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])
        _result, diagnostic = denoise_joint_authority_transport_2d(
            np.random.default_rng(820).random((16, 18)),
            maximum_continuations=3)
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])


if __name__ == "__main__":
    unittest.main()
