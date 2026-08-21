"""Invariants for strict direction-lane cross-fitting."""

from __future__ import annotations

import unittest

import numpy as np

from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_measure_2d,
    crossfit_characteristic_population_2d,
    denoise_crossfit_characteristic_transport_2d,
    primitive_directions_2d,
    primitive_direction_weights_2d,
)
from .continuous_tangent_transport_2d import continuous_tangent_jet_field_2d


class CrossfitCharacteristicTransport2DTests(unittest.TestCase):
    def test_strict_witness_jet_reproduces_affine_gradient(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        population, _ = crossfit_characteristic_population_2d(field)
        gradient_x, gradient_y, _ = continuous_tangent_jet_field_2d(population)
        np.testing.assert_allclose(gradient_x, 0.01, atol=4e-14, rtol=0.0)
        np.testing.assert_allclose(gradient_y, 0.015, atol=4e-14, rtol=0.0)

    def test_primitive_direction_quadrature_is_complete_and_nested(self):
        first = primitive_directions_2d(1)
        second = primitive_directions_2d(2)
        third = primitive_directions_2d(3)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 8)
        self.assertEqual(len(third), 16)
        self.assertTrue(set(first) < set(second) < set(third))
        for directions in (first, second, third):
            weight = np.asarray(primitive_direction_weights_2d(directions))
            self.assertTrue(np.all(weight > 0.0))
            self.assertAlmostEqual(float(np.sum(weight)), np.pi, places=14)
        np.testing.assert_allclose(
            primitive_direction_weights_2d(first), np.pi / 4.0,
            atol=2e-16, rtol=0.0)

    def test_constant_field_is_exact(self):
        field = np.full((18, 22), 0.43)
        for barycenter in ("mean", "median"):
            result, diagnostic = crossfit_characteristic_measure_2d(
                field, barycenter=barycenter)
            np.testing.assert_allclose(result, field, atol=2e-15, rtol=0.0)
            self.assertTrue(diagnostic["target_identity_excluded"])

    def test_target_change_cannot_change_its_own_predictive_law(self):
        rng = np.random.default_rng(812)
        field = rng.random((18, 22))
        changed = field.copy()
        target = (9, 11)
        changed[target] = 1.0 - changed[target]
        first, _diagnostic = crossfit_characteristic_population_2d(field)
        second, _diagnostic = crossfit_characteristic_population_2d(changed)
        np.testing.assert_allclose(
            first["prediction"][target], second["prediction"][target],
            atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            first["mass"][target], second["mass"][target],
            atol=0.0, rtol=0.0)

        first, _diagnostic = crossfit_characteristic_population_2d(
            field, angular_order=2)
        second, _diagnostic = crossfit_characteristic_population_2d(
            changed, angular_order=2)
        np.testing.assert_allclose(
            first["prediction"][target], second["prediction"][target],
            atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            first["mass"][target], second["mass"][target],
            atol=0.0, rtol=0.0)

    def test_affine_field_is_reproduced_where_population_exists(self):
        yy, xx = np.mgrid[:20, :24]
        field = 0.2 + 0.01 * xx + 0.015 * yy
        result, _diagnostic = crossfit_characteristic_measure_2d(
            field, barycenter="mean")
        np.testing.assert_allclose(result, field, atol=2e-14, rtol=0.0)

    def test_population_mass_is_exact(self):
        yy, xx = np.mgrid[:18, :22]
        field = 0.4 + 0.2 * np.sin(xx / 3.0) + 0.1 * (yy > 9)
        population, _diagnostic = crossfit_characteristic_population_2d(field)
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        self.assertTrue(np.all(population["mass"] >= 0.0))

    def test_accepted_continuations_descend_observation_action(self):
        rng = np.random.default_rng(813)
        field = rng.random((18, 22))
        _result, diagnostic = denoise_crossfit_characteristic_transport_2d(field)
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])


if __name__ == "__main__":
    unittest.main()
