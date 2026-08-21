"""Invariants for the deliberately minimal 2-D characteristic lift."""

from __future__ import annotations

import unittest

import numpy as np

from .cross_predictive_transport_2d import (
    CrossPredictive2DResolution,
    denoise_cross_predictive_transport_2d,
    heldout_relation_characteristic_measure_2d,
    relation_characteristic_measure_2d,
    relation_characteristic_population_2d,
    relation_transport_metric_2d,
)


class CrossPredictiveTransport2DTests(unittest.TestCase):
    def test_constant_field_is_exact(self):
        field = np.full((24, 32), 0.41)
        result, diagnostics = denoise_cross_predictive_transport_2d(field)
        np.testing.assert_allclose(result, field, atol=0.0, rtol=0.0)
        self.assertFalse(diagnostics["continuation_ceiling_hit"])

    def test_complete_primitive_scale_population_is_reported(self):
        field = np.arange(24 * 32, dtype=np.float64).reshape(24, 32)
        _result, diagnostics = relation_characteristic_measure_2d(field)
        expected = 3 * (16 + 12 + 12 + 12)
        self.assertEqual(diagnostics["direction_count"], 4)
        self.assertEqual(diagnostics["characteristic_count"], expected)

    def test_accepted_continuations_descend_action(self):
        yy, xx = np.mgrid[:32, :40]
        field = 0.3 + 0.2 * np.sin(xx / 3.0) + 0.1 * (yy > 16)
        _result, diagnostics = denoise_cross_predictive_transport_2d(field)
        for record in diagnostics["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"],
                )

    def test_ceiling_is_a_visible_failed_resolution(self):
        rng = np.random.default_rng(99)
        field = rng.random((20, 24))
        _result, diagnostics = denoise_cross_predictive_transport_2d(
            field, CrossPredictive2DResolution(maximum_continuations=1))
        self.assertTrue(diagnostics["continuation_ceiling_hit"])
        self.assertIn("rejected", diagnostics["status"])

    def test_relation_metric_has_unit_determinant(self):
        yy, xx = np.mgrid[:24, :28]
        image = 0.4 + 0.2 * np.sin(0.3 * xx + 0.17 * yy)
        geometry = relation_transport_metric_2d(image)
        np.testing.assert_allclose(
            geometry["metric_determinant"], 1.0,
            rtol=2e-13, atol=2e-13)

    def test_constant_relation_metric_is_identity(self):
        geometry = relation_transport_metric_2d(np.full((12, 14), 0.37))
        np.testing.assert_allclose(geometry["metric_xx"], 1.0)
        np.testing.assert_allclose(geometry["metric_xy"], 0.0)
        np.testing.assert_allclose(geometry["metric_yy"], 1.0)

    def test_population_barycenter_is_the_characteristic_measure(self):
        yy, xx = np.mgrid[:16, :18]
        field = 0.4 + 0.2 * np.sin(xx / 2.0) + 0.1 * (yy > 8)
        readout, _diagnostic = relation_characteristic_measure_2d(field)
        population, _population_diagnostic = (
            relation_characteristic_population_2d(field))
        np.testing.assert_allclose(np.sum(population["mass"], axis=-1), 1.0)
        np.testing.assert_allclose(
            np.sum(population["mass"] * population["prediction"], axis=-1),
            readout,
            rtol=3e-15,
            atol=3e-15,
        )

    def test_heldout_action_does_not_read_target_prediction_error(self):
        rng = np.random.default_rng(712)
        field = rng.random((16, 18))
        heldout, diagnostic = heldout_relation_characteristic_measure_2d(field)
        self.assertTrue(diagnostic["target_validation_excluded"])
        self.assertTrue(np.all(np.isfinite(heldout)))


if __name__ == "__main__":
    unittest.main()
