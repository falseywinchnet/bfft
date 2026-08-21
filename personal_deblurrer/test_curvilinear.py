"""Invariants for the curvilinear Eikonal exposure chart."""

from __future__ import annotations

import unittest

import numpy as np

from .curvilinear import (
    ReflectedPathOperator,
    curvilinear_endpoint_inverse,
    fit_curvilinear_exposure_chart,
    refine_curvilinear_exposure,
)
from .decomposition import apply_reflect
from .kernels import curved_path_kernel, line_kernel
from .synthetic import degrade
from .test_uncertainty import _fixture


class CurvilinearExposureTests(unittest.TestCase):
    def test_chart_orders_curve_and_records_nontrivial_jacobian(self) -> None:
        chart = fit_curvilinear_exposure_chart(
            curved_path_kernel(11.0, 30.0, 8.0))
        self.assertTrue(np.all(np.diff(chart.eikonal_coordinate) >= 0.0))
        self.assertGreater(chart.path_length, 10.0)
        self.assertGreater(chart.tangent_turn_degrees, 20.0)
        self.assertGreater(np.max(chart.path_jacobian), 1.01)
        self.assertGreater(chart.curvature_rms, 0.01)

    def test_reflected_forward_matches_psf_and_adjoint_closes(self) -> None:
        rng = np.random.default_rng(12)
        kernel = curved_path_kernel(9.0, 37.0, 6.0)
        chart = fit_curvilinear_exposure_chart(kernel)
        operator = ReflectedPathOperator(chart, (31, 29))
        latent = rng.normal(size=(31, 29))
        probe = rng.normal(size=(31, 29))
        np.testing.assert_allclose(
            operator.forward(latent),
            apply_reflect(latent, kernel),
            atol=2e-15,
            rtol=2e-15,
        )
        left = float(np.vdot(operator.forward(latent), probe).real)
        right = float(np.vdot(latent, operator.adjoint(probe)).real)
        self.assertAlmostEqual(left, right, places=11)

        latent_rgb = rng.normal(size=(31, 29, 3))
        probe_rgb = rng.normal(size=(31, 29, 3))
        np.testing.assert_allclose(
            operator.forward(latent_rgb),
            apply_reflect(latent_rgb, kernel),
            atol=2e-15,
            rtol=2e-15,
        )
        left_rgb = float(np.vdot(
            operator.forward(latent_rgb), probe_rgb).real)
        right_rgb = float(np.vdot(
            latent_rgb, operator.adjoint(probe_rgb)).real)
        self.assertAlmostEqual(left_rgb, right_rgb, places=10)

    def test_forward_preserves_dc_and_normalized_adjoint_fixes_dc(self) -> None:
        kernel = curved_path_kernel(11.0, 45.0, 8.0)
        operator = ReflectedPathOperator(
            fit_curvilinear_exposure_chart(kernel), (32, 30))
        constant = np.full((32, 30), 0.37)
        np.testing.assert_allclose(operator.forward(constant), constant, atol=1e-14)
        normalization = operator.adjoint_normalization()
        np.testing.assert_allclose(
            operator.adjoint(np.ones_like(constant)) / normalization,
            1.0,
            atol=1e-14,
        )

    def test_endpoint_seed_law_transports_uncertainty_without_mutation(self) -> None:
        truth = _fixture(48)
        kernel = curved_path_kernel(11.0, 30.0, 8.0)
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        before = observation.copy()
        result = curvilinear_endpoint_inverse(
            observation, kernel, passes=4)
        np.testing.assert_array_equal(observation, before)
        self.assertEqual(result.image.shape, truth.shape)
        self.assertEqual(result.uncertainty.shape, truth.shape)
        self.assertTrue(np.all(np.isfinite(result.image)))
        self.assertGreater(
            result.diagnostics["endpoint_disagreement_rms"], 0.0)
        self.assertAlmostEqual(
            sum(result.diagnostics["endpoint_weights"]), 1.0, places=12)

    def test_straight_path_has_unit_jacobian(self) -> None:
        chart = fit_curvilinear_exposure_chart(line_kernel(11.0, 30.0))
        self.assertLess(chart.tangent_turn_degrees, 1e-8)
        np.testing.assert_allclose(chart.path_jacobian, 1.0, atol=1e-8)

    def test_exact_refinement_improves_a_conservative_curve_basin(self) -> None:
        truth = _fixture(64)
        kernel = curved_path_kernel(11.0, 45.0, 8.0)
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        initial = observation.copy()
        result = refine_curvilinear_exposure(
            observation,
            kernel,
            initial,
            passes=4,
            endpoint_passes=2,
        )
        self.assertLess(
            np.mean((result.image - truth) ** 2),
            np.mean((initial - truth) ** 2),
        )
        self.assertEqual(
            result.diagnostics["operator_role"],
            "exact_ordered_path_gather_with_matched_reflect_scatter",
        )
        self.assertGreater(
            result.diagnostics["endpoint_seed_basin_rms"],
            result.diagnostics["endpoint_disagreement_rms"],
        )

    def test_signed_curvature_is_preserved(self) -> None:
        positive = fit_curvilinear_exposure_chart(
            curved_path_kernel(11.0, 30.0, 8.0))
        negative = fit_curvilinear_exposure_chart(
            curved_path_kernel(11.0, 30.0, -8.0))
        self.assertGreater(
            positive.quadratic_coefficients[0]
            * negative.quadratic_coefficients[0],
            -1.0,
        )
        self.assertLess(
            positive.quadratic_coefficients[0]
            * negative.quadratic_coefficients[0],
            0.0,
        )
        self.assertAlmostEqual(
            positive.path_length, negative.path_length, places=10)


if __name__ == "__main__":
    unittest.main()
