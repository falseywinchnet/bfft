"""Invariants for the family-free analytic transport basis."""

from __future__ import annotations

import unittest

import numpy as np

from .analytic_support import analyze_transport_support, fourier_eikonal_field
from .decomposition import two_stage_deblur_known
from .kernels import (
    curved_path_kernel,
    disk_kernel,
    wronski_binomial_kernel,
    wronski_separable_kernel,
)
from .synthetic import degrade
from .test_uncertainty import _fixture
from .workbench import BlurSpec


class AnalyticTransportSupportTests(unittest.TestCase):
    def test_wronski_atoms_are_exact_binomial_measures(self) -> None:
        single = wronski_binomial_kernel()
        repeated = wronski_binomial_kernel(stages=2)
        np.testing.assert_allclose(
            single.psf[single.psf.shape[0] // 2, 1:-1],
            (0.25, 0.5, 0.25),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            repeated.psf[repeated.psf.shape[0] // 2, 1:-1],
            np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0,
            atol=1e-15,
        )

    def test_direction_emerges_from_one_measure_without_family_label(self) -> None:
        angle = np.deg2rad(31.0)
        direction = np.asarray((np.cos(angle), np.sin(angle)))
        support = analyze_transport_support(
            wronski_binomial_kernel(direction, stages=2))
        self.assertFalse(support.diagnostics["family_classification"])
        self.assertGreater(
            abs(float(np.dot(support.principal_direction_xy, direction))),
            0.90,
        )
        axis_support = analyze_transport_support(
            wronski_binomial_kernel((1.0, 0.0), stages=2))
        self.assertEqual(axis_support.numerical_dimension, 1)

    def test_center_cloud_and_curve_use_the_same_cumulant_basis(self) -> None:
        cloud = analyze_transport_support(disk_kernel(3.0))
        curve = analyze_transport_support(
            curved_path_kernel(11.0, 20.0, 5.0))
        self.assertEqual(cloud.diagnostics["basis"], curve.diagnostics["basis"])
        self.assertEqual(cloud.numerical_dimension, 2)
        self.assertAlmostEqual(cloud.signed_bend_coupling, 0.0, delta=1e-12)
        self.assertGreater(abs(curve.signed_bend_coupling), 0.01)

    def test_fourier_eikonal_flow_is_exact_and_nulls_abstain(self) -> None:
        kernel = wronski_binomial_kernel()
        field = fourier_eikonal_field(
            kernel,
            np.asarray(((0.20, 0.13), (0.50, 0.00))),
        )
        self.assertTrue(field.supported[0])
        self.assertGreater(field.flow_xy[0, 0], 0.0)
        self.assertAlmostEqual(field.flow_xy[0, 1], 0.0, delta=1e-14)
        self.assertFalse(field.supported[1])
        self.assertTrue(np.isinf(field.attenuation_action[1]))
        np.testing.assert_allclose(field.flow_xy[1], 0.0, atol=0.0)

    def test_separable_composition_has_two_emergent_directions(self) -> None:
        support = analyze_transport_support(wronski_separable_kernel())
        self.assertEqual(support.numerical_dimension, 2)
        np.testing.assert_allclose(
            support.covariance_eigenvalues, (0.5, 0.5), atol=1e-15)

    def test_workbench_synthesizes_wronski_without_solver_classification(self) -> None:
        spec = BlurSpec(kind="Wronski repeated", angle_degrees=0.0)
        kernel = spec.kernel()
        truth = _fixture(64)
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        result = two_stage_deblur_known(
            observation, kernel, passes=12, reference=truth)
        self.assertFalse(result.diagnostics["blur_family_selected"])
        self.assertEqual(
            result.diagnostics["analytic_transport_support"]["basis"],
            "one_positive_displacement_measure",
        )
        self.assertGreater(result.diagnostics["psnr_gain"], 1.0)


if __name__ == "__main__":
    unittest.main()
