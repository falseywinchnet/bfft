from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .full_quartic_transport import (
    _covariance_square_root,
    directional_quartic_dictionary,
    estimate_full_quartic_transport,
)
from .multicapture_transport import _positive_sigma_measure
from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


class FullQuarticTransportTests(unittest.TestCase):
    def test_zero_quartic_gauge_is_exact_covariance_measure(self) -> None:
        covariance = np.asarray(((4.0, 1.25), (1.25, 2.0)))
        dictionary, _, _ = directional_quartic_dictionary(8)
        standard_points, standard_weights = dictionary[0]
        transported_points = standard_points @ _covariance_square_root(
            covariance).T
        covariance_points, covariance_weights = _positive_sigma_measure(
            covariance)
        np.testing.assert_allclose(
            transported_points, covariance_points, atol=2e-15)
        np.testing.assert_allclose(
            standard_weights, covariance_weights, atol=2e-15)

    def test_directional_dictionary_spans_full_symmetric_tensor(self) -> None:
        measures, tensors, _ = directional_quartic_dictionary(5)
        self.assertEqual(np.linalg.matrix_rank(tensors, tol=1e-10), 5)
        for points, weights in measures:
            self.assertGreaterEqual(float(np.min(weights)), 0.0)
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
            np.testing.assert_allclose(
                np.sum(weights[:, None] * points, axis=0), 0.0,
                atol=2e-15)
            np.testing.assert_allclose(
                np.einsum("k,ki,kj->ij", weights, points, points),
                np.eye(2),
                atol=2e-14,
            )

    def test_covariance_null_is_continuously_suppressed(self) -> None:
        truth = sources(64)["cameraman"]
        observations = []
        covariances = []
        for index in range(4):
            angle = np.deg2rad(17.0 * index)
            cosine = np.cos(angle)
            sine = np.sin(angle)
            rotation = np.asarray(((cosine, -sine), (sine, cosine)))
            covariance = rotation @ np.diag(
                (1.2 + 0.2 * index, 5.0 + index)) @ rotation.T
            covariances.append(covariance)
            points, weights = _positive_sigma_measure(covariance)
            field = SpatialExposureField.from_barycentric_paths(
                f"covariance_null_{index}",
                np.zeros((*truth.shape, 2)),
                points,
                weights,
                compact_global=True,
            )
            observations.append(
                SpatialReflectedExposureOperator(field).forward(truth))
        estimate = estimate_full_quartic_transport(
            observations, np.stack(covariances), maximum_frequency=0.25)
        self.assertLess(estimate.authority, 1e-3)
        self.assertLess(
            float(np.sqrt(np.sum(estimate.standardized_cumulants ** 2))),
            5e-4,
        )
        self.assertEqual(
            estimate.diagnostics["capture_role"],
            "all_directional_measures_have_positive_mass_no_shape_selection",
        )

    def test_directional_tensor_measure_is_positive_and_improves_consensus(
        self,
    ) -> None:
        truth = sources(96)["cameraman"]
        height, width = truth.shape
        dictionary, _, _ = directional_quartic_dictionary(8)
        mixture_specs = (
            ((0,), (1.0,)),
            ((0, 1), (0.1, 0.9)),
            ((0, 3), (0.1, 0.9)),
            ((0, 10), (0.1, 0.9)),
        )
        observations = []
        covariances = []
        for index, (indices, masses) in enumerate(mixture_specs):
            angle = np.deg2rad(17.0 * index)
            cosine = np.cos(angle)
            sine = np.sin(angle)
            rotation = np.asarray(((cosine, -sine), (sine, cosine)))
            covariance = rotation @ np.diag(
                (1.2 + 0.2 * index, 5.0 + index)) @ rotation.T
            covariances.append(covariance)
            root = _covariance_square_root(covariance)
            points = []
            weights = []
            for component, mass in zip(indices, masses):
                standard_points, standard_weights = dictionary[component]
                points.append(standard_points @ root.T)
                weights.append(mass * standard_weights)
            field = SpatialExposureField.from_barycentric_paths(
                f"directional_truth_{index}",
                np.zeros((height, width, 2)),
                np.concatenate(points),
                np.concatenate(weights),
                compact_global=True,
            )
            observations.append(
                SpatialReflectedExposureOperator(field).forward(truth))
        covariance_array = np.stack(covariances)
        estimate = estimate_full_quartic_transport(
            observations, covariance_array, maximum_frequency=0.14)
        self.assertGreater(estimate.authority, 0.05)
        self.assertLess(
            estimate.diagnostics["fitted_relative_log_magnitude_rms"],
            estimate.diagnostics["baseline_relative_log_magnitude_rms"],
        )
        self.assertEqual(
            estimate.diagnostics["capture_role"],
            "all_directional_measures_have_positive_mass_no_shape_selection",
        )
        self.assertLessEqual(estimate.diagnostics["maximum_atom_count"], 153)
        baseline_fields = []
        tensor_fields = []
        for index, covariance in enumerate(covariance_array):
            points = estimate.residual_displacements[index]
            weights = estimate.residual_weights[index]
            self.assertGreaterEqual(float(np.min(weights)), 0.0)
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
            np.testing.assert_allclose(
                np.sum(weights[:, None] * points, axis=0), 0.0,
                atol=2e-12)
            np.testing.assert_allclose(
                np.einsum("k,ki,kj->ij", weights, points, points),
                covariance,
                atol=2e-12,
            )
            baseline_points, baseline_weights = _positive_sigma_measure(covariance)
            baseline_fields.append(SpatialExposureField.from_barycentric_paths(
                f"baseline_{index}",
                np.zeros((height, width, 2)),
                baseline_points,
                baseline_weights,
                compact_global=True,
            ))
            tensor_fields.append(SpatialExposureField.from_barycentric_paths(
                f"full_tensor_{index}",
                np.zeros((height, width, 2)),
                points,
                weights,
                compact_global=True,
            ))
        baseline = solve_spatial_field_consensus(
            observations, baseline_fields, passes=32,
            descent_method="optimal_positive_line")
        tensor = solve_spatial_field_consensus(
            observations, tensor_fields, passes=32,
            descent_method="optimal_positive_line")
        baseline_error = float(np.mean((baseline.image - truth) ** 2))
        tensor_error = float(np.mean((tensor.image - truth) ** 2))
        self.assertLess(tensor_error, baseline_error)


if __name__ == "__main__":
    unittest.main()
