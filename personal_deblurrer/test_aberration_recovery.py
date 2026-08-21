"""Blind relative-aberration estimation controls with truth evaluation-only."""

from __future__ import annotations

import unittest

import numpy as np

from .aberration_recovery import (
    covariance_field_matrices,
    recover_affine_aberration_multicapture,
)
from .composed_transport import (
    compose_positive_transports,
    radial_scale_measure,
)
from .observation_anomalies import astigmatic_scale_measure, ghost_measure
from .test_uncertainty import _fixture
from .workbench import DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(mse))


def _operator_sequence(shape):
    parameters = (
        (0.015, 0.010, 0.0),
        (0.030, 0.018, 45.0),
        (0.045, 0.025, 90.0),
        (0.060, 0.015, 135.0),
    )
    result = []
    for radial_extent, astigmatic_extent, angle in parameters:
        radial = radial_scale_measure(
            shape, fractional_extent=radial_extent).to_transport(shape)
        astigmatic = astigmatic_scale_measure(
            shape,
            fractional_extent=astigmatic_extent,
            angle_degrees=angle,
        ).to_transport(shape)
        result.append(compose_positive_transports(radial, astigmatic))
    return tuple(result)


def _relative_tensor_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - np.mean(first, axis=0, keepdims=True)
    second = second - np.mean(second, axis=0, keepdims=True)
    margin = max(int(round(0.12 * first.shape[1])), 4)
    first = first[:, margin:-margin, margin:-margin].ravel()
    second = second[:, margin:-margin, margin:-margin].ravel()
    return float(np.dot(first, second) / np.sqrt(
        np.dot(first, first) * np.dot(second, second)))


class AberrationRecoveryTests(unittest.TestCase):
    def test_workbench_recovers_only_the_explicit_capture_set(self) -> None:
        truth = _fixture(64)
        operators = _operator_sequence(truth.shape[:2])
        session = DeblurSession()
        indices = []
        immutable = []
        for capture, operator in enumerate(operators):
            observation = operator.forward(truth)
            index = session.add_array(observation, f"aberration {capture}")
            session.use_as_is(index)
            indices.append(index)
            immutable.append(observation.copy())
        unrelated = session.add_array(
            np.zeros((31, 29), dtype=np.float64), "unrelated raster")

        target, result = session.recover_relative_aberration(
            indices,
            target_index=indices[2],
            passes=32,
        )

        self.assertEqual(target, indices[2])
        self.assertNotIn(unrelated, result.diagnostics.get(
            "selected_capture_indices", ()))
        for index, original in zip(indices, immutable):
            np.testing.assert_array_equal(
                session.sources[index].observation, original)
        diagnostic = session.sources[target].diagnostics
        self.assertEqual(diagnostic["selected_capture_indices"], indices)
        self.assertEqual(diagnostic["diagnostic_view_count"], 2 * len(indices))
        self.assertEqual(
            len(session.sources[target].diagnostic_views), 2 * len(indices))
        self.assertIn("common", diagnostic["common_aberration_warning"])
        self.assertFalse(diagnostic["common_aberration_identifiable"])
        self.assertFalse(diagnostic["truth_used_for_estimation"])

    def test_relative_affine_aberration_is_recovered_without_truth(self) -> None:
        truth = _fixture(64)
        operators = _operator_sequence(truth.shape[:2])
        observations = tuple(operator.forward(truth) for operator in operators)
        immutable = tuple(item.copy() for item in observations)
        result = recover_affine_aberration_multicapture(
            observations,
            passes=48,
            patch_size=24,
            stride=16,
        )
        for measured, original in zip(observations, immutable):
            np.testing.assert_array_equal(measured, original)
        average = np.mean(observations, axis=0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(average, truth) + 4.0)
        estimated = covariance_field_matrices(result.transport_result.fields)
        actual = np.stack([
            operator.local_moment_jet().covariance for operator in operators
        ])
        correlation = _relative_tensor_correlation(actual, estimated)
        fitted_correlation = _relative_tensor_correlation(
            actual,
            result.aberration_jet.fitted_covariance_fields,
        )
        self.assertGreater(correlation, 0.55)
        self.assertGreater(fitted_correlation, 0.50)
        self.assertGreater(
            result.aberration_jet.diagnostics[
                "crossfit_predictive_authority"],
            0.8,
        )
        self.assertFalse(result.diagnostics["truth_used_for_estimation"])
        self.assertFalse(result.diagnostics["family_classification"])
        self.assertFalse(result.diagnostics["common_aberration_identifiable"])

    def test_sparse_ghost_mass_remains_in_the_same_relative_atlas(self) -> None:
        truth = _fixture(64)
        base = _operator_sequence(truth.shape[:2])
        offsets = ((2.0, -1.0), (-1.5, 2.5), (3.0, 1.0), (-2.0, -2.5))
        operators = tuple(
            compose_positive_transports(
                operator,
                ghost_measure(offset, ghost_mass=0.025 + 0.012 * index)
                .to_transport(truth.shape[:2]),
            )
            for index, (operator, offset) in enumerate(zip(base, offsets))
        )
        observations = tuple(operator.forward(truth) for operator in operators)
        result = recover_affine_aberration_multicapture(
            observations,
            passes=48,
            patch_size=24,
            stride=16,
        )
        average = np.mean(observations, axis=0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(average, truth) + 2.0)
        self.assertEqual(result.diagnostics["capture_count"], 4)
        self.assertEqual(
            result.diagnostics["capture_role"],
            "all_captures_remain_positive_measures_no_frame_or_family_selection",
        )


if __name__ == "__main__":
    unittest.main()
