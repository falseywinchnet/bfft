from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .kernels import line_kernel, translated_kernel
from .multicapture_transport import (
    _spatial_positive_sigma_measure,
    deblur_multicapture_consensus,
)
from .spatial_transport import (
    SpatialExposureField,
    SpatialReflectedExposureOperator,
)
from .synthetic import degrade


class MultiCaptureTransportTests(unittest.TestCase):
    def _capture_set(self) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        truth = sources(64)["cameraman"]
        specifications = (
            (1.0, 0.0, (-1.5, -0.5)),
            (7.0, 0.0, (1.5, 0.5)),
            (7.0, 90.0, (0.5, -1.5)),
            (5.0, 45.0, (-0.5, 1.5)),
        )
        observations = tuple(degrade(
            truth,
            translated_kernel(line_kernel(length, angle), shift),
            boundary="reflect",
            clip=False,
        ) for length, angle, shift in specifications)
        return truth, observations

    def test_multicapture_positive_graph_improves_exchangeable_average(self) -> None:
        truth, observations = self._capture_set()
        result = deblur_multicapture_consensus(observations, passes=32)
        self.assertEqual(
            result.diagnostics["shared_spectral_preparation"],
            {
                "phase_fft_count": 4,
                "mixing_fft_count": 4,
                "total_observation_fft_count": 8,
                "former_pair_recomputed_fft_count": 24,
            },
        )
        self.assertEqual(result.diagnostics["compact_global_operator_count"], 4)
        self.assertGreater(
            result.diagnostics["compact_spatial_coefficient_bytes_avoided"], 0)
        average = np.mean(observations, axis=0)
        self.assertLess(
            float(np.mean((result.image - truth) ** 2)),
            0.8 * float(np.mean((average - truth) ** 2)),
        )
        self.assertEqual(result.diagnostics["capture_count"], 4)
        self.assertEqual(result.diagnostics["pair_count"], 6)
        self.assertEqual(
            result.diagnostics["capture_role"],
            "all_captures_remain_positive_measures_no_frame_or_family_selection",
        )
        self.assertGreaterEqual(
            result.diagnostics["minimum_frame_covariance_eigenvalue"], -1e-8)

    def test_capture_permutation_preserves_reconstruction(self) -> None:
        _, observations = self._capture_set()
        forward = deblur_multicapture_consensus(observations, passes=16)
        permutation = (2, 0, 3, 1)
        permuted = deblur_multicapture_consensus(
            tuple(observations[index] for index in permutation), passes=16)
        np.testing.assert_allclose(forward.image, permuted.image, atol=2e-5)

    def test_identical_capture_measure_is_exact_null_action(self) -> None:
        image = sources(48)["cameraman"]
        result = deblur_multicapture_consensus((image, image, image), passes=8)
        np.testing.assert_array_equal(result.image, image)
        np.testing.assert_allclose(
            result.diagnostics["frame_covariances"], 0.0, atol=1e-8)

    def test_optimal_positive_line_descends_faster_than_unit_transport(self) -> None:
        truth, observations = self._capture_set()
        unit = deblur_multicapture_consensus(
            observations, passes=8, descent_method="multiplicative")
        optimal = deblur_multicapture_consensus(
            observations, passes=8, descent_method="optimal_positive_line")
        self.assertLess(
            float(np.mean((optimal.image - truth) ** 2)),
            float(np.mean((unit.image - truth) ** 2)),
        )
        trace = np.asarray(optimal.diagnostics["residual_trace"])
        self.assertTrue(np.all(np.diff(trace) <= 1e-12))
        self.assertTrue(any(
            abs(step - 1.0) > 1e-3
            for step in optimal.diagnostics["optimal_step_trace"]
        ))

    def test_local_covariance_atlas_improves_spatially_varying_mixing(
        self,
    ) -> None:
        truth = sources(64)["cameraman"]
        height, width = truth.shape
        coordinate = np.linspace(0.0, 1.0, width)[None, :, None, None]
        horizontal = np.asarray(((5.0, 0.0), (0.0, 0.25)))
        vertical = np.asarray(((0.25, 0.0), (0.0, 5.0)))
        specifications = (
            (horizontal, vertical, 1.0),
            (vertical, horizontal, 0.85),
            (np.asarray(((2.5, 1.5), (1.5, 2.5))),
             np.asarray(((2.5, -1.5), (-1.5, 2.5))), 0.9),
            (0.35 * np.eye(2), 1.2 * np.eye(2), 1.0),
        )
        flow = np.zeros((height, width, 2), dtype=np.float64)
        observations = []
        for index, (left, right, scale) in enumerate(specifications):
            covariance = np.broadcast_to(
                ((1.0 - coordinate) * left + coordinate * right),
                (height, width, 2, 2),
            ).copy() * scale
            points, weights = _spatial_positive_sigma_measure(covariance)
            field = SpatialExposureField.from_barycentric_paths(
                f"spatial_mixing_{index}", flow, points, weights)
            observations.append(
                SpatialReflectedExposureOperator(field).forward(truth))
        global_result = deblur_multicapture_consensus(
            tuple(observations), passes=16)
        local_result = deblur_multicapture_consensus(
            tuple(observations), passes=16,
            mixing_patch_size=32, mixing_stride=16)
        global_error = float(np.mean((global_result.image - truth) ** 2))
        local_error = float(np.mean((local_result.image - truth) ** 2))
        self.assertLess(local_error, 0.8 * global_error)
        atlas = local_result.diagnostics["spatial_mixing_atlas"]
        self.assertEqual(atlas["chart_count"], 25)
        self.assertGreaterEqual(atlas["minimum_covariance_eigenvalue"], -1e-9)
        trace_minimum, trace_maximum = atlas[
            "spatial_covariance_trace_range"]
        self.assertGreater(trace_maximum, 4.0 * trace_minimum)
        self.assertEqual(local_result.diagnostics["compact_global_operator_count"], 0)
        self.assertEqual(
            local_result.diagnostics["generated_covariance_operator_count"], 4)
        self.assertEqual(
            local_result.diagnostics["generated_covariance_storage_bytes"],
            4 * (64 * 64 * 4 * 8 + 2 * 8),
        )


if __name__ == "__main__":
    unittest.main()
