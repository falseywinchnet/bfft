from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .kernels import line_kernel, translated_kernel
from .multicapture_transport import deblur_multicapture_consensus
from .synthetic import degrade


class QuarticShapeTransportTests(unittest.TestCase):
    def test_positive_quartic_shape_improves_line_exposure_consensus(self) -> None:
        truth = sources(64)["cameraman"]
        specifications = (
            (1.0, 0.0, (-1.5, -0.5)),
            (9.0, 0.0, (1.5, 0.5)),
            (9.0, 90.0, (0.5, -1.5)),
            (7.0, 45.0, (-0.5, 1.5)),
        )
        observations = tuple(degrade(
            truth,
            translated_kernel(line_kernel(length, angle), shift),
            boundary="reflect",
            clip=False,
        ) for length, angle, shift in specifications)
        covariance_only = deblur_multicapture_consensus(
            observations, passes=24)
        quartic = deblur_multicapture_consensus(
            observations, passes=24, quartic_shape=True)
        covariance_error = float(np.mean(
            (covariance_only.image - truth) ** 2))
        quartic_error = float(np.mean((quartic.image - truth) ** 2))
        self.assertLess(quartic_error, 0.995 * covariance_error)
        record = quartic.diagnostics["quartic_shape_transport"]
        self.assertGreater(record["shape_authority"], 0.05)
        self.assertLess(
            record["fitted_relative_log_magnitude_rms"],
            record["baseline_relative_log_magnitude_rms"],
        )
        self.assertEqual(
            record["capture_role"],
            "all_capture_axis_measures_remain_positive_no_family_selection",
        )
        for field, expected_covariance in zip(
            quartic.fields, quartic.diagnostics["frame_covariances"]
        ):
            self.assertGreaterEqual(float(np.min(field.weights)), 0.0)
            np.testing.assert_allclose(
                np.sum(field.weights, axis=0), 1.0, atol=1e-14)
            centered = field.centered_displacements_xy
            measured_covariance = np.mean(np.einsum(
                "khw,khwi,khwj->hwij",
                field.weights,
                centered,
                centered,
            ), axis=(0, 1))
            np.testing.assert_allclose(
                measured_covariance, expected_covariance,
                atol=2e-10, rtol=2e-10)


if __name__ == "__main__":
    unittest.main()
