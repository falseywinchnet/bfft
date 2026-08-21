from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .kernels import line_kernel
from .relative_mixing_transport import (
    estimate_adaptive_relative_mixing_from_spectra,
    estimate_relative_mixing_from_spectra,
    estimate_relative_mixing_transport,
    prepare_mixing_magnitude_spectrum,
)
from .synthetic import degrade


class RelativeMixingTransportTests(unittest.TestCase):
    def test_cumulant_radius_contracts_for_large_differential_extent(self) -> None:
        truth = sources(96)["cameraman"]
        first = degrade(
            truth, line_kernel(3.0, 25.0), boundary="reflect", clip=False)
        second = degrade(
            truth, line_kernel(13.0, 25.0), boundary="reflect", clip=False)
        estimate = estimate_adaptive_relative_mixing_from_spectra(
            prepare_mixing_magnitude_spectrum(first),
            prepare_mixing_magnitude_spectrum(second),
        )
        self.assertLess(
            estimate.diagnostics[
                "adaptive_maximum_frequency_cycles_per_pixel"],
            0.16,
        )
        self.assertEqual(
            estimate.diagnostics["frequency_radius_method"],
            "continuous_second_cumulant_validity_transport",
        )

    def test_prepared_spectrum_is_exactly_the_direct_estimator(self) -> None:
        truth = sources(64)["cameraman"]
        first = degrade(
            truth, line_kernel(3.0, 18.0), boundary="reflect", clip=False)
        second = degrade(
            truth, line_kernel(8.0, 67.0), boundary="reflect", clip=False)
        direct = estimate_relative_mixing_transport(first, second)
        prepared = estimate_relative_mixing_from_spectra(
            prepare_mixing_magnitude_spectrum(first),
            prepare_mixing_magnitude_spectrum(second),
        )
        np.testing.assert_array_equal(
            prepared.covariance_difference_second_minus_first,
            direct.covariance_difference_second_minus_first,
        )
        self.assertEqual(prepared.authority, direct.authority)

    def test_differential_line_blur_recovers_covariance_axis(self) -> None:
        truth = sources(96)["cameraman"]
        first = degrade(
            truth, line_kernel(3.0, 25.0), boundary="reflect", clip=False)
        second = degrade(
            truth, line_kernel(9.0, 25.0), boundary="reflect", clip=False)
        estimate = estimate_relative_mixing_transport(first, second)
        eigenvalues, eigenvectors = np.linalg.eigh(
            estimate.covariance_difference_second_minus_first)
        direction = eigenvectors[:, -1]
        expected = np.asarray((np.cos(np.deg2rad(25.0)),
                               np.sin(np.deg2rad(25.0))))
        self.assertGreater(abs(float(direction @ expected)), 0.98)
        self.assertGreater(eigenvalues[-1], 4.0)
        self.assertGreater(estimate.authority, 0.8)
        self.assertLess(np.trace(estimate.frame_covariances[0]), 0.2)
        self.assertGreater(np.trace(estimate.frame_covariances[1]), 4.0)

    def test_swap_is_exactly_exchange_symmetric(self) -> None:
        truth = sources(96)["cameraman"]
        first = degrade(
            truth, line_kernel(3.0, 70.0), boundary="reflect", clip=False)
        second = degrade(
            truth, line_kernel(7.0, 20.0), boundary="reflect", clip=False)
        forward = estimate_relative_mixing_transport(first, second)
        reverse = estimate_relative_mixing_transport(second, first)
        np.testing.assert_allclose(
            reverse.covariance_difference_second_minus_first,
            -forward.covariance_difference_second_minus_first,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            reverse.frame_covariances[0], forward.frame_covariances[1],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            reverse.frame_covariances[1], forward.frame_covariances[0],
            atol=1e-10,
        )
        self.assertAlmostEqual(reverse.authority, forward.authority, places=12)

    def test_identical_observations_have_zero_action(self) -> None:
        image = sources(64)["cameraman"]
        estimate = estimate_relative_mixing_transport(image, image)
        self.assertEqual(estimate.authority, 0.0)
        np.testing.assert_allclose(estimate.frame_covariances, 0.0, atol=1e-12)
        self.assertEqual([len(item) for item in estimate.residual_weights], [1, 1])


if __name__ == "__main__":
    unittest.main()
