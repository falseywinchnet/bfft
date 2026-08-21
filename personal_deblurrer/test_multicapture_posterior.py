from __future__ import annotations

import unittest

import numpy as np

from denoiser.fmmt_certified import denoise_fmmt
from denoiser.run_2d_denoiser_battery import sources

from .kernels import gaussian_kernel, line_kernel, translated_kernel
from .multicapture_posterior import (
    _denoise_channels,
    solve_multicapture_transport_posterior,
)
from .multicapture_transport import deblur_multicapture_consensus
from .synthetic import degrade


class MultiCapturePosteriorTests(unittest.TestCase):
    def test_parallel_rgb_fmmt_is_exact_serial_representation(self) -> None:
        gray = sources(24)["cameraman"]
        rgb = np.stack((gray, 0.8 * gray + 0.1, gray[::-1]), axis=2)
        parallel = _denoise_channels(rgb)
        serial = np.stack([
            denoise_fmmt(rgb[..., channel])[0]
            for channel in range(3)
        ], axis=2)
        np.testing.assert_array_equal(parallel, serial)

    def test_all_three_measures_remain_continuous_and_finite(self) -> None:
        truth = sources(32)["cameraman"]
        observations = tuple(degrade(
            truth,
            translated_kernel(
                line_kernel(7.0, 30.0 * capture),
                (capture - 1.5, 0.5 * (1.5 - capture))),
            gaussian_sigma=0.01,
            poisson_peak=100.0,
            seed=700 + capture,
            boundary="reflect",
        ) for capture in range(4))
        inverse = deblur_multicapture_consensus(
            observations,
            passes=8,
            mixing_patch_size=16,
            mixing_stride=12,
        )
        posterior = solve_multicapture_transport_posterior(inverse)
        self.assertEqual(posterior.image.shape, truth.shape)
        self.assertTrue(np.all(np.isfinite(posterior.image)))
        self.assertTrue(np.all(posterior.uncertainty >= 0.0))
        self.assertGreaterEqual(posterior.center_mass, 0.0)
        self.assertGreaterEqual(posterior.atlas_mass, 0.0)
        self.assertGreaterEqual(posterior.denoise_mass, 0.0)
        self.assertLessEqual(posterior.denoise_mass, 1.0)
        self.assertGreaterEqual(
            posterior.diagnostics["innovation_noise_authority_minimum"], 0.0)
        self.assertLessEqual(
            posterior.diagnostics["innovation_noise_authority_maximum"], 1.0)
        self.assertLessEqual(
            posterior.diagnostics["transported_noise_displacement_rms"],
            posterior.diagnostics["fmmt_displacement_rms"] + 1e-12,
        )
        self.assertAlmostEqual(
            posterior.center_mass + posterior.atlas_mass, 1.0)
        self.assertEqual(
            posterior.diagnostics["selection_policy"],
            "all_center_inverse_and_noise_measures_retained_no_winner_branch",
        )

    def test_common_blur_has_negligible_inverse_mass(self) -> None:
        truth = sources(32)["multiscale blobs"]
        kernel = gaussian_kernel(2.0)
        observations = tuple(degrade(
            truth,
            translated_kernel(kernel, shift),
            gaussian_sigma=0.001,
            seed=900 + index,
            boundary="reflect",
        ) for index, shift in enumerate((
            (-1.5, -0.5), (1.5, 0.5), (0.5, -1.5), (-0.5, 1.5))))
        inverse = deblur_multicapture_consensus(
            observations,
            passes=8,
            mixing_patch_size=16,
            mixing_stride=12,
        )
        posterior = solve_multicapture_transport_posterior(inverse)
        self.assertLess(posterior.atlas_mass, 0.15)


if __name__ == "__main__":
    unittest.main()
