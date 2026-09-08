from __future__ import annotations

import unittest

import numpy as np

from .radio_raster_transport import (
    estimate_radio_raster_profile,
    matched_noise_patch,
    transport_reference_raster,
)


class RadioRasterTransportTests(unittest.TestCase):
    def test_profile_uses_longest_silence_run_and_measured_ratio(self) -> None:
        trace = np.ones((4, 12), dtype=np.float64)
        active = np.zeros(12, dtype=bool)
        active[4:7] = True
        trace[:, active] = 4.0
        profile = estimate_radio_raster_profile(trace, active, rows=4, quantile=0.9)
        self.assertEqual(profile.noise_field.shape, (4, 5))
        self.assertAlmostEqual(profile.noise_to_active_ratio, 0.25)

    def test_matched_patch_and_transport_are_deterministic(self) -> None:
        trace = np.arange(48, dtype=np.float64).reshape(4, 12) + 1.0
        active = np.zeros(12, dtype=bool)
        active[5:8] = True
        profile = estimate_radio_raster_profile(trace, active, rows=4, quantile=0.9)
        patch = matched_noise_patch(profile, 9, phase=0.25)
        self.assertEqual(patch.shape, (4, 9))
        reference = np.ones((4, 9), dtype=np.float64)
        left = transport_reference_raster(reference, profile, noise_scale=0.5)
        right = transport_reference_raster(reference, profile, noise_scale=0.5)
        self.assertTrue(np.allclose(left, right))
        self.assertGreater(float(np.mean(left)), 1.0)

    def test_zero_noise_and_identity_photometry_preserve_reference(self) -> None:
        trace = np.ones((4, 12), dtype=np.float64)
        active = np.zeros(12, dtype=bool)
        active[4:8] = True
        trace[:, active] = 2.0
        profile = estimate_radio_raster_profile(trace, active, rows=4, quantile=0.9)
        reference = np.linspace(0.1, 1.0, 36).reshape(4, 9)
        transported = transport_reference_raster(reference, profile, noise_scale=0.0)
        self.assertTrue(np.allclose(reference, transported))


if __name__ == "__main__":
    unittest.main()
