from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .radiometric_transport import transport_radiometric_pair


class RadiometricTransportTests(unittest.TestCase):
    def test_identical_pair_preserves_exact_gauge(self) -> None:
        image = sources(32)["cameraman"]
        result = transport_radiometric_pair(image, image)
        np.testing.assert_array_equal(result.images[0], image)
        np.testing.assert_array_equal(result.images[1], image)
        np.testing.assert_array_equal(result.precision, 1.0)
        self.assertAlmostEqual(result.relative_gain_second_over_first, 1.0)
        self.assertEqual(result.authority, 0.0)

    def test_complementary_exposure_recovers_symmetric_gauge(self) -> None:
        image = sources(64)["multiscale blobs"]
        first = np.clip(0.7 * image, 0.0, 1.0)
        second = np.clip((1.0 / 0.7) * image, 0.0, 1.0)
        result = transport_radiometric_pair(first, second)
        self.assertAlmostEqual(
            result.relative_gain_second_over_first, 1.0 / 0.49, delta=0.06)
        self.assertGreater(result.authority, 0.99)
        saturated = second >= 1.0 - 1e-8
        self.assertLess(float(np.mean(result.precision[1][saturated])), 0.15)
        self.assertLess(float(np.mean(
            (result.images[0] - image) ** 2)), 2e-4)

    def test_swapping_frames_inverts_gain_without_selecting_one(self) -> None:
        image = sources(48)["cameraman"]
        first = np.clip(0.8 * image, 0.0, 1.0)
        second = np.clip(1.25 * image, 0.0, 1.0)
        forward = transport_radiometric_pair(first, second)
        reverse = transport_radiometric_pair(second, first)
        self.assertAlmostEqual(
            forward.relative_gain_second_over_first
            * reverse.relative_gain_second_over_first,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(forward.authority, reverse.authority, places=12)
        np.testing.assert_allclose(forward.images[0], reverse.images[1])
        np.testing.assert_allclose(forward.images[1], reverse.images[0])


if __name__ == "__main__":
    unittest.main()
