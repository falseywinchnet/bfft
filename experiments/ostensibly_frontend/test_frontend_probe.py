import unittest

import numpy as np

from experiments.ostensibly_frontend.run_frontend_probe import (
    exact_double_irfft_texture,
    scale_for_meyer,
)
from experiments.ostensibly_frontend.trace_geometry import (
    TraceGeometryConfig,
    descriptor_distance,
    extract_trace_components,
)
from experiments.ostensibly_frontend.full_recording import (
    activity_intervals,
    exact_low_rows_from_stft,
)
from experiments.ostensibly_frontend.phone_match import (
    fingerprint_phone_patch,
    phone_distance,
    registered_row_profile_distance,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    VOICE_PROBES,
    apply_ssb_probe_channel,
    synthesize_vowel,
)
from experiments.ostensibly_frontend.synthetic_consonants import (
    CONSONANTS,
    synthesize_consonant,
)


class FrontendProbeTests(unittest.TestCase):
    def test_exact_default_second_irfft_length_and_crop(self):
        sample_rate = 8192
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        samples = np.sin(2.0 * np.pi * 440.0 * time)
        spectrum, signed, texture = exact_double_irfft_texture(
            samples, n_fft=512, hop_length=128, crop_rows=64
        )
        self.assertEqual(spectrum.shape[0], 257)
        self.assertEqual(signed.shape[0], 1022)
        self.assertEqual(texture.shape, (64, spectrum.shape[1]))
        np.testing.assert_array_equal(texture, np.abs(signed[:64]))

    def test_meyer_scaling_is_linear(self):
        source = np.array([[0.0, 1.0], [2.0, 4.0]])
        scaled, peak = scale_for_meyer(source)
        self.assertEqual(peak, 4.0)
        np.testing.assert_allclose(scaled, source * (255.0 / 4.0))

    def test_trace_descriptor_is_translation_invariant(self):
        source = np.zeros((64, 80), dtype=np.float64)
        for column in range(8, 31):
            row = 14 + (column - 8) // 8
            source[row : row + 3, column] = 10.0
        for column in range(43, 66):
            row = 31 + (column - 43) // 8
            source[row : row + 3, column] = 10.0
        config = TraceGeometryConfig(
            opening_rows=9,
            score_percentile=50.0,
            minimum_area=8,
            minimum_frame_span=8,
        )
        _, _, components = extract_trace_components(source, config)
        self.assertEqual(len(components), 2)
        self.assertLess(descriptor_distance(components[0], components[1]), 1e-10)

    def test_blocked_low_rows_equal_unblocked_double_inverse(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=4096)
        spectrum = np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(samples, 512)[::128],
            axis=1,
        ).T
        blocked = exact_low_rows_from_stft(
            spectrum, n_fft=512, crop_rows=64, column_block=3
        )
        expected = np.abs(
            np.fft.irfft(np.fft.irfft(spectrum, axis=0), axis=0)[:64]
        )
        np.testing.assert_allclose(blocked, expected, rtol=0.0, atol=0.0)

    def test_activity_intervals_find_persistent_ridges(self):
        cartoon = np.zeros((64, 100), dtype=np.float64)
        cartoon[8:11, 20:45] = 10.0
        cartoon[14:17, 65:95] = 12.0
        _, active, intervals, _ = activity_intervals(
            cartoon, sample_rate=1000, hop_length=10
        )
        self.assertFalse(bool(active[0]))
        self.assertEqual(len(intervals), 2)
        self.assertLessEqual(intervals[0].frame0, 20)
        self.assertGreaterEqual(intervals[1].frame1, 95)

    def test_phone_fingerprint_removes_frequency_translation(self):
        source = np.zeros((192, 24), dtype=np.float64)
        source[20:24, 3:21] = 4.0
        source[36:39, 5:19] = 2.5
        shifted = np.zeros_like(source)
        shifted[47:51, 3:21] = 4.0
        shifted[63:66, 5:19] = 2.5
        left = fingerprint_phone_patch(source)
        right = fingerprint_phone_patch(shifted)
        self.assertEqual(right.anchor_row - left.anchor_row, 27)
        self.assertLess(phone_distance(left, right), 1e-10)

    def test_ordered_profile_registration_accepts_moderate_scale(self):
        profile = np.zeros(64, dtype=np.float64)
        profile[[8, 21, 39]] = [1.0, 0.7, 0.4]
        from scipy import ndimage as ndi
        profile = ndi.gaussian_filter1d(profile, 2.0)
        profile = np.asarray(profile / np.linalg.norm(profile))
        coordinates = np.arange(64, dtype=np.float64)
        scaled = np.interp(coordinates / 1.2, coordinates, profile, left=0, right=0)
        scaled /= np.linalg.norm(scaled)
        self.assertLess(registered_row_profile_distance(profile, scaled), 0.02)

    def test_acoustic_vowel_probe_is_finite_and_bounded(self):
        source = synthesize_vowel("IY", VOICE_PROBES[2], seed=7)
        channel = apply_ssb_probe_channel(source, seed=8)
        self.assertEqual(source.shape, channel.shape)
        self.assertTrue(np.isfinite(source).all())
        self.assertTrue(np.isfinite(channel).all())
        self.assertAlmostEqual(float(np.max(np.abs(source))), 1.0)

    def test_all_consonant_probes_are_finite_and_bounded(self):
        for index, phone in enumerate(CONSONANTS):
            source = synthesize_consonant(
                phone, VOICE_PROBES[2], seed=100 + index
            )
            self.assertTrue(np.isfinite(source).all(), phone)
            self.assertAlmostEqual(float(np.max(np.abs(source))), 1.0, msg=phone)


if __name__ == "__main__":
    unittest.main()
