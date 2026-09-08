import unittest

import numpy as np

from experiments.ostensibly_frontend.uncertainty_fusion import (
    centered_stft_lattice,
    double_irfft_low_rows,
    phase_lattice_observations_at_centers,
    registered_perceptual_maximum,
    resample_rfft_gain,
    texture_cartoon_cascade,
)
from experiments.ostensibly_frontend.phone_match import (
    fingerprint_whole_phone_patch,
    whole_patch_distance,
)
from experiments.ostensibly_frontend.full_recording import (
    TimeInterval,
    phone_boundary_proposals,
    supported_activity_intervals,
    waveform_rms_score,
)
from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    fourier_unrolled_densified_at_centers,
    reassigned_texture_baseline,
    reassigned_power_support_at_centers,
    resample_unrolled_rows,
    source_crop_for_target,
)
from experiments.ostensibly_frontend.supported_hodge_fusion import (
    _divergence,
    _forward_gradient,
    _integrate_longitudinal,
    supported_hodge_fusion,
)
from experiments.ostensibly_frontend.synthetic_consonants import (
    CONSONANTS,
    synthesize_consonant,
)
from experiments.ostensibly_frontend.synthetic_vowels import VOICE_PROBES
from experiments.ostensibly_frontend.multiscale_phone_geometry import (
    DURATION_LOG_WEIGHT,
    duration_coupled_similarity,
    multiscale_phone_descriptor,
    multiscale_phone_distance,
)
from experiments.ostensibly_frontend.run_arctic_segmentation_audit import (
    match_boundaries,
)
from experiments.ostensibly_frontend.sparse_gabor_geometry import (
    RealGaborConfig,
    bright_gabor_support,
    eroded_gabor_ridges,
    morphological_skeleton,
    real_gabor_kernels,
    real_gabor_responses,
    sparse_ridge_glyph,
)


class UncertaintyFusionTests(unittest.TestCase):
    def test_real_gabor_bank_is_three_by_three_and_zero_dc(self):
        kernels = real_gabor_kernels()
        self.assertEqual(len(kernels), 9)
        for kernel in kernels:
            self.assertFalse(np.iscomplexobj(kernel))
            self.assertAlmostEqual(float(np.sum(kernel)), 0.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(kernel)), 1.0)

    def test_sparse_real_gabor_geometry_retains_a_fixed_ridge_budget(self):
        field = np.zeros((96, 80), dtype=np.float64)
        field[31:34, 8:72] = 1.0
        field[61:63, 12:68] = 0.7
        config = RealGaborConfig(
            point_count=48,
            bright_percentile=65.0,
        )
        responses = real_gabor_responses(field, config)
        glyph = sparse_ridge_glyph(responses, config)
        self.assertEqual(responses.shape, (9, 96, 80))
        self.assertEqual(glyph.count, 48)
        self.assertEqual(glyph.points.shape, (48, 2))
        self.assertTrue(np.all(glyph.strengths > 0.0))
        self.assertTrue(set(np.unique(glyph.scales)).issubset({3.0, 6.0, 12.0}))

    def test_bright_gabor_support_is_binary_threshold_then_erosion_skeleton(self):
        responses = np.zeros((9, 31, 31), dtype=np.float64)
        responses[0, 8:23, 9:22] = 2.0
        config = RealGaborConfig(bright_percentile=0.0, erosion_iterations=0)
        _fused, bright, _threshold = bright_gabor_support(responses, config)
        returned_bright, skeleton, _strength, _threshold = eroded_gabor_ridges(
            responses, config
        )
        self.assertTrue(np.array_equal(bright, returned_bright))
        self.assertLess(np.count_nonzero(skeleton), np.count_nonzero(bright))
        self.assertGreater(np.count_nonzero(skeleton), 0)

    def test_morphological_skeleton_preserves_a_one_pixel_ridge(self):
        ridge = np.zeros((25, 25), dtype=bool)
        ridge[12, 3:22] = True
        self.assertTrue(np.array_equal(morphological_skeleton(ridge), ridge))

    def test_multiscale_phone_geometry_retains_raw_and_coarse_ridges(self):
        left = np.zeros((128, 24), dtype=np.float64)
        right = np.zeros_like(left)
        left[40:43, 4:20] = 1.0
        right[41:44, 4:20] = 1.0
        descriptor = multiscale_phone_descriptor(left)
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)
        self.assertTrue(np.all(np.isfinite(descriptor)))
        self.assertGreater(multiscale_phone_distance(left, right), 0.0)
        self.assertLess(multiscale_phone_distance(left, right), 0.25)

    def test_duration_coupling_prefers_matching_timing(self):
        descriptor = multiscale_phone_descriptor(np.eye(24))
        matched = duration_coupled_similarity(descriptor, 8, descriptor, 8)
        stretched = duration_coupled_similarity(descriptor, 8, descriptor, 16)
        self.assertAlmostEqual(matched, 1.0)
        self.assertAlmostEqual(
            matched - stretched, DURATION_LOG_WEIGHT * np.log(2.0)
        )

    def test_boundary_audit_is_one_to_one(self):
        result = match_boundaries(
            [0.10, 0.20, 0.30],
            [0.101, 0.102, 0.299],
            0.01,
        )
        self.assertEqual(result["matched"], 2)
        self.assertAlmostEqual(result["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result["recall"], 2.0 / 3.0)

    def test_phone_ceiling_cuts_snap_to_observed_novelty(self):
        field = np.ones((8, 31), dtype=np.float64)
        field[:, 9:] = np.arange(8, dtype=np.float64)[:, None] + 1.0
        field[:, 18:] = np.arange(8, 0, -1, dtype=np.float64)[:, None]
        speech = (TimeInterval(0, 31, 0.0, 3.1, 1.0),)
        phones = phone_boundary_proposals(
            field,
            speech,
            sample_rate=100,
            hop_length=10,
            minimum_seconds=0.4,
            maximum_seconds=1.2,
        )
        lengths = [phone.frame1 - phone.frame0 for phone in phones]
        self.assertLessEqual(max(lengths), 12)
        self.assertTrue(any(phone.frame1 in (9, 18) for phone in phones))

    def test_compact_consonant_probes_are_finite(self):
        for index, label in enumerate(CONSONANTS):
            samples = synthesize_consonant(
                label, VOICE_PROBES[index % len(VOICE_PROBES)],
                duration=0.06, seed=index,
            )
            self.assertEqual(samples.shape, (2880,))
            self.assertTrue(np.all(np.isfinite(samples)))
            self.assertGreater(float(np.max(np.abs(samples))), 0.0)

    def test_waveform_energy_supplies_independent_activity_support(self):
        hop = 64
        samples = np.zeros(64 * 48, dtype=np.float64)
        samples[64 * 22 : 64 * 38] = np.sin(
            2.0 * np.pi * 0.09 * np.arange(64 * 16))
        score = waveform_rms_score(
            samples, frame_count=48, hop_length=hop, window_length=128)
        self.assertGreater(float(np.mean(score[24:36])), float(np.mean(score[4:16])))

        trace = np.zeros((32, 48), dtype=np.float64)
        trace[8:11, 4:16] = 1.0
        _, active, intervals, diagnostics = supported_activity_intervals(
            trace,
            samples,
            sample_rate=6400,
            hop_length=hop,
            minimum_seconds=0.03,
            close_gap_seconds=0.02,
            rms_window_length=128,
        )
        self.assertTrue(np.any(active[24:36]))
        self.assertTrue(intervals)
        self.assertEqual(
            diagnostics["activity_method"],
            "harmonic_ridge_or_centered_waveform_log_rms",
        )

    def test_centered_lattice_has_exact_double_inverse_geometry(self):
        samples = np.sin(2.0 * np.pi * 220.0 * np.arange(4096) / 8192.0)
        spectrum = centered_stft_lattice(
            samples, n_fft=512, hop_length=128)
        field = double_irfft_low_rows(
            spectrum, n_fft=512, crop_rows=64)
        self.assertEqual(spectrum.shape, (257, 33))
        self.assertEqual(field.shape, (64, 33))
        self.assertTrue(np.all(field >= 0.0))

    def test_registered_maximum_seals_complementary_ridge_voids(self):
        rows, columns = 48, 72
        xx = np.arange(columns)
        ridge = np.exp(-0.5 * ((np.arange(rows)[:, None] - 17.0) / 1.2) ** 2)
        envelopes = (
            np.abs(np.cos(2.0 * np.pi * xx / 12.0)),
            np.abs(np.cos(2.0 * np.pi * xx / 12.0 + np.pi / 3.0)),
            np.abs(np.cos(2.0 * np.pi * xx / 12.0 + 2.0 * np.pi / 3.0)),
        )
        rng = np.random.default_rng(9)
        observations = np.stack([
            ridge * envelope[None, :] + 0.01 * rng.random((rows, columns))
            for envelope in envelopes
        ])
        result = registered_perceptual_maximum(
            observations, (-1, 0, 1), patch_size=24, stride=12)
        raw = observations[1, 17]
        fused = result.fused[17]
        self.assertLess(np.mean(fused < 0.5), 0.25 * np.mean(raw < 0.5))

    def test_texture_cartoon_cascade_preserves_both_complements(self):
        yy, xx = np.mgrid[:64, :80]
        field = 2.0 + np.sin(0.3 * xx) ** 2 + 4.0 * np.exp(
            -0.5 * ((yy - 18.0 - 0.03 * xx) / 1.4) ** 2)
        cascade = texture_cartoon_cascade(field, threads=1)
        np.testing.assert_allclose(
            cascade.input_scaled,
            cascade.first_cartoon + cascade.first_texture,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            cascade.first_texture,
            cascade.texture_cartoon + cascade.second_texture,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.abs(cascade.first_texture),
            cascade.magnitude_texture_cartoon
            + cascade.magnitude_second_texture,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            cascade.promoted_trace,
            cascade.magnitude_texture_cartoon
            * (cascade.input_peak / 255.0),
            atol=1e-12,
        )

    def test_whole_patch_geometry_registers_centroid_translation(self):
        source = np.zeros((256, 24), dtype=np.float64)
        source[34:38, 3:21] = 2.0
        source[61:64, 5:19] = 1.3
        shifted = np.zeros_like(source)
        shifted[51:55, 3:21] = 2.0
        shifted[78:81, 5:19] = 1.3
        left = fingerprint_whole_phone_patch(source)
        right = fingerprint_whole_phone_patch(shifted)
        self.assertAlmostEqual(right.centroid_row, left.centroid_row)
        self.assertLess(whole_patch_distance(left, right), 2e-3)

    def test_whole_patch_geometry_keeps_noise_in_terminal_field(self):
        source = np.zeros((256, 24), dtype=np.float64)
        source[40:44, 3:21] = 2.0
        clean = fingerprint_whole_phone_patch(source)
        rng = np.random.default_rng(3)
        noisy = fingerprint_whole_phone_patch(
            source + 0.35 * rng.random(source.shape))
        self.assertGreater(whole_patch_distance(clean, noisy), 1e-3)

    def test_unrolled_row_mapping_uses_second_inverse_length(self):
        target_row = 173
        source_row = target_row * (1024 - 1) / (2048 - 1)
        rows = source_crop_for_target(
            1024, target_n_fft=2048, target_crop_rows=256)
        source = np.zeros((rows, 3), dtype=np.float64)
        axis = np.arange(rows, dtype=np.float64)
        source[:, :] = np.exp(-0.5 * ((axis[:, None] - source_row) / 1.0) ** 2)
        mapped = resample_unrolled_rows(
            source,
            n_fft=1024,
            target_n_fft=2048,
            target_crop_rows=256,
        )
        self.assertLessEqual(abs(int(np.argmax(mapped[:, 1])) - target_row), 1)
        self.assertEqual(rows, 129)

    def test_reassigned_bin_maps_to_double_inverse_row(self):
        sample_rate = 8192.0
        frequency = 360.0
        samples = np.sin(
            2.0 * np.pi * frequency * np.arange(8192) / sample_rate)
        centers = np.asarray((2048, 2560, 3072, 3584), dtype=np.int64)
        peaks = []
        for n_fft in (1024, 2048, 4096):
            support = reassigned_power_support_at_centers(
                samples,
                centers,
                n_fft=n_fft,
                hop_length=512,
                target_n_fft=2048,
                target_crop_rows=256,
            )
            peaks.append(int(np.argmax(np.sum(support.power, axis=1))))
        expected = round(2.0 * (2048 - 1) * frequency / sample_rate)
        self.assertTrue(all(abs(peak - expected) <= 2 for peak in peaks))
        self.assertLessEqual(max(peaks) - min(peaks), 2)

    def test_spectral_gain_resampling_and_zero_gate_are_physical(self):
        gain = np.tile(np.linspace(0.0, 1.0, 17), (4, 1))
        resized = resample_rfft_gain(gain, 64)
        self.assertEqual(resized.shape, (4, 33))
        np.testing.assert_allclose(resized[:, 0], 0.0)
        np.testing.assert_allclose(resized[:, -1], 1.0)

        samples = np.sin(2.0 * np.pi * 0.1 * np.arange(1024))
        centers = np.asarray((256, 384, 512, 640), dtype=np.int64)
        observations = phase_lattice_observations_at_centers(
            samples,
            centers,
            n_fft=128,
            crop_rows=32,
            offsets=(0,),
            spectral_gain=np.zeros((centers.size, 65)),
        )
        np.testing.assert_array_equal(observations, 0.0)
        support = reassigned_power_support_at_centers(
            samples,
            centers,
            n_fft=128,
            hop_length=128,
            target_n_fft=128,
            target_crop_rows=32,
            spectral_gain=np.zeros((centers.size, 65)),
        )
        np.testing.assert_array_equal(support.power, 0.0)
        self.assertEqual(support.accepted_power, 0.0)

    def test_all_ones_spectral_gain_is_exact_multiscale_identity(self):
        sample_rate = 2048.0
        samples = np.sin(
            2.0 * np.pi * 120.0 * np.arange(2048) / sample_rate
        )
        centers = 128 * np.arange(2, 12, dtype=np.int64)
        arguments = dict(
            apertures=(128, 256, 512),
            target_n_fft=256,
            target_crop_rows=64,
            hop_length=128,
            offsets=(-31, 0, 43),
            patch_size=16,
            stride=8,
            ring_count=5,
        )
        baseline = fourier_unrolled_densified_at_centers(
            samples, centers, **arguments
        )
        identity = fourier_unrolled_densified_at_centers(
            samples,
            centers,
            spectral_gain=np.ones((centers.size, 65)),
            **arguments,
        )
        np.testing.assert_array_equal(identity.fused, baseline.fused)
        np.testing.assert_array_equal(identity.support_union, baseline.support_union)

    def test_reassigned_texture_baseline_is_exact_positive_sum(self):
        yy, xx = np.mgrid[:32, :40]
        fusion = 0.2 + np.exp(-0.5 * ((yy - 11.0) / 2.0) ** 2)
        support = (0.1 + np.exp(
            -0.5 * ((yy - 11.0) / 0.7) ** 2)) * (1.0 + 0.1 * xx)
        result = reassigned_texture_baseline(fusion, support)
        np.testing.assert_allclose(result.merged, result.cartoon + result.texture)
        self.assertTrue(np.all(result.merged >= 0.0))
        self.assertGreater(result.texture_amplitude_scale, 0.0)

    def test_multiscale_terminal_is_literal_maximum(self):
        sample_rate = 2048.0
        samples = (
            np.sin(2.0 * np.pi * 120.0 * np.arange(4096) / sample_rate)
            + 0.35 * np.sin(
                2.0 * np.pi * 230.0 * np.arange(4096) / sample_rate)
        )
        result = fourier_unrolled_densified_at_centers(
            samples,
            256 * np.arange(2, 14, dtype=np.int64),
            apertures=(256, 512, 1024),
            target_n_fft=512,
            target_crop_rows=128,
            hop_length=256,
            offsets=(-71, -19, 0, 37, 93),
            patch_size=16,
            stride=8,
            ring_count=5,
        )
        np.testing.assert_allclose(
            result.fused, np.max(result.aligned_fields, axis=0))
        self.assertTrue(np.all(result.fused >= 0.0))
        self.assertEqual(result.source_crop_rows, (65, 128, 256))

        supported = supported_hodge_fusion(result)
        self.assertTrue(np.all(np.isfinite(supported.field)))
        self.assertTrue(np.all(supported.field >= 0.0))
        self.assertFalse(np.allclose(supported.field, result.fused))
        transverse_divergence = _divergence(
            supported.transverse_time_derivative,
            supported.transverse_row_derivative,
        )
        self.assertLess(float(np.max(np.abs(transverse_divergence))), 1e-10)
        self.assertEqual(
            supported.diagnostics["feed_forward_residualizations"], 1)

    def test_hodge_projection_recovers_integrable_derivatives(self):
        yy, xx = np.mgrid[:48, :64]
        source = (
            0.7 * np.sin(2.0 * np.pi * xx / 17.0)
            + 0.4 * np.cos(2.0 * np.pi * yy / 13.0)
        )
        time_gradient, row_gradient = _forward_gradient(source)
        potential, projected_time, projected_row = _integrate_longitudinal(
            time_gradient, row_gradient)
        np.testing.assert_allclose(
            projected_time, time_gradient, atol=2e-12)
        np.testing.assert_allclose(
            projected_row, row_gradient, atol=2e-12)
        np.testing.assert_allclose(
            potential, source - np.mean(source), atol=2e-12)


if __name__ == "__main__":
    unittest.main()
