from __future__ import annotations

import unittest

import numpy as np

from experiments.self_geometric_harmonic_interpolation import (
    batched_cosine_hermite_characteristic_resize,
    bilinear_resample_to_shape,
    characteristic_transport_resize,
    cosine_phase_resize,
    directional_measure_transport_resize,
    local_phase_frame_resize,
    local_eikonal_phase_frame_resize,
    local_bruun_phase_frame_resize,
    bruun_connection_phase_resize,
    bruun_characteristic_resize,
    bruun_phase_jet_characteristic_resize,
    bruun_phase_jet_continuous_metric_resize,
    bruun_phase_jet_c1_characteristic_resize,
    projected_cosine_hermite_characteristic_resize,
    cosine_hermite_characteristic_resize,
    compact_spline_hermite_characteristic_resize,
    directional_variation_measure_resize,
    harmonic_resize,
    internally_measured_metric,
    lanczos_resample_to_shape,
    positive_restrict,
    riemannian_harmonic_adjoint_restrict,
    riemannian_lifting_restrict,
    riemannian_transport_restrict,
    riemannian_product_variation_resize,
    sbp42_hermite_characteristic_resize,
    transported_harmonic_resize,
    tensor_product_quintic_variation_resize,
    _evaluate_quintic_variation_lineage_profile,
    _quintic_variation_lineage_profile,
    quintic_variation_lifting_analysis,
    quintic_variation_lifting_synthesis,
)


class SelfGeometricHarmonicInterpolationTest(unittest.TestCase):
    def test_quintic_variation_lifting_is_exact_on_arbitrary_arrays(self) -> None:
        rng = np.random.default_rng(4404)
        fine = rng.normal(size=(17, 19, 3))
        for axis in (0, 1):
            coarse, detail = quintic_variation_lifting_analysis(fine, axis)
            reconstructed = quintic_variation_lifting_synthesis(
                coarse, detail, axis
            )
            np.testing.assert_allclose(
                reconstructed, fine, atol=4.0e-15, rtol=0.0
            )

    def test_quintic_variation_current_commutes_and_does_not_add_sign_pairs(self) -> None:
        rng = np.random.default_rng(4403)

        def sign_changes(value: np.ndarray) -> int:
            sign = np.sign(value)
            sign = sign[sign != 0.0]
            return int(np.sum(sign[1:] != sign[:-1]))

        for _ in range(128):
            source = rng.normal(size=(17, 3))
            anchored, increment = _quintic_variation_lineage_profile(source)
            np.testing.assert_array_equal(anchored, source)
            np.testing.assert_allclose(
                np.sum(increment, axis=1),
                np.diff(source, axis=0),
                atol=2.0e-15,
                rtol=0.0,
            )
            returned = np.stack([
                _evaluate_quintic_variation_lineage_profile(
                    (anchored, increment), float(node)
                )
                for node in range(source.shape[0])
            ])
            np.testing.assert_allclose(returned, source, atol=3.0e-15, rtol=0.0)
            for channel in range(source.shape[1]):
                self.assertLessEqual(
                    sign_changes(increment[:, :, channel].reshape(-1)),
                    sign_changes(np.diff(source[:, channel])),
                )

    def test_product_variation_current_is_cardinal_and_affine_exact(self) -> None:
        y, x = np.mgrid[:9, :11].astype(np.float64)
        source = 0.17 + 0.031 * x + 0.079 * y
        fy, fx = np.mgrid[:17, :21]
        truth = 0.17 + 0.031 * fx / 2.0 + 0.079 * fy / 2.0
        for operation in (
            tensor_product_quintic_variation_resize,
            riemannian_product_variation_resize,
            directional_variation_measure_resize,
        ):
            reconstructed = operation(source, 2)
            np.testing.assert_allclose(
                reconstructed[::2, ::2], source, atol=2.0e-14, rtol=0.0
            )
            np.testing.assert_allclose(
                reconstructed, truth, atol=2.0e-14, rtol=0.0
            )

    def test_harmonic_adjoint_restriction_is_positive_and_constant_exact(self) -> None:
        constant = np.full((17, 21, 3), 0.37)
        restricted = riemannian_harmonic_adjoint_restrict(constant, 2)
        self.assertEqual(restricted.shape, (9, 11, 3))
        np.testing.assert_allclose(restricted, 0.37, atol=2.0e-15, rtol=0.0)

        rng = np.random.default_rng(4399)
        source = rng.uniform(-0.2, 0.8, size=(17, 21, 3))
        restricted = riemannian_harmonic_adjoint_restrict(source, 2)
        self.assertGreaterEqual(float(np.min(restricted)), float(np.min(source)))
        self.assertLessEqual(float(np.max(restricted)), float(np.max(source)))

        transported = riemannian_transport_restrict(source, 2)
        self.assertGreaterEqual(float(np.min(transported)), float(np.min(source)))
        self.assertLessEqual(float(np.max(transported)), float(np.max(source)))

    def test_transport_restriction_reduces_to_harmonic_adjoint_on_constants(self) -> None:
        source = np.full((17, 21), 0.61)
        np.testing.assert_allclose(
            riemannian_transport_restrict(source, 2),
            riemannian_harmonic_adjoint_restrict(source, 2),
            atol=2.0e-15,
            rtol=0.0,
        )

        lifted = riemannian_lifting_restrict(source, 2)
        np.testing.assert_allclose(lifted, 0.61, atol=2.0e-15, rtol=0.0)

    def test_lifting_restriction_is_affine_exact_and_range_bounded(self) -> None:
        y, x = np.mgrid[:17, :21].astype(np.float64)
        affine = 0.13 + 0.021 * x + 0.037 * y
        np.testing.assert_allclose(
            riemannian_lifting_restrict(affine, 2),
            affine[::2, ::2],
            atol=2.0e-14,
            rtol=0.0,
        )
        oscillatory = np.sin(0.71 * x + 0.29 * y)
        restricted = riemannian_lifting_restrict(oscillatory, 2)
        self.assertGreaterEqual(
            float(np.min(restricted)), float(np.min(oscillatory))
        )
        self.assertLessEqual(
            float(np.max(restricted)), float(np.max(oscillatory))
        )
        for cy in range(restricted.shape[0]):
            for cx in range(restricted.shape[1]):
                support = oscillatory[
                    max(0, 2 * cy - 2) : min(17, 2 * cy + 3),
                    max(0, 2 * cx - 2) : min(21, 2 * cx + 3),
                ]
                self.assertGreaterEqual(restricted[cy, cx], np.min(support))
                self.assertLessEqual(restricted[cy, cx], np.max(support))

    def test_arbitrary_shape_resamplers_match_nested_upscale_geometry(self) -> None:
        rng = np.random.default_rng(4400)
        source = rng.normal(size=(7, 9, 3))
        target_shape = (13, 17)
        np.testing.assert_allclose(
            bilinear_resample_to_shape(source, target_shape),
            harmonic_resize(source, 2, adaptive=False),
            atol=2.0e-15,
            rtol=0.0,
        )
        from experiments.self_geometric_harmonic_interpolation import (
            lanczos_resize,
        )
        np.testing.assert_allclose(
            lanczos_resample_to_shape(source, target_shape, radius=3),
            lanczos_resize(source, 2, radius=3),
            atol=3.0e-15,
            rtol=0.0,
        )

    def test_compact_variational_jet_satisfies_its_exact_system(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _natural_cubic_variational_jet,
        )

        rng = np.random.default_rng(4401)
        source = rng.normal(size=(17, 4, 3))
        _, derivative = _natural_cubic_variational_jet(source)
        secant = np.diff(source, axis=0)
        np.testing.assert_allclose(
            2.0 * derivative[0] + derivative[1],
            3.0 * secant[0],
            atol=3.0e-15,
        )
        np.testing.assert_allclose(
            derivative[:-2] + 4.0 * derivative[1:-1] + derivative[2:],
            3.0 * (secant[:-1] + secant[1:]),
            atol=4.0e-15,
        )
        np.testing.assert_allclose(
            derivative[-2] + 2.0 * derivative[-1],
            3.0 * secant[-1],
            atol=3.0e-15,
        )

    def test_batched_cosine_factorization_matches_active_operator(self) -> None:
        rng = np.random.default_rng(4402)
        source = rng.uniform(size=(9, 11, 3))
        active = cosine_hermite_characteristic_resize(source, 2)
        batched = batched_cosine_hermite_characteristic_resize(source, 2)
        np.testing.assert_allclose(batched, active, atol=3.0e-15, rtol=0.0)

    def test_comparison_jets_retain_formal_invariants(self) -> None:
        y, x = np.mgrid[:9, :11].astype(np.float64)
        affine = 0.17 + 0.031 * x + 0.079 * y
        fy, fx = np.mgrid[:17, :21]
        truth = 0.17 + 0.031 * fx / 2.0 + 0.079 * fy / 2.0
        for operation in (
            compact_spline_hermite_characteristic_resize,
            sbp42_hermite_characteristic_resize,
        ):
            reconstructed = operation(affine, 2)
            np.testing.assert_allclose(reconstructed, truth, atol=2.0e-14)

            bounded_source = np.sin(0.41 * x + 0.23 * y)
            bounded = operation(bounded_source, 2)
            np.testing.assert_allclose(
                bounded[::2, ::2], bounded_source, atol=4.0e-15
            )
            self.assertGreaterEqual(
                float(np.min(bounded)), float(np.min(bounded_source)) - 2.0e-15
            )
            self.assertLessEqual(
                float(np.max(bounded)), float(np.max(bounded_source)) + 2.0e-15
            )

    def test_constant_has_identity_metric_and_is_reproduced(self) -> None:
        samples = np.full((7, 9), 0.37)
        metric = internally_measured_metric(samples)
        np.testing.assert_array_equal(
            metric, np.broadcast_to(np.eye(2), metric.shape)
        )
        reconstruction = harmonic_resize(samples, 3)
        np.testing.assert_allclose(reconstruction, 0.37, atol=2.0e-15)

    def test_affine_field_is_reproduced_with_internal_metric(self) -> None:
        y, x = np.mgrid[:6, :8].astype(np.float64)
        samples = 0.1 + 0.07 * x + 0.11 * y
        reconstruction = harmonic_resize(samples, 4)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.1 + 0.07 * fx / 4.0 + 0.11 * fy / 4.0
        np.testing.assert_allclose(reconstruction, truth, atol=3.0e-15)

    def test_metric_is_spd_and_determinant_one(self) -> None:
        y, x = np.mgrid[:9, :9].astype(np.float64)
        samples = np.tanh((x + 0.7 * y - 7.0) / 0.8)
        metric = internally_measured_metric(samples)
        eigenvalues = np.linalg.eigvalsh(metric)
        self.assertGreater(float(eigenvalues.min()), 0.0)
        np.testing.assert_allclose(np.linalg.det(metric), 1.0, atol=8.0e-16)

    def test_reconstruction_is_cardinal_and_cell_range_preserving(self) -> None:
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(7, 8, 3))
        scale = 4
        reconstruction = harmonic_resize(samples, scale)
        np.testing.assert_allclose(
            reconstruction[::scale, ::scale], samples, atol=2.0e-15
        )
        for iy in range(samples.shape[0] - 1):
            for ix in range(samples.shape[1] - 1):
                cell = reconstruction[
                    iy * scale : (iy + 1) * scale + 1,
                    ix * scale : (ix + 1) * scale + 1,
                ]
                corners = samples[iy : iy + 2, ix : ix + 2]
                low = np.min(corners, axis=(0, 1))
                high = np.max(corners, axis=(0, 1))
                self.assertGreaterEqual(float(np.min(cell - low)), -2.0e-15)
                self.assertLessEqual(float(np.max(cell - high)), 2.0e-15)

    def test_rotated_structure_changes_the_interior_not_the_boundary(self) -> None:
        y, x = np.mgrid[:8, :8].astype(np.float64)
        samples = np.tanh((x + 0.63 * y - 6.0) / 0.7)
        adaptive = harmonic_resize(samples, 4)
        isotropic = harmonic_resize(samples, 4, adaptive=False)
        self.assertGreater(float(np.max(np.abs(adaptive - isotropic))), 1.0e-6)
        np.testing.assert_allclose(adaptive[::4, :], isotropic[::4, :], atol=2.0e-15)
        np.testing.assert_allclose(adaptive[:, ::4], isotropic[:, ::4], atol=2.0e-15)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(91)
        samples = rng.uniform(size=(6, 7))
        first = harmonic_resize(samples, 3)
        second = harmonic_resize(samples.copy(), 3)
        np.testing.assert_array_equal(first, second)

    def test_positive_restriction_preserves_constant(self) -> None:
        refined = np.full((25, 33, 2), (0.2, 0.8))
        restricted = positive_restrict(refined, 4)
        np.testing.assert_allclose(
            restricted,
            np.broadcast_to((0.2, 0.8), restricted.shape),
            atol=2.0e-15,
        )

    def test_transported_traces_are_cardinal_and_cell_range_preserving(self) -> None:
        rng = np.random.default_rng(611)
        samples = rng.uniform(size=(8, 9, 2))
        scale = 4
        reconstruction = transported_harmonic_resize(samples, scale)
        np.testing.assert_allclose(
            reconstruction[::scale, ::scale], samples, atol=2.0e-15
        )
        for iy in range(samples.shape[0] - 1):
            for ix in range(samples.shape[1] - 1):
                cell = reconstruction[
                    iy * scale : (iy + 1) * scale + 1,
                    ix * scale : (ix + 1) * scale + 1,
                ]
                corners = samples[iy : iy + 2, ix : ix + 2]
                low = np.min(corners, axis=(0, 1))
                high = np.max(corners, axis=(0, 1))
                self.assertGreaterEqual(float(np.min(cell - low)), -3.0e-15)
                self.assertLessEqual(float(np.max(cell - high)), 3.0e-15)

    def test_transported_affine_field_is_exact(self) -> None:
        y, x = np.mgrid[:7, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = transported_harmonic_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=4.0e-15)

    def test_characteristic_transport_is_cardinal_and_globally_bounded(self) -> None:
        rng = np.random.default_rng(902)
        samples = rng.uniform(size=(8, 9, 3))
        reconstruction = characteristic_transport_resize(samples, 4)
        np.testing.assert_allclose(
            reconstruction[::4, ::4], samples, atol=3.0e-15
        )
        tolerance = 4.0 * np.finfo(float).eps
        self.assertGreaterEqual(
            float(np.min(reconstruction)), float(np.min(samples)) - tolerance
        )
        self.assertLessEqual(
            float(np.max(reconstruction)), float(np.max(samples)) + tolerance
        )

    def test_characteristic_transport_preserves_affine_field(self) -> None:
        y, x = np.mgrid[:7, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = characteristic_transport_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=8.0e-15)

    def test_directional_measure_transport_is_cardinal_and_bounded(self) -> None:
        rng = np.random.default_rng(119)
        samples = rng.uniform(size=(7, 8, 3))
        reconstruction = directional_measure_transport_resize(samples, 3)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=3.0e-15
        )
        tolerance = 4.0 * np.finfo(float).eps
        self.assertGreaterEqual(
            float(np.min(reconstruction)), float(np.min(samples)) - tolerance
        )
        self.assertLessEqual(
            float(np.max(reconstruction)), float(np.max(samples)) + tolerance
        )

    def test_directional_measure_transport_preserves_affine_field(self) -> None:
        y, x = np.mgrid[:7, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = directional_measure_transport_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=8.0e-15)

    def test_cosine_phase_continuation_is_cardinal(self) -> None:
        rng = np.random.default_rng(812)
        samples = rng.normal(size=(7, 9, 2))
        reconstruction = cosine_phase_resize(samples, 4)
        np.testing.assert_allclose(
            reconstruction[::4, ::4], samples, atol=7.0e-15
        )

    def test_cosine_phase_continuation_reproduces_constant(self) -> None:
        samples = np.full((7, 9), 0.37)
        reconstruction = cosine_phase_resize(samples, 4)
        np.testing.assert_allclose(reconstruction, 0.37, atol=3.0e-15)

    def test_local_phase_frame_is_cardinal(self) -> None:
        rng = np.random.default_rng(404)
        samples = rng.normal(size=(9, 10, 2))
        for frame_size in (4, 6, 8):
            reconstruction = local_phase_frame_resize(
                samples, 3, frame_size
            )
            np.testing.assert_allclose(
                reconstruction[::3, ::3], samples, atol=2.0e-14
            )

    def test_local_phase_frame_reproduces_constant(self) -> None:
        samples = np.full((9, 10), 0.37)
        for frame_size in (4, 6, 8):
            reconstruction = local_phase_frame_resize(
                samples, 3, frame_size
            )
            np.testing.assert_allclose(
                reconstruction, 0.37, atol=1.0e-14
            )

    def test_local_eikonal_phase_frame_is_cardinal(self) -> None:
        y, x = np.mgrid[:9, :10].astype(np.float64)
        samples = 0.5 + 0.2 * np.sin(0.8 * x + 0.35 * y)
        reconstruction = local_eikonal_phase_frame_resize(samples, 3, 4)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=2.0e-12
        )

    def test_local_eikonal_phase_frame_reproduces_constant(self) -> None:
        samples = np.full((9, 10), 0.37)
        reconstruction = local_eikonal_phase_frame_resize(samples, 3, 4)
        np.testing.assert_allclose(reconstruction, 0.37, atol=2.0e-12)

    def test_frozen_eikonal_phase_is_a_coordinate_gauge(self) -> None:
        y, x = np.mgrid[:9, :10].astype(np.float64)
        samples = (
            0.5
            + 0.21 * np.sin(0.81 * x + 0.37 * y)
            + 0.09 * np.cos(0.22 * x - 1.03 * y)
        )
        flat = local_phase_frame_resize(samples, 3, 6)
        eikonal = local_eikonal_phase_frame_resize(samples, 3, 6)
        np.testing.assert_allclose(eikonal, flat, atol=2.0e-12)

    def test_bruun_pairs_match_numpy_rfft_without_complex_state(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_rfft_pairs,
        )

        rng = np.random.default_rng(73)
        source = rng.normal(size=(16, 5))
        real, negative_imaginary = _bruun_rfft_pairs(source)
        reference = np.fft.rfft(source, axis=0)
        np.testing.assert_allclose(real, reference.real, atol=8.0e-15)
        np.testing.assert_allclose(
            negative_imaginary, -reference.imag, atol=8.0e-15
        )

    def test_local_bruun_phase_frame_is_cardinal_and_constant_exact(self) -> None:
        rng = np.random.default_rng(907)
        samples = rng.normal(size=(9, 10, 2))
        for frame_size in (4, 8):
            reconstruction = local_bruun_phase_frame_resize(
                samples, 3, frame_size
            )
            np.testing.assert_allclose(
                reconstruction[::3, ::3], samples, atol=2.0e-14
            )
            constant = local_bruun_phase_frame_resize(
                np.full((9, 10), 0.37), 3, frame_size
            )
            np.testing.assert_allclose(constant, 0.37, atol=2.0e-14)

    def test_bruun_rfft2_pairs_match_numpy(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_rfft2_pairs,
        )

        rng = np.random.default_rng(211)
        source = rng.normal(size=(8, 8, 3))
        real, pair = _bruun_rfft2_pairs(source)
        reference = np.fft.rfft2(source, axes=(0, 1))
        np.testing.assert_allclose(real, reference.real, atol=2.0e-14)
        np.testing.assert_allclose(pair, -reference.imag, atol=2.0e-14)

    def test_bruun_connection_phase_is_cardinal_and_constant_exact(self) -> None:
        rng = np.random.default_rng(311)
        samples = rng.normal(size=(7, 8, 2))
        reconstruction = bruun_connection_phase_resize(samples, 3, 4)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=3.0e-14
        )
        constant = bruun_connection_phase_resize(
            np.full((7, 8), 0.37), 3, 4
        )
        np.testing.assert_allclose(constant, 0.37, atol=3.0e-14)

    def test_bruun_characteristic_is_cardinal_and_affine_exact(self) -> None:
        y, x = np.mgrid[:9, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = bruun_characteristic_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=2.0e-14)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=2.0e-14
        )

    def test_bruun_phase_jet_profile_is_interval_range_preserving(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_limited_profile,
            _evaluate_limited_bruun_profile,
        )

        rng = np.random.default_rng(818)
        source = rng.normal(size=(9, 3))
        profile = _bruun_limited_profile(source)
        for interval in range(source.shape[0] - 1):
            positions = interval + np.linspace(0.0, 1.0, 101)
            values = np.stack([
                _evaluate_limited_bruun_profile(profile, position)
                for position in positions
            ])
            low = np.minimum(source[interval], source[interval + 1])
            high = np.maximum(source[interval], source[interval + 1])
            self.assertGreaterEqual(float(np.min(values - low)), -2.0e-15)
            self.assertLessEqual(float(np.max(values - high)), 2.0e-15)

    def test_bruun_phase_jet_is_cardinal_affine_and_globally_bounded(self) -> None:
        y, x = np.mgrid[:9, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = bruun_phase_jet_characteristic_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=2.0e-14)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=2.0e-14
        )

        rng = np.random.default_rng(77)
        random_samples = rng.uniform(size=(9, 9, 3))
        random_reconstruction = bruun_phase_jet_characteristic_resize(
            random_samples, 2
        )
        tolerance = 8.0 * np.finfo(float).eps
        self.assertGreaterEqual(
            float(np.min(random_reconstruction)),
            float(np.min(random_samples)) - tolerance,
        )
        self.assertLessEqual(
            float(np.max(random_reconstruction)),
            float(np.max(random_samples)) + tolerance,
        )

    def test_continuous_metric_phase_jet_preserves_invariants(self) -> None:
        y, x = np.mgrid[:9, :9].astype(np.float64)
        samples = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = bruun_phase_jet_continuous_metric_resize(samples, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=2.0e-14)
        np.testing.assert_allclose(
            reconstruction[::3, ::3], samples, atol=2.0e-14
        )

    def test_c1_phase_jet_is_shared_and_interval_range_preserving(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_c1_limited_profile,
            _evaluate_c1_bruun_profile,
        )

        rng = np.random.default_rng(941)
        source = rng.normal(size=(9, 3))
        profile = _bruun_c1_limited_profile(source)
        self.assertEqual(profile[1].shape, source.shape)
        for interval in range(source.shape[0] - 1):
            positions = interval + np.linspace(0.0, 1.0, 101)
            values = np.stack([
                _evaluate_c1_bruun_profile(profile, position)
                for position in positions
            ])
            low = np.minimum(source[interval], source[interval + 1])
            high = np.maximum(source[interval], source[interval + 1])
            self.assertGreaterEqual(float(np.min(values - low)), -2.0e-15)
            self.assertLessEqual(float(np.max(values - high)), 2.0e-15)

        y, x = np.mgrid[:9, :9].astype(np.float64)
        affine = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = bruun_phase_jet_c1_characteristic_resize(affine, 3)
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 3.0 + 0.09 * fy / 3.0
        np.testing.assert_allclose(reconstruction, truth, atol=2.0e-14)

    def test_cosine_spectral_jet_is_the_bruun_derivative(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_limited_profile,
            _cosine_spectral_jet,
        )

        rng = np.random.default_rng(317)
        source = rng.normal(size=(17, 4))
        _, bruun = _bruun_limited_profile(source)
        _, cosine = _cosine_spectral_jet(source)
        np.testing.assert_allclose(cosine, bruun, atol=2.0e-14)

    def test_admissible_jet_projection_is_feasible_and_closest_so_far(self) -> None:
        from experiments.self_geometric_harmonic_interpolation import (
            _bruun_c1_limited_profile,
            _cosine_spectral_jet,
            _project_admissible_hermite_jet,
        )

        rng = np.random.default_rng(733)
        source = rng.normal(size=(17, 3))
        _, proposed = _cosine_spectral_jet(source)
        projected, information = _project_admissible_hermite_jet(
            source, proposed, diagnostics=True
        )
        _, retracted = _bruun_c1_limited_profile(source)
        self.assertLessEqual(
            float(np.linalg.norm(projected - proposed)),
            float(np.linalg.norm(retracted - proposed)) + 2.0e-13,
        )
        self.assertLessEqual(information["feasibility_residual"], 2.0e-13)
        repeated = _project_admissible_hermite_jet(source, projected)
        np.testing.assert_allclose(repeated, projected, atol=2.0e-13)

    def test_projected_cosine_hermite_operator_preserves_invariants(self) -> None:
        y, x = np.mgrid[:10, :12].astype(np.float64)
        affine = 0.15 + 0.03 * x + 0.09 * y
        reconstruction = projected_cosine_hermite_characteristic_resize(
            affine, 2
        )
        fy, fx = np.mgrid[: reconstruction.shape[0], : reconstruction.shape[1]]
        truth = 0.15 + 0.03 * fx / 2.0 + 0.09 * fy / 2.0
        np.testing.assert_allclose(reconstruction, truth, atol=3.0e-14)
        np.testing.assert_allclose(
            reconstruction[::2, ::2], affine, atol=3.0e-14
        )

    def test_cosine_hermite_stack_replaces_bruun_factorization(self) -> None:
        rng = np.random.default_rng(116)
        samples = rng.uniform(size=(9, 9, 3))
        legacy = bruun_phase_jet_c1_characteristic_resize(samples, 2)
        formal = cosine_hermite_characteristic_resize(samples, 2)
        np.testing.assert_allclose(formal, legacy, atol=3.0e-14)
        np.testing.assert_allclose(
            formal[::2, ::2], samples, atol=3.0e-14
        )
        tolerance = 8.0 * np.finfo(float).eps
        self.assertGreaterEqual(
            float(np.min(formal)), float(np.min(samples)) - tolerance
        )
        self.assertLessEqual(
            float(np.max(formal)), float(np.max(samples)) + tolerance
        )

        y, x = np.mgrid[:10, :12].astype(np.float64)
        affine = 0.15 + 0.03 * x + 0.09 * y
        non_dyadic = cosine_hermite_characteristic_resize(affine, 2)
        fy, fx = np.mgrid[: non_dyadic.shape[0], : non_dyadic.shape[1]]
        truth = 0.15 + 0.03 * fx / 2.0 + 0.09 * fy / 2.0
        np.testing.assert_allclose(non_dyadic, truth, atol=3.0e-14)


if __name__ == "__main__":
    unittest.main()
