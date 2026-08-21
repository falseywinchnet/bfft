"""Parity and closure tests for the optional native exposure operator ABI."""

from __future__ import annotations

import unittest

import numpy as np

from .curvilinear import ReflectedPathOperator, fit_curvilinear_exposure_chart
from .kernels import curved_path_kernel, disk_kernel, gaussian_kernel, line_kernel
from .native_backend import native_available
from .spatial_transport import (
    CovarianceExposureOperatorBatch,
    CovarianceReflectedExposureOperator,
    SpatialExposureField,
    SpatialExposureOperatorBatch,
    SpatialReflectedExposureOperator,
    rotational_exposure,
)


class CompactGlobalOperatorTests(unittest.TestCase):
    def test_compact_matches_full_spatial_for_fractional_offsets(self) -> None:
        shape = (73, 89)
        image = np.random.default_rng(91).random(shape)
        flow = np.broadcast_to(
            np.asarray((0.37, -1.0000000003))[None, None, :],
            (*shape, 2),
        )
        residual = np.asarray(((-2.15, 0.4), (0.6, -0.7), (1.3, 1.8)))
        weights = np.asarray((0.21, 0.47, 0.32))
        operators = []
        for compact in (False, True):
            operators.append(SpatialReflectedExposureOperator(
                SpatialExposureField.from_barycentric_paths(
                    f"global_{compact}", flow, residual, weights,
                    compact_global=compact,
                )
            ))
        spatial, compact = operators
        np.testing.assert_allclose(
            compact.forward(image), spatial.forward(image), atol=2e-14)
        np.testing.assert_allclose(
            compact.adjoint(image), spatial.adjoint(image), atol=2e-14)


@unittest.skipUnless(native_available(), "native exposure operator not built")
class NativePathOperatorTests(unittest.TestCase):
    def test_native_covariance_batch_matches_individual_operators(self) -> None:
        rng = np.random.default_rng(642)
        shape = (29, 33)
        operators = []
        for _ in range(4):
            factors = rng.normal(size=(*shape, 2, 2))
            covariance = 0.3 * np.einsum(
                "...ik,...jk->...ij", factors, factors)
            operators.append(CovarianceReflectedExposureOperator(covariance))
        batch = CovarianceExposureOperatorBatch(tuple(operators))
        self.assertTrue(batch.backend.endswith("_covariance_generated_batch"))
        for channels in (None, 3):
            image_shape = (
                (len(operators), *shape)
                if channels is None else
                (len(operators), *shape, channels))
            images = rng.random(image_shape)
            expected_forward = np.stack([
                operator.forward(image)
                for operator, image in zip(operators, images)])
            expected_adjoint = np.stack([
                operator.adjoint(image)
                for operator, image in zip(operators, images)])
            np.testing.assert_allclose(
                batch.forward(images), expected_forward,
                atol=3e-12, rtol=3e-12)
            np.testing.assert_allclose(
                batch.adjoint(images), expected_adjoint,
                atol=3e-12, rtol=3e-12)

    def test_spatial_quartic_side_weights_match_numpy_oracle(self) -> None:
        rng = np.random.default_rng(772)
        shape = (31, 35)
        factors = rng.normal(size=(*shape, 2, 2))
        covariance = 0.4 * np.einsum(
            "...ik,...jk->...ij", factors, factors)
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        side_weights = np.stack((
            0.12 + 0.20 * xx / max(shape[1] - 1, 1),
            0.15 + 0.18 * yy / max(shape[0] - 1, 1),
        ), axis=-1)
        operator = CovarianceReflectedExposureOperator(
            covariance, side_weights)
        self.assertTrue(operator.backend.endswith("_covariance_generated"))
        image = rng.random((*shape, 3))
        dual = rng.random(image.shape)
        np.testing.assert_allclose(
            operator.forward(image), operator._forward_numpy(image),
            atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(
            operator.adjoint(dual), operator._adjoint_numpy(dual),
            atol=3e-12, rtol=3e-12)
        self.assertAlmostEqual(
            float(np.vdot(operator.forward(image), dual)),
            float(np.vdot(image, operator.adjoint(dual))),
            places=10,
        )

    def test_generated_covariance_operator_matches_explicit_positive_measure(
        self,
    ) -> None:
        rng = np.random.default_rng(442)
        shape = (37, 41)
        factors = rng.normal(size=(*shape, 2, 2))
        covariance = np.einsum(
            "...ik,...jk->...ij", factors, factors) * 0.7
        generated = CovarianceReflectedExposureOperator(covariance)
        low_axis, high_axis = generated._axis_displacements()
        coordinates = (-1.0, 0.0, 1.0)
        sigma_weights = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
        atoms = np.stack([
            low * low_axis + high * high_axis
            for low in coordinates for high in coordinates
        ])
        weights = np.stack([
            np.full(shape, sigma_weights[low] * sigma_weights[high])
            for low in range(3) for high in range(3)
        ])
        explicit = SpatialReflectedExposureOperator(
            SpatialExposureField.from_barycentric_paths(
                "explicit_nine_atom_covariance",
                np.zeros((*shape, 2), dtype=np.float64),
                atoms,
                weights,
            ))
        self.assertTrue(generated.backend.endswith("_covariance_generated"))
        self.assertLess(
            generated.storage_bytes,
            explicit._source_indices.nbytes + explicit._coefficients.nbytes)
        for channels in (None, 3):
            with self.subTest(channels=channels):
                image_shape = shape if channels is None else (*shape, channels)
                image = rng.random(image_shape)
                dual = rng.random(image_shape)
                np.testing.assert_allclose(
                    generated.forward(image), explicit.forward(image),
                    atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(
                    generated.adjoint(image), explicit.adjoint(image),
                    atol=2e-12, rtol=2e-12)
                self.assertAlmostEqual(
                    float(np.vdot(generated.forward(image), dual)),
                    float(np.vdot(image, generated.adjoint(dual))),
                    places=10,
                )

    def test_native_matches_numpy_oracle_for_gray_and_rgb(self) -> None:
        rng = np.random.default_rng(3187)
        for kernel in (
            gaussian_kernel(2.0),
            disk_kernel(3.0),
            line_kernel(11.0, 30.0),
            curved_path_kernel(11.0, 30.0, 8.0),
        ):
            for channels in (None, 3):
                with self.subTest(kernel=kernel.name, channels=channels):
                    shape = (33, 31) if channels is None else (33, 31, channels)
                    image = rng.random(shape)
                    operator = ReflectedPathOperator(
                        fit_curvilinear_exposure_chart(kernel), shape[:2])
                    self.assertTrue(operator.backend.startswith("native_cxx_"))
                    np.testing.assert_allclose(
                        operator.forward(image),
                        operator._forward_numpy(image),
                        atol=2e-14,
                        rtol=2e-14,
                    )
                    np.testing.assert_allclose(
                        operator.adjoint(image),
                        operator._adjoint_numpy(image),
                        atol=2e-14,
                        rtol=2e-14,
                    )

    def test_native_adjoint_and_dc_normalization_close(self) -> None:
        rng = np.random.default_rng(913)
        operator = ReflectedPathOperator(
            fit_curvilinear_exposure_chart(
                curved_path_kernel(13.0, 47.0, -12.0)),
            (35, 37),
        )
        latent = rng.random((35, 37, 3))
        dual = rng.random((35, 37, 3))
        left = float(np.vdot(operator.forward(latent), dual))
        right = float(np.vdot(latent, operator.adjoint(dual)))
        self.assertAlmostEqual(left, right, places=11)
        normalization = operator.adjoint_normalization(3)
        self.assertGreater(float(np.min(normalization)), 0.0)
        self.assertAlmostEqual(
            float(np.sum(normalization)),
            float(np.prod(latent.shape)),
            places=10,
        )

    def test_native_spatial_operator_matches_numpy_oracle(self) -> None:
        rng = np.random.default_rng(617)
        field = rotational_exposure(
            (35, 37),
            mean_angle_degrees=4.0,
            exposure_degrees=8.0,
            atoms=9,
        )
        operator = SpatialReflectedExposureOperator(field)
        self.assertTrue(operator.backend.startswith("native_cxx_"))
        for channels in (None, 3):
            with self.subTest(channels=channels):
                shape = (35, 37) if channels is None else (35, 37, channels)
                image = rng.random(shape)
                np.testing.assert_allclose(
                    operator.forward(image),
                    operator._forward_numpy(image),
                    atol=3e-14,
                    rtol=3e-14,
                )
                np.testing.assert_allclose(
                    operator.adjoint(image),
                    operator._adjoint_numpy(image),
                    atol=3e-14,
                    rtol=3e-14,
                )

    def test_global_fractional_measure_uses_compact_native_path(self) -> None:
        points = np.asarray(((-1.3, 0.4), (0.0, 0.0), (1.3, -0.4)))
        weights = np.asarray((0.2, 0.6, 0.2))
        field = SpatialExposureField.from_barycentric_paths(
            name="compact_global_fractional_measure",
            barycentric_flow_xy=np.zeros((33, 31, 2), dtype=np.float64),
            residual_displacements_xy=points,
            weights=weights,
            compact_global=True,
        )
        operator = SpatialReflectedExposureOperator(field)
        self.assertIsNotNone(operator._scalar_coefficients)
        self.assertIsNone(operator._coefficients)
        self.assertTrue(operator.backend.startswith("native_cxx_"))
        image = np.random.default_rng(991).random((33, 31, 3))
        np.testing.assert_allclose(
            operator.forward(image), operator._forward_numpy(image),
            atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(
            operator.adjoint(image), operator._adjoint_numpy(image),
            atol=3e-14, rtol=3e-14)

    def test_native_spatial_batch_matches_exchangeable_oracle(self) -> None:
        rng = np.random.default_rng(812)
        operators = tuple(
            SpatialReflectedExposureOperator(rotational_exposure(
                (31, 29),
                mean_angle_degrees=angle,
                exposure_degrees=6.0,
                atoms=7,
            ))
            for angle in (-3.0, 0.0, 3.0)
        )
        batch = SpatialExposureOperatorBatch(operators)
        self.assertTrue(batch.backend.endswith("_batch"))
        for channels in (None, 3):
            with self.subTest(channels=channels):
                shape = (
                    (3, 31, 29)
                    if channels is None else (3, 31, 29, channels))
                images = rng.random(shape)
                expected_forward = np.stack([
                    operator._forward_numpy(image)
                    for operator, image in zip(operators, images)
                ], axis=0)
                expected_adjoint = np.stack([
                    operator._adjoint_numpy(image)
                    for operator, image in zip(operators, images)
                ], axis=0)
                np.testing.assert_allclose(
                    batch.forward(images), expected_forward,
                    atol=3e-14, rtol=3e-14)
                np.testing.assert_allclose(
                    batch.adjoint(images), expected_adjoint,
                    atol=3e-14, rtol=3e-14)
                dual = rng.random(shape)
                self.assertAlmostEqual(
                    float(np.vdot(batch.forward(images), dual)),
                    float(np.vdot(images, batch.adjoint(dual))),
                    places=11,
                )


if __name__ == "__main__":
    unittest.main()
