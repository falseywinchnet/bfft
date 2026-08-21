"""Parity gates for the compact generated global exposure operator."""

from __future__ import annotations

import unittest

import numpy as np

from personal_deblurrer.spatial_transport import (
    CompactGlobalExposureField,
    CompactGlobalExposureOperatorBatch,
    CompactGlobalReflectedExposureOperator,
    SpatialExposureField,
    SpatialReflectedExposureOperator,
    pullback_barycentric_coordinates,
    pullback_compact_global_values,
)


class CompactGlobalExposureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.shape = (17, 19)
        self.points = np.asarray([
            [0.0, 0.0],
            [1.3, -2.2],
            [-1.3, 2.2],
            [3.1, 0.7],
            [-3.1, -0.7],
        ])
        self.weights = np.asarray([0.20, 0.15, 0.15, 0.25, 0.25])
        materialized = SpatialExposureField.from_barycentric_paths(
            "materialized_reference",
            np.zeros((*self.shape, 2), dtype=np.float64),
            self.points,
            self.weights,
            compact_global=True,
        )
        compact = CompactGlobalExposureField(
            "compact_generated", self.shape, self.points, self.weights)
        self.reference = SpatialReflectedExposureOperator(materialized)
        self.compact = CompactGlobalReflectedExposureOperator(compact)

    def test_forward_adjoint_and_rgb_match_materialized_reference(self) -> None:
        rng = np.random.default_rng(12)
        for channels in (None, 3):
            shape = self.shape if channels is None else (*self.shape, channels)
            source = rng.normal(size=shape)
            cotangent = rng.normal(size=shape)
            reference_forward = self.reference.forward(source)
            compact_forward = self.compact.forward(source)
            np.testing.assert_allclose(
                compact_forward, reference_forward, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                self.compact.adjoint(cotangent),
                self.reference.adjoint(cotangent),
                atol=1e-12,
                rtol=1e-12,
            )
            self.assertAlmostEqual(
                float(np.vdot(compact_forward, cotangent)),
                float(np.vdot(source, self.compact.adjoint(cotangent))),
                places=11,
            )

    def test_unit_mass_and_storage_reduction(self) -> None:
        np.testing.assert_allclose(
            self.compact.forward(np.ones(self.shape)), 1.0,
            atol=1e-13, rtol=1e-13)
        reference_plan_bytes = (
            self.reference._source_indices.nbytes
            + self.reference._scalar_coefficients.nbytes)
        self.assertLess(self.compact.storage_bytes, reference_plan_bytes / 3)

    def test_translation_plus_mixing_matches_materialized_joint_field(self) -> None:
        translation = np.asarray((2.25, -1.4))
        flow = np.broadcast_to(translation, (*self.shape, 2))
        materialized = SpatialExposureField.from_barycentric_paths(
            "translated_materialized_reference",
            flow,
            self.points,
            self.weights,
            compact_global=True,
        )
        compact_field = CompactGlobalExposureField(
            "translated_compact",
            self.shape,
            self.points,
            self.weights,
            translation,
        )
        reference = SpatialReflectedExposureOperator(materialized)
        compact = CompactGlobalReflectedExposureOperator(compact_field)
        random = np.random.default_rng(91)
        source = random.normal(size=(*self.shape, 3))
        cotangent = random.normal(size=source.shape)
        np.testing.assert_allclose(
            compact.forward(source), reference.forward(source),
            atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            compact.adjoint(cotangent), reference.adjoint(cotangent),
            atol=1e-12, rtol=1e-12)

        observation = compact.forward(source)
        pulled, record = pullback_compact_global_values(
            observation, compact_field)
        reference_pulled, _, _ = pullback_barycentric_coordinates(
            observation, materialized.barycentric_field())
        np.testing.assert_allclose(
            pulled, reference_pulled,
            atol=1e-12, rtol=1e-12)
        self.assertEqual(
            record["method"], "analytic_constant_translation_pullback")

    def test_parallel_batch_matches_individual_compact_operators(self) -> None:
        fields = tuple(CompactGlobalExposureField(
            f"compact_batch_{index}",
            self.shape,
            self.points * (1.0 + 0.1 * index),
            self.weights,
        ) for index in range(4))
        operators = tuple(
            CompactGlobalReflectedExposureOperator(field)
            for field in fields)
        batch = CompactGlobalExposureOperatorBatch(operators)
        random = np.random.default_rng(918)
        for channels in (None, 3):
            shape = (
                (len(operators), *self.shape)
                if channels is None else
                (len(operators), *self.shape, channels))
            images = random.normal(size=shape)
            np.testing.assert_allclose(
                batch.forward(images),
                np.stack([
                    operator.forward(image)
                    for operator, image in zip(operators, images)]),
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                batch.adjoint(images),
                np.stack([
                    operator.adjoint(image)
                    for operator, image in zip(operators, images)]),
                atol=1e-12,
                rtol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
