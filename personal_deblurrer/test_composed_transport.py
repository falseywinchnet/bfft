"""Composition and double-radial controls for one observation transport."""

from __future__ import annotations

import unittest

import numpy as np

from .composed_transport import (
    PositiveObservationTransport,
    compose_affine_measures,
    compose_positive_transports,
    radial_scale_measure,
    refine_consolidated_transport,
)
from .spatial_transport import SpatialReflectedExposureOperator
from .test_uncertainty import _fixture
from .workbench import BlurSpec, DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(mse))


class ComposedObservationTransportTests(unittest.TestCase):
    def test_exact_discrete_composition_closes_forward_and_adjoint(self) -> None:
        shape = (31, 33)
        first = radial_scale_measure(shape, fractional_extent=0.035).to_transport(
            shape)
        second = radial_scale_measure(shape, fractional_extent=0.055).to_transport(
            shape)
        composed = compose_positive_transports(first, second)
        rng = np.random.default_rng(194)
        latent = rng.random((*shape, 3))
        dual = rng.random(latent.shape)
        np.testing.assert_allclose(
            composed.forward(latent),
            second.forward(first.forward(latent)),
            atol=8e-16,
            rtol=8e-16,
        )
        self.assertAlmostEqual(
            float(np.vdot(composed.forward(latent), dual)),
            float(np.vdot(latent, composed.adjoint(dual))),
            places=11,
        )
        self.assertFalse(composed.diagnostics()["family_classification"])

    def test_affine_radial_composition_adds_log_scale_coordinates(self) -> None:
        shape = (41, 43)
        first = radial_scale_measure(shape, fractional_extent=0.04)
        second = radial_scale_measure(shape, fractional_extent=0.07)
        composed = compose_affine_measures(first, second)
        first_log = np.log(first.matrices[:, 0, 0])
        second_log = np.log(second.matrices[:, 0, 0])
        expected = np.asarray([
            left + right for right in second_log for left in first_log
        ])
        np.testing.assert_allclose(
            np.log(composed.matrices[:, 0, 0]), expected, atol=2e-16)
        np.testing.assert_allclose(
            composed.matrices[:, 0, 1], 0.0, atol=0.0)
        self.assertEqual(composed.atom_count, 9)
        self.assertFalse(composed.diagnostics()["family_classification"])

    def test_double_radial_direction_is_intrinsic_local_moment_flow(self) -> None:
        shape = (65, 67)
        stage = radial_scale_measure(
            shape, fractional_extent=0.055).to_transport(shape)
        composed = compose_positive_transports(stage, stage)
        jet = composed.local_moment_jet()
        height, width = shape
        yy, xx = np.mgrid[:height, :width]
        center_x = 0.5 * (width - 1)
        center_y = 0.5 * (height - 1)
        radial = np.stack((xx - center_x, yy - center_y), axis=-1)
        radius = np.linalg.norm(radial, axis=-1)
        radial /= np.maximum(radius[..., None], 1.0)
        alignment = np.abs(np.sum(
            jet.principal_direction_xy * radial, axis=-1))
        interior = (
            (yy >= 6) & (yy < height - 6)
            & (xx >= 6) & (xx < width - 6)
            & (radius >= 6.0) & jet.supported
        )
        self.assertGreater(float(np.mean(alignment[interior])), 0.985)
        self.assertFalse(jet.diagnostics["family_classification"])
        self.assertGreater(float(np.mean(jet.supported)), 0.9)

    def test_double_radial_is_recovered_as_one_consolidated_operator(self) -> None:
        truth = _fixture(64)
        stage_field = radial_scale_measure(
            truth.shape[:2], fractional_extent=0.05).to_spatial_field(
                truth.shape[:2])
        stage_native = SpatialReflectedExposureOperator(stage_field)
        stage = PositiveObservationTransport.from_spatial_field(stage_field)
        composed = compose_positive_transports(stage, stage)
        observation = stage_native.forward(stage_native.forward(truth))
        immutable = observation.copy()
        result = refine_consolidated_transport(
            observation,
            composed,
            passes=96,
            ratio_limit=4.0,
        )
        np.testing.assert_array_equal(observation, immutable)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(observation, truth) + 4.0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(result.adjoint_seed, truth) + 2.0)
        self.assertEqual(
            result.diagnostics["method"],
            "one_consolidated_positive_observation_transport_inverse",
        )
        self.assertFalse(result.diagnostics["operator_decomposition"])
        self.assertFalse(result.diagnostics["family_classification"])
        self.assertTrue(result.diagnostics["observation_unchanged"])
        self.assertLess(result.diagnostics["forward_rms"], 0.003)

    def test_identity_is_exact_fixed_point(self) -> None:
        truth = _fixture(33)
        identity = PositiveObservationTransport.identity(truth.shape[:2])
        result = refine_consolidated_transport(truth, identity, passes=8)
        np.testing.assert_allclose(result.image, truth, atol=2e-14)
        np.testing.assert_allclose(identity.forward(truth), truth, atol=0.0)

    def test_large_radial_composition_keeps_a_matrix_free_plan(self) -> None:
        shape = (1024, 1024)
        stage = radial_scale_measure(shape).to_transport(shape)
        composed = compose_positive_transports(stage, stage)
        self.assertLess(stage.storage_bytes, 1024)
        self.assertLess(composed.storage_bytes, 1024)
        self.assertTrue(composed.diagnostics()["matrix_free_composition"])
        self.assertEqual(composed.contribution_count, 144)

    def test_workbench_double_radial_uses_one_consolidated_measure(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        index = session.add_array(truth, "double radial fixture")
        record = session.synthesize(
            index,
            BlurSpec(
                kind="Double radial exposure",
                radial_fractional_extent=0.05,
            ),
        )
        immutable = record.observation.copy()
        self.assertIsNone(record.kernel)
        self.assertIsNone(record.spatial_field)
        self.assertIsNotNone(record.observation_transport)
        result = session.deblur_active(index, passes=96)
        np.testing.assert_array_equal(record.observation, immutable)
        self.assertGreater(record.diagnostics["psnr_gain"], 8.0)
        self.assertEqual(
            result.diagnostics["method"],
            "one_consolidated_positive_observation_transport_inverse",
        )


if __name__ == "__main__":
    unittest.main()
