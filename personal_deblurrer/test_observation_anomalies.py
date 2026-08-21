"""Unified affine blur and censored-sensor anomaly controls."""

from __future__ import annotations

import unittest

import numpy as np

from .composed_transport import (
    compose_positive_transports,
    radial_scale_measure,
    refine_consolidated_transport,
)
from .observation_anomalies import (
    astigmatic_scale_measure,
    bounded_linear_sensor_observation,
    ghost_measure,
    rotation_exposure_measure,
    shear_exposure_measure,
    translation_mixture_measure,
)
from .test_uncertainty import _fixture
from .workbench import BlurSpec, DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(mse))


class ObservationAnomalyTests(unittest.TestCase):
    def test_workbench_transports_bounded_sensor_anomalies(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        index = session.add_array(truth, "bounded sensor fixture")
        record = session.synthesize(
            index,
            BlurSpec(
                kind="Double radial exposure",
                radial_fractional_extent=0.045,
                exposure_gain=1.8,
                sensor_quantization_levels=32,
                dead_pixel_period=53,
            ),
        )
        immutable = record.observation.copy()
        self.assertIsNotNone(record.transport_observation)
        self.assertIsNotNone(record.observation_bounds)
        result = session.deblur_active(index, passes=96)
        np.testing.assert_array_equal(record.observation, immutable)
        self.assertGreater(record.diagnostics["psnr_gain"], 8.0)
        self.assertTrue(
            result.diagnostics["observation_bounds"]["interval_censored"])

    def test_workbench_anomaly_catalog_maps_to_the_same_transport_interface(
        self,
    ) -> None:
        shape = (48, 50)
        for kind in (
            "Decentered double radial",
            "Ghost copy anomaly",
            "Shear exposure",
            "Astigmatic scale exposure",
            "Radial rotation ghost",
            "Compound lens anomaly",
        ):
            with self.subTest(kind=kind):
                transport = BlurSpec(kind=kind).observation_transport(shape)
                self.assertIsNotNone(transport)
                assert transport is not None
                self.assertFalse(
                    transport.diagnostics()["family_classification"])
                self.assertLess(transport.storage_bytes, 2048)

    def test_affine_generators_are_positive_normalized_measures(self) -> None:
        shape = (35, 37)
        measures = (
            translation_mixture_measure(
                np.asarray(((0.0, 0.0), (2.0, -1.0))),
                weights=np.asarray((0.85, 0.15))),
            ghost_measure((3.0, -2.0), ghost_mass=0.12),
            rotation_exposure_measure(shape, exposure_degrees=5.0),
            shear_exposure_measure(shape, fractional_extent=0.035),
            astigmatic_scale_measure(
                shape, fractional_extent=0.04, angle_degrees=27.0),
        )
        rng = np.random.default_rng(78)
        latent = rng.random((*shape, 3))
        dual = rng.random(latent.shape)
        for measure in measures:
            with self.subTest(measure=measure.name):
                transport = measure.to_transport(shape)
                np.testing.assert_allclose(
                    transport.forward(np.ones(shape)), 1.0, atol=2e-15)
                self.assertAlmostEqual(
                    float(np.vdot(transport.forward(latent), dual)),
                    float(np.vdot(latent, transport.adjoint(dual))),
                    places=11,
                )
                self.assertFalse(
                    transport.diagnostics()["family_classification"])

    def test_radial_rotation_ghost_compose_without_a_solver_branch(self) -> None:
        truth = _fixture(64)
        shape = truth.shape[:2]
        radial = radial_scale_measure(
            shape,
            fractional_extent=0.035,
            center_xy=(29.0, 34.0),
        ).to_transport(shape)
        rotation = rotation_exposure_measure(
            shape,
            exposure_degrees=2.0,
            center_xy=(33.0, 30.0),
        ).to_transport(shape)
        ghost = ghost_measure((2.5, -1.5), ghost_mass=0.06).to_transport(shape)
        compound = compose_positive_transports(
            compose_positive_transports(radial, rotation), ghost)
        sequential = ghost.forward(rotation.forward(radial.forward(truth)))
        np.testing.assert_allclose(
            compound.forward(truth), sequential, atol=0.0, rtol=0.0)
        immutable = sequential.copy()
        result = refine_consolidated_transport(
            sequential, compound, passes=96, ratio_limit=4.0)
        np.testing.assert_array_equal(sequential, immutable)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(sequential, truth) + 3.0)
        self.assertFalse(result.diagnostics["operator_decomposition"])
        self.assertFalse(result.diagnostics["family_classification"])
        self.assertLess(compound.storage_bytes, 1024)

    def test_interval_censoring_handles_saturation_quantization_and_dead_pixels(
        self,
    ) -> None:
        truth = _fixture(64)
        shape = truth.shape[:2]
        stage = radial_scale_measure(
            shape, fractional_extent=0.045).to_transport(shape)
        transport = compose_positive_transports(stage, stage)
        clean_observation = transport.forward(truth)
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        invalid = ((7 * xx + 11 * yy) % 53) == 0
        sensor = bounded_linear_sensor_observation(
            clean_observation,
            exposure_gain=1.8,
            quantization_levels=32,
            invalid_mask=invalid,
        )
        immutable = sensor.measured.copy()
        bounded = refine_consolidated_transport(
            sensor.transport_center,
            transport,
            passes=96,
            observation_bounds=sensor.bounds,
        )
        naive = refine_consolidated_transport(
            sensor.measured,
            transport,
            passes=96,
        )
        np.testing.assert_array_equal(sensor.measured, immutable)
        self.assertGreater(_psnr(bounded.image, truth), _psnr(naive.image, truth) + 4.0)
        self.assertGreater(_psnr(bounded.image, truth), _psnr(sensor.measured, truth))
        diagnostics = bounded.diagnostics
        self.assertTrue(
            diagnostics["observation_bounds"]["interval_censored"])
        self.assertLess(diagnostics["observation_authority_fraction"], 0.99)
        self.assertGreater(diagnostics["mean_interval_width"], 0.0)
        self.assertGreater(
            diagnostics["observation_bounds"]["maximum_code_fraction"], 0.2)
        self.assertEqual(
            diagnostics["forward_rms_semantics"],
            "distance_to_admissible_observation_interval",
        )


if __name__ == "__main__":
    unittest.main()
