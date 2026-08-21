"""Exactness and recovery controls for unified spatial exposure transport."""

from __future__ import annotations

import unittest

import numpy as np

from .decomposition import apply_reflect
from .kernels import curved_path_kernel, line_kernel
from .spatial_transport import (
    SpatialExposureField,
    SpatialReflectedExposureOperator,
    refine_spatial_exposure,
    rotational_exposure,
    shear_path_exposure,
)
from .test_uncertainty import _fixture
from .workbench import BlurSpec, DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(mse))


class SpatialExposureTransportTests(unittest.TestCase):
    def test_global_kernel_is_exact_limit_of_spatial_operator(self) -> None:
        rng = np.random.default_rng(417)
        latent = rng.random((35, 37, 3))
        dual = rng.random(latent.shape)
        for kernel in (
            line_kernel(11.0, 31.0),
            curved_path_kernel(11.0, 31.0, 8.0),
        ):
            with self.subTest(kernel=kernel.name):
                field = SpatialExposureField.from_global_kernel(
                    kernel, latent.shape[:2])
                operator = SpatialReflectedExposureOperator(field)
                np.testing.assert_allclose(
                    operator.forward(latent),
                    apply_reflect(latent, kernel),
                    atol=2e-15,
                    rtol=2e-15,
                )
                left = float(np.vdot(operator.forward(latent), dual))
                right = float(np.vdot(latent, operator.adjoint(dual)))
                self.assertAlmostEqual(left, right, places=11)

    def test_deterministic_shear_is_single_atom_limit(self) -> None:
        truth = _fixture(64)
        field = shear_path_exposure(
            truth.shape[:2], shear=0.1, residual_length=0.0, atoms=1)
        observation = SpatialReflectedExposureOperator(field).forward(truth)
        result = refine_spatial_exposure(observation, field, passes=64)
        self.assertEqual(field.atom_count, 1)
        self.assertLess(
            field.diagnostics()["centered_mixing_rms"], 1e-14)
        self.assertGreater(
            _psnr(result.barycentric_seed, truth),
            _psnr(observation, truth) + 10.0,
        )
        self.assertEqual(result.diagnostics["passes_used"], 0)

    def test_warp_and_mixing_recover_in_one_continuous_law(self) -> None:
        truth = _fixture(64)
        field = shear_path_exposure(
            truth.shape[:2], shear=0.1, residual_length=7.0, atoms=9)
        operator = SpatialReflectedExposureOperator(field)
        observation = operator.forward(truth)
        immutable = observation.copy()
        result = refine_spatial_exposure(
            observation, field, passes=64, ratio_limit=4.0)
        np.testing.assert_array_equal(observation, immutable)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(observation, truth) + 5.0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(result.barycentric_seed, truth) + 5.0)
        diagnostic = result.diagnostics
        self.assertEqual(
            diagnostic["method"],
            "barycentric_first_spatial_positive_exposure_transport",
        )
        self.assertGreater(
            diagnostic["field"]["centered_mixing_rms"], 1.0)
        self.assertLess(
            diagnostic["barycentric_pullback"][
                "terminal_coordinate_residual_max"],
            1e-10,
        )
        self.assertNotIn("selected", diagnostic)
        self.assertEqual(result.uncertainty.shape, observation.shape)
        self.assertTrue(np.all(np.isfinite(result.uncertainty)))
        self.assertGreater(diagnostic["uncertainty_q95"], 0.0)

    def test_rotational_exposure_recovers_spatial_warp_and_mix(self) -> None:
        truth = _fixture(64)
        field = rotational_exposure(
            truth.shape[:2],
            mean_angle_degrees=4.0,
            exposure_degrees=8.0,
            atoms=9,
        )
        observation = SpatialReflectedExposureOperator(field).forward(truth)
        result = refine_spatial_exposure(
            observation, field, passes=64, ratio_limit=4.0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(observation, truth) + 8.0)
        self.assertGreater(
            _psnr(result.image, truth), _psnr(result.barycentric_seed, truth) + 3.0)
        field_record = result.diagnostics["field"]
        self.assertGreater(field_record["barycentric_flow_rms"], 1.0)
        self.assertGreater(field_record["centered_mixing_rms"], 1.0)
        self.assertEqual(field_record["fold_fraction"], 0.0)
        self.assertLess(
            result.diagnostics["barycentric_pullback"][
                "terminal_coordinate_residual_max"],
            1e-6,
        )
        self.assertEqual(result.uncertainty.shape, observation.shape)

    def test_folded_barycentric_map_abstains_without_overwriting(self) -> None:
        truth = _fixture(48)
        height, width = truth.shape[:2]
        _, xx = np.mgrid[:height, :width]
        flow = np.zeros((height, width, 2), dtype=np.float64)
        flow[..., 0] = 2.0 * (xx - 0.5 * (width - 1))
        field = SpatialExposureField.from_barycentric_paths(
            name="orientation_reversing_fold_control",
            barycentric_flow_xy=flow,
            residual_displacements_xy=np.zeros((1, 2), dtype=np.float64),
            weights=np.ones(1, dtype=np.float64),
        )
        observation = SpatialReflectedExposureOperator(field).forward(truth)
        immutable = observation.copy()
        result = refine_spatial_exposure(observation, field, passes=64)
        np.testing.assert_array_equal(observation, immutable)
        np.testing.assert_array_equal(result.image, observation)
        self.assertEqual(
            result.diagnostics["estimation_decision"],
            "abstain_noninvertible_barycentric_map",
        )
        self.assertEqual(result.diagnostics["passes_used"], 0)
        self.assertEqual(
            result.diagnostics["stopped_by"], "geometry_fold_abstention")
        self.assertGreater(
            result.diagnostics["field"]["fold_fraction"], 0.99)
        self.assertGreater(result.diagnostics["uncertainty_q95"], 0.99)

    def test_workbench_runs_rotational_exposure_without_new_input_role(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        index = session.add_array(truth, "spatial fixture")
        record = session.synthesize(
            index,
            BlurSpec(
                kind="Rotational exposure",
                rotation_mean_degrees=4.0,
                rotation_exposure_degrees=8.0,
            ),
            read_noise_sigma=0.002,
            seed=5,
        )
        fingerprint = record.observation.copy()
        self.assertIsNone(record.kernel)
        self.assertIsNotNone(record.spatial_field)
        result = session.deblur_active(index, passes=64)
        np.testing.assert_array_equal(record.observation, fingerprint)
        self.assertGreater(record.diagnostics["psnr_gain"], 5.0)
        self.assertEqual(
            result.diagnostics["method"],
            "barycentric_first_spatial_positive_exposure_transport",
        )

    def test_workbench_runs_rolling_shutter_as_one_spatial_measure(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        index = session.add_array(truth, "rolling shutter fixture")
        record = session.synthesize(
            index,
            BlurSpec(
                kind="Rolling shutter exposure",
                rolling_mean_x=4.0,
                rolling_row_acceleration=1.0,
                rolling_exposure_extent=2.0,
            ),
            read_noise_sigma=0.002,
            seed=9,
        )
        immutable = record.observation.copy()
        self.assertIsNone(record.kernel)
        self.assertIsNotNone(record.spatial_field)
        assert record.spatial_field is not None
        self.assertGreater(float(np.std(
            record.spatial_field.barycentric_flow_xy[..., 0])), 0.5)
        result = session.deblur_active(index, passes=64)
        np.testing.assert_array_equal(record.observation, immutable)
        self.assertGreater(record.diagnostics["psnr_gain"], 4.0)
        self.assertEqual(
            result.diagnostics["method"],
            "barycentric_first_spatial_positive_exposure_transport",
        )


if __name__ == "__main__":
    unittest.main()
