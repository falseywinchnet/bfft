"""Consensus and gauge controls for multi-observation spatial estimation."""

from __future__ import annotations

import unittest

import numpy as np

from .spatial_estimation import (
    deblur_rotation_consensus,
    estimate_rotation_consensus,
)
from .spatial_transport import (
    SpatialReflectedExposureOperator,
    rotational_exposure,
)
from .test_uncertainty import _fixture
from .workbench import BlurSpec, DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    mse = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(mse))


class SpatialEstimationTests(unittest.TestCase):
    def _sequence(self) -> tuple[np.ndarray, list[np.ndarray]]:
        truth = _fixture(96)
        observations = []
        for index, angle in enumerate((-4.0, 0.0, 4.0)):
            field = rotational_exposure(
                truth.shape[:2],
                mean_angle_degrees=angle,
                exposure_degrees=4.0,
                atoms=9,
            )
            blurred = SpatialReflectedExposureOperator(field).forward(truth)
            observations.append(np.clip(
                blurred
                + np.random.default_rng(index).normal(0.0, 0.002, blurred.shape),
                0.0,
                1.0,
            ))
        return truth, observations

    def test_rotation_consensus_recovers_continuous_trajectory(self) -> None:
        _, observations = self._sequence()
        estimate = estimate_rotation_consensus(observations, duty_cycle=1.0)
        np.testing.assert_allclose(
            estimate.relative_mean_angles_degrees,
            (-4.0, 0.0, 4.0),
            atol=0.01,
        )
        np.testing.assert_allclose(
            estimate.exposure_extents_degrees,
            (4.0, 4.0, 4.0),
            atol=0.01,
        )
        self.assertLess(estimate.cycle_rms_degrees, 0.001)
        self.assertGreater(estimate.confidence, 0.9)
        self.assertTrue(estimate.relative_motion_observable)
        self.assertTrue(estimate.common_rotation_gauge_unidentifiable)

    def test_shared_latent_consensus_beats_every_capture(self) -> None:
        truth, observations = self._sequence()
        result = deblur_rotation_consensus(
            observations, duty_cycle=1.0, passes=64)
        self.assertGreater(
            _psnr(result.image, truth),
            max(_psnr(item, truth) for item in observations) + 2.0,
        )
        self.assertGreater(
            _psnr(result.image, truth),
            _psnr(np.mean(observations, axis=0), truth) + 10.0,
        )
        self.assertEqual(result.uncertainty.shape, truth.shape)
        self.assertGreater(result.diagnostics["uncertainty_q95"], 0.0)
        self.assertNotIn("selected", result.diagnostics)

    def test_identical_observations_preserve_common_exposure_gauge(self) -> None:
        truth = _fixture(64)
        field = rotational_exposure(
            truth.shape[:2],
            mean_angle_degrees=0.0,
            exposure_degrees=6.0,
            atoms=9,
        )
        observation = SpatialReflectedExposureOperator(field).forward(truth)
        estimate = estimate_rotation_consensus(
            [observation, observation, observation], duty_cycle=1.0)
        self.assertFalse(estimate.relative_motion_observable)
        self.assertLess(float(np.ptp(
            estimate.relative_mean_angles_degrees)), 1e-4)
        self.assertLess(float(np.max(
            estimate.exposure_extents_degrees)), 1e-4)
        result = deblur_rotation_consensus(
            [observation, observation, observation], passes=16)
        self.assertEqual(
            result.diagnostics["estimation_decision"],
            "abstain_common_rotation_and_exposure_gauge",
        )
        self.assertLess(float(np.max(np.abs(
            result.image - observation))), 1e-10)

    def test_workbench_consensus_uses_explicit_registered_captures(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        indices = []
        observations = []
        for capture, angle in enumerate((-4.0, 0.0, 4.0)):
            index = session.add_array(truth, f"capture {capture}")
            record = session.synthesize(
                index,
                BlurSpec(
                    kind="Rotational exposure",
                    rotation_mean_degrees=angle,
                    rotation_exposure_degrees=4.0,
                ),
                read_noise_sigma=0.002,
                seed=capture,
            )
            indices.append(index)
            observations.append(record.observation.copy())
        target, result = session.deblur_rotation_consensus(
            indices, reference_index=1, passes=64)
        self.assertEqual(target, indices[1])
        for index, observation in zip(indices, observations):
            np.testing.assert_array_equal(
                session.sources[index].observation, observation)
        self.assertGreater(
            _psnr(result.image, truth),
            max(_psnr(item, truth) for item in observations) + 1.0,
        )
        self.assertIs(session.sources[target].result, result.image)
        self.assertEqual(
            session.sources[target].diagnostics["truth_role"],
            "evaluation_only",
        )

    def test_workbench_rejects_unrelated_synthetic_truths(self) -> None:
        session = DeblurSession()
        first = session.add_array(_fixture(48), "first scene")
        second = session.add_array(1.0 - _fixture(48), "second scene")
        spec = BlurSpec(
            kind="Rotational exposure",
            rotation_mean_degrees=0.0,
            rotation_exposure_degrees=4.0,
        )
        session.synthesize(first, spec)
        session.synthesize(second, spec)
        with self.assertRaisesRegex(ValueError, "one source truth"):
            session.deblur_rotation_consensus([first, second], passes=2)


if __name__ == "__main__":
    unittest.main()
