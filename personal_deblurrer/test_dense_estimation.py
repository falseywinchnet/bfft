"""Controls for continuous dense flow estimation and shared reconstruction."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import shift

from .dense_estimation import (
    deblur_dense_pair_consensus,
    estimate_dense_pair_exposure,
)
from .spatial_transport import (
    SpatialExposureField,
    SpatialReflectedExposureOperator,
)
from .test_uncertainty import _fixture
from .workbench import DeblurSession


def _psnr(image: np.ndarray, truth: np.ndarray) -> float:
    error = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
    return float(-10.0 * np.log10(error))


class DenseExposureEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = _fixture(64)
        height, width = cls.truth.shape
        yy, xx = np.mgrid[:height, :width]
        center_x = 0.5 * (width - 1)
        center_y = 0.5 * (height - 1)
        cls.relative_flow = np.empty((height, width, 2), dtype=np.float64)
        # Translation, shear, cross-axis affine motion, and smooth local warp
        # coexist in one field. No generator label is supplied to estimation.
        cls.relative_flow[..., 0] = (
            2.5
            + 0.025 * (yy - center_y)
            + 0.55 * np.sin(2.0 * np.pi * yy / height)
        )
        cls.relative_flow[..., 1] = (
            -1.5
            + 0.018 * (xx - center_x)
            + 0.35 * np.sin(2.0 * np.pi * xx / width)
        )
        duty_cycle = 0.5
        times = np.linspace(-0.5, 0.5, 7)[:, None, None, None]
        residual = times * duty_cycle * cls.relative_flow[None, ...]
        fields = tuple(
            SpatialExposureField.from_barycentric_paths(
                name=f"dense_control_{index}",
                barycentric_flow_xy=sign * 0.5 * cls.relative_flow,
                residual_displacements_xy=residual,
                weights=np.ones((7, height, width), dtype=np.float64),
            )
            for index, sign in enumerate((-1.0, 1.0))
        )
        cls.observations = tuple(
            SpatialReflectedExposureOperator(field).forward(cls.truth)
            for field in fields
        )
        cls.fields = fields
        cls.estimate = estimate_dense_pair_exposure(
            *cls.observations, duty_cycle=duty_cycle)
        cls.result = deblur_dense_pair_consensus(
            *cls.observations, duty_cycle=duty_cycle, passes=64)

    def test_one_field_recovers_affine_and_local_motion_together(self) -> None:
        endpoint_error = np.sqrt(np.sum(
            (self.estimate.forward_sampling_flow_xy - self.relative_flow) ** 2,
            axis=2,
        ))
        self.assertLess(float(np.mean(endpoint_error)), 0.3)
        self.assertLess(float(np.quantile(endpoint_error, 0.9)), 0.5)
        self.assertTrue(self.estimate.relative_motion_observable)
        self.assertGreater(
            self.estimate.diagnostics["transport_authority_mean"], 0.75)
        self.assertNotIn("selected", self.estimate.diagnostics)
        self.assertNotIn("motion_class", self.estimate.diagnostics)
        for direction in ("forward", "reverse"):
            levels = self.estimate.diagnostics[direction]["levels"]
            self.assertEqual(
                [level["gradient_constancy_weight"] for level in levels],
                [0.0] * (len(levels) - 1) + [0.2],
            )
            self.assertEqual(
                [level["robust_flow_action"] for level in levels],
                [False] * (len(levels) - 1) + [True],
            )
            for level in self.estimate.diagnostics[direction]["levels"]:
                energy = np.asarray(level["energy_trace"])
                self.assertTrue(np.all(np.diff(energy) <= 1e-14))
                self.assertLessEqual(max(level["cg_iterations"]), 60)

    def test_dense_exposure_consensus_beats_both_inputs_and_average(self) -> None:
        best_capture = max(
            _psnr(item, self.truth) for item in self.observations)
        average = _psnr(np.mean(self.observations, axis=0), self.truth)
        restored = _psnr(self.result.image, self.truth)
        self.assertGreater(restored, best_capture + 8.0)
        self.assertGreater(restored, average + 4.0)
        self.assertEqual(self.result.uncertainty.shape, self.truth.shape)
        self.assertTrue(np.all(np.isfinite(self.result.uncertainty)))
        self.assertEqual(
            self.result.diagnostics["reconstruction_method"],
            "shared_latent_spatial_positive_exposure_transport",
        )
        self.assertNotIn("selected", self.result.diagnostics)

    def test_identical_pair_preserves_the_common_warp_gauge(self) -> None:
        result = deblur_dense_pair_consensus(
            self.truth, self.truth, passes=16, warp_iterations=3)
        self.assertFalse(result.estimate.relative_motion_observable)
        self.assertEqual(
            result.diagnostics["estimation_decision"],
            "abstain_common_warp_and_exposure_gauge",
        )
        self.assertLess(float(np.max(np.abs(
            result.image - self.truth))), 1e-12)
        self.assertEqual(result.diagnostics["passes_used"], 0)
        self.assertEqual(
            result.estimate.diagnostics["fast_path"],
            "machine_precision_identical_observations",
        )
        self.assertEqual(
            result.estimate.diagnostics["forward"]["levels"], [])

    def test_workbench_pair_path_preserves_inputs_and_truth_gauge(self) -> None:
        session = DeblurSession()
        indices = [
            session.add_array(self.truth, f"dense capture {index}")
            for index in range(2)
        ]
        immutable = []
        for index, observation, field in zip(
            indices, self.observations, self.fields
        ):
            record = session.sources[index]
            record.observation = observation.copy()
            record.spatial_field = field
            record.mode = "synthetic_blur"
            record.synthetic_truth_available = True
            immutable.append(record.observation.copy())
        target, result = session.deblur_dense_pair_consensus(
            indices[0], indices[1], target_index=indices[1], passes=64)
        self.assertEqual(target, indices[1])
        for index, observation in zip(indices, immutable):
            np.testing.assert_array_equal(
                session.sources[index].observation, observation)
        self.assertTrue(
            session.sources[target].diagnostics[
                "synthetic_truth_gauge_matches_output"])
        self.assertGreater(
            session.sources[target].diagnostics[
                "psnr_gain_over_best_capture"],
            8.0,
        )
        self.assertIs(session.sources[target].result, result.image)

    def test_joint_coverage_continues_through_individual_visibility_fold(self) -> None:
        size = 64
        background = _fixture(size)
        yy, xx = np.mgrid[:size, :size]
        mask = (
            (xx - 0.5 * size) ** 2 / (0.22 * size) ** 2
            + (yy - 0.5 * size) ** 2 / (0.28 * size) ** 2
            < 1.0
        ).astype(np.float64)
        foreground = (
            0.15
            + 0.75 * ((np.floor(xx / 4) + np.floor(yy / 5)) % 2)
        )
        truth = mask * foreground + (1.0 - mask) * background
        observations = []
        for displacement in (-3.0, 3.0):
            moved_mask = shift(
                mask, (0.0, displacement), order=1,
                mode="constant", cval=0.0, prefilter=False)
            moved_foreground = shift(
                foreground, (0.0, displacement), order=1,
                mode="reflect", prefilter=False)
            observations.append(
                moved_mask * moved_foreground
                + (1.0 - moved_mask) * background)
        result = deblur_dense_pair_consensus(
            observations[0], observations[1], duty_cycle=0.0, passes=64)
        self.assertGreater(
            max(result.diagnostics["fold_fractions"]), 0.0)
        self.assertEqual(
            result.diagnostics["execution_chart"], "direct_joint_operator")
        self.assertEqual(
            result.diagnostics["estimation_decision"],
            "relative_dense_flow_supported",
        )
        self.assertEqual(
            result.diagnostics["coverage_decision"],
            "joint_coverage_direct_transport_over_individual_folds",
        )
        self.assertEqual(
            result.diagnostics["unsupported_visibility_fraction"], 0.0)
        self.assertLess(result.diagnostics["ownership_entropy_min"], 0.1)
        self.assertGreater(
            _psnr(result.image, truth),
            _psnr(np.mean(observations, axis=0), truth) + 0.3,
        )
        self.assertNotIn("selected", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
