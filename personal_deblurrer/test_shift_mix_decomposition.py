"""Invariants for deterministic-transport then centered-mixing recovery."""

from __future__ import annotations

import unittest

import numpy as np

from .decomposition import (
    estimate_relative_shift,
    factor_transport_mix,
    image_fingerprint,
    shift_image_reflect,
    single_observation_shift_policy,
    two_stage_deblur_blind,
    two_stage_deblur_known,
)
from .kernels import (
    curved_path_kernel,
    disk_kernel,
    gaussian_kernel,
    line_kernel,
    translated_kernel,
)
from .synthetic import degrade
from .test_uncertainty import _fixture
from .workbench import BlurSpec, DeblurSession, DeblurWorkbenchApp


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first) - np.asarray(second)) ** 2))


class ShiftMixDecompositionTests(unittest.TestCase):
    def test_gui_shift_controls_enter_blur_spec_not_session_kwargs(self) -> None:
        class Values:
            fields = {
                "blur_kind": "Line", "blur_sigma": 2.0,
                "blur_radius": 3.0, "blur_length": 9.0,
                "blur_angle": 30.0, "blur_bend": 4.0,
                "blur_extent": 7.0, "shift_x": 3.0,
                "shift_y": -2.0, "synthetic_seed": 0,
            }

            def get_value(self, tag):
                return self.fields[tag]

        spec = DeblurWorkbenchApp(Values()).spec()
        self.assertEqual(spec.shift_x, 3.0)
        self.assertEqual(spec.shift_y, -2.0)
        np.testing.assert_allclose(spec.kernel().centroid, (3.0, -2.0), atol=1e-12)

    def test_positive_kernel_factorization_separates_shift_and_mix(self) -> None:
        centered = line_kernel(9.0, 30.0)
        shifted = translated_kernel(centered, (3.0, -2.0))
        factor = factor_transport_mix(shifted)
        np.testing.assert_allclose(
            factor.deterministic_shift_xy, (3.0, -2.0), atol=1e-12)
        np.testing.assert_allclose(
            factor.mixing_covariance, centered.covariance, atol=1e-12)
        self.assertTrue(factor.shift_detected)
        self.assertLess(factor.phase_factorization_error, 1e-12)

    def test_relative_shift_is_observed_from_shared_scene_coordinates(self) -> None:
        reference = _fixture(96)
        moving = shift_image_reflect(reference, (4.0, -3.0))
        estimate = estimate_relative_shift(reference, moving)
        self.assertTrue(estimate.observable)
        np.testing.assert_allclose(estimate.shift_xy, (-4.0, 3.0), atol=0.25)

    def test_single_observation_absolute_shift_is_declared_gauge(self) -> None:
        estimate = single_observation_shift_policy()
        self.assertFalse(estimate.observable)
        self.assertEqual(
            estimate.reason, "absolute_translation_is_single_image_gauge")

    def test_known_center_mixes_recover_without_mutating_observation(self) -> None:
        truth = _fixture(96)
        for kernel in (
            gaussian_kernel(2.0),
            disk_kernel(3.0),
            line_kernel(11.0, 0.0),
            curved_path_kernel(11.0, 0.0, 4.0),
        ):
            with self.subTest(kernel=kernel.name):
                observation = degrade(
                    truth, kernel, boundary="reflect", clip=False)
                before = image_fingerprint(observation)
                result = two_stage_deblur_known(
                    observation, kernel, passes=12, reference=truth)
                self.assertEqual(before, image_fingerprint(observation))
                self.assertTrue(result.diagnostics["observation_unchanged"])
                self.assertGreater(result.diagnostics["psnr_gain"], 2.5)
                self.assertLess(_mse(result.image, truth), _mse(observation, truth))
                support = result.diagnostics["support_gate"]
                self.assertEqual(support["otf_evaluations"], 1)
                self.assertEqual(
                    support["forward_evaluations"],
                    support["passes_used"] + 1,
                )
                self.assertEqual(
                    support["adjoint_evaluations"], support["passes_used"])
                self.assertEqual(
                    support["redundant_forward_evaluations_removed"],
                    support["passes_used"],
                )
                self.assertEqual(support["descent_method"], "optimal_positive_line")
                self.assertEqual(support["moment_transport_evaluations"], 1)
                self.assertEqual(support["separate_moment_transports_avoided"], 1)
                self.assertTrue(np.all(np.isfinite(support["step_trace"])))
                self.assertTrue(np.all(np.asarray(support["step_trace"]) >= 0.0))
                trace = np.asarray(result.diagnostics["mixing_residual_trace"])
                self.assertTrue(np.all(np.diff(trace) <= 1e-12))
                if kernel.name.startswith("line_"):
                    self.assertGreater(support["dead_fraction"], 0.05)
                    characteristic = result.diagnostics[
                        "characteristic_transport"]
                    self.assertTrue(characteristic["selected"])
                    line_constraint = characteristic["line_constraint"]
                    self.assertEqual(
                        line_constraint["seed_policy"],
                        "minimum_correction_plus_longitudinal_flux_action",
                    )
                    self.assertGreater(
                        characteristic["line_constraint_authority"], 0.0)
                    self.assertGreater(result.diagnostics["psnr_gain"], 5.0)
                elif kernel.name.startswith(("gaussian_", "disk_")):
                    characteristic = result.diagnostics[
                        "characteristic_transport"]
                    self.assertTrue(characteristic["selected"])
                    self.assertEqual(
                        characteristic["method"],
                        "continuous_positive_exposure_transport",
                    )
                    self.assertLess(
                        characteristic["line_constraint_authority"], 1e-12)

    def test_shift_is_recovered_before_centered_mix(self) -> None:
        truth = _fixture(96)
        kernel = translated_kernel(line_kernel(9.0, 30.0), (3.0, -2.0))
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        result = two_stage_deblur_known(
            observation, kernel, passes=12, reference=truth)
        np.testing.assert_allclose(
            result.factorization.deterministic_shift_xy,
            (3.0, -2.0),
            atol=1e-12,
        )
        self.assertEqual(
            result.diagnostics["decision_order"],
            ["deterministic_transport", "centered_mixing"],
        )
        self.assertGreater(result.diagnostics["psnr_gain"], 6.0)

    def test_oblique_line_uses_expanded_transport_coordinates(self) -> None:
        truth = _fixture(96)
        kernel = line_kernel(11.0, 30.0)
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        positive = two_stage_deblur_known(
            observation, kernel, passes=24, path_authority_scale=0.0)
        result = two_stage_deblur_known(observation, kernel, passes=24)
        characteristic = result.diagnostics["characteristic_transport"]
        self.assertTrue(characteristic["selected"])
        line_constraint = characteristic["line_constraint"]
        self.assertEqual(
            line_constraint["chart"],
            "expanded_rotated_transport_coordinates",
        )
        self.assertGreater(
            line_constraint["chart_shape"][0], observation.shape[0])
        self.assertLess(_mse(result.image, truth), _mse(positive.image, truth))
        self.assertIsNotNone(result.uncertainty)
        self.assertEqual(result.uncertainty.shape, observation.shape)

    def test_vertical_line_keeps_exact_native_chart(self) -> None:
        truth = _fixture(64)
        kernel = line_kernel(9.0, 90.0)
        observation = degrade(truth, kernel, boundary="reflect", clip=False)
        result = two_stage_deblur_known(observation, kernel, passes=12)
        line_constraint = result.diagnostics[
            "characteristic_transport"]["line_constraint"]
        self.assertEqual(line_constraint["chart"], "native_raster_axis")
        self.assertEqual(line_constraint["axis"], "vertical")

    def test_curve_uses_one_exact_curvilinear_operator(self) -> None:
        truth = _fixture(96)
        mild = curved_path_kernel(11.0, 30.0, 4.0)
        observation = degrade(truth, mild, boundary="reflect", clip=False)
        positive = two_stage_deblur_known(
            observation, mild, passes=24, path_authority_scale=0.0)
        result = two_stage_deblur_known(observation, mild, passes=24)
        characteristic = result.diagnostics["characteristic_transport"]
        self.assertEqual(
            characteristic["method"],
            "continuous_positive_exposure_transport",
        )
        self.assertEqual(
            characteristic["operator_role"],
            "exact_ordered_path_gather_with_matched_reflect_scatter",
        )
        self.assertGreater(characteristic["fitted_tangent_turn_degrees"], 20.0)
        self.assertLess(characteristic["line_constraint_authority"], 1e-12)
        self.assertLess(_mse(result.image, truth), _mse(positive.image, truth))

        strong = curved_path_kernel(11.0, 30.0, 8.0)
        strong_observation = degrade(
            truth, strong, boundary="reflect", clip=False)
        strong_result = two_stage_deblur_known(
            strong_observation, strong, passes=12)
        strong_characteristic = strong_result.diagnostics[
            "characteristic_transport"]
        self.assertTrue(strong_characteristic["selected"])
        self.assertGreater(strong_characteristic["jacobian_max"], 1.1)
        self.assertIsNotNone(strong_result.uncertainty)

    def test_one_noise_discrepancy_law_limits_every_transport_action(self) -> None:
        truth = _fixture(64)
        kernel = line_kernel(11.0, 0.0)
        clean = degrade(truth, kernel, boundary="reflect", clip=False)
        noisy = degrade(
            truth,
            kernel,
            gaussian_sigma=0.04,
            seed=7,
            boundary="reflect",
            clip=False,
        )
        clean_result = two_stage_deblur_known(clean, kernel, passes=64)
        noisy_result = two_stage_deblur_known(noisy, kernel, passes=64)
        clean_support = clean_result.diagnostics["support_gate"]
        noisy_support = noisy_result.diagnostics["support_gate"]
        noisy_characteristic = noisy_result.diagnostics[
            "characteristic_transport"]
        self.assertEqual(clean_support["stopped_by"], "maximum_passes")
        self.assertEqual(noisy_support["stopped_by"], "noise_discrepancy")
        self.assertLess(noisy_support["passes_used"], clean_support["passes_used"])
        self.assertLess(
            noisy_characteristic["line_constraint_authority"], 1e-12)
        self.assertEqual(
            noisy_characteristic["refinement_stopped_by"],
            "noise_discrepancy",
        )

    def test_continuous_line_descent_reduces_step_edge_halo(self) -> None:
        size = 96
        truth = np.zeros((size, size, 3), dtype=np.float64)
        truth[:, size // 2 :] = 1.0
        kernel = line_kernel(11.0, 0.0)
        observation = degrade(
            truth,
            kernel,
            gaussian_sigma=0.002,
            seed=3,
            boundary="reflect",
        )
        positive = two_stage_deblur_known(
            observation,
            kernel,
            passes=64,
            path_authority_scale=0.0,
        )
        continuous = two_stage_deblur_known(
            observation, kernel, passes=64)
        x = np.arange(size)
        halo_band = (
            ((x >= size // 2 - 15) & (x <= size // 2 - 2))
            | ((x >= size // 2 + 1) & (x <= size // 2 + 14))
        )
        positive_halo = float(np.sqrt(np.mean(
            (positive.image[:, halo_band] - truth[:, halo_band]) ** 2)))
        continuous_halo = float(np.sqrt(np.mean(
            (continuous.image[:, halo_band] - truth[:, halo_band]) ** 2)))
        self.assertLess(continuous_halo, 0.6 * positive_halo)
        self.assertLess(_mse(continuous.image, truth), _mse(positive.image, truth))

    def test_local_constancy_transport_suppresses_remote_line_echoes(self) -> None:
        size = 96
        truth = np.zeros((size, size), dtype=np.float64)
        truth[:, size // 2 :] = 1.0
        kernel = line_kernel(11.0, 30.0)
        observation = degrade(
            truth,
            kernel,
            gaussian_sigma=0.002,
            seed=1700,
            boundary="reflect",
        )
        ungated = two_stage_deblur_known(
            observation,
            kernel,
            passes=64,
            local_constancy_floor=0.0,
        )
        gated = two_stage_deblur_known(observation, kernel, passes=64)
        x = np.arange(size)
        remote = (x < size // 2 - 16) | (x > size // 2 + 16)
        ungated_echo = float(np.sqrt(np.mean(
            (ungated.image[:, remote] - truth[:, remote]) ** 2)))
        gated_echo = float(np.sqrt(np.mean(
            (gated.image[:, remote] - truth[:, remote]) ** 2)))
        self.assertLess(gated_echo, 0.5 * ungated_echo)
        self.assertLess(_mse(gated.image, truth), _mse(ungated.image, truth))
        support = gated.diagnostics["support_gate"]
        characteristic = gated.diagnostics["characteristic_transport"]
        self.assertLess(support["local_constancy_authority_range"][0], 0.5)
        self.assertLess(
            characteristic["local_constancy_authority_range"][0], 0.5)
        self.assertEqual(characteristic["moment_transport_evaluations"], 1)

    def test_blind_estimation_only_selects_mixing_geometry(self) -> None:
        truth = _fixture(96)
        for kernel in (gaussian_kernel(2.0), line_kernel(11.0, 0.0)):
            with self.subTest(kernel=kernel.name):
                observation = degrade(
                    truth, kernel, boundary="reflect", clip=False)
                result = two_stage_deblur_blind(observation, passes=8)
                self.assertEqual(
                    result.diagnostics["shift_observability"],
                    "absolute_translation_is_single_image_gauge",
                )
                self.assertLess(_mse(result.image, truth), _mse(observation, truth))
                self.assertEqual(result.diagnostics["path_authority_scale"], 0.0)
                self.assertEqual(
                    result.diagnostics["path_recurrence_policy"],
                    "disabled_until_known_operator_or_multi_observation_consensus",
                )
                self.assertEqual(
                    result.diagnostics["characteristic_transport"]["reason"],
                    "path_authority_withheld_by_operator_trust_policy",
                )

    def test_workbench_has_explicit_and_immutable_input_roles(self) -> None:
        truth = _fixture(64)
        session = DeblurSession()
        index = session.add_array(truth, "fixture")
        record = session.use_as_is(index)
        self.assertEqual(record.mode, "as_is_observation")
        self.assertFalse(record.synthetic_truth_available)
        self.assertFalse(np.shares_memory(record.original, record.observation))
        original_fingerprint = image_fingerprint(record.original)
        observation_fingerprint = image_fingerprint(record.observation)
        session.synthesize(index, BlurSpec(kind="Disk", radius=3.0))
        self.assertEqual(original_fingerprint, image_fingerprint(record.original))
        self.assertNotEqual(observation_fingerprint, image_fingerprint(record.observation))
        blurred_fingerprint = image_fingerprint(record.observation)
        session.deblur_active(index, passes=8)
        self.assertEqual(blurred_fingerprint, image_fingerprint(record.observation))


if __name__ == "__main__":
    unittest.main()
