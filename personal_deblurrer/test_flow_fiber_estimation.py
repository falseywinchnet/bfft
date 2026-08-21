from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus
from .kernels import line_kernel
from .run_curved_flow_fiber_benchmark import _curved_layer_pair
from .run_dense_estimation_benchmark import _fields, _nominal_flow
from .run_rolling_shutter_flow_atlas_benchmark import (
    _rolling_shutter_layer_pair,
)
from .run_visibility_benchmark import _layered_pair
from .spatial_transport import SpatialReflectedExposureOperator
from .synthetic import degrade


class FlowFiberEstimationTests(unittest.TestCase):
    def test_identical_pair_abstains_without_change(self) -> None:
        image = sources(32)["cameraman"]
        result = deblur_flow_fiber_consensus(image, image)
        np.testing.assert_array_equal(result.image, image)
        self.assertIsNone(result.fiber_solution)
        self.assertEqual(result.diagnostics["support_count"], 0)

    def test_blind_flow_measure_is_soft_and_improves_folded_pair(self) -> None:
        background = sources(48)["cameraman"]
        truth, observations = _layered_pair(
            background, 3.0, noise_sigma=0.002, seed=21000)
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1], passes=48)
        average = np.mean(observations, axis=0)
        result_mse = float(np.mean((result.image - truth) ** 2))
        average_mse = float(np.mean((average - truth) ** 2))
        self.assertLess(result_mse, 0.8 * average_mse)
        self.assertIsNotNone(result.fiber_solution)
        assert result.fiber_solution is not None
        np.testing.assert_allclose(
            np.sum(result.fiber_solution.reference_ownership, axis=0),
            1.0,
            atol=1e-12,
        )
        self.assertGreater(
            result.diagnostics["latent_measure_entropy_mean"], 0.05)
        self.assertGreater(result.diagnostics["correction_authority_max"], 0.0)
        self.assertEqual(result.diagnostics["fiber_passes"], 1)
        self.assertNotIn("selected", repr(result.diagnostics).lower())

    def test_smooth_single_connection_preserves_dense_gauge(self) -> None:
        truth = sources(48)["cameraman"]
        fields = _fields(
            _nominal_flow((48, 48)), duty_cycle=0.5, atoms=7)
        observations = [
            SpatialReflectedExposureOperator(field).forward(truth)
            for field in fields
        ]
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1],
            duty_cycle=0.5, passes=32)
        self.assertIsNotNone(result.fiber_solution)
        self.assertLess(
            result.diagnostics["coherent_disagreement_authority"], 0.002)
        self.assertLess(float(np.max(np.abs(
            result.image - result.common_image))), 0.001)

    def test_fourier_circles_recover_nonfold_woven_motion(self) -> None:
        truth, observations = _layered_pair(
            sources(48)["woven chirps"],
            3.0,
            noise_sigma=0.002,
            seed=21000,
        )
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1], passes=32)
        self.assertEqual(max(result.diagnostics["fold_fractions"]), 0.0)
        common_mse = float(np.mean((result.common_image - truth) ** 2))
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.5 * common_mse)
        self.assertGreater(
            result.diagnostics["coherent_disagreement_authority"], 0.9)
        np.testing.assert_allclose(
            result.diagnostics["fourier_circle_translation_xy"],
            (6.0, 0.0),
            atol=0.1,
        )

    def test_local_circle_atlas_recovers_curved_layer_with_zero_global_shift(
        self,
    ) -> None:
        truth, observations = _curved_layer_pair(
            sources(48)["cameraman"],
            3.0,
            noise_sigma=0.002,
            seed=31000,
        )
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1], passes=32)
        common_mse = float(np.mean((result.common_image - truth) ** 2))
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.80 * common_mse)
        np.testing.assert_allclose(
            result.diagnostics["fourier_circle_translation_xy"],
            (0.0, 0.0),
            atol=0.1,
        )
        self.assertGreater(
            result.diagnostics["fourier_circle_atlas_transport"][
                "flow_rms_pixels"],
            0.25,
        )
        self.assertEqual(result.diagnostics["support_count"], 5)

    def test_radiometric_atlas_recovers_complementary_clipped_pair(self) -> None:
        truth, observations = _layered_pair(
            sources(48)["cameraman"],
            3.0,
            noise_sigma=0.002,
            seed=41000,
        )
        clipped = (
            np.clip(0.7 * observations[0], 0.0, 1.0),
            np.clip((1.0 / 0.7) * observations[1], 0.0, 1.0),
        )
        result = deblur_flow_fiber_consensus(*clipped, passes=32)
        average_mse = float(np.mean(
            (np.mean(clipped, axis=0) - truth) ** 2))
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.20 * average_mse)
        self.assertAlmostEqual(
            result.diagnostics["relative_gain_second_over_first"],
            1.0 / 0.49,
            delta=0.15,
        )
        self.assertEqual(
            result.diagnostics["radiometric_role"],
            "continuous_precision_measure_not_frame_selection",
        )

    def test_atlas_recovers_observable_rolling_shutter_acceleration(self) -> None:
        truth, observations = _rolling_shutter_layer_pair(
            sources(48)["cameraman"],
            3.0,
            1.2,
            exposure_extent=1.0,
            noise_sigma=0.002,
            seed=51000,
        )
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1],
            duty_cycle=1.0 / 6.0,
            passes=32,
        )
        average_mse = float(np.mean(
            (np.mean(observations, axis=0) - truth) ** 2))
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.35 * average_mse)
        charts = result.diagnostics[
            "fourier_circle_atlas_transport"]["chart_records"]
        chart_x = np.asarray([
            record["translation_xy"][0] for record in charts
        ])
        self.assertGreater(float(np.std(chart_x)), 0.25)

    def test_relative_mixing_follows_center_transport_without_family_choice(
        self,
    ) -> None:
        truth = sources(64)["cameraman"]
        observations = (
            degrade(truth, line_kernel(3.0, 25.0),
                    boundary="reflect", clip=False),
            degrade(truth, line_kernel(9.0, 25.0),
                    boundary="reflect", clip=False),
        )
        result = deblur_flow_fiber_consensus(
            *observations, passes=32, automatic_relative_mixing=True)
        average_mse = float(np.mean(
            (np.mean(observations, axis=0) - truth) ** 2))
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.75 * average_mse)
        self.assertGreater(
            result.diagnostics["relative_mixing_authority"], 0.8)
        self.assertEqual(
            result.diagnostics["role"],
            "exchange_symmetric_positive_measure_not_frame_or_family_selection",
        )


if __name__ == "__main__":
    unittest.main()
