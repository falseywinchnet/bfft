"""Invariant tests for the 1-D/2-D continuous support transport form."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .transport_support import (
        TransportResolution,
        denoise_1d,
        _lane_scale_space,
        _residue_lanes,
        support_density,
        transport_support_birth,
    )
except ImportError:
    from transport_support import (
        TransportResolution,
        denoise_1d,
        _lane_scale_space,
        _residue_lanes,
        support_density,
        transport_support_birth,
    )


FAST = TransportResolution(scale_samples=5, histogram_bins=32, maximum_steps=512)


class ContinuousSupportTransportTests(unittest.TestCase):
    def test_complement_lane_filter_matches_two_lane_reference(self):
        from scipy import ndimage

        rng = np.random.default_rng(7)
        field = rng.random((27, 31))
        lanes = _residue_lanes(field.shape)
        expected = np.stack([
            ndimage.gaussian_filter(
                field * lane, 2.3, mode="reflect")
            / ndimage.gaussian_filter(lane, 2.3, mode="reflect")
            for lane in lanes
        ])
        actual = _lane_scale_space(field, 2.3)
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.0)

    def test_constant_state_is_fixed_in_one_dimension(self):
        field = np.full(128, 0.37)
        output, _barrier, diagnostics = transport_support_birth(field, field, FAST)
        np.testing.assert_array_equal(output, field)
        self.assertEqual(diagnostics["transport_stop"], "zero_transport")
        self.assertEqual(diagnostics["transport_action_spent"], 0.0)

    def test_one_dimensional_form_preserves_mass_and_range(self):
        x = np.linspace(0.0, 1.0, 256, endpoint=False)
        truth = 0.2 + 0.45 * (x > 0.43) + 0.08 * np.sin(18.0 * np.pi * x) * (x > 0.62)
        rng = np.random.default_rng(71)
        observed = np.clip(truth + rng.uniform(-0.22, 0.22, x.shape), 0.0, 1.0)
        output, diagnostics = denoise_1d(observed, FAST)
        self.assertEqual(output.shape, observed.shape)
        self.assertLess(abs(diagnostics["mass_conservation_error"]), 1e-10)
        self.assertGreaterEqual(float(np.min(output)), float(np.min(observed)) - 1e-12)
        self.assertLessEqual(float(np.max(output)), float(np.max(observed)) + 1e-12)

    def test_one_dimensional_extent_controls_are_effective(self):
        x = np.linspace(0.0, 1.0, 512, endpoint=False)
        truth = 0.25 + 0.25 / (1.0 + np.exp(-(x - 0.5) * 80.0))
        rng = np.random.default_rng(91)
        observed = np.clip(truth + rng.uniform(-0.24, 0.24, x.shape), 0.0, 1.0)
        conservative, conservative_diag = denoise_1d(
            observed,
            FAST,
            provisional_sigma=1.0,
            action_budget_multiplier=0.0,
            continuation_rounds=1,
        )
        extended, extended_diag = denoise_1d(
            observed,
            FAST,
            provisional_sigma=2.0,
            action_budget_multiplier=8.0,
            continuation_rounds=4,
        )
        conservative_roughness = float(np.mean(np.diff(conservative) ** 2))
        extended_roughness = float(np.mean(np.diff(extended) ** 2))
        self.assertLess(extended_roughness, conservative_roughness)
        self.assertEqual(conservative_diag["total_transport_steps"], 0)
        self.assertGreater(extended_diag["total_transport_steps"], 0)
        self.assertEqual(extended_diag["continuation_rounds"], 4)

    def test_two_dimensional_flow_spends_observation_budget(self):
        yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
        truth = 0.62 + 0.18 * xx
        truth[(xx + 0.20) ** 2 + (yy + 0.05) ** 2 < 0.22] = 0.12
        rng = np.random.default_rng(19)
        observed = np.clip(truth + rng.uniform(-0.24, 0.24, truth.shape), 0.0, 1.0)
        provisional = (
            observed
            + np.roll(observed, 1, 0) + np.roll(observed, -1, 0)
            + np.roll(observed, 1, 1) + np.roll(observed, -1, 1)
        ) / 5.0
        output, barrier, diagnostics = transport_support_birth(observed, provisional, FAST)
        self.assertEqual(output.shape, truth.shape)
        self.assertEqual(barrier.shape, truth.shape)
        self.assertTrue(np.all((barrier >= 0.0) & (barrier <= 1.0)))
        self.assertLess(abs(diagnostics["mass_conservation_error"]), 1e-9)
        if diagnostics["transport_stop"] == "action_budget":
            self.assertAlmostEqual(
                diagnostics["transport_action_spent"],
                diagnostics["transport_action_budget"],
                places=9,
            )

    def test_support_is_continuous_and_finite_on_tapered_edge(self):
        yy, xx = np.mgrid[-1:1:72j, -1:1:72j]
        field = 0.72 + 0.06 * yy
        taper = (xx > -0.55) & (xx < 0.25 + 0.38 * yy) & (yy < 0.45)
        field[taper] = 0.12
        support, diagnostics = support_density(field, FAST)
        self.assertTrue(np.all(np.isfinite(support)))
        self.assertTrue(np.all((support >= 0.0) & (support <= 1.0)))
        self.assertGreater(diagnostics["mean_support_density"], 0.0)

    def test_precomputed_support_reuses_the_identical_physical_field(self):
        rng = np.random.default_rng(101)
        observed = rng.random((24, 28))
        provisional = 0.5 * observed + 0.5 * np.mean(observed)
        expected, expected_barrier, _ = transport_support_birth(
            observed, provisional, FAST)
        support, support_diagnostic = support_density(observed, FAST)
        actual, actual_barrier, _ = transport_support_birth(
            observed,
            provisional,
            FAST,
            support_field=support,
            support_diagnostics=support_diagnostic,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_barrier, expected_barrier)


if __name__ == "__main__":
    unittest.main()
