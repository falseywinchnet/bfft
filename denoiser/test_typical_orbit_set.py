"""Invariants for the first post-FMMT typical-orbit checkpoint."""

from __future__ import annotations

import unittest

import numpy as np

from .typical_orbit_set import (
    LocalOrbitResolution,
    TypicalOrbitResolution,
    denoise_local_orbit_survival,
    denoise_typical_orbit_set,
    infer_local_orbit_survival,
    infer_typical_orbit_set,
)


class TypicalOrbitSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolution = TypicalOrbitResolution(
            ring_radii=(1, 2), candidate_count=5, query_multiplier=5)
        self.local_resolution = LocalOrbitResolution(radii=(1, 2))

    def test_constant_is_exact(self) -> None:
        image = np.full((24, 27), 0.375)
        output, diagnostics = denoise_typical_orbit_set(image, self.resolution)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(diagnostics["changed_fraction"], 0.0)

    def test_isolated_replacement_is_removed_without_target_in_descriptor(self) -> None:
        image = np.full((25, 25), 0.4)
        image[12, 12] = 1.0
        output, _ = denoise_typical_orbit_set(image, self.resolution)
        self.assertEqual(output[12, 12], 0.4)
        self.assertEqual(output[3, 3], 0.4)

    def test_readout_is_observation_or_actual_reference_component(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.random((22, 23))
        state = infer_typical_orbit_set(image, self.resolution)
        self.assertTrue(np.all(state.estimate >= state.lower - 1e-15))
        self.assertTrue(np.all(state.estimate <= state.upper + 1e-15))
        np.testing.assert_array_equal(
            state.estimate[state.retained_observation],
            image[state.retained_observation],
        )

    def test_bounds_and_diagnostics_are_finite(self) -> None:
        yy, xx = np.mgrid[:28, :30]
        image = np.clip(0.2 + 0.01 * xx + 0.2 * (yy > xx), 0.0, 1.0)
        state = infer_typical_orbit_set(image, self.resolution)
        self.assertTrue(np.all(np.isfinite(state.estimate)))
        self.assertTrue(np.all(state.lower <= state.upper))
        self.assertTrue(np.all((state.estimate >= 0.0) & (state.estimate <= 1.0)))

    def test_local_survival_is_exact_on_constant(self) -> None:
        image = np.full((25, 27), 0.375)
        output, diagnostics = denoise_local_orbit_survival(
            image, self.local_resolution)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(diagnostics["conclusively_falsified_fraction"], 0.0)

    def test_local_survival_removes_isolated_replacement(self) -> None:
        image = np.full((29, 31), 0.4)
        image[14, 15] = 1.0
        output, _ = denoise_local_orbit_survival(image, self.local_resolution)
        self.assertEqual(output[14, 15], 0.4)

    def test_local_survival_preserves_clean_step(self) -> None:
        image = np.full((31, 33), 0.2)
        image[:, 16:] = 0.8
        output, _ = denoise_local_orbit_survival(image, self.local_resolution)
        np.testing.assert_array_equal(output, image)

    def test_retained_samples_are_byte_exact(self) -> None:
        rng = np.random.default_rng(13)
        image = rng.random((27, 29))
        state = infer_local_orbit_survival(image, self.local_resolution)
        np.testing.assert_array_equal(
            state.estimate[state.retained_observation],
            image[state.retained_observation],
        )

    def test_straight_step_has_cross_scale_orbit_coherence(self) -> None:
        image = np.full((31, 33), 0.2)
        image[:, 16:] = 0.8
        state = infer_local_orbit_survival(image, self.local_resolution)
        self.assertIsNotNone(state.orbit_coherence)
        self.assertTrue(np.all(state.orbit_coherence[5:-5, 15:18]))


if __name__ == "__main__":
    unittest.main()
