"""Foundational invariants for DCNT."""

from __future__ import annotations

import unittest

import numpy as np

from .dcnt import (
    DCNTResolution,
    _target_excluded_conv_family_batch,
    denoise_dcnt,
    minimal_supported_noise,
    target_excluded_conv_family,
    transport_uncertainty,
    tv_chambolle_reference,
)


class DCNTTest(unittest.TestCase):
    def test_component_batched_observers_match_scalar_charts(self) -> None:
        rng = np.random.default_rng(20260828)
        images = rng.normal(size=(4, 18, 20))
        expected = np.stack([
            target_excluded_conv_family(image) for image in images])
        actual = _target_excluded_conv_family_batch(images)
        np.testing.assert_allclose(
            actual, expected, atol=2e-14, rtol=0.0, equal_nan=True)

    def test_tv_reconstruction_matches_scikit(self) -> None:
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            self.skipTest("scikit-image is not installed in this runtime")
        rng = np.random.default_rng(20260827)
        image = rng.normal(size=(17, 19))
        expected = denoise_tv_chambolle(
            image, weight=0.13, eps=2e-4, max_num_iter=41)
        actual = tv_chambolle_reference(
            image, weight=0.13, eps=2e-4, maximum_iterations=41)
        np.testing.assert_array_equal(actual, expected)

    def test_observer_predictions_exclude_the_target(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.normal(size=(18, 20))
        original = target_excluded_conv_family(image)
        for row, column in ((0, 0), (0, 19), (17, 0), (17, 19), (9, 11)):
            changed = image.copy()
            changed[row, column] += 1000.0
            after = target_excluded_conv_family(changed)[:, row, column]
            before = original[:, row, column]
            np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
            np.testing.assert_allclose(
                before[np.isfinite(before)], after[np.isfinite(after)],
                atol=0.0, rtol=0.0)

    def test_every_pixel_has_three_target_excluded_predictions(self) -> None:
        image = np.arange(18 * 20, dtype=float).reshape(18, 20)
        family = target_excluded_conv_family(image)
        np.testing.assert_array_equal(
            np.sum(np.isfinite(family), axis=0),
            np.full(image.shape, 3),
        )

    def test_affine_field_is_exactly_supported(self) -> None:
        y, x = np.indices((18, 20), dtype=float)
        image = 0.2 + 0.013 * x - 0.009 * y
        family = target_excluded_conv_family(image)
        for chart in family:
            # The compact interior jet is affine exact.  Symmetric image
            # closure intentionally bends the extrapolation near a boundary.
            interior = chart[6:-6, 6:-6]
            expected = image[6:-6, 6:-6]
            finite = np.isfinite(interior)
            np.testing.assert_allclose(
                interior[finite], expected[finite], atol=4e-14, rtol=0.0)

    def test_zonotope_contains_every_transport_witness(self) -> None:
        rng = np.random.default_rng(13)
        law = transport_uncertainty(rng.normal(size=(18, 20)))
        distance = np.abs(law.family - law.centre[None, ...])
        self.assertTrue(np.all(
            distance[np.isfinite(distance)]
            <= np.broadcast_to(law.zonotope_radius, distance.shape)[
                np.isfinite(distance)] + 1e-15))

    def test_minimal_noise_is_interval_distance(self) -> None:
        observation = np.array((0.0, 0.3, 0.7, 1.0))
        centre = np.full(4, 0.5)
        radius = np.full(4, 0.2)
        noise, admitted = minimal_supported_noise(
            observation, centre, radius)
        np.testing.assert_allclose(admitted, (0.3, 0.3, 0.7, 0.7))
        np.testing.assert_allclose(noise, (-0.3, 0.0, 0.0, 0.3))

    def test_uncertainty_mode_fixes_affine_fields(self) -> None:
        y, x = np.indices((18, 20), dtype=float)
        image = 0.2 + 0.013 * x - 0.009 * y
        estimate, diagnostic = denoise_dcnt(
            image,
            mode="uncertainty",
            resolution=DCNTResolution(maximum_cycles=2),
        )
        np.testing.assert_allclose(estimate, image, atol=8e-14, rtol=0.0)
        self.assertLessEqual(diagnostic["recomposition_error"], 1e-15)

    def test_isolated_replacement_is_contracted(self) -> None:
        truth = np.full((18, 20), 0.4)
        observation = truth.copy()
        observation[9, 11] = 1.0
        estimate, _diagnostic = denoise_dcnt(
            observation,
            mode="uncertainty",
            resolution=DCNTResolution(maximum_cycles=1),
        )
        self.assertLess(
            np.mean((estimate - truth) ** 2),
            np.mean((observation - truth) ** 2),
        )

    def test_exact_observation_partition(self) -> None:
        rng = np.random.default_rng(19)
        observation = rng.random((18, 20))
        estimate, diagnostic = denoise_dcnt(
            observation,
            mode="uncertainty",
            resolution=DCNTResolution(maximum_cycles=1),
        )
        np.testing.assert_allclose(
            estimate + diagnostic["residual"], observation,
            atol=2e-16, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
