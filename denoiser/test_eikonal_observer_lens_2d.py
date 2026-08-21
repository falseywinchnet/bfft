"""Invariants for the invertible eikonal observer lens."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .eikonal_observer_lens_2d import (
    denoise_phase_eikonal_observer_lens_2d,
    denoise_eikonal_observer_lens_2d,
    eikonal_lens_analysis_2d,
    eikonal_lens_synthesis_2d,
    smooth_eikonal_lens_detail_2d,
)


class EikonalObserverLens2DTests(unittest.TestCase):
    def test_analysis_synthesis_is_exact(self):
        rng = np.random.default_rng(17)
        image = rng.random((8, 8))
        _law, diagnostic = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        coarse, detail, _ = eikonal_lens_analysis_2d(
            image, diagnostic["forest"])
        reconstructed = eikonal_lens_synthesis_2d(
            coarse, detail, diagnostic["forest"])
        np.testing.assert_allclose(
            reconstructed, image, atol=3e-15, rtol=0.0)

    def test_constant_is_completely_absorbed(self):
        image = np.full((8, 8), 0.37)
        _law, diagnostic = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        coarse, detail, _ = eikonal_lens_analysis_2d(
            image, diagnostic["forest"])
        np.testing.assert_allclose(detail, 0.0, atol=2e-15, rtol=0.0)
        reconstructed = eikonal_lens_synthesis_2d(
            coarse, detail, diagnostic["forest"])
        np.testing.assert_allclose(reconstructed, image, atol=2e-15, rtol=0.0)

    def test_exact_affine_jet_is_completely_absorbed(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.2 + 0.03 * xx - 0.017 * yy
        _law, diagnostic = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        forest = diagnostic["forest"]
        first = np.asarray(forest["parent_first"]).reshape(-1)
        second = np.asarray(forest["parent_second"]).reshape(-1)
        fraction = np.asarray(forest["parent_fraction"]).reshape(-1)
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        offset = np.zeros(image.size)
        for child in np.asarray(forest["acceptance_order"]).reshape(-1):
            if first[child] < 0:
                continue
            if second[child] < 0:
                base_x = flat_x[first[child]]
                base_y = flat_y[first[child]]
            else:
                t = fraction[child]
                base_x = ((1.0 - t) * flat_x[first[child]]
                          + t * flat_x[second[child]])
                base_y = ((1.0 - t) * flat_y[first[child]]
                          + t * flat_y[second[child]])
            offset[child] = (
                0.03 * (flat_x[child] - base_x)
                - 0.017 * (flat_y[child] - base_y))
        coarse, detail, _ = eikonal_lens_analysis_2d(
            image, forest, prediction_offset=offset.reshape(image.shape))
        np.testing.assert_allclose(detail, 0.0, atol=2e-15, rtol=0.0)
        reconstructed = eikonal_lens_synthesis_2d(
            coarse, detail, forest,
            prediction_offset=offset.reshape(image.shape))
        np.testing.assert_allclose(reconstructed, image, atol=2e-15, rtol=0.0)

    def test_observer_smoothing_contracts_only_detail_action(self):
        rng = np.random.default_rng(23)
        image = rng.normal(size=(8, 8))
        _law, diagnostic = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        coarse, detail, _ = eikonal_lens_analysis_2d(
            image, diagnostic["forest"])
        smoothed, record = smooth_eikonal_lens_detail_2d(
            detail, diagnostic["forest"])
        self.assertLessEqual(
            record["forest_action_after"], record["forest_action_before"])
        first = np.asarray(
            diagnostic["forest"]["parent_first"]).reshape(image.shape)
        np.testing.assert_allclose(smoothed[first < 0], 0.0, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(np.isfinite(eikonal_lens_synthesis_2d(
            coarse, smoothed, diagnostic["forest"]))))

    def test_full_constant_denoiser_is_exact(self):
        image = np.full((8, 8), 0.29)
        estimate, diagnostic = denoise_eikonal_observer_lens_2d(
            image, angular_count=4, quantile_count=8)
        np.testing.assert_allclose(estimate, image, atol=2e-15, rtol=0.0)
        self.assertLess(
            diagnostic["exact_analysis_synthesis_maximum_error"], 2e-15)

    def test_phase_absorbing_constant_is_exact(self):
        image = np.full((8, 8), 0.31)
        estimate, diagnostic = denoise_phase_eikonal_observer_lens_2d(
            image, angular_count=4, quantile_count=8)
        np.testing.assert_allclose(estimate, image, atol=2e-14, rtol=0.0)
        self.assertLess(
            diagnostic["exact_analysis_synthesis_maximum_error"], 2e-14)
        self.assertLess(
            diagnostic["phase"]["maximum_projection_normal_error"], 2e-12)


if __name__ == "__main__":
    unittest.main()
