"""Invariants for continuous-scale zonotope push-forward and flux patterns."""

import unittest

import numpy as np

from .continuous_scale_zonotope_transport_2d import (
    _continuous_scale_lineage_generators,
    _flux_pattern_coordinates,
    continuous_scale_zonotope_transport_state_2d,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)


class ContinuousScaleZonotopeTransport2DTests(unittest.TestCase):
    def test_scale_lineages_recompose_exactly_under_nested_refinement(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.1 * np.sin(0.8 * xx) + 0.04 * np.cos(0.6 * yy)
        coarse, _labels0, diagnostic0 = _continuous_scale_lineage_generators(
            field, "test", 0)
        refined, _labels1, diagnostic1 = _continuous_scale_lineage_generators(
            field, "test", 1)
        np.testing.assert_allclose(
            np.sum(coarse, axis=1).reshape(field.shape),
            field,
            atol=3e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.sum(refined, axis=1).reshape(field.shape),
            field,
            atol=3e-14,
            rtol=0.0,
        )
        self.assertGreater(refined.shape[1], coarse.shape[1])
        self.assertLess(
            diagnostic0["exact_recomposition_maximum_error"], 3e-14)
        self.assertLess(
            diagnostic1["exact_recomposition_maximum_error"], 3e-14)

    def test_dense_generators_reexpress_as_exact_flux_patterns(self):
        yy, xx = np.mgrid[:8, :8]
        posterior = 0.3 + 0.08 * np.sin(0.5 * xx) + 0.03 * yy
        residual = 0.02 * np.cos(0.7 * xx - 0.4 * yy)
        metric = continual_transport_metric(posterior, residual * residual)
        laplacian, _markov, _stencil = _continual_flux_laplacian(
            metric, np.ones_like(posterior))
        generator = np.stack((
            residual.reshape(-1),
            (0.01 + 0.03 * np.sin(0.9 * xx + 0.2 * yy)).reshape(-1),
        ), axis=1)
        diagnostic = _flux_pattern_coordinates(laplacian, generator)
        self.assertLess(diagnostic["reconstruction_maximum_error"], 3e-13)
        self.assertLess(diagnostic["antisymmetric_flux_sum_error"], 2e-15)
        self.assertEqual(diagnostic["generator_count"], 2)

    def test_pushforward_is_linear_and_preserves_complete_lineage(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.13 * np.sin(0.7 * xx) + 0.05 * np.cos(0.6 * yy)
        state = continuous_scale_zonotope_transport_state_2d(image)
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["full_lineage_recomposition_error"], 5e-14)
        self.assertLess(state["pushforward_center_linearity_error"], 2e-15)
        self.assertLess(
            state["pushed_flux_patterns"]["reconstruction_maximum_error"],
            5e-13,
        )
        self.assertTrue(
            state["contraction"]["feasible_outer_component"])
        self.assertTrue(np.all(state["coefficient_lower"] >= 0.0))
        self.assertTrue(np.all(state["coefficient_upper"] <= 1.0))
        self.assertTrue(np.all(
            state["coefficient_lower"] <= state["coefficient_upper"]))

    def test_reported_enclosures_contain_sampled_coefficient_members(self):
        rng = np.random.default_rng(1149)
        yy, xx = np.mgrid[:8, :8]
        image = 0.45 + 0.1 * np.sin(0.8 * xx + 0.3 * yy)
        state = continuous_scale_zonotope_transport_state_2d(image)
        lower = state["coefficient_lower"]
        upper = state["coefficient_upper"]
        for _ in range(4):
            coefficient = lower + rng.random(lower.size) * (upper - lower)
            transfer = (state["generator"] @ coefficient).reshape(image.shape)
            self.assertTrue(np.all(
                transfer >= state["transfer_enclosure_lower"] - 2e-15))
            self.assertTrue(np.all(
                transfer <= state["transfer_enclosure_upper"] + 2e-15))
            pushed = (
                state["pushed_generator"] @ coefficient
            ).reshape(image.shape)
            self.assertTrue(np.all(
                pushed >= state["pushed_transfer_enclosure_lower"] - 2e-15))
            self.assertTrue(np.all(
                pushed <= state["pushed_transfer_enclosure_upper"] + 2e-15))

    def test_constant_scene_remains_a_complete_zero_lineage(self):
        image = np.full((8, 8), 0.37)
        state = continuous_scale_zonotope_transport_state_2d(image)
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["full_lineage_recomposition_error"], 5e-14)
        self.assertLess(
            np.max(np.abs(state["pushed_center_posterior_for_geometry_only"]
                          - image)),
            5e-14,
        )


if __name__ == "__main__":
    unittest.main()
