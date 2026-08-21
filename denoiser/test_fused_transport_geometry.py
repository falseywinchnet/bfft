"""Invariants for the band-free transported-information geometry."""

from __future__ import annotations

import unittest

import numpy as np

from .fused_transport_geometry import (
    combine_information_geometries,
    observation_graph_measure,
    predictive_directional_jet_sasaki_geometry,
    predictive_horizontal_wasserstein_geometry,
    predictive_information_geometry,
    predictive_jet_horizontal_wasserstein_geometry,
    predictive_lineage_jet_geometry,
    predictive_lineage_prolongation_geometry,
    predictive_wasserstein_geometry,
    weighted_empirical_quantiles,
    weighted_support_quantiles,
)


class FusedTransportGeometryTests(unittest.TestCase):
    def test_spatially_invariant_law_is_exactly_one_support_unit(self):
        probability = np.empty((18, 30, 7), dtype=np.float64)
        probability[...] = np.arange(1.0, 8.0)
        geometry = predictive_information_geometry(probability)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)
        np.testing.assert_allclose(
            geometry["metric_determinant"], 1.0, atol=2e-15, rtol=0.0)
        self.assertAlmostEqual(float(np.sum(geometry["measure"])), 1.0, places=14)

    def test_observation_lift_conserves_mass_and_exact_equation(self):
        rng = np.random.default_rng(4)
        probability = rng.random((9, 11, 13))
        observation = rng.random((9, 11))
        values = np.linspace(0.0, 1.0, 13)
        state = observation_graph_measure(probability, observation, values)
        np.testing.assert_allclose(
            np.sum(state["mass"], axis=-1), 1.0, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            state["signal"] + state["residual"],
            np.broadcast_to(observation[..., None], state["signal"].shape),
            atol=2e-16,
            rtol=0.0,
        )

    def test_predictable_translation_creates_oriented_support_volume(self):
        height, width, atoms = 20, 36, 41
        values = np.linspace(-1.0, 1.0, atoms)
        center = np.linspace(-0.65, 0.65, width)[None, :, None]
        probability = np.exp(-0.5 * ((values[None, None, :] - center) / 0.12) ** 2)
        probability = np.broadcast_to(probability, (height, width, atoms)).copy()
        geometry = predictive_information_geometry(probability)
        self.assertGreater(geometry["implied_support"], 1.0)
        self.assertGreater(
            float(np.mean(geometry["precision_xx"])),
            float(np.mean(geometry["precision_yy"])),
        )

    def test_raw_random_atoms_are_not_a_valid_predictive_state(self):
        rng = np.random.default_rng(31)
        height, width, atoms = 18, 24, 17
        raw = np.zeros((height, width, atoms), dtype=np.float64)
        index = rng.integers(0, atoms, size=(height, width))
        yy, xx = np.indices((height, width))
        raw[yy, xx, index] = 1.0
        invariant = np.mean(raw, axis=(0, 1), keepdims=True)
        invariant = np.broadcast_to(invariant, raw.shape)
        raw_geometry = predictive_information_geometry(raw)
        transported_noise_geometry = predictive_information_geometry(invariant)
        self.assertGreater(
            raw_geometry["implied_support"],
            10.0 * transported_noise_geometry["implied_support"],
        )

    def test_wasserstein_law_is_particle_order_invariant(self):
        rng = np.random.default_rng(72)
        particles = rng.normal(size=(13, 19, 9))
        permutation = np.argsort(rng.random(particles.shape), axis=-1)
        shuffled = np.take_along_axis(particles, permutation, axis=-1)
        original = predictive_wasserstein_geometry(particles)
        reordered = predictive_wasserstein_geometry(shuffled)
        for key in ("measure", "precision_xx", "precision_xy", "precision_yy"):
            np.testing.assert_allclose(
                original[key], reordered[key], atol=0.0, rtol=0.0)

    def test_wasserstein_law_has_exact_uniform_particle_refinement(self):
        rng = np.random.default_rng(73)
        particles = rng.normal(size=(11, 17, 7))
        refined = np.repeat(particles, 4, axis=-1)
        coarse = predictive_wasserstein_geometry(particles)
        fine = predictive_wasserstein_geometry(refined)
        for key in ("measure", "precision_xx", "precision_xy", "precision_yy"):
            np.testing.assert_allclose(
                coarse[key], fine[key], atol=2e-15, rtol=2e-15)

    def test_spatially_invariant_particles_are_one_support_unit(self):
        law = np.array([-0.8, -0.2, 0.1, 0.7, 1.4], dtype=np.float64)
        particles = np.broadcast_to(law, (12, 21, law.size))
        geometry = predictive_wasserstein_geometry(particles)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)
        np.testing.assert_allclose(
            geometry["metric_determinant"], 1.0, atol=2e-15, rtol=0.0)

    def test_translating_particles_create_oriented_wasserstein_pullback(self):
        height, width = 15, 27
        law = np.array([-0.2, 0.0, 0.3, 0.9], dtype=np.float64)
        shift = np.linspace(-0.5, 0.5, width)[None, :, None]
        particles = np.broadcast_to(
            law[None, None, :] + shift, (height, width, law.size)).copy()
        geometry = predictive_wasserstein_geometry(particles)
        self.assertGreater(geometry["implied_support"], 1.0)
        self.assertGreater(
            float(np.mean(geometry["precision_xx"])),
            float(np.mean(geometry["precision_yy"])),
        )

    def test_horizontal_geometry_quotients_arbitrary_translation(self):
        height, width = 15, 27
        law = np.array([-0.2, 0.0, 0.3, 0.9], dtype=np.float64)
        yy, xx = np.mgrid[:height, :width]
        shift = (0.2 * np.sin(xx / 4.0) + 0.03 * yy)[..., None]
        particles = law[None, None, :] + shift
        geometry = predictive_horizontal_wasserstein_geometry(particles)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)
        np.testing.assert_allclose(
            geometry["metric_determinant"], 1.0, atol=2e-15, rtol=0.0)

    def test_horizontal_geometry_retains_shape_change(self):
        height, width = 15, 27
        law = np.array([-1.0, -0.2, 0.2, 1.0], dtype=np.float64)
        scale = np.linspace(0.2, 1.0, width)[None, :, None]
        particles = np.broadcast_to(
            law[None, None, :] * scale, (height, width, law.size)).copy()
        geometry = predictive_horizontal_wasserstein_geometry(particles)
        self.assertGreater(geometry["implied_support"], 1.0)
        self.assertGreater(
            float(np.mean(geometry["precision_xx"])),
            float(np.mean(geometry["precision_yy"])),
        )

    def test_weighted_quantiles_are_invariant_to_particle_order(self):
        rng = np.random.default_rng(74)
        values = rng.normal(size=(8, 10, 11))
        mass = rng.random(values.shape)
        order = np.argsort(rng.random(values.shape), axis=-1)
        expected = weighted_empirical_quantiles(values, mass, 17)
        actual = weighted_empirical_quantiles(
            np.take_along_axis(values, order, axis=-1),
            np.take_along_axis(mass, order, axis=-1),
            17,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_jet_horizontal_geometry_annihilates_affine_transport(self):
        height, width, atoms = 14, 23, 9
        yy, xx = np.mgrid[:height, :width]
        slope_x = 0.017
        slope_y = -0.011
        law = np.linspace(-0.2, 0.3, atoms)
        values = (
            law[None, None, :]
            + slope_x * xx[..., None]
            + slope_y * yy[..., None]
        )
        mass = np.broadcast_to(
            np.linspace(1.0, 2.0, atoms), values.shape).copy()
        geometry = predictive_jet_horizontal_wasserstein_geometry(
            values,
            mass,
            np.full((height, width), slope_x),
            np.full((height, width), slope_y),
            quantile_count=19,
        )
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)
        np.testing.assert_allclose(
            geometry["metric_determinant"], 1.0, atol=2e-15, rtol=0.0)

    def test_jet_horizontal_geometry_retains_unexplained_shape_change(self):
        height, width, atoms = 14, 23, 9
        law = np.linspace(-1.0, 1.0, atoms)
        scale = np.linspace(0.2, 1.0, width)[None, :, None]
        values = np.broadcast_to(
            law[None, None, :] * scale, (height, width, atoms)).copy()
        mass = np.ones_like(values)
        zero = np.zeros((height, width))
        geometry = predictive_jet_horizontal_wasserstein_geometry(
            values, mass, zero, zero, quantile_count=17)
        self.assertGreater(geometry["implied_support"], 1.0)
        self.assertGreater(
            float(np.mean(geometry["precision_xx"])),
            float(np.mean(geometry["precision_yy"])),
        )

    def test_directional_sasaki_geometry_annihilates_affine_jet_law(self):
        height, width, atoms = 13, 21, 8
        yy, xx = np.mgrid[:height, :width]
        slope_x = 0.013
        slope_y = -0.019
        law = np.linspace(-0.3, 0.4, atoms)
        values = (
            law[None, None, :]
            + slope_x * xx[..., None]
            + slope_y * yy[..., None]
        )
        mass = np.ones_like(values)
        tangent = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (atoms // 2, 1))
        derivative = np.broadcast_to(
            slope_x * tangent[:, 1] + slope_y * tangent[:, 0],
            values.shape,
        ).copy()
        geometry = predictive_directional_jet_sasaki_geometry(
            values, mass, derivative, tangent, quantile_count=17)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)

    def test_directional_sasaki_geometry_retains_curvature(self):
        height, width, atoms = 13, 21, 8
        yy, xx = np.mgrid[:height, :width]
        law = np.linspace(-0.1, 0.1, atoms)
        values = law[None, None, :] + 0.001 * xx[..., None] ** 2
        values = np.broadcast_to(values, (height, width, atoms)).copy()
        mass = np.ones_like(values)
        tangent = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (atoms // 2, 1))
        derivative = np.broadcast_to(
            0.002 * xx[..., None] * tangent[:, 1], values.shape).copy()
        geometry = predictive_directional_jet_sasaki_geometry(
            values, mass, derivative, tangent, quantile_count=17)
        self.assertGreater(geometry["implied_support"], 1.0)
        self.assertGreater(geometry["vertical_jet_trace_mean"], 0.0)

    def test_shared_support_quantiles_are_source_order_invariant(self):
        rng = np.random.default_rng(75)
        support = rng.normal(size=13)
        weights = rng.random((7, 9, support.size))
        order = rng.permutation(support.size)
        expected = weighted_support_quantiles(weights, support, 19)
        actual = weighted_support_quantiles(
            weights[..., order], support[order], 19)
        np.testing.assert_array_equal(actual, expected)

    def test_lineage_jet_geometry_annihilates_constant_source_jet(self):
        height, width = 6, 8
        pixels = height * width
        rng = np.random.default_rng(76)
        lineage = rng.random((height, width, pixels))
        lineage /= np.sum(lineage, axis=-1, keepdims=True)
        gradient_x = np.full((height, width), 0.017)
        gradient_y = np.full((height, width), -0.009)
        geometry = predictive_lineage_jet_geometry(
            lineage, gradient_x, gradient_y, quantile_count=17)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=12)

    def test_post_lineage_prolongation_annihilates_affine_section(self):
        height, width = 7, 9
        yy, xx = np.mgrid[:height, :width]
        source = (0.13 + 0.017 * xx - 0.009 * yy).ravel()
        lineage = np.eye(height * width).reshape(
            height, width, height * width)
        geometry = predictive_lineage_prolongation_geometry(lineage, source)
        self.assertAlmostEqual(geometry["implied_support"], 1.0, places=11)

    def test_post_lineage_prolongation_retains_curvature(self):
        height, width = 7, 9
        yy, xx = np.mgrid[:height, :width]
        source = (0.001 * xx * xx + 0.002 * yy * yy).ravel()
        lineage = np.eye(height * width).reshape(
            height, width, height * width)
        geometry = predictive_lineage_prolongation_geometry(lineage, source)
        self.assertGreater(geometry["implied_support"], 1.0)

    def test_combining_topological_geometries_counts_domain_base_once(self):
        particles = np.broadcast_to(
            np.array([-0.2, 0.1, 0.7]), (8, 11, 3)).copy()
        first = predictive_wasserstein_geometry(particles)
        second = predictive_horizontal_wasserstein_geometry(particles)
        combined = combine_information_geometries(first, second)
        self.assertAlmostEqual(combined["implied_support"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
