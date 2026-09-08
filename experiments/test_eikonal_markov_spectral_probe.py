from __future__ import annotations

import unittest

import numpy as np

from experiments.eikonal_markov_spectral_probe import (
    build_metric_graph,
    common_refinement_sites,
    hs_optimal_positive_polynomial,
    interpolate_kernel,
    metric_curved,
    metric_rotated,
    nearest_sources,
    rational_interface_signal,
    rational_smooth_signal,
    sample_sites,
    sampling_stable_spectral_reconstruction,
    sampling_trace_polynomial,
    selling_stencil,
)


class EikonalMarkovSpectralProbeTests(unittest.TestCase):
    def test_factor_two_center_lattices_are_exact_and_disjoint(self) -> None:
        side, sources, queries = common_refinement_sites(3)
        self.assertEqual(side, 13)
        self.assertEqual(sources.size, 9)
        self.assertEqual(queries.size, 36)
        self.assertEqual(np.intersect1d(sources, queries).size, 0)

        source_x = np.unique(sources % side)
        query_x = np.unique(queries % side)
        np.testing.assert_array_equal(source_x, np.array([2, 6, 10]))
        np.testing.assert_array_equal(query_x, np.array([1, 3, 5, 7, 9, 11]))

    def test_spacing_and_anisotropy_form_one_spd_quadratic(self) -> None:
        metric = metric_rotated(0.0, 0.0)
        spacing = np.diag([1.0, 2.0])
        displacement = np.array([1.0, -1.0])
        physical = spacing @ displacement
        self.assertEqual(float(physical @ metric @ physical), 2.0)
        pulled_back = spacing.T @ metric @ spacing
        directions, coefficients = selling_stencil(np.linalg.inv(pulled_back))
        reconstructed = sum(
            (
                coefficient * np.outer(direction, direction)
                for coefficient, direction in zip(coefficients, directions)
            ),
            np.zeros((2, 2)),
        )
        np.testing.assert_allclose(reconstructed, np.linalg.inv(pulled_back), atol=1.0e-14)
        self.assertGreaterEqual(float(coefficients.min()), 0.0)

    def test_sampling_trace_polynomial_is_markov_psd_and_trace_matched(self) -> None:
        _, _, laplacian = build_metric_graph(8, 8, (1.0, 1.7), metric_curved, False)
        samples = sample_sites(8, 8, 3, "stratified")
        kernel, _, _, _, minimum_mass = sampling_trace_polynomial(laplacian, samples)
        tolerance = 2.0e-12
        self.assertGreaterEqual(float(kernel.min()), -tolerance)
        self.assertLessEqual(float(np.max(np.abs(kernel.sum(axis=1) - 1.0))), tolerance)
        self.assertLessEqual(float(np.max(np.abs(kernel - kernel.T))), tolerance)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(kernel)[0]), -tolerance)
        self.assertAlmostEqual(float(np.trace(kernel)), float(samples.size), places=10)
        self.assertGreater(minimum_mass, 0.0)

    def test_hs_optimal_polynomial_is_positive_markov_psd_and_trace_matched(self) -> None:
        _, _, laplacian = build_metric_graph(8, 8, (1.0, 1.7), metric_curved, False)
        samples = sample_sites(8, 8, 3, "stratified")
        kernel, coefficients, _, _, minimum_mass = hs_optimal_positive_polynomial(
            laplacian, samples
        )
        tolerance = 3.0e-10
        self.assertGreaterEqual(float(coefficients.min()), 0.0)
        self.assertAlmostEqual(float(coefficients.sum()), 1.0, places=11)
        self.assertGreaterEqual(float(kernel.min()), -tolerance)
        self.assertLessEqual(float(np.max(np.abs(kernel.sum(axis=1) - 1.0))), tolerance)
        self.assertLessEqual(float(np.max(np.abs(kernel - kernel.T))), tolerance)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(kernel)[0]), -tolerance)
        self.assertAlmostEqual(float(np.trace(kernel)), float(samples.size), places=8)
        self.assertGreater(minimum_mass, 0.0)

    def test_final_weights_are_convex_and_preserve_sample_range(self) -> None:
        coordinates, owners, laplacian = build_metric_graph(
            10, 10, (1.0, 1.7), metric_curved, True
        )
        samples = sample_sites(10, 10, 3, "stratified")
        truth = rational_interface_signal(coordinates, owners)
        values = truth[samples]
        kernel, _, _, _, _ = sampling_trace_polynomial(laplacian, samples)
        nearest = nearest_sources(laplacian, samples)
        estimate, weights, denominator = interpolate_kernel(
            kernel, samples, values, nearest
        )
        self.assertGreater(float(denominator.min()), 0.0)
        self.assertGreaterEqual(float(weights.min()), -2.0e-13)
        self.assertLessEqual(float(np.max(np.abs(weights.sum(axis=1) - 1.0))), 2.0e-12)
        for owner in np.unique(owners):
            source_values = values[owners[samples] == owner]
            nodes = owners == owner
            self.assertGreaterEqual(float(estimate[nodes].min()), float(source_values.min()) - 2.0e-13)
            self.assertLessEqual(float(estimate[nodes].max()), float(source_values.max()) + 2.0e-13)

    def test_observation_values_are_retained_separately(self) -> None:
        coordinates, _, laplacian = build_metric_graph(
            8, 8, (1.0, 1.7), metric_curved, False
        )
        samples = sample_sites(8, 8, 3, "regular")
        truth = rational_smooth_signal(coordinates)
        kernel, _, _, _, _ = sampling_trace_polynomial(laplacian, samples)
        nearest = nearest_sources(laplacian, samples)
        estimate, _, _ = interpolate_kernel(kernel, samples, truth[samples], nearest)
        np.testing.assert_array_equal(estimate[samples], truth[samples])

    def test_sampling_stable_spectrum_reproduces_constants_and_retains_samples(self) -> None:
        _, _, laplacian = build_metric_graph(
            8, 8, (1.0, 1.7), metric_curved, False
        )
        samples = sample_sites(8, 8, 3, "stratified")
        values = np.full(samples.size, 0.375)
        estimate, weights, audits, projector = sampling_stable_spectral_reconstruction(
            laplacian, samples, values
        )
        np.testing.assert_allclose(estimate, 0.375, atol=3.0e-13)
        np.testing.assert_array_equal(estimate[samples], values)
        self.assertLessEqual(float(np.max(np.abs(weights.sum(axis=1) - 1.0))), 3.0e-12)
        self.assertEqual(len(audits), 1)
        self.assertGreater(audits[0]["minimum_sampling_singular_value"], 0.0)
        self.assertLessEqual(float(np.max(np.abs(projector - projector.T))), 3.0e-13)
        self.assertLessEqual(
            float(np.max(np.abs(projector @ projector - projector))), 3.0e-13
        )


if __name__ == "__main__":
    unittest.main()
