from __future__ import annotations

import unittest

import numpy as np

from experiments.eikonal_geometry_only_probe import (
    discrete_riemannian_natural_neighbor_weights,
    graph_amle,
    harmonic_barycentric_weights,
    metric_path_graph,
    riemannian_delaunay_p1_weights,
    riemannian_harmonic_cell_weights,
    topological_neighbor_graph,
    weighted_midrange,
)
from experiments.eikonal_markov_spectral_probe import (
    build_metric_graph,
    common_refinement_sites,
    metric_isotropic,
    metric_rotated,
)


class GeometryOnlyProbeTests(unittest.TestCase):
    def test_weighted_midrange_balances_unequal_metric_slopes(self) -> None:
        values = np.array([0.0, 1.0])
        lengths = np.array([1.0, 3.0])
        midpoint = weighted_midrange(values, lengths)
        self.assertEqual(midpoint, 0.25)
        self.assertEqual((midpoint - values[0]) / lengths[0], (values[1] - midpoint) / lengths[1])

    def test_natural_neighbor_weights_are_geometry_only_convex_and_cardinal(self) -> None:
        side, samples, _ = common_refinement_sites(3)
        coordinates, owners, laplacian = build_metric_graph(
            side, side, (1.0, 1.7), metric_rotated, False
        )
        owners[:] = 0
        graph = metric_path_graph(coordinates, owners, laplacian, metric_rotated)
        weights = discrete_riemannian_natural_neighbor_weights(graph, samples)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=3.0e-15)
        np.testing.assert_array_equal(weights[samples], np.eye(samples.size))
        first = weights @ np.arange(samples.size, dtype=float)
        second = weights @ np.arange(samples.size, dtype=float)[::-1]
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(weights, weights.copy())

    def test_scalar_amle_is_cardinal_bounded_and_infinity_harmonic(self) -> None:
        side, samples, _ = common_refinement_sites(3)
        coordinates, owners, laplacian = build_metric_graph(
            side, side, (1.0, 1.7), metric_isotropic, False
        )
        owners[:] = 0
        graph = metric_path_graph(coordinates, owners, laplacian, metric_isotropic)
        weights = discrete_riemannian_natural_neighbor_weights(graph, samples)
        values = np.linspace(0.1, 0.9, samples.size)
        initial = weights @ values
        estimate, _, residual = graph_amle(graph, samples, values, initial)
        np.testing.assert_array_equal(estimate[samples], values)
        self.assertGreaterEqual(float(estimate.min()), float(values.min()) - 2.0e-12)
        self.assertLessEqual(float(estimate.max()), float(values.max()) + 2.0e-12)
        self.assertLessEqual(residual, 1.0e-11)

    def test_harmonic_coordinates_are_geometry_only_convex_cardinal_and_harmonic(self) -> None:
        side, samples, _ = common_refinement_sites(3)
        _, _, laplacian = build_metric_graph(
            side, side, (1.0, 1.7), metric_rotated, False
        )
        weights = harmonic_barycentric_weights(laplacian, samples)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=3.0e-15)
        np.testing.assert_array_equal(weights[samples], np.eye(samples.size))
        free = np.setdiff1d(np.arange(side * side), samples)
        np.testing.assert_allclose(
            laplacian[np.ix_(free, np.arange(side * side))] @ weights,
            0.0,
            atol=2.0e-12,
        )
        values = np.array([0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.6, 0.4, 0.5])
        estimate = weights @ values
        adjacency = laplacian < 0.0
        for node in free:
            neighbors = np.flatnonzero(adjacency[node])
            self.assertFalse(np.all(estimate[node] > estimate[neighbors]))
            self.assertFalse(np.all(estimate[node] < estimate[neighbors]))

    def test_metric_delaunay_p1_is_convex_cardinal_and_affine_exact(self) -> None:
        side, samples, _ = common_refinement_sites(3)
        coordinates, owners, _ = build_metric_graph(
            side, side, (1.0, 1.7), metric_rotated, False
        )
        owners[:] = 0
        weights = riemannian_delaunay_p1_weights(
            coordinates, owners, samples, metric_rotated
        )
        self.assertGreaterEqual(float(weights.min()), 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=3.0e-15)
        np.testing.assert_array_equal(weights[samples], np.eye(samples.size))
        # Affine precision is asserted only inside the convex hull; replicated
        # boundary queries intentionally reproduce the nearest boundary value.
        x_min, y_min = coordinates[samples].min(axis=0)
        x_max, y_max = coordinates[samples].max(axis=0)
        interior = np.flatnonzero(
            (coordinates[:, 0] >= x_min)
            & (coordinates[:, 0] <= x_max)
            & (coordinates[:, 1] >= y_min)
            & (coordinates[:, 1] <= y_max)
        )
        affine = 0.25 + 0.03 * coordinates[:, 0] - 0.02 * coordinates[:, 1]
        np.testing.assert_allclose(
            (weights @ affine[samples])[interior], affine[interior], atol=3.0e-14
        )
        topology = topological_neighbor_graph(coordinates, owners)
        estimate = weights @ np.array(
            [0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.6, 0.4, 0.5]
        )
        free = np.setdiff1d(np.arange(side * side), samples)
        for node in free:
            start, end = topology.indptr[node], topology.indptr[node + 1]
            neighbors = topology.indices[start:end]
            self.assertFalse(np.all(estimate[node] > estimate[neighbors]))
            self.assertFalse(np.all(estimate[node] < estimate[neighbors]))

    def test_harmonic_cells_are_convex_cardinal_affine_and_metric_continuous(self) -> None:
        side, samples, _ = common_refinement_sites(3)
        coordinates, owners, _ = build_metric_graph(
            side, side, (1.0, 1.7), metric_rotated, False
        )
        owners[:] = 0
        weights = riemannian_harmonic_cell_weights(
            coordinates, owners, samples, metric_rotated
        )
        self.assertGreaterEqual(float(weights.min()), 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=3.0e-15)
        np.testing.assert_array_equal(weights[samples], np.eye(samples.size))
        x_min, y_min = coordinates[samples].min(axis=0)
        x_max, y_max = coordinates[samples].max(axis=0)
        interior = np.flatnonzero(
            (coordinates[:, 0] >= x_min)
            & (coordinates[:, 0] <= x_max)
            & (coordinates[:, 1] >= y_min)
            & (coordinates[:, 1] <= y_max)
        )
        affine = 0.25 + 0.03 * coordinates[:, 0] - 0.02 * coordinates[:, 1]
        np.testing.assert_allclose(
            (weights @ affine[samples])[interior], affine[interior], atol=4.0e-14
        )

        def near_flip(epsilon: float):
            return lambda _x, _y: np.array(
                ((1.0, epsilon), (epsilon, 1.0)), dtype=float
            )

        minus = riemannian_harmonic_cell_weights(
            coordinates, owners, samples, near_flip(-1.0e-8)
        )
        zero = riemannian_harmonic_cell_weights(
            coordinates, owners, samples, near_flip(0.0)
        )
        plus = riemannian_harmonic_cell_weights(
            coordinates, owners, samples, near_flip(1.0e-8)
        )
        self.assertLess(float(np.max(np.abs(minus - zero))), 2.0e-8)
        self.assertLess(float(np.max(np.abs(plus - zero))), 2.0e-8)


if __name__ == "__main__":
    unittest.main()
