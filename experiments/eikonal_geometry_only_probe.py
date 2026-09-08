"""Isolated comparison of value-independent Riemannian interpolation laws.

The manuscript is intentionally not imported or modified.  The strict candidate in
this probe is a discrete Riemannian natural-neighbour operator: its weights depend
only on the SPD metric, owners, source sites, and query site.  Scalar graph AMLE is
included as a fixed-law but value-adaptive comparator.  The existing positive Markov
and sampling-stable spectral constructions are reused as controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from experiments.eikonal_markov_spectral_probe import (
    Array,
    build_metric_graph,
    clamp_by_owner,
    common_refinement_sites,
    determinant_one,
    metric_curved,
    metric_isotropic,
    metric_rotated,
    rational_interface_signal,
    rational_smooth_signal,
    sampling_stable_spectral_reconstruction,
    sampling_trace_polynomial,
    selling_stencil,
    trigonometric_signal,
    unresolved_extremum_signal,
)


@dataclass(frozen=True)
class GeometryOnlyAudit:
    mse: float
    maximum_sample_range_overshoot: float
    maximum_sample_error: float
    minimum_weight: float
    maximum_row_mass_error: float
    new_strict_extrema: int


def metric_path_graph(
    coordinates: Array,
    owners: Array,
    laplacian: Array,
    metric_fn: Callable[[float, float], Array],
) -> sparse.csr_matrix:
    """Give every admitted stencil edge its midpoint Riemannian length."""

    scale = coordinates.max(axis=0)
    normalized = coordinates / scale
    metrics = np.stack(
        [determinant_one(metric_fn(float(x), float(y))) for x, y in normalized]
    )
    rows, columns = np.nonzero(np.triu(laplacian < 0.0, k=1))
    graph_rows: list[int] = []
    graph_columns: list[int] = []
    graph_lengths: list[float] = []
    for first, second in zip(rows, columns):
        if owners[first] != owners[second]:
            continue
        displacement = coordinates[second] - coordinates[first]
        midpoint_metric = 0.5 * (metrics[first] + metrics[second])
        length = float(np.sqrt(displacement @ midpoint_metric @ displacement))
        if not length > 0.0:
            raise RuntimeError("Riemannian edge length must be positive")
        graph_rows.extend((int(first), int(second)))
        graph_columns.extend((int(second), int(first)))
        graph_lengths.extend((length, length))
    graph = sparse.coo_matrix(
        (graph_lengths, (graph_rows, graph_columns)), shape=laplacian.shape
    ).tocsr()
    component_count, labels = csgraph.connected_components(graph, directed=False)
    for component in range(component_count):
        if np.count_nonzero(labels == component) < 2:
            raise RuntimeError("metric path graph contains an isolated node")
    return graph


def topological_neighbor_graph(coordinates: Array, owners: Array) -> sparse.csr_matrix:
    """Full owner-preserving Cartesian 8-neighborhood for extrema audits."""

    x_axis = np.unique(coordinates[:, 0])
    y_axis = np.unique(coordinates[:, 1])
    index = {
        (float(point[0]), float(point[1])): node
        for node, point in enumerate(coordinates)
    }
    rows: list[int] = []
    columns: list[int] = []
    for iy, y in enumerate(y_axis):
        for ix, x in enumerate(x_axis):
            node = index[(float(x), float(y))]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < x_axis.size and 0 <= jy < y_axis.size):
                        continue
                    neighbor = index[(float(x_axis[jx]), float(y_axis[jy]))]
                    if owners[node] == owners[neighbor]:
                        rows.append(node)
                        columns.append(neighbor)
    return sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(coordinates.shape[0], coordinates.shape[0]),
    ).tocsr()


def _tie_mask(values: Array, minima: Array) -> Array:
    scale = np.maximum(1.0, np.abs(minima))
    tolerance = 64.0 * np.finfo(float).eps * scale
    return np.abs(values - minima) <= tolerance


def discrete_riemannian_natural_neighbor_weights(
    graph: sparse.csr_matrix,
    samples: Array,
) -> Array:
    """Voronoi-volume-transfer weights on a finite Riemannian path space.

    Every graph vertex has equal determinant-one volume.  Voronoi ties are
    divided equally, which is invariant under permutations and preserves exact
    grid symmetries.  No observed value is read by this function.
    """

    node_count = graph.shape[0]
    sample_distances = csgraph.dijkstra(graph, directed=False, indices=samples)
    old_minimum = np.min(sample_distances, axis=0)
    if np.any(~np.isfinite(old_minimum)):
        raise ValueError("every metric component must contain a source")
    old_membership = _tie_mask(sample_distances, old_minimum[None, :])
    old_share = old_membership / old_membership.sum(axis=0, keepdims=True)
    all_distances = csgraph.dijkstra(graph, directed=False)

    weights = np.zeros((node_count, samples.size), dtype=float)
    sample_lookup = {int(site): column for column, site in enumerate(samples)}
    for query in range(node_count):
        if query in sample_lookup:
            weights[query, sample_lookup[query]] = 1.0
            continue
        query_distance = all_distances[query]
        finite = np.isfinite(query_distance) & np.isfinite(old_minimum)
        comparison_scale = np.maximum(1.0, np.maximum(query_distance, old_minimum))
        tolerance = 64.0 * np.finfo(float).eps * comparison_scale
        strictly_new = finite & (query_distance < old_minimum - tolerance)
        tied = finite & (np.abs(query_distance - old_minimum) <= tolerance)
        query_share = np.zeros(node_count, dtype=float)
        query_share[strictly_new] = 1.0
        query_share[tied] = 1.0 / (old_membership[:, tied].sum(axis=0) + 1.0)
        transferred = (old_share * query_share[None, :]).sum(axis=1)
        mass = float(transferred.sum())
        if not mass > 0.0:
            raise RuntimeError("inserted query acquired no Voronoi volume")
        weights[query] = transferred / mass

    # Close only floating summation error; the defining construction is a
    # nonnegative partition of unity in exact arithmetic.
    weights[weights < 0.0] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def harmonic_barycentric_weights(laplacian: Array, samples: Array) -> Array:
    """Riemannian Dirichlet harmonic coordinates with point observations.

    The columns are geometry-only basis functions.  Each equals one at its own
    source, zero at the other sources, and is graph harmonic elsewhere.  The
    M-matrix maximum principle makes the exact weights nonnegative and mass one.
    """

    node_count = laplacian.shape[0]
    fixed = np.zeros(node_count, dtype=bool)
    fixed[samples] = True
    free = np.flatnonzero(~fixed)
    weights = np.zeros((node_count, samples.size), dtype=float)
    weights[samples] = np.eye(samples.size)
    if free.size:
        interior = laplacian[np.ix_(free, free)]
        boundary = laplacian[np.ix_(free, samples)]
        weights[free] = np.linalg.solve(interior, -boundary)
    tolerance = 256.0 * np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(weights, ord=np.inf))
    )
    if float(weights.min()) < -tolerance:
        raise RuntimeError("harmonic coordinates violated the M-matrix maximum principle")
    weights[np.abs(weights) <= tolerance] = 0.0
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _bracket(axis: Array, coordinate: float) -> tuple[float, float, float]:
    """Bracket with exact boundary replication and return normalized position."""

    if coordinate <= axis[0]:
        return float(axis[0]), float(axis[0]), 0.0
    if coordinate >= axis[-1]:
        return float(axis[-1]), float(axis[-1]), 0.0
    upper_index = int(np.searchsorted(axis, coordinate, side="right"))
    lower, upper = float(axis[upper_index - 1]), float(axis[upper_index])
    return lower, upper, (coordinate - lower) / (upper - lower)


def _triangle_barycentric(point: Array, vertices: Array) -> Array:
    system = np.vstack((vertices.T, np.ones(3)))
    right = np.array((point[0], point[1], 1.0), dtype=float)
    return np.linalg.solve(system, right)


def _opposite_angle(first: Array, vertex: Array, third: Array) -> float:
    left, right = first - vertex, third - vertex
    cosine = float(left @ right) / float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def riemannian_delaunay_p1_weights(
    coordinates: Array,
    owners: Array,
    samples: Array,
    metric_fn: Callable[[float, float], Array],
) -> Array:
    """Geometry-only metric-Delaunay P1 coordinates on a Cartesian acquisition.

    Each owner is treated independently.  A cell's determinant-one midpoint
    metric whitens its four source vertices; the Delaunay diagonal is selected
    in that metric.  P1 barycentric weights are affine-invariant, hence their
    numerical values are the same before and after whitening.  A co-circular
    cell has no distinguished diagonal and uses its symmetric bilinear limit.
    """

    node_count = coordinates.shape[0]
    weights = np.zeros((node_count, samples.size), dtype=float)
    scale = coordinates.max(axis=0)
    sample_column = {int(node): column for column, node in enumerate(samples)}
    for owner in np.unique(owners):
        owner_columns = np.flatnonzero(owners[samples] == owner)
        if owner_columns.size == 0:
            raise ValueError("every owner must contain a source")
        owner_samples = samples[owner_columns]
        x_axis = np.unique(coordinates[owner_samples, 0])
        y_axis = np.unique(coordinates[owner_samples, 1])
        lookup = {
            (float(coordinates[node, 0]), float(coordinates[node, 1])): sample_column[int(node)]
            for node in owner_samples
        }
        for node in np.flatnonzero(owners == owner):
            if node in sample_column:
                weights[node, sample_column[node]] = 1.0
                continue
            x, y = (float(value) for value in coordinates[node])
            x0, x1, tx = _bracket(x_axis, x)
            y0, y1, ty = _bracket(y_axis, y)

            if x0 == x1 and y0 == y1:
                weights[node, lookup[(x0, y0)]] = 1.0
                continue
            if x0 == x1:
                weights[node, lookup[(x0, y0)]] = 1.0 - ty
                weights[node, lookup[(x0, y1)]] = ty
                continue
            if y0 == y1:
                weights[node, lookup[(x0, y0)]] = 1.0 - tx
                weights[node, lookup[(x1, y0)]] = tx
                continue

            corners = np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
            columns = [lookup[tuple(corner)] for corner in corners]
            midpoint = corners.mean(axis=0) / scale
            metric = determinant_one(metric_fn(float(midpoint[0]), float(midpoint[1])))
            whitener = np.linalg.cholesky(metric).T
            transformed = corners @ whitener.T
            angle_br = _opposite_angle(transformed[0], transformed[1], transformed[2])
            angle_tl = _opposite_angle(transformed[0], transformed[3], transformed[2])
            angle_sum = angle_br + angle_tl
            tie_tolerance = 256.0 * np.finfo(float).eps

            if abs(angle_sum - np.pi) <= tie_tolerance:
                local = np.array(
                    (
                        (1.0 - tx) * (1.0 - ty),
                        tx * (1.0 - ty),
                        tx * ty,
                        (1.0 - tx) * ty,
                    )
                )
                weights[node, columns] = local
                continue

            point = np.array((x, y))
            if angle_sum < np.pi:  # Delaunay diagonal bottom-left to top-right.
                triangle = (0, 1, 2) if tx >= ty else (0, 2, 3)
            else:  # Delaunay diagonal bottom-right to top-left.
                triangle = (0, 1, 3) if tx + ty <= 1.0 else (1, 2, 3)
            local = _triangle_barycentric(point, corners[list(triangle)])
            for coefficient, corner_index in zip(local, triangle):
                weights[node, columns[corner_index]] = coefficient

    tolerance = 512.0 * np.finfo(float).eps
    if float(weights.min()) < -tolerance:
        raise RuntimeError("metric-Delaunay P1 coordinates left their simplex")
    weights[np.abs(weights) <= tolerance] = 0.0
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def riemannian_harmonic_cell_weights(
    coordinates: Array,
    owners: Array,
    samples: Array,
    metric_fn: Callable[[float, float], Array],
) -> Array:
    """Monotone local Riemannian harmonic coordinates on Cartesian cells.

    The target lattice supplies the Dirichlet grid, so there is no resolution
    parameter.  Every source cell is kept unsplit.  Its midpoint metric is
    pulled back through the physical target spacing, inverted to the
    Laplace--Beltrami diffusion tensor, and decomposed into nonnegative axial
    plus directional second differences.  The unit-stencil decomposition is
    used when available; otherwise Selling's nonnegative lattice decomposition
    supplies a boundary-truncated wide stencil.  Nonuniform endpoint factors
    preserve affine functions and the M-matrix sign pattern exactly in the
    defining arithmetic.
    """

    x_axis = np.unique(coordinates[:, 0])
    y_axis = np.unique(coordinates[:, 1])
    nx, ny = x_axis.size, y_axis.size
    if nx * ny != coordinates.shape[0]:
        raise ValueError("coordinates must form a complete Cartesian lattice")
    dx = float(x_axis[1] - x_axis[0]) if nx > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if ny > 1 else 1.0
    hmat = np.diag((dx, dy))
    physical_scale = np.array(
        [max(float(x_axis[-1]), dx), max(float(y_axis[-1]), dy)]
    )
    sample_column = {int(node): column for column, node in enumerate(samples)}
    weights = np.zeros((coordinates.shape[0], samples.size), dtype=float)
    assigned = np.zeros(coordinates.shape[0], dtype=bool)

    def node(ix: int, iy: int) -> int:
        return iy * nx + ix

    def boundary_hats(
        lx: float, ly: float, width: int, height: int
    ) -> Array:
        boundary_tolerance = 2.0e-12
        tx, ty = np.clip(lx / width, 0.0, 1.0), np.clip(ly / height, 0.0, 1.0)
        if abs(ly) <= boundary_tolerance:
            return np.array((1.0 - tx, tx, 0.0, 0.0))
        if abs(lx - width) <= boundary_tolerance:
            return np.array((0.0, 1.0 - ty, ty, 0.0))
        if abs(ly - height) <= boundary_tolerance:
            return np.array((0.0, 0.0, tx, 1.0 - tx))
        if abs(lx) <= boundary_tolerance:
            return np.array((1.0 - ty, 0.0, 0.0, ty))
        raise ValueError("local point is not on the cell boundary")

    for owner in np.unique(owners):
        owner_sample_nodes = samples[owners[samples] == owner]
        if owner_sample_nodes.size == 0:
            raise ValueError("every owner must contain a source")
        sample_x = np.unique(owner_sample_nodes % nx)
        sample_y = np.unique(owner_sample_nodes // nx)
        if sample_x.size == 1 or sample_y.size == 1:
            # A thin owner has a canonical lower-dimensional harmonic law:
            # geodesic-linear interpolation along its sampled axis and exact
            # replication in the absent axis.
            for global_node in np.flatnonzero(owners == owner):
                ix, iy = global_node % nx, global_node // nx
                row = np.zeros(samples.size, dtype=float)
                if sample_x.size == 1 and sample_y.size == 1:
                    row[sample_column[node(int(sample_x[0]), int(sample_y[0]))]] = 1.0
                elif sample_x.size == 1:
                    yc = int(np.clip(iy, sample_y[0], sample_y[-1]))
                    upper = int(np.searchsorted(sample_y, yc, side="right"))
                    upper = min(max(upper, 1), sample_y.size - 1)
                    y0, y1 = int(sample_y[upper - 1]), int(sample_y[upper])
                    t = (yc - y0) / (y1 - y0)
                    row[sample_column[node(int(sample_x[0]), y0)]] = 1.0 - t
                    row[sample_column[node(int(sample_x[0]), y1)]] = t
                else:
                    xc = int(np.clip(ix, sample_x[0], sample_x[-1]))
                    upper = int(np.searchsorted(sample_x, xc, side="right"))
                    upper = min(max(upper, 1), sample_x.size - 1)
                    x0, x1 = int(sample_x[upper - 1]), int(sample_x[upper])
                    t = (xc - x0) / (x1 - x0)
                    row[sample_column[node(x0, int(sample_y[0]))]] = 1.0 - t
                    row[sample_column[node(x1, int(sample_y[0]))]] = t
                weights[global_node] = row
                assigned[global_node] = True
            continue
        for iy0, iy1 in zip(sample_y, sample_y[1:]):
            for ix0, ix1 in zip(sample_x, sample_x[1:]):
                corners = (
                    node(int(ix0), int(iy0)),
                    node(int(ix1), int(iy0)),
                    node(int(ix1), int(iy1)),
                    node(int(ix0), int(iy1)),
                )
                if any(corner not in sample_column for corner in corners):
                    continue
                if any(owners[corner] != owner for corner in corners):
                    continue
                width, height = int(ix1 - ix0), int(iy1 - iy0)
                midpoint = np.array(
                    (
                        0.5 * (x_axis[ix0] + x_axis[ix1]),
                        0.5 * (y_axis[iy0] + y_axis[iy1]),
                    )
                ) / physical_scale
                metric = determinant_one(
                    metric_fn(float(midpoint[0]), float(midpoint[1]))
                )
                diffusion = np.linalg.inv(hmat.T @ metric @ hmat)
                cross = float(diffusion[0, 1])
                diagonal = abs(cross)
                horizontal = float(diffusion[0, 0]) - diagonal
                vertical = float(diffusion[1, 1]) - diagonal
                tolerance = 512.0 * np.finfo(float).eps * max(
                    1.0, float(np.linalg.norm(diffusion, ord=np.inf))
                )
                if horizontal >= -tolerance and vertical >= -tolerance:
                    horizontal, vertical = max(horizontal, 0.0), max(vertical, 0.0)
                    diagonal_direction = (1, 1) if cross >= 0.0 else (1, -1)
                    directions = (
                        (np.array((1, 0), dtype=int), horizontal),
                        (np.array((0, 1), dtype=int), vertical),
                        (np.array(diagonal_direction, dtype=int), diagonal),
                    )
                else:
                    wide_directions, wide_coefficients = selling_stencil(diffusion)
                    directions = tuple(
                        (np.asarray(direction, dtype=int), float(coefficient))
                        for direction, coefficient in zip(
                            wide_directions, wide_coefficients
                        )
                        if coefficient > tolerance
                    )
                interior = [
                    (lx, ly)
                    for ly in range(1, height)
                    for lx in range(1, width)
                ]
                row_of = {point: row for row, point in enumerate(interior)}
                matrix = np.zeros((len(interior), len(interior)), dtype=float)
                right = np.zeros((len(interior), 4), dtype=float)

                def ray_fraction(point: tuple[int, int], direction: Array) -> float:
                    fraction = 1.0
                    for coordinate, step, limit in zip(point, direction, (width, height)):
                        if step > 0:
                            fraction = min(fraction, (limit - coordinate) / float(step))
                        elif step < 0:
                            fraction = min(fraction, coordinate / float(-step))
                    if not fraction > 0.0:
                        raise RuntimeError("wide stencil did not reach a positive boundary distance")
                    return fraction

                for point, row in row_of.items():
                    for direction, coefficient in directions:
                        if coefficient == 0.0:
                            continue
                        plus = ray_fraction(point, direction)
                        minus = ray_fraction(point, -direction)
                        endpoint_coefficients = (
                            (plus, 2.0 / (plus * (plus + minus))),
                            (-minus, 2.0 / (minus * (plus + minus))),
                        )
                        for signed_fraction, directional_weight in endpoint_coefficients:
                            endpoint = np.asarray(point, dtype=float) + signed_fraction * direction
                            total_weight = coefficient * directional_weight
                            matrix[row, row] += total_weight
                            rounded = tuple(int(round(value)) for value in endpoint)
                            if np.allclose(endpoint, rounded, rtol=0.0, atol=2.0e-12) and rounded in row_of:
                                matrix[row, row_of[rounded]] -= total_weight
                            else:
                                right[row] += total_weight * boundary_hats(
                                    float(endpoint[0]), float(endpoint[1]), width, height
                                )
                solution = np.linalg.solve(matrix, right) if interior else right
                corner_columns = [sample_column[corner] for corner in corners]
                for ly in range(height + 1):
                    for lx in range(width + 1):
                        global_node = node(int(ix0) + lx, int(iy0) + ly)
                        if owners[global_node] != owner:
                            continue
                        local = (
                            solution[row_of[(lx, ly)]]
                            if (lx, ly) in row_of
                            else boundary_hats(lx, ly, width, height)
                        )
                        row = np.zeros(samples.size, dtype=float)
                        row[corner_columns] = local
                        if assigned[global_node]:
                            if not np.allclose(
                                weights[global_node], row, rtol=2.0e-12, atol=2.0e-12
                            ):
                                raise RuntimeError("neighboring harmonic cells disagree")
                        else:
                            weights[global_node] = row
                            assigned[global_node] = True

        # Exact replication from the ownerwise source hull supplies all exterior nodes.
        for global_node in np.flatnonzero(owners == owner):
            if assigned[global_node]:
                continue
            ix, iy = global_node % nx, global_node // nx
            clamped_x = int(np.clip(ix, sample_x[0], sample_x[-1]))
            clamped_y = int(np.clip(iy, sample_y[0], sample_y[-1]))
            clamped = node(clamped_x, clamped_y)
            if not assigned[clamped]:
                raise RuntimeError("ownerwise harmonic cell coverage is incomplete")
            weights[global_node] = weights[clamped]
            assigned[global_node] = True

    if not np.all(assigned):
        raise RuntimeError("harmonic cell operator left target nodes unassigned")
    tolerance = 2048.0 * np.finfo(float).eps
    if float(weights.min()) < -tolerance:
        raise RuntimeError("harmonic cell coordinates lost positivity")
    weights[np.abs(weights) <= tolerance] = 0.0
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def weighted_midrange(neighbor_values: Array, lengths: Array) -> float:
    """Unique scalar minimizing the largest incident absolute metric slope."""

    if neighbor_values.size == 1:
        return float(neighbor_values[0])
    best_slope = -np.inf
    best_value = float(neighbor_values[0])
    for high in range(neighbor_values.size):
        for low in range(neighbor_values.size):
            slope = (neighbor_values[high] - neighbor_values[low]) / (
                lengths[high] + lengths[low]
            )
            candidate = (
                lengths[low] * neighbor_values[high]
                + lengths[high] * neighbor_values[low]
            ) / (lengths[high] + lengths[low])
            if slope > best_slope:
                best_slope = float(slope)
                best_value = float(candidate)
    return best_value


def graph_amle(
    graph: sparse.csr_matrix,
    samples: Array,
    values: Array,
    initial: Array,
    tolerance: float = 2.0e-12,
    maximum_sweeps: int = 50000,
) -> tuple[Array, int, float]:
    """Solve the weighted graph infinity equation by deterministic GS sweeps."""

    estimate = np.asarray(initial, dtype=float).copy()
    estimate[samples] = values
    fixed = np.zeros(graph.shape[0], dtype=bool)
    fixed[samples] = True
    unconstrained = np.flatnonzero(~fixed)

    for sweep in range(1, maximum_sweeps + 1):
        maximum_change = 0.0
        for node in unconstrained:
            start, end = graph.indptr[node], graph.indptr[node + 1]
            neighbors = graph.indices[start:end]
            lengths = graph.data[start:end]
            updated = weighted_midrange(estimate[neighbors], lengths)
            maximum_change = max(maximum_change, abs(updated - estimate[node]))
            estimate[node] = updated
        if maximum_change <= tolerance:
            break
    else:
        raise RuntimeError("graph AMLE iteration did not converge")

    residual = 0.0
    for node in unconstrained:
        start, end = graph.indptr[node], graph.indptr[node + 1]
        neighbors = graph.indices[start:end]
        lengths = graph.data[start:end]
        residual = max(
            residual,
            abs(estimate[node] - weighted_midrange(estimate[neighbors], lengths)),
        )
    return estimate, sweep, residual


def new_strict_extrema(
    estimate: Array,
    truth: Array,
    graph: sparse.csr_matrix,
    samples: Array,
    evaluated: Array,
) -> int:
    fixed = np.zeros(estimate.size, dtype=bool)
    fixed[samples] = True
    count = 0
    for node in evaluated:
        if fixed[node]:
            continue
        start, end = graph.indptr[node], graph.indptr[node + 1]
        neighbors = graph.indices[start:end]
        estimate_scale = max(
            1.0, abs(float(estimate[node])), float(np.max(np.abs(estimate[neighbors])))
        )
        truth_scale = max(
            1.0, abs(float(truth[node])), float(np.max(np.abs(truth[neighbors])))
        )
        estimate_tolerance = 512.0 * np.finfo(float).eps * estimate_scale
        truth_tolerance = 512.0 * np.finfo(float).eps * truth_scale
        estimate_max = bool(
            np.all(estimate[node] > estimate[neighbors] + estimate_tolerance)
        )
        estimate_min = bool(
            np.all(estimate[node] < estimate[neighbors] - estimate_tolerance)
        )
        truth_max = bool(np.all(truth[node] > truth[neighbors] + truth_tolerance))
        truth_min = bool(np.all(truth[node] < truth[neighbors] - truth_tolerance))
        count += int((estimate_max and not truth_max) or (estimate_min and not truth_min))
    return count


def audit_method(
    estimate: Array,
    weights: Array,
    truth: Array,
    samples: Array,
    owners: Array,
    graph: sparse.csr_matrix,
    evaluated: Array,
) -> GeometryOnlyAudit:
    values = truth[samples]
    overshoot = 0.0
    for owner in np.unique(owners[evaluated]):
        source_values = values[owners[samples] == owner]
        nodes = evaluated[owners[evaluated] == owner]
        lower, upper = float(source_values.min()), float(source_values.max())
        overshoot = max(
            overshoot,
            float(np.max(np.maximum(lower - estimate[nodes], estimate[nodes] - upper))),
        )
    if weights.size:
        minimum_weight = float(weights.min())
        row_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    else:
        minimum_weight = float("nan")
        row_error = float("nan")
    return GeometryOnlyAudit(
        mse=float(np.mean(np.square(estimate[evaluated] - truth[evaluated]))),
        maximum_sample_range_overshoot=max(0.0, overshoot),
        maximum_sample_error=float(np.max(np.abs(estimate[samples] - values))),
        minimum_weight=minimum_weight,
        maximum_row_mass_error=row_error,
        new_strict_extrema=new_strict_extrema(
            estimate, truth, graph, samples, evaluated
        ),
    )


def run_geometry_only_case(
    name: str,
    metric_fn: Callable[[float, float], Array],
    signal_kind: str,
    split_owner: bool,
    coarse_size: int,
) -> dict[str, object]:
    side, samples, queries = common_refinement_sites(coarse_size)
    coordinates, owners, laplacian = build_metric_graph(
        side, side, (1.0, 1.7), metric_fn, split_owner
    )
    if not split_owner:
        owners = np.zeros_like(owners)
    graph = metric_path_graph(coordinates, owners, laplacian, metric_fn)
    topology = topological_neighbor_graph(coordinates, owners)

    if signal_kind == "rational":
        truth = rational_smooth_signal(coordinates)
    elif signal_kind == "trigonometric":
        truth = trigonometric_signal(coordinates)
    elif signal_kind == "unresolved_extremum":
        truth = unresolved_extremum_signal(coordinates)
    elif signal_kind == "interface":
        truth = rational_interface_signal(coordinates, owners)
    else:
        raise ValueError(signal_kind)
    values = truth[samples]

    natural_weights = discrete_riemannian_natural_neighbor_weights(graph, samples)
    natural_estimate = natural_weights @ values
    natural_estimate[samples] = values
    weight_digest = hashlib.sha256(natural_weights.tobytes()).hexdigest()
    # This second value vector is deliberately unrelated.  The weights must be
    # bit-identical because the strict candidate is not permitted to inspect it.
    alternate_values = values[::-1].copy()
    if hashlib.sha256(natural_weights.tobytes()).hexdigest() != weight_digest:
        raise AssertionError("natural-neighbour weights changed with values")
    _ = natural_weights @ alternate_values

    delaunay_weights = riemannian_delaunay_p1_weights(
        coordinates, owners, samples, metric_fn
    )
    delaunay_estimate = delaunay_weights @ values
    delaunay_estimate[samples] = values
    delaunay_weight_digest = hashlib.sha256(delaunay_weights.tobytes()).hexdigest()
    _ = delaunay_weights @ alternate_values

    harmonic_cell_weights = riemannian_harmonic_cell_weights(
        coordinates, owners, samples, metric_fn
    )
    harmonic_cell_estimate = harmonic_cell_weights @ values
    harmonic_cell_estimate[samples] = values
    harmonic_cell_weight_digest = hashlib.sha256(
        harmonic_cell_weights.tobytes()
    ).hexdigest()
    _ = harmonic_cell_weights @ alternate_values

    harmonic_weights = harmonic_barycentric_weights(laplacian, samples)
    harmonic_estimate = harmonic_weights @ values
    harmonic_estimate[samples] = values
    harmonic_weight_digest = hashlib.sha256(harmonic_weights.tobytes()).hexdigest()
    _ = harmonic_weights @ alternate_values

    amle_estimate, amle_sweeps, amle_residual = graph_amle(
        graph, samples, values, natural_estimate
    )

    markov_kernel, markov_degree, markov_alpha, _, _ = sampling_trace_polynomial(
        laplacian, samples
    )
    markov_sampled = markov_kernel[:, samples]
    markov_weights = markov_sampled / markov_sampled.sum(axis=1, keepdims=True)
    markov_estimate = markov_weights @ values
    markov_estimate[samples] = values

    spectral_estimate, spectral_weights, spectral_components, _ = (
        sampling_stable_spectral_reconstruction(laplacian, samples, values)
    )
    projected_spectral, activated = clamp_by_owner(
        spectral_estimate, samples, values, owners
    )
    projected_spectral[samples] = values

    empty = np.empty((0, 0), dtype=float)
    return {
        "case": name,
        "coarse_size": coarse_size,
        "graph_side": side,
        "sample_count": int(samples.size),
        "query_count": int(queries.size),
        "strict_rule": "weights depend only on metric, owners, sites, and query",
        "natural_weight_sha256": weight_digest,
        "delaunay_p1_weight_sha256": delaunay_weight_digest,
        "harmonic_cell_weight_sha256": harmonic_cell_weight_digest,
        "harmonic_weight_sha256": harmonic_weight_digest,
        "amle": {
            "role": "fixed-law value-adaptive comparator",
            "sweeps": amle_sweeps,
            "infinity_residual": amle_residual,
        },
        "positive_markov": {
            "degree": markov_degree,
            "convex_power_coefficient": markov_alpha,
        },
        "projected_spectral": {
            "projection_activation_rate": float(np.mean(activated[queries])),
            "components": spectral_components,
        },
        "methods": {
            "riemannian_harmonic_cells": asdict(
                audit_method(
                    harmonic_cell_estimate,
                    harmonic_cell_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "riemannian_delaunay_p1": asdict(
                audit_method(
                    delaunay_estimate,
                    delaunay_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "riemannian_harmonic_coordinates": asdict(
                audit_method(
                    harmonic_estimate,
                    harmonic_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "riemannian_natural_neighbor": asdict(
                audit_method(
                    natural_estimate,
                    natural_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "scalar_graph_amle": asdict(
                audit_method(
                    amle_estimate,
                    empty,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "positive_markov": asdict(
                audit_method(
                    markov_estimate,
                    markov_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
            "sampling_stable_spectral_projected": asdict(
                audit_method(
                    projected_spectral,
                    spectral_weights,
                    truth,
                    samples,
                    owners,
                    topology,
                    queries,
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sizes", default="3,4")
    arguments = parser.parse_args()
    sizes = [int(value) for value in arguments.sizes.split(",")]
    specifications = (
        ("isotropic_rational", metric_isotropic, "rational", False),
        ("rotated_rational", metric_rotated, "rational", False),
        ("curved_rational", metric_curved, "rational", False),
        ("isotropic_trigonometric", metric_isotropic, "trigonometric", False),
        ("rotated_trigonometric", metric_rotated, "trigonometric", False),
        ("curved_trigonometric", metric_curved, "trigonometric", False),
        ("curved_unresolved_extremum", metric_curved, "unresolved_extremum", False),
        ("curved_owner_interface", metric_curved, "interface", True),
    )
    cases = [
        run_geometry_only_case(name, metric, signal, owner, size)
        for size in sizes
        for name, metric, signal, owner in specifications
    ]
    payload = {
        "status": "diagnostic floating-point screen; formal claims live in the exact certificate",
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
