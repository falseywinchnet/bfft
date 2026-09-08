"""Isolated validation probe for Markov-constrained Riemannian spectral interpolation.

This file is deliberately independent of the manuscript.  It tests whether the
sampling-derived finite spectral projector can be projected onto the cone of
entrywise-nonnegative, mass-preserving spectral kernels without collapsing its
effective rank.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

try:
    import cvxpy as cp
except ModuleNotFoundError:  # Exact/geometry-only probes do not require CVXPY.
    cp = None
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csgraph


Array = np.ndarray


@dataclass(frozen=True)
class SpectralGroup:
    eigenvalue: float
    multiplicity: int
    projector: Array


@dataclass(frozen=True)
class KernelAudit:
    solver_status: str
    node_count: int
    sample_count: int
    admitted_mode_count: int
    admitted_group_count: int
    cutoff_eigenvalue: float
    effective_rank: float
    effective_rank_ratio: float
    minimum_entry_before_closure: float
    minimum_entry_after_closure: float
    maximum_row_sum_error: float
    closure_scale: float
    objective_before_closure: float
    objective_after_closure: float
    minimum_sample_denominator: float


@dataclass(frozen=True)
class MethodAudit:
    mse_unsampled: float
    maximum_overshoot: float
    negative_weight_mass: float
    projection_activation_rate: float
    maximum_sample_error: float


def determinant_one(metric: Array) -> Array:
    det = float(np.linalg.det(metric))
    if not det > 0.0:
        raise ValueError("metric must be SPD")
    return metric / det ** 0.5


def metric_isotropic(_: float, __: float) -> Array:
    return np.eye(2, dtype=float)


def metric_rotated(_: float, __: float) -> Array:
    # Full SPD form with determinant one and non-axis-aligned eigenvectors.
    return np.array([[2.0, 1.0], [1.0, 1.0]], dtype=float)


def metric_curved(x: float, y: float) -> Array:
    theta = np.pi * (0.15 + 0.55 * x + 0.20 * np.sin(2.0 * np.pi * y))
    ratio = np.exp(1.05 * np.cos(2.0 * np.pi * (x - 0.35 * y)))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    shape = np.diag([ratio, 1.0 / ratio])
    return rotation @ shape @ rotation.T


def grid_coordinates(nx: int, ny: int, spacing: tuple[float, float]) -> Array:
    hx, hy = spacing
    return np.array(
        [(ix * hx, iy * hy) for iy in range(ny) for ix in range(nx)], dtype=float
    )


def selling_stencil(diffusion: Array) -> tuple[Array, Array]:
    """Decompose a 2-D SPD tensor into nonnegative lattice rank-one forms.

    The first Euclidean-radius shell containing a unimodular superbase with a
    nonnegative decomposition is selected; ties are lexicographic.  Thus the
    stencil radius is discovered from the tensor rather than supplied as a
    parameter.  The returned directions ``e_k`` and coefficients ``gamma_k``
    satisfy ``diffusion = sum_k gamma_k e_k e_k^T`` up to roundoff.
    """

    value = np.asarray(diffusion, dtype=float)
    if value.shape != (2, 2) or not np.allclose(value, value.T, atol=1.0e-13):
        raise ValueError("diffusion tensor must be symmetric 2x2")
    if float(np.linalg.eigvalsh(value)[0]) <= 0.0:
        raise ValueError("diffusion tensor must be SPD")

    for radius in range(1, 33):
        vectors = [
            np.array((x, y), dtype=int)
            for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1)
            if (x != 0 or y != 0) and max(abs(x), abs(y)) <= radius
        ]
        candidates: list[tuple[tuple[object, ...], Array, Array]] = []
        for first in vectors:
            for second in vectors:
                if abs(int(first[0] * second[1] - first[1] * second[0])) != 1:
                    continue
                third = -first - second
                if max(abs(int(third[0])), abs(int(third[1]))) > radius:
                    continue
                directions = np.stack((first, second, third))
                design = np.stack(
                    (
                        directions[:, 0] ** 2,
                        directions[:, 0] * directions[:, 1],
                        directions[:, 1] ** 2,
                    ),
                    axis=0,
                ).astype(float)
                coefficients = np.linalg.solve(
                    design,
                    np.array((value[0, 0], value[0, 1], value[1, 1])),
                )
                tolerance = 2.0e-12 * max(1.0, float(np.linalg.norm(value, ord=2)))
                if float(coefficients.min()) < -tolerance:
                    continue
                coefficients = np.maximum(coefficients, 0.0)
                canonical = []
                for direction in directions:
                    item = tuple(int(x) for x in direction)
                    if item[0] < 0 or (item[0] == 0 and item[1] < 0):
                        item = (-item[0], -item[1])
                    canonical.append(item)
                canonical.sort()
                score: tuple[object, ...] = (
                    max(int(direction @ direction) for direction in directions),
                    sum(int(direction @ direction) for direction in directions),
                    tuple(canonical),
                )
                candidates.append((score, directions, coefficients))
        if candidates:
            _, directions, coefficients = min(candidates, key=lambda item: item[0])
            reconstructed = sum(
                (
                    coefficient * np.outer(direction, direction)
                    for coefficient, direction in zip(coefficients, directions)
                ),
                np.zeros((2, 2), dtype=float),
            )
            if not np.allclose(reconstructed, value, rtol=2.0e-12, atol=2.0e-12):
                raise RuntimeError("lattice decomposition failed its tensor identity")
            return directions, coefficients
    raise RuntimeError("no nonnegative lattice decomposition found")


def build_metric_graph(
    nx: int,
    ny: int,
    spacing: tuple[float, float],
    metric_fn: Callable[[float, float], Array],
    split_owner: bool,
) -> tuple[Array, Array, Array]:
    """Build a Neumann graph discretization of the Riemannian Dirichlet form.

    The single dimensionless metric in lattice coordinates is ``G=H^T M H``.
    Its inverse is decomposed into nonnegative lattice directional forms, so
    spacing and anisotropy are combined before the square root/inverse and no
    commutation of their eigenframes is assumed.  Missing physical- or owner-
    boundary edges implement zero flux, preserving constants componentwise.
    """

    coordinates = grid_coordinates(nx, ny, spacing)
    normalized = coordinates / np.array(
        [max((nx - 1) * spacing[0], spacing[0]), max((ny - 1) * spacing[1], spacing[1])]
    )
    metrics = np.stack([determinant_one(metric_fn(x, y)) for x, y in normalized])
    owners = (np.arange(nx)[None, :] >= nx // 2).repeat(ny, axis=0).ravel()

    hmat = np.diag(spacing)

    def node(ix: int, iy: int) -> int:
        return iy * nx + ix

    edge_weights: dict[tuple[int, int], float] = {}
    for iy in range(ny):
        for ix in range(nx):
            i = node(ix, iy)
            pulled_back_metric = hmat.T @ metrics[i] @ hmat
            directions, coefficients = selling_stencil(np.linalg.inv(pulled_back_metric))
            for direction, coefficient in zip(directions, coefficients):
                if coefficient <= 0.0:
                    continue
                dx, dy = (int(direction[0]), int(direction[1]))
                for sign in (-1, 1):
                    jx, jy = ix + sign * dx, iy + sign * dy
                    if not (0 <= jx < nx and 0 <= jy < ny):
                        continue
                    j = node(jx, jy)
                    if split_owner and owners[i] != owners[j]:
                        continue
                    edge = (min(i, j), max(i, j))
                    edge_weights[edge] = edge_weights.get(edge, 0.0) + 0.5 * float(coefficient)

    count = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (i, j), weight in edge_weights.items():
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((weight, weight))
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - adjacency
    return coordinates, owners, laplacian.toarray()


def component_projector(laplacian: Array) -> tuple[Array, Array]:
    adjacency = sparse.csr_matrix(np.where(laplacian < 0.0, -laplacian, 0.0))
    component_count, labels = csgraph.connected_components(adjacency, directed=False)
    projector = np.zeros_like(laplacian)
    for component in range(component_count):
        indices = np.flatnonzero(labels == component)
        projector[np.ix_(indices, indices)] = 1.0 / indices.size
    return projector, labels


def spectral_groups(laplacian: Array, labels: Array) -> list[SpectralGroup]:
    values, vectors = np.linalg.eigh(laplacian)
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 2.0e-9 * scale
    positive = np.flatnonzero(values > tolerance)
    groups: list[SpectralGroup] = []
    cursor = 0
    while cursor < positive.size:
        start = cursor
        base = values[positive[cursor]]
        while cursor + 1 < positive.size and abs(values[positive[cursor + 1]] - base) <= tolerance:
            cursor += 1
        members = positive[start : cursor + 1]
        projector = vectors[:, members] @ vectors[:, members].T
        # Exact block structure is part of the owner-masked operator.  Remove
        # eigensolver roundoff between disconnected components.
        projector[labels[:, None] != labels[None, :]] = 0.0
        groups.append(
            SpectralGroup(
                eigenvalue=float(np.mean(values[members])),
                multiplicity=int(members.size),
                projector=projector,
            )
        )
        cursor += 1
    return groups


def sampling_cutoff(groups: list[SpectralGroup], zero_modes: int, sample_count: int) -> list[SpectralGroup]:
    admitted: list[SpectralGroup] = []
    dimension = zero_modes
    for group in groups:
        if dimension + group.multiplicity > sample_count:
            break
        admitted.append(group)
        dimension += group.multiplicity
    if not admitted:
        raise ValueError("sampling structure admits no nonconstant complete eigenspace")
    return admitted


def close_kernel(projector_zero: Array, variable_kernel: Array) -> tuple[Array, float]:
    negative = variable_kernel < 0.0
    if not np.any(negative):
        return projector_zero + variable_kernel, 1.0
    ratios = projector_zero[negative] / (-variable_kernel[negative])
    scale = min(1.0, float(np.min(ratios)))
    scale = np.nextafter(scale, 0.0)
    for _ in range(128):
        kernel = projector_zero + scale * variable_kernel
        if float(kernel.min()) >= 0.0:
            return kernel, scale
        scale = np.nextafter(scale, 0.0)
    raise RuntimeError("failed to close the nonnegative kernel in floating arithmetic")


def markov_spectral_projection(
    laplacian: Array, sample_count: int
) -> tuple[Array, Array, list[SpectralGroup], KernelAudit]:
    if cp is None:
        raise RuntimeError("markov_spectral_projection requires CVXPY")
    projector_zero, labels = component_projector(laplacian)
    zero_modes = int(np.unique(labels).size)
    groups = sampling_cutoff(spectral_groups(laplacian, labels), zero_modes, sample_count)
    projectors = np.stack([group.projector for group in groups], axis=-1)
    upper = np.triu_indices(laplacian.shape[0])
    design = projectors[upper[0], upper[1], :]
    base = projector_zero[upper]
    multiplicities = np.array([group.multiplicity for group in groups], dtype=float)

    coefficients = cp.Variable(len(groups))
    objective = cp.Minimize(cp.sum(cp.multiply(multiplicities, cp.square(1.0 - coefficients))))
    constraints = [coefficients >= 0.0, coefficients <= 1.0, base + design @ coefficients >= 0.0]
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1.0e-10,
        tol_feas=1.0e-10,
        tol_gap_rel=1.0e-10,
        max_iter=500,
        verbose=False,
    )
    if coefficients.value is None:
        raise RuntimeError(f"Markov projection failed: {problem.status}")
    raw_coefficients = np.clip(np.asarray(coefficients.value, dtype=float), 0.0, 1.0)
    variable_kernel = np.tensordot(projectors, raw_coefficients, axes=([-1], [0]))
    raw_kernel = projector_zero + variable_kernel
    kernel, closure_scale = close_kernel(projector_zero, variable_kernel)
    closed_coefficients = raw_coefficients * closure_scale

    admitted_modes = zero_modes + sum(group.multiplicity for group in groups)
    effective_rank = zero_modes + float(multiplicities @ closed_coefficients)
    objective_before = float(multiplicities @ np.square(1.0 - raw_coefficients))
    objective_after = float(multiplicities @ np.square(1.0 - closed_coefficients))
    audit = KernelAudit(
        solver_status=str(problem.status),
        node_count=laplacian.shape[0],
        sample_count=sample_count,
        admitted_mode_count=admitted_modes,
        admitted_group_count=len(groups) + 1,
        cutoff_eigenvalue=groups[-1].eigenvalue,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank / admitted_modes,
        minimum_entry_before_closure=float(raw_kernel.min()),
        minimum_entry_after_closure=float(kernel.min()),
        maximum_row_sum_error=float(np.max(np.abs(kernel.sum(axis=1) - 1.0))),
        closure_scale=closure_scale,
        objective_before_closure=objective_before,
        objective_after_closure=objective_after,
        minimum_sample_denominator=float("nan"),
    )
    return kernel, closed_coefficients, groups, audit


def sharp_projector(laplacian: Array, admitted_groups: list[SpectralGroup]) -> Array:
    projector_zero, _ = component_projector(laplacian)
    return projector_zero + sum((group.projector for group in admitted_groups), np.zeros_like(laplacian))


def sampling_stable_spectral_reconstruction(
    laplacian: Array, samples: Array, values: Array
) -> tuple[Array, Array, list[dict[str, float | int]], Array]:
    """Reconstruct in the largest complete low-frequency sampling subspace.

    Each owner-connected component uses its Neumann graph spectrum.  Complete
    eigenspaces are admitted in increasing eigenvalue order while their sample
    restriction remains injective and their total dimension does not exceed
    the number of observations in that component.  Coefficients are the unique
    Euclidean least-squares solution in that subspace.  Observed values are
    retained separately after evaluating the spectral field.
    """

    adjacency = sparse.csr_matrix(np.where(laplacian < 0.0, -laplacian, 0.0))
    component_count, labels = csgraph.connected_components(adjacency, directed=False)
    estimate = np.zeros(laplacian.shape[0], dtype=float)
    weights = np.zeros((laplacian.shape[0], samples.size), dtype=float)
    projector = np.zeros_like(laplacian)
    audits: list[dict[str, float | int]] = []
    for component in range(component_count):
        nodes = np.flatnonzero(labels == component)
        sample_columns = np.flatnonzero(labels[samples] == component)
        if sample_columns.size == 0:
            raise ValueError("every owner-connected component must contain an observation")
        local_samples_global = samples[sample_columns]
        node_lookup = {int(node): index for index, node in enumerate(nodes)}
        local_samples = np.array(
            [node_lookup[int(node)] for node in local_samples_global], dtype=int
        )
        local_laplacian = laplacian[np.ix_(nodes, nodes)]
        eigenvalues, eigenvectors = np.linalg.eigh(local_laplacian)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        eigen_tolerance = 2.0e-9 * scale

        admitted = 0
        cursor = 0
        while cursor < eigenvalues.size:
            end = cursor + 1
            while end < eigenvalues.size and abs(eigenvalues[end] - eigenvalues[cursor]) <= eigen_tolerance:
                end += 1
            if end > sample_columns.size:
                break
            candidate_basis = eigenvectors[:, :end]
            candidate = candidate_basis[local_samples]
            singular_values = np.linalg.svd(candidate, compute_uv=False)
            residual = np.linalg.norm(
                local_laplacian @ candidate_basis
                - candidate_basis * eigenvalues[:end][None, :],
                ord=2,
            )
            orthogonality_error = np.linalg.norm(
                candidate_basis.T @ candidate_basis - np.eye(end), ord=2
            )
            spectral_gap = float(eigenvalues[end] - eigenvalues[end - 1])
            if spectral_gap <= 2.0 * residual:
                break
            subspace_error_bound = 2.0 * residual / spectral_gap + orthogonality_error
            certified_sampling_lower_bound = float(singular_values[-1]) - subspace_error_bound
            if certified_sampling_lower_bound <= 0.0:
                break
            admitted = end
            cursor = end
        if admitted == 0:
            raise RuntimeError("constant mode was not sampling-injective")

        basis = eigenvectors[:, :admitted]
        projector[np.ix_(nodes, nodes)] = basis @ basis.T
        sample_operator = basis[local_samples]
        singular_values = np.linalg.svd(sample_operator, compute_uv=False)
        residual = np.linalg.norm(
            local_laplacian @ basis - basis * eigenvalues[:admitted][None, :],
            ord=2,
        )
        orthogonality_error = np.linalg.norm(
            basis.T @ basis - np.eye(admitted), ord=2
        )
        spectral_gap = float(eigenvalues[admitted] - eigenvalues[admitted - 1])
        subspace_error_bound = 2.0 * residual / spectral_gap + orthogonality_error
        certified_sampling_lower_bound = float(singular_values[-1]) - subspace_error_bound
        dual = np.linalg.pinv(sample_operator)
        local_weights = basis @ dual
        estimate[nodes] = local_weights @ values[sample_columns]
        weights[np.ix_(nodes, sample_columns)] = local_weights
        audits.append(
            {
                "component": component,
                "node_count": int(nodes.size),
                "sample_count": int(sample_columns.size),
                "admitted_mode_count": admitted,
                "cutoff_eigenvalue": float(eigenvalues[admitted - 1]),
                "minimum_sampling_singular_value": float(singular_values[-1]),
                "certified_sampling_singular_lower_bound": certified_sampling_lower_bound,
                "sampling_condition_number": float(singular_values[0] / singular_values[-1]),
                "eigenspace_residual_norm": float(residual),
                "eigenspace_gap": spectral_gap,
                "least_squares_residual": float(
                    np.linalg.norm(sample_operator @ dual @ values[sample_columns] - values[sample_columns])
                ),
            }
        )
    estimate[samples] = values
    return estimate, weights, audits, projector


def nearest_sources(laplacian: Array, samples: Array) -> Array:
    graph_data = np.zeros_like(laplacian)
    edges = laplacian < 0.0
    graph_data[edges] = np.sqrt(1.0 / (-laplacian[edges]))
    graph = sparse.csr_matrix(graph_data)
    distances = csgraph.dijkstra(graph, directed=False, indices=samples)
    return samples[np.argmin(distances, axis=0)]


def heat_kernel_matched(laplacian: Array, target_trace: float) -> tuple[Array, float]:
    eigenvalues = np.linalg.eigvalsh(laplacian)
    zero_count = int(np.count_nonzero(eigenvalues <= 2.0e-9 * max(1.0, eigenvalues[-1])))
    if not zero_count < target_trace <= laplacian.shape[0]:
        raise ValueError("target trace must lie between the nullity and node count")
    lower, upper = 0.0, 1.0
    while float(np.exp(-upper * eigenvalues).sum()) > target_trace:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if float(np.exp(-midpoint * eigenvalues).sum()) > target_trace:
            lower = midpoint
        else:
            upper = midpoint
    time = 0.5 * (lower + upper)
    kernel = linalg.expm(-time * laplacian)
    return kernel, time


def lazy_walk_kernel_matched(laplacian: Array, target_trace: float) -> tuple[Array, int, float]:
    degree = np.diag(laplacian)
    tau = 1.0 / float(degree.max())
    step = np.eye(laplacian.shape[0]) - tau * laplacian
    kernel = np.eye(laplacian.shape[0])
    best_kernel = kernel.copy()
    best_power = 0
    best_error = abs(float(np.trace(kernel)) - target_trace)
    for power in range(1, 513):
        kernel = kernel @ step
        error = abs(float(np.trace(kernel)) - target_trace)
        if error < best_error:
            best_kernel = kernel.copy()
            best_power = power
            best_error = error
        if float(np.trace(kernel)) < target_trace and power > best_power + 8:
            break
    return best_kernel, best_power, tau


def canonical_coverage_polynomial(
    laplacian: Array, samples: Array
) -> tuple[Array, int, float, float]:
    """Return the least-degree sampling-covering Markov spectral polynomial.

    B = I - L/lambda_max is simultaneously symmetric, entrywise nonnegative,
    mass preserving, and positive semidefinite.  The degree is the first power
    for which every row receives positive mass from the observed sites.
    """

    lambda_max = float(np.linalg.eigvalsh(laplacian)[-1])
    step = np.eye(laplacian.shape[0]) - laplacian / lambda_max
    kernel = np.eye(laplacian.shape[0])
    for power in range(laplacian.shape[0] + 1):
        sampled_mass = kernel[:, samples].sum(axis=1)
        if float(sampled_mass.min()) > 0.0:
            return kernel, power, 1.0 / lambda_max, float(sampled_mass.min())
        kernel = kernel @ step
        # Matrix products of nonnegative operands are nonnegative in exact
        # arithmetic.  Remove only negative roundoff before the next product.
        kernel[kernel < 0.0] = 0.0
    raise RuntimeError("no sampling-covering polynomial found")


def sampling_trace_polynomial(
    laplacian: Array, samples: Array
) -> tuple[Array, int, float, float, float]:
    """Match the Markov polynomial's effective dimension to observations.

    Consecutive powers of B bracket the sample count in trace.  Their unique
    convex combination has trace exactly equal to the number of observations,
    remains a nonnegative mass-one PSD polynomial, and introduces no fitted
    bandwidth or signal-dependent choice.
    """

    eigenvalues = np.linalg.eigvalsh(laplacian)
    lambda_max = float(eigenvalues[-1])
    step = np.eye(laplacian.shape[0]) - laplacian / lambda_max
    powers = [np.eye(laplacian.shape[0])]
    previous = powers[0]
    previous_trace = float(np.trace(previous))
    bracket: tuple[int, float, Array] | None = None
    coverage_degree: int | None = None
    target = float(samples.size)
    for degree in range(1, laplacian.shape[0] + 1):
        current = previous @ step
        current[current < 0.0] = 0.0
        powers.append(current)
        current_trace = float(np.trace(current))
        if bracket is None and current_trace <= target <= previous_trace:
            denominator = previous_trace - current_trace
            alpha = 1.0 if denominator == 0.0 else (previous_trace - target) / denominator
            kernel = (1.0 - alpha) * previous + alpha * current
            bracket = (degree, alpha, kernel)
        if coverage_degree is None and float(current[:, samples].sum(axis=1).min()) > 0.0:
            coverage_degree = degree
        if bracket is not None and coverage_degree is not None:
            trace_degree, alpha, trace_kernel = bracket
            if trace_degree >= coverage_degree:
                sampled_mass = trace_kernel[:, samples].sum(axis=1)
                return trace_kernel, trace_degree, alpha, 1.0 / lambda_max, float(sampled_mass.min())
            coverage_kernel = powers[coverage_degree]
            sampled_mass = coverage_kernel[:, samples].sum(axis=1)
            return coverage_kernel, coverage_degree, 1.0, 1.0 / lambda_max, float(sampled_mass.min())
        previous = current
        previous_trace = current_trace
    raise RuntimeError("sample-count trace was not reached by the Markov polynomial")


def hs_optimal_positive_polynomial(
    laplacian: Array, samples: Array
) -> tuple[Array, Array, int, float, float]:
    """Best finite-propagation positive polynomial for spectral admissibility.

    The admissible family is the convex hull of ``I, B, ..., B^m`` where
    ``B = I - L/lambda_max`` and ``m`` is the first degree that both reaches
    every query from the observations and can attain trace ``|S|``.  Within
    that family, the coefficients uniquely minimize Hilbert--Schmidt distance
    to the basis-invariant rank-|S| low-frequency target.  A repeated cutoff
    eigenspace receives one common fractional coefficient, so the definition
    never depends on an arbitrary eigenbasis.
    """

    if cp is None:
        raise RuntimeError("hs_optimal_positive_polynomial requires CVXPY")
    eigenvalues = np.linalg.eigvalsh(laplacian)
    lambda_max = float(eigenvalues[-1])
    step_eigenvalues = np.clip(1.0 - eigenvalues / lambda_max, 0.0, 1.0)
    step = np.eye(laplacian.shape[0]) - laplacian / lambda_max
    target_trace = float(samples.size)

    powers = [np.eye(laplacian.shape[0])]
    trace_degree: int | None = None
    coverage_degree: int | None = None
    for degree in range(1, laplacian.shape[0] + 1):
        current = powers[-1] @ step
        current[current < 0.0] = 0.0
        powers.append(current)
        if trace_degree is None and float(np.trace(current)) <= target_trace:
            trace_degree = degree
        if coverage_degree is None and float(current[:, samples].sum(axis=1).min()) > 0.0:
            coverage_degree = degree
        if trace_degree is not None and coverage_degree is not None:
            break
    if trace_degree is None or coverage_degree is None:
        raise RuntimeError("positive polynomial degree was not determined")
    maximum_degree = max(trace_degree, coverage_degree)

    tolerance = 2.0e-9 * max(1.0, float(eigenvalues[-1]))
    cutoff = eigenvalues[samples.size - 1]
    below = eigenvalues < cutoff - tolerance
    tied = np.abs(eigenvalues - cutoff) <= tolerance
    target = np.zeros_like(eigenvalues)
    target[below] = 1.0
    target[tied] = (samples.size - int(np.count_nonzero(below))) / int(np.count_nonzero(tied))

    exponents = np.arange(maximum_degree + 1, dtype=float)
    spectral_design = step_eigenvalues[:, None] ** exponents[None, :]
    trace_design = spectral_design.sum(axis=0)
    coefficients = cp.Variable(maximum_degree + 1)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(spectral_design @ coefficients - target)),
        [coefficients >= 0.0, cp.sum(coefficients) == 1.0, trace_design @ coefficients == target_trace],
    )
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1.0e-11,
        tol_feas=1.0e-11,
        tol_gap_rel=1.0e-11,
        max_iter=500,
        verbose=False,
    )
    if coefficients.value is None:
        raise RuntimeError(f"positive polynomial optimization failed: {problem.status}")
    values = np.asarray(coefficients.value, dtype=float)
    values[np.abs(values) < 5.0e-12] = 0.0
    if float(values.min()) < -5.0e-10:
        raise RuntimeError("positive polynomial optimizer violated coefficient nonnegativity")
    values = np.maximum(values, 0.0)
    values /= values.sum()
    kernel = sum(
        (coefficient * power for coefficient, power in zip(values, powers)),
        np.zeros_like(laplacian),
    )
    minimum_mass = float(kernel[:, samples].sum(axis=1).min())
    return kernel, values, maximum_degree, float(problem.value), minimum_mass


def interpolate_kernel(
    kernel: Array,
    samples: Array,
    values: Array,
    nearest: Array,
    enforce_observations: bool = True,
) -> tuple[Array, Array, Array]:
    sampled_kernel = kernel[:, samples]
    denominator = sampled_kernel.sum(axis=1)
    estimate = np.empty(kernel.shape[0], dtype=float)
    weights = np.zeros_like(sampled_kernel)
    valid = np.abs(denominator) > 32.0 * np.finfo(float).eps
    weights[valid] = sampled_kernel[valid] / denominator[valid, None]
    estimate[valid] = weights[valid] @ values
    sample_lookup = {int(site): float(value) for site, value in zip(samples, values)}
    for node in np.flatnonzero(~valid):
        estimate[node] = sample_lookup[int(nearest[node])]
    if enforce_observations:
        estimate[samples] = values
    return estimate, weights, denominator


def clamp_by_owner(
    estimate: Array, samples: Array, values: Array, owners: Array
) -> tuple[Array, Array]:
    projected = estimate.copy()
    activated = np.zeros(estimate.size, dtype=bool)
    for owner in np.unique(owners):
        source_values = values[owners[samples] == owner]
        lower, upper = float(source_values.min()), float(source_values.max())
        nodes = owners == owner
        clipped = np.clip(projected[nodes], lower, upper)
        activated[nodes] = clipped != projected[nodes]
        projected[nodes] = clipped
    return projected, activated


def method_audit(
    estimate: Array,
    weights: Array,
    truth: Array,
    samples: Array,
    values: Array,
    owners: Array,
    projection_activated: Array | None = None,
    evaluation_indices: Array | None = None,
) -> MethodAudit:
    evaluated = np.ones(truth.size, dtype=bool)
    evaluated[samples] = False
    if evaluation_indices is not None:
        evaluated[:] = False
        evaluated[evaluation_indices] = True
    overshoot = 0.0
    for owner in np.unique(owners):
        source_values = values[owners[samples] == owner]
        lower, upper = float(source_values.min()), float(source_values.max())
        nodes = (owners == owner) & evaluated
        if not np.any(nodes):
            continue
        overshoot = max(
            overshoot,
            float(np.max(np.maximum(lower - estimate[nodes], estimate[nodes] - upper))),
        )
    negative_mass = float(np.max(np.sum(np.maximum(-weights[evaluated], 0.0), axis=1)))
    activation_rate = (
        0.0
        if projection_activated is None
        else float(np.mean(projection_activated[evaluated]))
    )
    return MethodAudit(
        mse_unsampled=float(np.mean(np.square(estimate[evaluated] - truth[evaluated]))),
        maximum_overshoot=max(0.0, overshoot),
        negative_weight_mass=negative_mass,
        projection_activation_rate=activation_rate,
        maximum_sample_error=float(np.max(np.abs(estimate[samples] - values))),
    )


def sample_sites(nx: int, ny: int, stride: int, sampling_kind: str) -> Array:
    if sampling_kind == "regular":
        xs = sorted(set(range(0, nx, stride)) | {nx - 1})
        ys = sorted(set(range(0, ny, stride)) | {ny - 1})
        return np.array([iy * nx + ix for iy in ys for ix in xs], dtype=int)
    if sampling_kind == "stratified":
        rng = np.random.default_rng(20260826)
        sites: list[int] = []
        for y0 in range(0, ny, stride):
            for x0 in range(0, nx, stride):
                width = min(stride, nx - x0)
                height = min(stride, ny - y0)
                ix = x0 + int(rng.integers(width))
                iy = y0 + int(rng.integers(height))
                sites.append(iy * nx + ix)
        return np.array(sorted(sites), dtype=int)
    raise ValueError(f"unknown sampling kind: {sampling_kind}")


def common_refinement_sites(coarse_size: int) -> tuple[int, Array, Array]:
    """Embed factor-two source and query centers in their exact common lattice."""

    side = 4 * coarse_size + 1
    source_axis = [4 * index + 2 for index in range(coarse_size)]
    query_axis = [2 * index + 1 for index in range(2 * coarse_size)]
    sources = np.array([iy * side + ix for iy in source_axis for ix in source_axis], dtype=int)
    queries = np.array([iy * side + ix for iy in query_axis for ix in query_axis], dtype=int)
    if np.intersect1d(sources, queries).size:
        raise AssertionError("factor-two center grids must be disjoint")
    return side, sources, queries


def rational_smooth_signal(coordinates: Array) -> Array:
    scale = coordinates.max(axis=0)
    x = coordinates[:, 0] / scale[0]
    y = coordinates[:, 1] / scale[1]
    return (2.0 + 2.0 * x + y + x * y + x * x + 0.5 * y * y) / 7.5


def rational_interface_signal(coordinates: Array, owners: Array) -> Array:
    scale = coordinates.max(axis=0)
    x = coordinates[:, 0] / scale[0]
    y = coordinates[:, 1] / scale[1]
    left = (1.0 + x + y + x * y) / 8.0
    right = (5.0 + x - 0.5 * y + 0.5 * x * y) / 8.0
    return np.where(owners == 0, left, right)


def trigonometric_signal(coordinates: Array) -> Array:
    scale = coordinates.max(axis=0)
    x = coordinates[:, 0] / scale[0]
    y = coordinates[:, 1] / scale[1]
    return (
        0.5
        + 0.18 * np.sin(2.0 * np.pi * x)
        + 0.13 * np.cos(2.0 * np.pi * y)
        + 0.09 * np.sin(2.0 * np.pi * (x + y))
    )


def unresolved_extremum_signal(coordinates: Array) -> Array:
    scale = coordinates.max(axis=0)
    x = coordinates[:, 0] / scale[0]
    y = coordinates[:, 1] / scale[1]
    exponent = -((x - 0.57) ** 2 / 0.0225 + (y - 0.43) ** 2 / 0.0529)
    return 0.12 + 0.76 * np.exp(exponent)


def run_case(
    name: str,
    metric_fn: Callable[[float, float], Array],
    split_owner: bool,
    signal_kind: str,
    sampling_kind: str,
    evaluation_mode: str,
    nx: int,
    ny: int,
    stride: int,
    include_fields: bool = False,
) -> dict[str, object]:
    if evaluation_mode == "sparse":
        graph_nx, graph_ny = nx, ny
        evaluation_indices = None
    elif evaluation_mode == "resample":
        if nx != ny:
            raise ValueError("common-refinement resampling probe currently uses square grids")
        graph_nx, samples, evaluation_indices = common_refinement_sites(nx)
        graph_ny = graph_nx
    else:
        raise ValueError(f"unknown evaluation mode: {evaluation_mode}")

    coordinates, owners, laplacian = build_metric_graph(
        nx=graph_nx,
        ny=graph_ny,
        spacing=(1.0, 1.7),
        metric_fn=metric_fn,
        split_owner=split_owner,
    )
    if not split_owner:
        owners = np.zeros_like(owners)
    if evaluation_mode == "sparse":
        samples = sample_sites(nx, ny, stride, sampling_kind)
    if signal_kind == "interface":
        truth = rational_interface_signal(coordinates, owners)
    elif signal_kind == "rational":
        truth = rational_smooth_signal(coordinates)
    elif signal_kind == "trigonometric":
        truth = trigonometric_signal(coordinates)
    elif signal_kind == "unresolved_extremum":
        truth = unresolved_extremum_signal(coordinates)
    else:
        raise ValueError(f"unknown signal kind: {signal_kind}")
    values = truth[samples]
    nearest = nearest_sources(laplacian, samples)

    markov_kernel, coefficients, groups, kernel_audit = markov_spectral_projection(
        laplacian, samples.size
    )
    markov_estimate, markov_weights, denominator = interpolate_kernel(
        markov_kernel, samples, values, nearest
    )
    kernel_audit = KernelAudit(
        **{
            **asdict(kernel_audit),
            "minimum_sample_denominator": float(np.min(denominator)),
        }
    )

    sharp_kernel = sharp_projector(laplacian, groups)
    sharp_estimate, sharp_weights, _ = interpolate_kernel(sharp_kernel, samples, values, nearest)
    sharp_projected, activated = clamp_by_owner(sharp_estimate, samples, values, owners)
    sharp_projected[samples] = values

    spectral_estimate, spectral_weights, spectral_audits, stable_projector = (
        sampling_stable_spectral_reconstruction(laplacian, samples, values)
    )
    spectral_projected, spectral_activated = clamp_by_owner(
        spectral_estimate, samples, values, owners
    )
    spectral_projected[samples] = values
    stable_kernel_estimate, stable_kernel_weights, _ = interpolate_kernel(
        stable_projector, samples, values, nearest
    )
    stable_kernel_projected, stable_kernel_activated = clamp_by_owner(
        stable_kernel_estimate, samples, values, owners
    )
    stable_kernel_projected[samples] = values

    nearest_estimate = truth[nearest]
    nearest_estimate[samples] = values
    empty_weights = np.zeros((truth.size, samples.size), dtype=float)

    heat_kernel, heat_time = heat_kernel_matched(laplacian, kernel_audit.effective_rank)
    heat_estimate, heat_weights, _ = interpolate_kernel(heat_kernel, samples, values, nearest)

    walk_kernel, walk_power, walk_tau = lazy_walk_kernel_matched(
        laplacian, kernel_audit.effective_rank
    )
    walk_estimate, walk_weights, _ = interpolate_kernel(walk_kernel, samples, values, nearest)

    coverage_kernel, coverage_power, coverage_tau, coverage_minimum_mass = (
        canonical_coverage_polynomial(laplacian, samples)
    )
    coverage_estimate, coverage_weights, _ = interpolate_kernel(
        coverage_kernel, samples, values, nearest
    )

    trace_kernel, trace_degree, trace_alpha, trace_tau, trace_minimum_mass = (
        sampling_trace_polynomial(laplacian, samples)
    )
    trace_estimate, trace_weights, _ = interpolate_kernel(
        trace_kernel, samples, values, nearest
    )

    optimal_kernel, optimal_coefficients, optimal_degree, optimal_objective, optimal_minimum_mass = (
        hs_optimal_positive_polynomial(laplacian, samples)
    )
    optimal_estimate, optimal_weights, _ = interpolate_kernel(
        optimal_kernel, samples, values, nearest
    )

    payload: dict[str, object] = {
        "case": name,
        "signal_kind": signal_kind,
        "sampling_kind": sampling_kind,
        "evaluation_mode": evaluation_mode,
        "kernel": asdict(kernel_audit),
        "coefficients": coefficients.tolist(),
        "sampling_stable_spectral_components": spectral_audits,
        "matched_positive_baselines": {
            "heat_time": heat_time,
            "lazy_walk_power": walk_power,
            "lazy_walk_tau": walk_tau,
            "heat_trace": float(np.trace(heat_kernel)),
            "lazy_walk_trace": float(np.trace(walk_kernel)),
            "coverage_power": coverage_power,
            "coverage_tau": coverage_tau,
            "coverage_trace": float(np.trace(coverage_kernel)),
            "coverage_minimum_sample_mass": coverage_minimum_mass,
            "trace_degree": trace_degree,
            "trace_alpha": trace_alpha,
            "trace_tau": trace_tau,
            "trace_effective_rank": float(np.trace(trace_kernel)),
            "trace_minimum_sample_mass": trace_minimum_mass,
            "optimal_positive_degree": optimal_degree,
            "optimal_positive_coefficients": optimal_coefficients.tolist(),
            "optimal_positive_hs_objective": optimal_objective,
            "optimal_positive_trace": float(np.trace(optimal_kernel)),
            "optimal_positive_minimum_sample_mass": optimal_minimum_mass,
        },
        "methods": {
            "markov": asdict(
                method_audit(
                    markov_estimate,
                    markov_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "sharp": asdict(
                method_audit(
                    sharp_estimate,
                    sharp_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "sharp_range_projected": asdict(
                method_audit(
                    sharp_projected,
                    sharp_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    activated,
                    evaluation_indices,
                )
            ),
            "sampling_stable_spectral": asdict(
                method_audit(
                    spectral_estimate,
                    spectral_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "sampling_stable_spectral_projected": asdict(
                method_audit(
                    spectral_projected,
                    spectral_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    spectral_activated,
                    evaluation_indices,
                )
            ),
            "sampling_stable_projector_kernel": asdict(
                method_audit(
                    stable_kernel_estimate,
                    stable_kernel_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "sampling_stable_projector_kernel_projected": asdict(
                method_audit(
                    stable_kernel_projected,
                    stable_kernel_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    stable_kernel_activated,
                    evaluation_indices,
                )
            ),
            "nearest": asdict(
                method_audit(
                    nearest_estimate,
                    empty_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "heat_matched_rank": asdict(
                method_audit(
                    heat_estimate,
                    heat_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "lazy_walk_matched_rank": asdict(
                method_audit(
                    walk_estimate,
                    walk_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "canonical_coverage_polynomial": asdict(
                method_audit(
                    coverage_estimate,
                    coverage_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "sampling_trace_polynomial": asdict(
                method_audit(
                    trace_estimate,
                    trace_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
            "hs_optimal_positive_polynomial": asdict(
                method_audit(
                    optimal_estimate,
                    optimal_weights,
                    truth,
                    samples,
                    values,
                    owners,
                    evaluation_indices=evaluation_indices,
                )
            ),
        },
    }
    if include_fields:
        if evaluation_indices is None:
            raise ValueError("field export is defined for disjoint resampling mode")
        query_side = 2 * nx
        payload["query_fields"] = {
            "shape": [query_side, query_side],
            "truth": truth[evaluation_indices].tolist(),
            "nearest": nearest_estimate[evaluation_indices].tolist(),
            "positive_markov": trace_estimate[evaluation_indices].tolist(),
            "stable_projector_raw": stable_kernel_estimate[evaluation_indices].tolist(),
            "stable_projector_projected": stable_kernel_projected[evaluation_indices].tolist(),
            "count_cutoff_projected": sharp_projected[evaluation_indices].tolist(),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--sampling", choices=("regular", "stratified"), default="regular")
    parser.add_argument("--mode", choices=("sparse", "resample"), default="sparse")
    parser.add_argument("--include-fields", action="store_true")
    arguments = parser.parse_args()
    cases = [
        run_case("isotropic_rational", metric_isotropic, False, "rational", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("rotated_spd_rational", metric_rotated, False, "rational", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("curved_spd_rational", metric_curved, False, "rational", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("isotropic_trigonometric", metric_isotropic, False, "trigonometric", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("rotated_spd_trigonometric", metric_rotated, False, "trigonometric", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("curved_spd_trigonometric", metric_curved, False, "trigonometric", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("curved_spd_unresolved_extremum", metric_curved, False, "unresolved_extremum", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
        run_case("curved_spd_owner_interface", metric_curved, True, "interface", arguments.sampling, arguments.mode, arguments.size, arguments.size, arguments.stride, arguments.include_fields),
    ]
    payload = {"cases": cases}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
