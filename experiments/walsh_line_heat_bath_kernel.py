#!/usr/bin/env python3
"""Finite audit of a syndrome-zero random-line heat-bath kernel.

For an affine parity coset A and a nonzero direction v in its difference
lattice, the heat-bath update from x resamples the whole discrete line

    x + Z v

with its conditional discrete-Gaussian law.  This is an exact one-dimensional
shifted DGS, so it is efficiently samplable at every width.  Averaging these
line projections over a short, nearly isotropic family of kernel directions
is a candidate replacement for the unavailable n-dimensional OU proposal.

The finite coefficient box audit measures the exact reversible spectral gap,
the direction-frame lower bound, and boundary mass.  It also records the
asymptotic dense-self-dual obstruction: fixed-rank blocks can perform well in
small dimensions while having exponentially small overlap in the worst case.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import (
    _gf2_inverse,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)
from experiments.walsh_radial_matched_filter import optimal_gaussian_target_width


T0 = 0.23147
TARGET_WIDTH = optimal_gaussian_target_width(2.0 * T0)
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_line_heat_bath_kernel.json"


def constant_rank_dense_lattice_obstruction(
    r: float,
    normalized_minimum_squared: float = 1.0 / (2.0 * math.pi * math.e),
) -> dict[str, float]:
    """Asymptotic fixed-rank overlap bound for a dense self-dual lattice.

    If lambda_1(L)^2/n -> c^2 and s=xi_r(lambda_1(L)), then every nonzero
    vector in a fixed-rank block costs at least

        exp(-pi c^2 n/(4 s^2)) = 2^(-delta n)

    in the Hellinger-overlap estimate for an exact block heat bath.  The
    Conway--Thompson asymptotic is c^2=1/(2*pi*e).
    """
    if r <= 0.0 or normalized_minimum_squared <= 0.0:
        raise ValueError("r and the normalized squared minimum must be positive")
    s_squared = (
        4.0 * r * math.log(2.0)
        / (math.pi * normalized_minimum_squared)
    )
    exponent_bits = (
        math.pi * normalized_minimum_squared
        / (4.0 * s_squared * math.log(2.0))
    )
    return {
        "normalized_minimum_squared": normalized_minimum_squared,
        "xi_r_squared": s_squared,
        "fixed_rank_overlap_exponent_bits": exponent_bits,
    }


def coefficient_box(n: int, cutoff: int) -> np.ndarray:
    values = range(-cutoff, cutoff + 1)
    return np.asarray(list(itertools.product(values, repeat=n)), dtype=np.int16)


def transformed_bits(coefficients: np.ndarray, inverse_transform: np.ndarray) -> np.ndarray:
    return ((coefficients & 1).astype(np.uint8) @ inverse_transform) & 1


def canonical_sign(vector: np.ndarray) -> tuple[int, ...]:
    vector = np.asarray(vector, dtype=np.int16)
    first = int(np.flatnonzero(vector)[0])
    if vector[first] < 0:
        vector = -vector
    return tuple(int(x) for x in vector)


def kernel_direction_candidates(
    n: int,
    h: int,
    inverse_transform: np.ndarray,
    inverse_basis: np.ndarray,
    *,
    search_cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates: dict[tuple[int, ...], float] = {}
    for row in itertools.product(range(-search_cutoff, search_cutoff + 1), repeat=n):
        z = np.asarray(row, dtype=np.int16)
        if not np.any(z):
            continue
        if np.any(transformed_bits(z[None, :], inverse_transform)[0, :h]):
            continue
        key = canonical_sign(z)
        point = np.asarray(key, dtype=np.float64) @ inverse_basis
        candidates[key] = float(point @ point)
    ordered = sorted(candidates, key=lambda key: (candidates[key], key))
    return (
        np.asarray(ordered, dtype=np.int16),
        np.asarray([candidates[key] for key in ordered], dtype=np.float64),
    )


def isotropic_kernel_directions(
    n: int,
    h: int,
    inverse_transform: np.ndarray,
    inverse_basis: np.ndarray,
    *,
    search_cutoff: int,
    count: int,
    s: float | None = None,
) -> np.ndarray:
    """Greedily cover weak directions, weighted by 1D Gaussian mobility."""
    candidates, norm2 = kernel_direction_candidates(
        n,
        h,
        inverse_transform,
        inverse_basis,
        search_cutoff=search_cutoff,
    )
    if len(candidates) < count:
        raise ValueError("direction search box is too small")
    physical = candidates @ inverse_basis
    unit = physical / np.sqrt(norm2)[:, None]
    if s is None:
        mobility = np.ones(len(candidates), dtype=np.float64)
    else:
        mobility = np.asarray(
            [centered_line_mobility(s / math.sqrt(value)) for value in norm2]
        )
    frame = 1e-6 * np.eye(n)
    available = np.ones(len(candidates), dtype=bool)
    chosen: list[int] = []
    for _ in range(count):
        inverse_frame = np.linalg.inv(frame)
        leverage = np.einsum("ij,jk,ik->i", unit, inverse_frame, unit)
        score = mobility * leverage
        score[~available] = -math.inf
        index = int(np.argmax(score))
        chosen.append(index)
        available[index] = False
        frame += mobility[index] * np.outer(unit[index], unit[index])
    return candidates[chosen]


def phase_aware_kernel_directions(
    n: int,
    h: int,
    inverse_transform: np.ndarray,
    inverse_basis: np.ndarray,
    *,
    search_cutoff: int,
    count: int,
    s: float,
) -> np.ndarray:
    """Greedily cover physical leverage and all tail Walsh characters."""
    candidates, norm2 = kernel_direction_candidates(
        n,
        h,
        inverse_transform,
        inverse_basis,
        search_cutoff=search_cutoff,
    )
    if len(candidates) < count:
        raise ValueError("direction search box is too small")
    physical = candidates @ inverse_basis
    unit = physical / np.sqrt(norm2)[:, None]
    mobility = np.asarray(
        [centered_line_mobility(s / math.sqrt(value)) for value in norm2]
    )
    tail = transformed_bits(candidates, inverse_transform)[:, h:]
    character_count = (1 << tail.shape[1]) - 1
    toggles = np.empty((len(candidates), character_count), dtype=np.float64)
    for theta in range(1, character_count + 1):
        theta_bits = (
            (theta >> np.arange(tail.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        toggles[:, theta - 1] = (tail @ theta_bits) & 1

    frame = 1e-6 * np.eye(n)
    character_coverage = np.full(character_count, 1e-9, dtype=np.float64)
    available = np.ones(len(candidates), dtype=bool)
    chosen: list[int] = []
    for _ in range(count):
        inverse_frame = np.linalg.inv(frame)
        physical_gain = mobility * np.einsum(
            "ij,jk,ik->i", unit, inverse_frame, unit)
        character_gain = mobility * (
            toggles @ (1.0 / character_coverage)
        )
        physical_gain /= max(float(np.max(physical_gain[available])), 1e-300)
        character_gain /= max(float(np.max(character_gain[available])), 1e-300)
        score = physical_gain + character_gain
        score[~available] = -math.inf
        index = int(np.argmax(score))
        chosen.append(index)
        available[index] = False
        frame += mobility[index] * np.outer(unit[index], unit[index])
        character_coverage += mobility[index] * toggles[index]
    return candidates[chosen]


def centered_line_mobility(tau: float) -> float:
    """Variance fraction of a centered D_{Z,tau} versus a continuous line."""
    if tau <= 0.0:
        return 0.0
    cutoff = max(4, int(math.ceil(8.0 * tau)))
    k = np.arange(-cutoff, cutoff + 1, dtype=np.float64)
    mass = np.exp(-math.pi * k * k / (tau * tau))
    variance = float(np.sum(k * k * mass) / np.sum(mass))
    return 2.0 * math.pi * variance / (tau * tau)


def generator_directions(transform: np.ndarray, h: int) -> np.ndarray:
    """Return a redundant generating family for the integer parity kernel."""
    n = transform.shape[0]
    rows = [2 * np.eye(n, dtype=np.int16)[i] for i in range(n)]
    rows.extend(np.asarray(transform[i], dtype=np.int16) for i in range(h, n))
    unique = {canonical_sign(row) for row in rows if np.any(row)}
    return np.asarray(sorted(unique), dtype=np.int16)


def line_heat_bath_matrix(
    coefficients: np.ndarray,
    target_mass: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    lookup = {tuple(int(x) for x in row): i for i, row in enumerate(coefficients)}
    transition = np.zeros((len(coefficients), len(coefficients)), dtype=np.float64)
    for direction in directions:
        unseen = set(range(len(coefficients)))
        while unseen:
            origin_index = next(iter(unseen))
            origin = coefficients[origin_index]
            line = []
            # The coefficient box is finite, so walking until the first miss in
            # each direction finds the entire intersection with this line.
            k = 0
            while True:
                index = lookup.get(tuple(int(x) for x in origin + k * direction))
                if index is None:
                    break
                line.append(index)
                k += 1
            k = -1
            while True:
                index = lookup.get(tuple(int(x) for x in origin + k * direction))
                if index is None:
                    break
                line.append(index)
                k -= 1
            line = sorted(set(line))
            unseen.difference_update(line)
            conditional = target_mass[line] / np.sum(target_mass[line])
            transition[np.ix_(line, line)] += conditional[None, :] / len(directions)
    return transition


def line_heat_bath_matrices(
    coefficients: np.ndarray,
    target_mass: np.ndarray,
    directions: np.ndarray,
) -> list[np.ndarray]:
    """Return the individual conditional-expectation projections ``H_v``."""
    return [
        line_heat_bath_matrix(
            coefficients,
            target_mass,
            np.asarray([direction], dtype=directions.dtype),
        )
        for direction in directions
    ]


def block_heat_bath_matrix(
    coefficients: np.ndarray,
    target_mass: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Heat-bath projection on cosets of the span of a direction block."""
    lookup = {tuple(int(x) for x in row): i for i, row in enumerate(coefficients)}
    parent = np.arange(len(coefficients), dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for index, coefficient in enumerate(coefficients):
        for direction in directions:
            neighbor = lookup.get(tuple(int(x) for x in coefficient + direction))
            if neighbor is None:
                continue
            first = root(index)
            second = root(neighbor)
            if first != second:
                parent[second] = first
    components: dict[int, list[int]] = {}
    for index in range(len(coefficients)):
        components.setdefault(root(index), []).append(index)
    transition = np.zeros((len(coefficients), len(coefficients)), dtype=np.float64)
    for component in components.values():
        conditional = target_mass[component] / np.sum(target_mass[component])
        transition[np.ix_(component, component)] = conditional[None, :]
    return transition


def block_heat_bath_matrices(
    coefficients: np.ndarray,
    target_mass: np.ndarray,
    directions: np.ndarray,
    block_size: int,
) -> list[np.ndarray]:
    """Partition an ordered direction frame into low-rank heat-bath blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [
        block_heat_bath_matrix(
            coefficients,
            target_mass,
            directions[start:start + block_size],
        )
        for start in range(0, len(directions), block_size)
    ]


def linear_relaxation_bound(
    coefficients: np.ndarray,
    points: np.ndarray,
    target_mass: np.ndarray,
    directions: np.ndarray,
) -> float:
    """Best Dirichlet/variance ratio among physical linear observables."""
    stationary = target_mass / np.sum(target_mass)
    mean = stationary @ points
    centered = points - mean
    covariance = (centered.T * stationary) @ centered
    lookup = {tuple(int(x) for x in row): i for i, row in enumerate(coefficients)}
    conditional_covariance = np.zeros_like(covariance)
    for direction in directions:
        unseen = set(range(len(coefficients)))
        direction_covariance = np.zeros_like(covariance)
        while unseen:
            origin_index = next(iter(unseen))
            origin = coefficients[origin_index]
            line = []
            for sign in (1, -1):
                k = 0 if sign == 1 else -1
                while True:
                    index = lookup.get(tuple(int(x) for x in origin + k * direction))
                    if index is None:
                        break
                    line.append(index)
                    k += sign
            line = sorted(set(line))
            unseen.difference_update(line)
            line_probability = float(np.sum(stationary[line]))
            conditional = stationary[line] / line_probability
            line_mean = conditional @ points[line]
            line_centered = points[line] - line_mean
            direction_covariance += line_probability * (
                (line_centered.T * conditional) @ line_centered
            )
        conditional_covariance += direction_covariance / len(directions)
    values, vectors = np.linalg.eigh(covariance)
    keep = values > 1e-13 * values[-1]
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    comparison = whitener.T @ conditional_covariance @ whitener
    return float(np.linalg.eigvalsh(0.5 * (comparison + comparison.T))[0])


def transition_linear_relaxation_bound(
    points: np.ndarray,
    stationary: np.ndarray,
    transition: np.ndarray,
) -> float:
    """Best one-step Dirichlet ratio among physical linear observables."""
    mean = stationary @ points
    centered = points - mean
    covariance = (centered.T * stationary) @ centered
    propagated = transition @ centered
    cross = (centered.T * stationary) @ propagated
    dirichlet = covariance - 0.5 * (cross + cross.T)
    values, vectors = np.linalg.eigh(covariance)
    keep = values > 1e-13 * values[-1]
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    comparison = whitener.T @ dirichlet @ whitener
    comparison = 0.5 * (comparison + comparison.T)
    return float(np.linalg.eigvalsh(comparison)[0])


def reversible_gap(transition: np.ndarray, stationary: np.ndarray) -> float:
    root = np.sqrt(stationary)
    symmetric = root[:, None] * transition / root[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return float(1.0 - eigenvalues[-2])


def observable_relaxation_inflation(
    transition: np.ndarray,
    stationary: np.ndarray,
    values: np.ndarray,
) -> float:
    """Worst asymptotic-variance inflation on a supplied feature space.

    For a stationary reversible chain and centered scalar observable ``f``,
    the asymptotic variance of its empirical mean is

        <f, (I+K)(I-K)^+ f>_pi / M.

    This routine returns the largest ratio of that quadratic form to
    ``Var_pi(f)`` over the linear span of the columns of ``values``.  It can
    be far smaller than the inverse absolute gap because slow eigenfunctions
    orthogonal to the requested observables are irrelevant.
    """
    stationary = np.asarray(stationary, dtype=np.float64)
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    centered = features - stationary @ features
    root = np.sqrt(stationary)
    symmetric = root[:, None] * transition / root[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_features = eigenvectors.T @ (root[:, None] * centered)
    keep_mode = eigenvalues < 1.0 - 1e-11
    frozen_energy = float(np.sum(spectral_features[~keep_mode] ** 2))
    total_energy = float(np.sum(spectral_features ** 2))
    if frozen_energy > 1e-10 * max(total_energy, 1e-300):
        return math.inf
    spectral_features = spectral_features[keep_mode]
    eigenvalues = eigenvalues[keep_mode]
    covariance = spectral_features.T @ spectral_features
    inflation_weight = (1.0 + eigenvalues) / (1.0 - eigenvalues)
    asymptotic = (
        spectral_features.T * inflation_weight
    ) @ spectral_features
    covariance = 0.5 * (covariance + covariance.T)
    asymptotic = 0.5 * (asymptotic + asymptotic.T)
    values_cov, vectors_cov = np.linalg.eigh(covariance)
    if not len(values_cov) or values_cov[-1] <= 1e-20:
        return 0.0
    keep = values_cov > 1e-12 * values_cov[-1]
    whitener = (
        vectors_cov[:, keep] / np.sqrt(values_cov[keep])[None, :]
    )
    comparison = whitener.T @ asymptotic @ whitener
    comparison = 0.5 * (comparison + comparison.T)
    return float(np.linalg.eigvalsh(comparison)[-1])


def traceless_hessian_features(points: np.ndarray) -> np.ndarray:
    """Vectorize ``xx^T-||x||^2 I/n`` for every physical lattice point."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[1]
    outer = np.einsum("ni,nj->nij", points, points)
    norm2 = np.einsum("ni,ni->n", points, points)
    outer[:, np.arange(n), np.arange(n)] -= norm2[:, None] / n
    return outer.reshape(len(points), n * n)


def walsh_hessian_observable_inflation(
    transition: np.ndarray,
    stationary: np.ndarray,
    points: np.ndarray,
    tail_bits: np.ndarray,
) -> dict[str, object]:
    """Audit every remaining Walsh character of the Hessian observable."""
    tail_bits = np.asarray(tail_bits, dtype=np.uint8)
    hessian = traceless_hessian_features(points)
    rows = []
    for theta in range(1 << tail_bits.shape[1]):
        theta_bits = (
            (theta >> np.arange(tail_bits.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        parity = (tail_bits @ theta_bits) & 1
        sign = 1.0 - 2.0 * parity.astype(np.float64)
        inflation = observable_relaxation_inflation(
            transition,
            stationary,
            sign[:, None] * hessian,
        )
        rows.append({
            "theta": theta,
            "asymptotic_variance_inflation": inflation,
        })
    return {
        "worst_asymptotic_variance_inflation": max(
            row["asymptotic_variance_inflation"] for row in rows
        ),
        "walsh_characters": rows,
    }


def sweep_observable_contraction(
    projections: list[np.ndarray],
    stationary: np.ndarray,
    values: np.ndarray,
    *,
    sweeps: int = 8,
) -> dict[str, object]:
    """Measure ordered lifting-residual contraction on a feature subspace.

    Each ``H_v`` is an orthogonal projection in ``L^2(pi)``.  One sweep
    applies all of them in order.  The reported norm is the worst residual
    energy after each number of sweeps, divided by the initial variance, over
    the complete linear span of ``values``.
    """
    stationary = np.asarray(stationary, dtype=np.float64)
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    features = features - stationary @ features
    covariance = (features.T * stationary) @ features
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not len(eigenvalues) or eigenvalues[-1] <= 1e-20:
        return {"sweep_contraction": [0.0] * sweeps}
    keep = eigenvalues > 1e-12 * eigenvalues[-1]
    whitened = features @ (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    )
    residual = whitened
    contraction = []
    for _ in range(sweeps):
        for projection in projections:
            residual = projection @ residual
        gram = (residual.T * stationary) @ residual
        gram = 0.5 * (gram + gram.T)
        contraction.append(float(max(np.linalg.eigvalsh(gram)[-1], 0.0)))
    roots = [
        value ** (1.0 / (2.0 * (index + 1)))
        for index, value in enumerate(contraction)
    ]
    return {
        "sweep_contraction": contraction,
        "per_sweep_norm_roots": roots,
        "best_observed_per_sweep_norm": min(roots),
    }


def lifting_flow_certificate(
    projections: list[np.ndarray],
    stationary: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    """Certify observable autocorrelation by a lifting-detail decomposition.

    Put ``Q=H_m...H_1`` and solve ``(I-Q)z=f`` on centered functions.  One
    analysis sweep stores details ``g_i=(I-H_i)H_(i-1)...H_1 z``.  They obey
    ``f=sum_i g_i`` and ``H_i g_i=0``.  For

        L = I-K = (1/m) sum_i (I-H_i),

    the dual Dirichlet principle gives

        <f,L^+f>_pi <= m sum_i ||g_i||_pi^2.

    The returned generalized eigenvalue is the worst right-hand energy over
    the supplied feature span; ``2*m*C-1`` bounds asymptotic variance
    inflation for every observable in that span.
    """
    stationary = np.asarray(stationary, dtype=np.float64)
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    features = features - stationary @ features
    covariance = (features.T * stationary) @ features
    covariance = 0.5 * (covariance + covariance.T)
    values_cov, vectors_cov = np.linalg.eigh(covariance)
    if not len(values_cov) or values_cov[-1] <= 1e-20:
        return {
            "stable_detail_energy": 0.0,
            "asymptotic_variance_inflation_bound": 0.0,
            "decomposition_residual": 0.0,
        }
    keep = values_cov > 1e-12 * values_cov[-1]
    features = features @ vectors_cov[:, keep]
    covariance = np.diag(values_cov[keep])

    sweep = np.eye(len(stationary))
    for projection in projections:
        sweep = projection @ sweep
    # The rank-one term fixes the constant gauge.  For an irreducible family,
    # I-Q+1*pi^T is nonsingular and agrees with I-Q on centered functions.
    gauge_fixed = (
        np.eye(len(stationary))
        - sweep
        + np.ones((len(stationary), 1)) * stationary[None, :]
    )
    potential = np.linalg.solve(gauge_fixed, features)
    residual = potential
    detail_energy = np.zeros_like(covariance)
    detail_sum = np.zeros_like(features)
    for projection in projections:
        updated = projection @ residual
        detail = residual - updated
        detail_sum += detail
        detail_energy += (detail.T * stationary) @ detail
        residual = updated
    decomposition_error = features - detail_sum
    decomposition_residual = float(np.sqrt(np.sum(
        stationary[:, None] * decomposition_error * decomposition_error
    )))
    whitener = np.diag(1.0 / np.sqrt(np.diag(covariance)))
    comparison = whitener @ detail_energy @ whitener
    comparison = 0.5 * (comparison + comparison.T)
    stable_energy = float(np.linalg.eigvalsh(comparison)[-1])
    count = len(projections)
    return {
        "stable_detail_energy": stable_energy,
        "asymptotic_variance_inflation_bound": max(
            2.0 * count * stable_energy - 1.0,
            0.0,
        ),
        "decomposition_residual": decomposition_residual,
    }


def walsh_hessian_sweep_contraction(
    projections: list[np.ndarray],
    stationary: np.ndarray,
    points: np.ndarray,
    tail_bits: np.ndarray,
    *,
    sweeps: int = 8,
) -> dict[str, object]:
    """Audit ordered projection sweeps for every Walsh--Hessian subspace."""
    tail_bits = np.asarray(tail_bits, dtype=np.uint8)
    hessian = traceless_hessian_features(points)
    rows = []
    for theta in range(1 << tail_bits.shape[1]):
        theta_bits = (
            (theta >> np.arange(tail_bits.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        parity = (tail_bits @ theta_bits) & 1
        sign = 1.0 - 2.0 * parity.astype(np.float64)
        rows.append({
            "theta": theta,
            **sweep_observable_contraction(
                projections,
                stationary,
                sign[:, None] * hessian,
                sweeps=sweeps,
            ),
        })
    return {
        "worst_best_observed_per_sweep_norm": max(
            row["best_observed_per_sweep_norm"] for row in rows
        ),
        "walsh_characters": rows,
    }


def walsh_hessian_lifting_certificate(
    projections: list[np.ndarray],
    stationary: np.ndarray,
    points: np.ndarray,
    tail_bits: np.ndarray,
) -> dict[str, object]:
    """Prove finite autocorrelation bounds by exact lifting decompositions."""
    tail_bits = np.asarray(tail_bits, dtype=np.uint8)
    hessian = traceless_hessian_features(points)
    rows = []
    for theta in range(1 << tail_bits.shape[1]):
        theta_bits = (
            (theta >> np.arange(tail_bits.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        parity = (tail_bits @ theta_bits) & 1
        sign = 1.0 - 2.0 * parity.astype(np.float64)
        rows.append({
            "theta": theta,
            **lifting_flow_certificate(
                projections,
                stationary,
                sign[:, None] * hessian,
            ),
        })
    return {
        "worst_asymptotic_variance_inflation_bound": max(
            row["asymptotic_variance_inflation_bound"] for row in rows
        ),
        "maximum_decomposition_residual": max(
            row["decomposition_residual"] for row in rows
        ),
        "walsh_characters": rows,
    }


def projection_feature_contraction(
    projection: np.ndarray,
    stationary: np.ndarray,
    values: np.ndarray,
) -> float:
    """Worst ``||Hf||_pi/||f||_pi`` on a centered feature span."""
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    features = features - stationary @ features
    covariance = (features.T * stationary) @ features
    covariance = 0.5 * (covariance + covariance.T)
    values_cov, vectors_cov = np.linalg.eigh(covariance)
    if not len(values_cov) or values_cov[-1] <= 1e-20:
        return 0.0
    keep = values_cov > 1e-12 * values_cov[-1]
    whitener = vectors_cov[:, keep] / np.sqrt(values_cov[keep])[None, :]
    propagated = projection @ features @ whitener
    gram = (propagated.T * stationary) @ propagated
    gram = 0.5 * (gram + gram.T)
    return float(math.sqrt(max(np.linalg.eigvalsh(gram)[-1], 0.0)))


def walsh_hessian_local_block_contraction(
    projections: list[np.ndarray],
    stationary: np.ndarray,
    points: np.ndarray,
    tail_bits: np.ndarray,
    block_direction_tail_bits: list[np.ndarray],
) -> dict[str, object]:
    """Best active local block for each Walsh--Hessian character."""
    tail_bits = np.asarray(tail_bits, dtype=np.uint8)
    hessian = traceless_hessian_features(points)
    rows = []
    for theta in range(1 << tail_bits.shape[1]):
        theta_bits = (
            (theta >> np.arange(tail_bits.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        parity = (tail_bits @ theta_bits) & 1
        sign = 1.0 - 2.0 * parity.astype(np.float64)
        contractions = []
        for projection, block_tail in zip(
            projections, block_direction_tail_bits, strict=True
        ):
            active = theta == 0 or np.any((block_tail @ theta_bits) & 1)
            if active:
                contractions.append(projection_feature_contraction(
                    projection,
                    stationary,
                    sign[:, None] * hessian,
                ))
        rows.append({
            "theta": theta,
            "best_active_block_contraction": min(contractions),
        })
    return {
        "worst_best_active_block_contraction": max(
            row["best_active_block_contraction"] for row in rows
        ),
        "zero_character_best_block_contraction": rows[0][
            "best_active_block_contraction"
        ],
        "worst_best_nonzero_character_contraction": max(
            row["best_active_block_contraction"] for row in rows[1:]
        ) if len(rows) > 1 else 0.0,
        "walsh_characters": rows,
    }


def frame_statistics(
    directions: np.ndarray,
    inverse_basis: np.ndarray,
    s: float,
    direction_tail_bits: np.ndarray | None = None,
) -> dict[str, float]:
    physical = directions @ inverse_basis
    norms = np.linalg.norm(physical, axis=1)
    unit = physical / norms[:, None]
    frame = unit.T @ unit / len(unit)
    eigenvalues = np.linalg.eigvalsh(frame)
    mobility = np.asarray([centered_line_mobility(s / norm) for norm in norms])
    mobile_frame = (unit.T * mobility) @ unit / len(unit)
    mobile_eigenvalues = np.linalg.eigvalsh(mobile_frame)
    report = {
        "directions": int(len(directions)),
        "minimum_frame_eigenvalue": float(eigenvalues[0]),
        "maximum_frame_eigenvalue": float(eigenvalues[-1]),
        "frame_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "minimum_centered_line_mobility": float(np.min(mobility)),
        "minimum_mobile_frame_eigenvalue": float(mobile_eigenvalues[0]),
        "maximum_direction_length_over_s": float(np.max(norms) / s),
    }
    if direction_tail_bits is not None and direction_tail_bits.shape[1]:
        tail_bits = np.asarray(direction_tail_bits, dtype=np.uint8)
        coverage = []
        mobile_coverage = []
        for theta in range(1, 1 << tail_bits.shape[1]):
            theta_bits = (
                (theta >> np.arange(tail_bits.shape[1], dtype=np.int64)) & 1
            ).astype(np.uint8)
            toggles = ((tail_bits @ theta_bits) & 1).astype(np.float64)
            coverage.append(float(np.mean(toggles)))
            mobile_coverage.append(float(np.mean(toggles * mobility)))
        report.update({
            "minimum_nonzero_walsh_character_edge_fraction": min(coverage),
            "minimum_mobile_walsh_character_coverage": min(mobile_coverage),
        })
    return report


def block_phase_spectral_statistics(
    directions: np.ndarray,
    inverse_basis: np.ndarray,
    direction_tail_bits: np.ndarray,
    s: float,
    block_size: int,
) -> dict[str, float]:
    """Measure half-dual separation for every nonzero Walsh character.

    On a block lattice with physical row basis ``C``, a restricted parity
    character ``b`` has frequency coset ``C^+ (Z^r+b/2)`` in the block dual.
    Its distance from the ordinary dual controls the twisted theta numerator
    under Poisson summation.  We report the worst character's best block.
    """
    tail = np.asarray(direction_tail_bits, dtype=np.uint8)
    blocks = []
    for start in range(0, len(directions), block_size):
        block_directions = directions[start:start + block_size]
        block_tail = tail[start:start + block_size]
        physical = block_directions @ inverse_basis
        rank = int(np.linalg.matrix_rank(physical.astype(np.float64)))
        if rank != len(block_directions):
            continue
        gram_inverse = np.linalg.inv(physical @ physical.T)
        blocks.append((block_tail, gram_inverse))
    if not blocks or not tail.shape[1]:
        return {
            "minimum_best_twisted_dual_margin_times_s": 0.0,
            "minimum_character_block_coverage": 0.0,
        }
    best_margins = []
    coverage = []
    maximum_dual_search_radius = 0
    dual_theta_tail_upper_bounds = []
    for _, gram_inverse in blocks:
        minimum_eigenvalue = float(np.linalg.eigvalsh(gram_inverse)[0])
        exponent = math.pi * s * s * minimum_eigenvalue
        radius = max(4, int(math.ceil(math.sqrt(
            math.log(1e14) / max(exponent, 1e-300)
        ))))
        values = range(-radius, radius + 1)
        total = 0.0
        for integer in itertools.product(values, repeat=len(gram_inverse)):
            vector = np.asarray(integer, dtype=np.float64)
            total += math.exp(-math.pi * s * s * float(
                vector @ gram_inverse @ vector
            ))
        first_omitted = math.exp(-exponent * (radius + 1) ** 2)
        ratio = math.exp(-exponent * (2 * radius + 3))
        one_sided_tail = first_omitted / max(1.0 - ratio, 1e-300)
        one_dimensional_upper = 1.0 + 2.0 * sum(
            math.exp(-exponent * k * k) for k in range(1, radius + 1)
        ) + 2.0 * one_sided_tail
        outside_upper = (
            len(gram_inverse)
            * 2.0
            * one_sided_tail
            * one_dimensional_upper ** (len(gram_inverse) - 1)
        )
        dual_theta_tail_upper_bounds.append(max(total + outside_upper - 1.0, 0.0))
    for theta in range(1, 1 << tail.shape[1]):
        theta_bits = (
            (theta >> np.arange(tail.shape[1], dtype=np.int64)) & 1
        ).astype(np.uint8)
        margins = []
        active = 0
        for block_tail, gram_inverse in blocks:
            parity = (block_tail @ theta_bits) & 1
            if not np.any(parity):
                continue
            active += 1
            offset = 0.5 * parity.astype(np.float64)
            best = float(offset @ gram_inverse @ offset)
            minimum_eigenvalue = float(np.linalg.eigvalsh(gram_inverse)[0])
            radius = int(math.ceil(math.sqrt(
                best / max(minimum_eigenvalue, 1e-300)
            ) + 1.0))
            maximum_dual_search_radius = max(maximum_dual_search_radius, radius)
            values = range(-radius, radius + 1)
            for integer in itertools.product(values, repeat=len(offset)):
                candidate = np.asarray(integer, dtype=np.float64) + offset
                best = min(best, float(candidate @ gram_inverse @ candidate))
            margins.append(s * math.sqrt(max(best, 0.0)))
        best_margins.append(max(margins) if margins else 0.0)
        coverage.append(active / len(blocks))
    return {
        "minimum_best_twisted_dual_margin_times_s": min(best_margins),
        "minimum_character_block_coverage": min(coverage),
        "maximum_exact_dual_search_radius": maximum_dual_search_radius,
        "maximum_block_dual_theta_tail_upper_bound": max(
            dual_theta_tail_upper_bounds
        ),
    }


def audit_family(
    coefficients: np.ndarray,
    points: np.ndarray,
    target_mass: np.ndarray,
    inverse_basis: np.ndarray,
    directions: np.ndarray,
    s: float,
    name: str,
    tail_bits: np.ndarray | None = None,
    direction_tail_bits: np.ndarray | None = None,
    block_size: int = 1,
) -> dict[str, object]:
    projections = block_heat_bath_matrices(
        coefficients,
        target_mass,
        directions,
        block_size,
    )
    transition = sum(projections) / len(projections)
    stationary = target_mass / np.sum(target_mass)
    flow = stationary[:, None] * transition
    frame = frame_statistics(
        directions,
        inverse_basis,
        s,
        direction_tail_bits,
    )
    gap = reversible_gap(transition, stationary)
    linear_bound = transition_linear_relaxation_bound(
        points,
        stationary,
        transition,
    )
    report = {
        "family": name,
        "heat_bath_blocks": len(projections),
        "maximum_directions_per_block": block_size,
        "minimum_block_rank": min(
            int(np.linalg.matrix_rank(
                directions[start:start + block_size].astype(np.float64)
            ))
            for start in range(0, len(directions), block_size)
        ),
        "maximum_block_rank": max(
            int(np.linalg.matrix_rank(
                directions[start:start + block_size].astype(np.float64)
            ))
            for start in range(0, len(directions), block_size)
        ),
        **frame,
        "row_stochastic_error": float(np.max(np.abs(np.sum(transition, axis=1) - 1.0))),
        "detailed_balance_error": float(np.max(np.abs(flow - flow.T))),
        "absolute_spectral_gap": gap,
        "linear_observable_gap_upper_bound": linear_bound,
        "nonlinear_to_linear_gap_ratio": gap / linear_bound,
    }
    if direction_tail_bits is not None:
        report.update(block_phase_spectral_statistics(
            directions,
            inverse_basis,
            direction_tail_bits,
            s,
            block_size,
        ))
    if tail_bits is not None:
        report["hessian_observable"] = walsh_hessian_observable_inflation(
            transition,
            stationary,
            points,
            tail_bits,
        )
        report["hessian_sweep_certificate"] = walsh_hessian_sweep_contraction(
            projections,
            stationary,
            points,
            tail_bits,
        )
        report["hessian_lifting_flow_certificate"] = (
            walsh_hessian_lifting_certificate(
                projections,
                stationary,
                points,
                tail_bits,
            )
        )
        block_tail_bits = [
            direction_tail_bits[start:start + block_size]
            for start in range(0, len(directions), block_size)
        ]
        report["hessian_local_block_contraction"] = (
            walsh_hessian_local_block_contraction(
                projections,
                stationary,
                points,
                tail_bits,
                block_tail_bits,
            )
        )
    return report


def audit_dimension(
    n: int,
    *,
    h: int,
    cutoff: int,
    direction_cutoff: int,
    direction_multiple: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    inverse_basis = np.linalg.inv(basis)
    shortest, _ = shortest_vector_coefficients(basis)
    transform = random_gl2(n, rng)
    inverse_transform = _gf2_inverse(transform)
    all_coefficients = coefficient_box(n, cutoff)
    all_points = all_coefficients @ inverse_basis
    all_norm2 = np.einsum("ij,ij->i", all_points, all_points)
    s2 = 4.0 * n * TARGET_WIDTH * math.log(2.0) / (math.pi * shortest * shortest)
    s = math.sqrt(s2)
    all_mass = np.exp(-math.pi * all_norm2 / s2)
    bits = transformed_bits(all_coefficients, inverse_transform)
    prefix_values = bits[:, :h]
    prefix_indices = prefix_values @ (1 << np.arange(h, dtype=np.int64))
    generators = generator_directions(transform, h)
    isotropic_directions = isotropic_kernel_directions(
        n,
        h,
        inverse_transform,
        inverse_basis,
        search_cutoff=direction_cutoff,
        count=direction_multiple * n,
        s=s,
    )
    phase_aware_directions = phase_aware_kernel_directions(
        n,
        h,
        inverse_transform,
        inverse_basis,
        search_cutoff=direction_cutoff,
        count=direction_multiple * n,
        s=s,
    )
    reports = []
    for prefix in range(1 << h):
        mask = prefix_indices == prefix
        coefficients = all_coefficients[mask]
        points = all_points[mask]
        mass = all_mass[mask]
        tail_bits = bits[mask, h:]
        boundary = np.any(np.abs(coefficients) == cutoff, axis=1)
        reports.append({
            "prefix": prefix,
            "states": int(len(coefficients)),
            "prefix_probability": float(np.sum(mass) / np.sum(all_mass)),
            "boundary_mass": float(np.sum(mass[boundary]) / np.sum(mass)),
            "families": [
                audit_family(
                    coefficients,
                    points,
                    mass,
                    inverse_basis,
                    directions,
                    s,
                    name,
                    tail_bits,
                    transformed_bits(
                        directions,
                        inverse_transform,
                    )[:, h:],
                    block_size,
                )
                for name, directions, block_size in (
                    ("kernel_generators", generators, 1),
                    ("short_isotropic_pool", isotropic_directions, 1),
                    ("physical_walsh_phase_pool", phase_aware_directions, 1),
                    ("physical_walsh_rank2_blocks", phase_aware_directions, 2),
                    ("physical_walsh_rank3_blocks", phase_aware_directions, 3),
                )
            ],
        })
    return {
        "dimension": n,
        "h": h,
        "cutoff": cutoff,
        "direction_cutoff": direction_cutoff,
        "target_width": TARGET_WIDTH,
        "s_over_lambda1": s / shortest,
        "required_gap_exponent": 0.5 - 2.0 * TARGET_WIDTH,
        "cosets": reports,
    }


def audit() -> dict[str, object]:
    gap_budget = 0.5 - 2.0 * TARGET_WIDTH
    return {
        "experiment": "walsh_line_heat_bath_kernel",
        "gap_budget": gap_budget,
        "constant_rank_dense_lattice_obstruction": (
            constant_rank_dense_lattice_obstruction(TARGET_WIDTH)
        ),
        "necessary_mobile_radius_coefficient_times_s_sqrt_n": math.sqrt(
            4.0 * gap_budget * math.log(2.0) / math.pi
        ),
        "rows": [
            audit_dimension(3, h=1, cutoff=3, direction_cutoff=2, direction_multiple=3, seed=23),
            audit_dimension(4, h=2, cutoff=2, direction_cutoff=2, direction_multiple=3, seed=23),
            audit_dimension(5, h=2, cutoff=2, direction_cutoff=2, direction_multiple=3, seed=23),
        ],
        "interpretation": (
            "A polynomial-sized isotropic pool of syndrome-zero directions gives "
            "exact low-rank heat-bath kernels.  The audit separates the absolute "
            "gap from the integrated autocorrelation of the actual Walsh-Hessian "
            "observables and certifies the latter by a reversible lifting-detail "
            "decomposition.  Rank-two and rank-three blocks test whether locally "
            "coupled details cross barriers that freeze individual lines.  The "
            "dense-self-dual overlap calculation shows why this finite gain cannot "
            "be promoted to a worst-case 2^o(n) constant-rank theorem."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
