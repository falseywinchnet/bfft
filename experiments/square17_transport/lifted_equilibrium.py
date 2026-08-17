#!/usr/bin/env python3
"""Lifted entropy/eikonal equilibrium search for seventeen unit squares.

This is the packing analogue of two existing BFFT research paths:

* Night Vision keeps a posterior over the complete motion orbit, projects to
  a prescribed entropy, and only then backprojects a belief.
* Segmenting v3 propagates all first-arrival sources simultaneously; a hard
  owner is a terminal readout of equal-action fronts.

Here every square pair retains all four SAT separating-axis branches.  A
population of complete configurations carries those relaxed branch measures.
Each local update projects all near-capacity incidences toward one equal yield
in the eigenbasis of their contact metric.  The basis is whitened, so a unit
coefficient has equal local action in every admitted direction.  Persistent
slack/dual state prevents a branch residual from being forgotten between
sweeps.  Configuration lanes are resampled only through a fixed entropy
budget and are never ranked to a single answer until terminal emission.

The algorithm is classical and finite.  It is "quantum-like" only in the
useful mathematical sense: a free-energy relaxation over superposed discrete
branches followed by a zero-temperature measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import legalize_capacity, polish_pose_capacity_slsqp, write_svg
from geometry import (
    PAIR_I,
    PAIR_J,
    SQUARE_COUNT,
    capacity_state,
    pair_witness_state,
    wrap_square_phase,
)


CONSTRAINT_COUNT = 4 * SQUARE_COUNT + len(PAIR_I)
POSE_DIMENSION = 3 * SQUARE_COUNT

# Low-complexity occupancy sectors, not geometric answers.  Their transposes
# and the random 17-of-25 sectors below ensure that the population contains
# both row and column gauges before transport begins.
OCCUPANCY_SECTORS = (
    (4, 4, 5, 4),
    (5, 4, 4, 4),
    (4, 5, 4, 4),
    (4, 4, 4, 5),
    (3, 4, 3, 4, 3),
    (4, 3, 4, 3, 3),
    (3, 3, 4, 3, 4),
    (5, 3, 3, 3, 3),
)


@dataclass(frozen=True)
class LiftedConfig:
    lanes: int = 64
    sweeps: int = 1600
    start_side: float = 5.0
    target_side: float = 4.67
    shrink_fraction: float = 0.35
    cool_start_fraction: float = 0.35
    start_axis_temperature: float = 0.18
    end_axis_temperature: float = 0.0002
    target_clearance: float = 2.0e-5
    activation_clearance: float = 0.12
    activation_temperature: float = 0.025
    absolute_smoothing: float = 2.0e-5
    metric_ridge: float = 2.0e-4
    translation_trust: float = 0.028
    phase_trust: float = 0.018
    momentum: float = 0.72
    dual_cap: float = 0.12
    posterior_interval: int = 20
    mutation_interval: int = 20
    effective_lane_fraction: float = 0.68
    mutation_translation: float = 0.025
    mutation_phase: float = 0.030
    finite_dimension: float = 6.0
    emit_lanes: int = 8
    terminal_measurement: bool = True
    initialization: str = "mixed"
    seed: int = 0


@dataclass
class ConstraintChart:
    clearance: np.ndarray
    jacobian: np.ndarray
    axis_probability: np.ndarray
    mean_axis_entropy: float
    effective_axes: float


def normalized_bruun_dip_basis(length: int = 64, level: int = 3) -> np.ndarray:
    """Return the normalized real Bruun DIP intermediate operator.

    This is the real residue-pair walk from ``experiments/dip_numba.py`` made
    explicit as a small fixed matrix.  The packed real transform has two
    normalization classes (the walking DC/Nyquist ridge and its interior
    packets), so every residue row is normalized by its own exact norm.  The
    resulting matrix is the fixed orthogonal chart used to carry momentum
    while the contact eigenbasis itself changes.
    """

    if length < 4 or length & (length - 1):
        raise ValueError("DIP length must be a power of two")
    maximum_level = int(math.log2(length))
    if level < 0 or level > maximum_level:
        raise ValueError("invalid DIP level")
    state = np.eye(length, dtype=np.float64)
    for stage in range(level):
        e = 1 << stage
        q = length >> stage
        q2 = q >> 1
        output = np.empty_like(state)
        for column in range(q2):
            output[column] = state[column] + state[q2 + column]
            output[q2 + column] = state[column] - state[q2 + column]
        if e >= 2:
            d = e >> 1
            output[(2 * d) * q2:(2 * d + 1) * q2] = state[q:q + q2]
            output[(2 * d + 1) * q2:(2 * d + 2) * q2] = state[q + q2:2 * q]
        for d in range(1, e >> 1):
            theta = math.pi * d / e
            cosine = math.cos(theta)
            sine = math.sin(theta)
            source_a = state[(2 * d) * q:(2 * d + 1) * q]
            source_b = state[(2 * d + 1) * q:(2 * d + 2) * q]
            even_a, odd_a = source_a[:q2], source_a[q2:]
            even_b, odd_b = source_b[:q2], source_b[q2:]
            rotated_a = cosine * odd_a - sine * odd_b
            rotated_b = sine * odd_a + cosine * odd_b
            low_a = (2 * d) * q2
            low_b = (2 * d + 1) * q2
            high_a = (2 * (e - d)) * q2
            high_b = (2 * (e - d) + 1) * q2
            output[low_a:low_a + q2] = even_a + rotated_a
            output[low_b:low_b + q2] = even_b + rotated_b
            output[high_a:high_a + q2] = even_a - rotated_a
            output[high_b:high_b + q2] = rotated_b - even_b
        state = output
    row_norm = np.linalg.norm(state, axis=1)
    if np.any(row_norm <= 0.0):
        raise RuntimeError("degenerate Bruun DIP residue row")
    return state / row_norm[:, None]


def pack_pose_direction(direction: np.ndarray) -> np.ndarray:
    result = np.zeros(64, dtype=np.float64)
    result[:SQUARE_COUNT] = direction[:, 0]
    result[SQUARE_COUNT:2 * SQUARE_COUNT] = direction[:, 1]
    result[2 * SQUARE_COUNT:3 * SQUARE_COUNT] = 0.65 * direction[:, 2]
    return result


def unpack_pose_direction(packed: np.ndarray) -> np.ndarray:
    result = np.zeros((SQUARE_COUNT, 3), dtype=np.float64)
    result[:, 0] = packed[:SQUARE_COUNT]
    result[:, 1] = packed[SQUARE_COUNT:2 * SQUARE_COUNT]
    result[:, 2] = packed[2 * SQUARE_COUNT:3 * SQUARE_COUNT] / 0.65
    return result


def entropy_project_scores(
    scores: np.ndarray,
    target_effective: float,
    *,
    steps: int = 36,
) -> tuple[np.ndarray, dict[str, float]]:
    """Night-Vision entropy projection for one finite hypothesis orbit."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    candidate_count = values.shape[1]
    target = float(np.clip(target_effective, 1.0, candidate_count))
    target_entropy = math.log(target)
    centered = values - np.max(values, axis=1, keepdims=True)

    def probability_at(beta: float) -> np.ndarray:
        logits = beta * centered
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(np.clip(logits, -60.0, 0.0))
        probability /= np.maximum(np.sum(probability, axis=1, keepdims=True), 1.0e-30)
        return probability

    def entropy_at(beta: float) -> float:
        probability = probability_at(beta)
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1.0e-30)), axis=1)
        return float(np.mean(entropy))

    if target >= candidate_count * (1.0 - 1.0e-12):
        probability = np.full_like(values, 1.0 / candidate_count)
        return probability.squeeze(axis=0) if scores.ndim == 1 else probability, {
            "inverse_temperature": 0.0,
            "mean_entropy": math.log(candidate_count),
            "effective_support": float(candidate_count),
        }
    low, high = 0.0, 1.0
    while entropy_at(high) > target_entropy and high < 1.0e10:
        high *= 2.0
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if entropy_at(middle) > target_entropy:
            low = middle
        else:
            high = middle
    beta = 0.5 * (low + high)
    probability = probability_at(beta)
    entropy = entropy_at(beta)
    result = probability.squeeze(axis=0) if scores.ndim == 1 else probability
    return result, {
        "inverse_temperature": beta,
        "mean_entropy": entropy,
        "effective_support": math.exp(entropy),
    }


def _boundary_chart(
    poses: np.ndarray,
    side: float,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(poses[:, 2])
    sine = np.sin(poses[:, 2])
    epsilon = max(float(smoothing), 0.0)
    sine_absolute = np.sqrt(sine * sine + epsilon * epsilon)
    sine_sign = sine / np.maximum(sine_absolute, 1.0e-30)
    half_width = 0.5 * (cosine + sine_absolute)
    half_width_prime = 0.5 * (-sine + sine_sign * cosine)
    clearance = np.column_stack((
        poses[:, 0] - half_width,
        side - poses[:, 0] - half_width,
        poses[:, 1] - half_width,
        side - poses[:, 1] - half_width,
    )).ravel()
    jacobian = np.zeros((4 * SQUARE_COUNT, POSE_DIMENSION), dtype=np.float64)
    gradients = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    for square in range(SQUARE_COUNT):
        for face, gradient in enumerate(gradients):
            row = 4 * square + face
            jacobian[row, 3 * square:3 * square + 2] = gradient
            jacobian[row, 3 * square + 2] = -half_width_prime[square]
    return clearance, jacobian


def lifted_constraint_chart(
    poses: np.ndarray,
    side: float,
    axis_temperature: float,
    *,
    absolute_smoothing: float,
) -> ConstraintChart:
    """Lift four SAT branches per pair and return their free-energy chart."""

    boundary, boundary_jacobian = _boundary_chart(
        poses, side, absolute_smoothing
    )
    witnesses = pair_witness_state(
        poses, absolute_smoothing=absolute_smoothing
    )
    temperature = max(float(axis_temperature), 1.0e-8)
    normalized = witnesses.clearance / temperature
    maximum = np.max(normalized, axis=1, keepdims=True)
    exponential = np.exp(np.clip(normalized - maximum, -60.0, 0.0))
    probability = exponential / np.maximum(
        np.sum(exponential, axis=1, keepdims=True), 1.0e-30
    )
    # This is deliberately the unnormalised branch partition function.  Its
    # entropy term is the tunnelling allowance: a high-temperature relaxed
    # state can pass through configurations that are not exact packings.  It
    # is never reported as geometric feasibility, and converges monotonically
    # to the exact SAT maximum under the terminal zero-temperature anneal.
    pair_clearance = temperature * (
        maximum[:, 0] + np.log(np.sum(exponential, axis=1))
    )
    branch_gradient_weight = probability
    pair_jacobian = np.zeros((len(PAIR_I), POSE_DIMENSION), dtype=np.float64)
    center_i = np.einsum(
        "pa,pad->pd", branch_gradient_weight, witnesses.center_i_gradient,
        optimize=True
    )
    center_j = np.einsum(
        "pa,pad->pd", branch_gradient_weight, witnesses.center_j_gradient,
        optimize=True
    )
    theta_i = np.einsum(
        "pa,pa->p", branch_gradient_weight, witnesses.theta_i_gradient,
        optimize=True
    )
    theta_j = np.einsum(
        "pa,pa->p", branch_gradient_weight, witnesses.theta_j_gradient,
        optimize=True
    )
    for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
        pair_jacobian[pair, 3 * first:3 * first + 2] = center_i[pair]
        pair_jacobian[pair, 3 * first + 2] = theta_i[pair]
        pair_jacobian[pair, 3 * second:3 * second + 2] = center_j[pair]
        pair_jacobian[pair, 3 * second + 2] = theta_j[pair]
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1.0e-30)), axis=1
    )
    return ConstraintChart(
        clearance=np.concatenate((boundary, pair_clearance)),
        jacobian=np.vstack((boundary_jacobian, pair_jacobian)),
        axis_probability=probability,
        mean_axis_entropy=float(np.mean(entropy)),
        effective_axes=float(np.exp(np.mean(entropy))),
    )


def equal_action_projection(
    chart: ConstraintChart,
    dual: np.ndarray,
    config: LiftedConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One simultaneous Bregman projection in the local eikonal eigenbasis."""

    clearance = chart.clearance
    gate = 1.0 / (1.0 + np.exp(np.clip(
        (clearance - config.activation_clearance)
        / max(config.activation_temperature, 1.0e-12),
        -60.0,
        60.0,
    )))
    slack = np.maximum(
        clearance - config.target_clearance + dual, 0.0
    )
    target = config.target_clearance + slack - clearance - dual
    root = np.sqrt(np.maximum(gate, 1.0e-12))
    weighted_jacobian = root[:, None] * chart.jacobian
    weighted_target = root * target
    metric = weighted_jacobian.T @ weighted_jacobian
    right = weighted_jacobian.T @ weighted_target
    eigenvalue, eigenvector = np.linalg.eigh(metric)
    coefficient = eigenvector.T @ right
    direction = eigenvector @ (
        coefficient / (eigenvalue + config.metric_ridge)
    )
    prediction = chart.jacobian @ direction
    new_dual = np.clip(
        dual + clearance + prediction - config.target_clearance - slack,
        -config.dual_cap,
        config.dual_cap,
    )
    new_dual[gate < 1.0e-4] *= 0.9
    return (
        direction.reshape(SQUARE_COUNT, 3),
        new_dual,
        eigenvalue,
        eigenvector,
    )


def _limit_direction(
    direction: np.ndarray,
    translation_trust: float,
    phase_trust: float,
) -> np.ndarray:
    result = np.asarray(direction, dtype=np.float64).copy()
    translation = np.linalg.norm(result[:, :2], axis=1)
    scale = min(
        1.0,
        float(translation_trust) / max(float(np.max(translation)), 1.0e-30),
        float(phase_trust) / max(float(np.max(np.abs(result[:, 2]))), 1.0e-30),
    )
    return scale * result


def _constraint_merit(chart: ConstraintChart, target: float) -> float:
    violation = np.minimum(chart.clearance - target, 0.0)
    return 0.5 * float(np.dot(violation, violation))


def _accepted_first_arrival_step(
    poses: np.ndarray,
    direction: np.ndarray,
    side: float,
    axis_temperature: float,
    config: LiftedConfig,
    baseline: float,
) -> tuple[np.ndarray, float, float]:
    """Return the first decreasing action along one simultaneous direction."""

    scale = 1.0
    best = poses
    best_merit = baseline
    for _ in range(12):
        candidate = poses + scale * direction
        candidate[:, 2] = wrap_square_phase(candidate[:, 2])
        candidate[:, :2] = np.clip(candidate[:, :2], -0.25, side + 0.25)
        candidate_chart = lifted_constraint_chart(
            candidate,
            side,
            axis_temperature,
            absolute_smoothing=config.absolute_smoothing,
        )
        merit = _constraint_merit(candidate_chart, config.target_clearance)
        if merit < best_merit - 1.0e-18:
            return candidate, merit, scale
        scale *= 0.5
    return best, best_merit, 0.0


def _structured_sector_lane(
    rng: np.random.Generator,
    sector: int,
    side: float,
) -> np.ndarray:
    counts = OCCUPANCY_SECTORS[sector % len(OCCUPANCY_SECTORS)]
    y_values = np.linspace(0.5, 4.5, len(counts))
    rows = []
    for row, (count, y_value) in enumerate(zip(counts, y_values)):
        x_values = np.linspace(0.5, 4.5, count)
        if count < 5 and row % 2:
            x_values = x_values[::-1]
        rows.extend((x_value, y_value) for x_value in x_values)
    centers = np.asarray(rows, dtype=np.float64)
    if sector >= len(OCCUPANCY_SECTORS):
        centers = centers[:, ::-1]
    centers += rng.normal(0.0, 1.0e-5, (SQUARE_COUNT, 2))
    centers = 0.5 * side + (side - 1.0) / 4.0 * (centers - 2.5)
    phase = rng.normal(0.0, 1.0e-5, SQUARE_COUNT)
    return np.column_stack((centers, phase))


def initial_configuration_lanes(lanes: int, side: float, seed: int) -> np.ndarray:
    """Mix sixteen row/column gauges with continuous farthest-point sectors."""

    rng = np.random.default_rng(seed)
    population = np.empty((lanes, SQUARE_COUNT, 3), dtype=np.float64)
    for lane in range(lanes):
        if lane < 2 * len(OCCUPANCY_SECTORS):
            population[lane] = _structured_sector_lane(rng, lane, side)
        else:
            # A continuous, permutation-free support chart: choose points by
            # maximin action from a fresh low-discrepancy-sized cloud.  It has
            # no row, column, or 5x5 ownership inherited from initialization.
            candidates = rng.uniform(0.62, 4.38, (768, 2))
            selected = [int(rng.integers(len(candidates)))]
            distance_squared = np.sum(
                (candidates - candidates[selected[0]]) ** 2, axis=1
            )
            for _ in range(1, SQUARE_COUNT):
                chosen = int(np.argmax(distance_squared))
                selected.append(chosen)
                distance_squared = np.minimum(
                    distance_squared,
                    np.sum((candidates - candidates[chosen]) ** 2, axis=1),
                )
            centers = candidates[selected]
            phase = rng.uniform(-0.25 * math.pi, 0.25 * math.pi, SQUARE_COUNT)
            centers = 0.5 * side + (side - 1.0) / 4.0 * (centers - 2.5)
            population[lane] = np.column_stack((centers, phase))
    return population


def replicated_sector_lanes(lanes: int, side: float, seed: int) -> np.ndarray:
    """Independent replicas of all sixteen discrete row/column gauges."""

    rng = np.random.default_rng(seed)
    return np.asarray([
        _structured_sector_lane(
            rng, lane % (2 * len(OCCUPANCY_SECTORS)), side
        )
        for lane in range(lanes)
    ])


def equidistant_simplex_lanes(
    lanes: int,
    side: float,
    *,
    translation_radius: float = 0.55,
    phase_radius: float = 0.22,
) -> np.ndarray:
    """Exactly equidistant pose hypotheses around the coincident state.

    At most ``POSE_DIMENSION + 1`` points can be mutually equidistant in the
    51-dimensional physical chart.  A centered regular simplex attains that
    bound.  A fixed Bruun/DIP-derived orthogonal rotation prevents its axes
    from inheriting the x/y/theta packing order.
    """

    if lanes < 2 or lanes > POSE_DIMENSION + 1:
        raise ValueError("simplex initialization requires 2..52 lanes")
    centering = np.eye(lanes) - np.ones((lanes, lanes)) / lanes
    eigenvalue, eigenvector = np.linalg.eigh(centering)
    simplex = eigenvector[:, eigenvalue > 0.5]
    if simplex.shape[1] < POSE_DIMENSION:
        simplex = np.pad(
            simplex,
            ((0, 0), (0, POSE_DIMENSION - simplex.shape[1])),
        )
    elif simplex.shape[1] > POSE_DIMENSION:
        simplex = simplex[:, :POSE_DIMENSION]

    dip = normalized_bruun_dip_basis()
    rotation, _ = np.linalg.qr(dip[:POSE_DIMENSION, :POSE_DIMENSION].T)
    physical = simplex @ rotation.T
    direction = physical.reshape(lanes, SQUARE_COUNT, 3)
    chart_rms = math.sqrt(float(np.mean(direction ** 2)))
    population = np.empty_like(direction)
    population[:, :, :2] = (
        0.5 * side
        + translation_radius * direction[:, :, :2] / chart_rms
    )
    # Keep the covering-space phase for the exactly equidistant initial
    # projection.  Square-periodic wrapping happens only after its first
    # physical relaxation step.
    population[:, :, 2] = phase_radius * direction[:, :, 2] / chart_rms
    return population


def _metric_mutation(
    eigenvalue: np.ndarray,
    eigenvector: np.ndarray,
    config: LiftedConfig,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = max(float(config.finite_dimension), 1.0e-6)
    coefficient = rng.standard_t(dimension, POSE_DIMENSION)
    coefficient /= np.sqrt(eigenvalue + config.metric_ridge)
    direction = (eigenvector @ coefficient).reshape(SQUARE_COUNT, 3)
    translation_rms = math.sqrt(float(np.mean(direction[:, :2] ** 2)))
    phase_rms = math.sqrt(float(np.mean(direction[:, 2] ** 2)))
    if translation_rms > 1.0e-15:
        direction[:, :2] *= (
            scale * config.mutation_translation / translation_rms
        )
    if phase_rms > 1.0e-15:
        direction[:, 2] *= scale * config.mutation_phase / phase_rms
    return direction


def _exact_population_state(
    population: np.ndarray,
    side: float,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.empty(len(population), dtype=np.float64)
    residual = np.empty(len(population), dtype=np.float64)
    for lane, poses in enumerate(population):
        state = capacity_state(poses, side)
        minimum[lane] = state.minimum_clearance
        residual[lane] = state.overlap_residual
    return minimum, residual


def polish_measured_axis_capacity_slsqp(
    poses: np.ndarray,
    side: float,
    *,
    rounds: int = 4,
    iterations: int = 2400,
) -> tuple[np.ndarray, dict[str, object]]:
    """Terminal hard-owner preimage after the lifted evolution has stopped."""

    try:
        from scipy.optimize import Bounds, minimize
    except ImportError as error:  # pragma: no cover - optional Mini readout
        raise RuntimeError("SciPy is required for terminal measured-axis readout") from error

    anchor = np.asarray(poses, dtype=np.float64).copy()
    current = anchor.copy()
    coordinate_scale = np.tile(np.asarray([1.0, 1.0, 0.65]), SQUARE_COUNT)
    lower = np.tile(np.asarray([-0.5, -0.5, -0.25 * math.pi]), SQUARE_COUNT)
    upper = np.tile(
        np.asarray([side + 0.5, side + 0.5, 0.25 * math.pi]), SQUARE_COUNT
    )
    audit_rounds = []

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        delta = (flat - anchor.ravel()) * coordinate_scale
        return 0.5 * float(np.dot(delta, delta)), delta * coordinate_scale

    for measurement in range(rounds):
        witness = pair_witness_state(current, absolute_smoothing=1.0e-10)
        owner = np.argmax(witness.clearance, axis=1)

        def constraints(flat: np.ndarray) -> np.ndarray:
            chart = flat.reshape(SQUARE_COUNT, 3)
            boundary, _ = _boundary_chart(chart, side, 1.0e-10)
            branches = pair_witness_state(chart, absolute_smoothing=1.0e-10)
            pair = branches.clearance[np.arange(len(PAIR_I)), owner]
            return np.concatenate((boundary, pair))

        def constraint_jacobian(flat: np.ndarray) -> np.ndarray:
            chart = flat.reshape(SQUARE_COUNT, 3)
            _, boundary_jacobian = _boundary_chart(chart, side, 1.0e-10)
            branches = pair_witness_state(chart, absolute_smoothing=1.0e-10)
            jacobian = np.zeros((len(PAIR_I), POSE_DIMENSION), dtype=np.float64)
            for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
                axis = owner[pair]
                jacobian[pair, 3 * first:3 * first + 2] = (
                    branches.center_i_gradient[pair, axis]
                )
                jacobian[pair, 3 * first + 2] = (
                    branches.theta_i_gradient[pair, axis]
                )
                jacobian[pair, 3 * second:3 * second + 2] = (
                    branches.center_j_gradient[pair, axis]
                )
                jacobian[pair, 3 * second + 2] = (
                    branches.theta_j_gradient[pair, axis]
                )
            return np.vstack((boundary_jacobian, jacobian))

        result = minimize(
            objective,
            current.ravel(),
            method="SLSQP",
            jac=True,
            bounds=Bounds(lower, upper),
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": iterations, "ftol": 1.0e-12, "disp": False},
        )
        current = np.asarray(result.x).reshape(SQUARE_COUNT, 3)
        current[:, 2] = wrap_square_phase(current[:, 2])
        exact = capacity_state(current, side)
        audit_rounds.append({
            "measurement": measurement + 1,
            "solver_success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "minimum_clearance": exact.minimum_clearance,
            "overlap_residual": exact.overlap_residual,
            "owner_changes": int(np.sum(
                np.argmax(
                    pair_witness_state(
                        current, absolute_smoothing=1.0e-10
                    ).clearance,
                    axis=1,
                ) != owner
            )),
        })
        if exact.minimum_clearance >= -1.0e-9:
            break

    exact = capacity_state(current, side)
    return current, {
        "success": bool(exact.minimum_clearance >= -1.0e-8),
        "minimum_clearance": exact.minimum_clearance,
        "overlap_residual": exact.overlap_residual,
        "anchor_distance": float(np.linalg.norm(
            (current - anchor).ravel() * coordinate_scale
        )),
        "rounds": audit_rounds,
    }


def solve_lifted_equilibrium(
    config: LiftedConfig,
    initial_poses: np.ndarray | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    resumed = initial_poses is not None
    if initial_poses is None:
        if config.initialization == "mixed":
            population = initial_configuration_lanes(
                config.lanes, config.start_side, config.seed
            )
        elif config.initialization == "replicated":
            population = replicated_sector_lanes(
                config.lanes, config.start_side, config.seed
            )
        elif config.initialization == "simplex":
            population = equidistant_simplex_lanes(
                config.lanes, config.start_side
            )
        else:
            raise ValueError(
                "initialization must be 'mixed', 'replicated', or 'simplex'"
            )
    else:
        source = np.asarray(initial_poses, dtype=np.float64)
        if source.shape != (SQUARE_COUNT, 3):
            raise ValueError("resume pose chart must have shape (17, 3)")
        population = np.tile(source, (config.lanes, 1, 1))
        for lane in range(1, config.lanes):
            population[lane, :, :2] += rng.normal(
                0.0, config.mutation_translation, (SQUARE_COUNT, 2)
            )
            population[lane, :, 2] = wrap_square_phase(
                population[lane, :, 2]
                + rng.normal(0.0, config.mutation_phase, SQUARE_COUNT)
            )
    dual = np.zeros((config.lanes, CONSTRAINT_COUNT), dtype=np.float64)
    dip_basis = normalized_bruun_dip_basis()
    momentum = np.zeros((config.lanes, 64), dtype=np.float64)
    eigenvalues = np.tile(np.ones(POSE_DIMENSION), (config.lanes, 1))
    eigenvectors = np.tile(np.eye(POSE_DIMENSION), (config.lanes, 1, 1))
    trace: list[dict[str, object]] = []
    lane_projection_info = {
        "effective_support": float(config.lanes),
        "inverse_temperature": 0.0,
    }
    lane_probability = np.full(config.lanes, 1.0 / config.lanes)

    for sweep in range(config.sweeps):
        fraction = sweep / max(config.sweeps - 1, 1)
        shrink_progress = min(fraction / max(config.shrink_fraction, 1.0e-6), 1.0)
        smooth = shrink_progress * shrink_progress * (3.0 - 2.0 * shrink_progress)
        side = config.start_side + smooth * (config.target_side - config.start_side)
        cool_progress = np.clip(
            (fraction - config.cool_start_fraction)
            / max(1.0 - config.cool_start_fraction, 1.0e-6),
            0.0,
            1.0,
        )
        axis_temperature = config.start_axis_temperature * (
            config.end_axis_temperature / config.start_axis_temperature
        ) ** cool_progress
        axis_support = []
        accepted_action = []

        for lane in range(config.lanes):
            chart = lifted_constraint_chart(
                population[lane],
                side,
                axis_temperature,
                absolute_smoothing=config.absolute_smoothing,
            )
            baseline = _constraint_merit(chart, config.target_clearance)
            direction, proposed_dual, values, vectors = equal_action_projection(
                chart, dual[lane], config
            )
            eigenvalues[lane] = values
            eigenvectors[lane] = vectors
            direction = _limit_direction(
                direction, config.translation_trust, config.phase_trust
            )
            coefficient = dip_basis @ pack_pose_direction(direction)
            momentum[lane] = (
                config.momentum * momentum[lane]
                + (1.0 - config.momentum) * coefficient
            )
            transported = unpack_pose_direction(dip_basis.T @ momentum[lane])
            transported = _limit_direction(
                transported, config.translation_trust, config.phase_trust
            )
            candidate, _, action = _accepted_first_arrival_step(
                population[lane],
                transported,
                side,
                axis_temperature,
                config,
                baseline,
            )
            if action == 0.0:
                candidate, _, action = _accepted_first_arrival_step(
                    population[lane],
                    direction,
                    side,
                    axis_temperature,
                    config,
                    baseline,
                )
            population[lane] = candidate
            if action > 0.0:
                dual[lane] = proposed_dual
            else:
                dual[lane] *= 0.85
                momentum[lane] *= 0.5
            axis_support.append(chart.effective_axes)
            accepted_action.append(action)

        minimum, residual = _exact_population_state(population, side)
        if (
            config.posterior_interval > 0
            and (sweep + 1) % config.posterior_interval == 0
        ):
            score_scale = max(float(np.median(residual ** 2)), 1.0e-12)
            lane_probability, lane_projection_info = entropy_project_scores(
                -(residual ** 2) / score_scale,
                max(2.0, config.effective_lane_fraction * config.lanes),
            )
        # Exploration never clones a winning lane or deletes a losing one.
        # If enabled, every complete sector receives its own metric-whitened
        # perturbation while retaining its dual and DIP momentum history.
        if (
            config.mutation_interval > 0
            and (sweep + 1) % config.mutation_interval == 0
            and sweep + 1 < config.sweeps
        ):
            mutation_scale = max(0.08, 1.0 - fraction)
            for lane in range(config.lanes):
                population[lane] += _metric_mutation(
                    eigenvalues[lane],
                    eigenvectors[lane],
                    config,
                    mutation_scale,
                    rng,
                )
                population[lane, :, 2] = wrap_square_phase(
                    population[lane, :, 2]
                )

        if sweep == 0 or (sweep + 1) % 20 == 0 or sweep + 1 == config.sweeps:
            best = int(np.argmin(residual))
            trace.append({
                "sweep": sweep + 1,
                "side": side,
                "axis_temperature": axis_temperature,
                "mean_effective_axes": float(np.mean(axis_support)),
                "mean_accepted_action": float(np.mean(accepted_action)),
                "effective_configuration_lanes": float(
                    lane_projection_info["effective_support"]
                ),
                "best_lane": best,
                "best_minimum_clearance": float(minimum[best]),
                "best_overlap_residual": float(residual[best]),
                "median_overlap_residual": float(np.median(residual)),
                "feasible_lanes": int(np.sum(minimum >= -1.0e-8)),
            })

    final_side = config.target_side
    minimum, residual = _exact_population_state(population, final_side)
    global_order = np.argsort(residual)
    representative = []
    sector_count = min(2 * len(OCCUPANCY_SECTORS), config.lanes)
    for sector in range(sector_count):
        if config.initialization == "replicated" and not resumed:
            members = np.flatnonzero(
                np.arange(config.lanes)
                % (2 * len(OCCUPANCY_SECTORS)) == sector
            )
            representative.append(int(members[np.argmin(residual[members])]))
        else:
            representative.append(sector)
    order = np.asarray(
        (
            representative
            + [int(lane) for lane in global_order if int(lane) not in representative]
            if (
                config.terminal_measurement
                and not resumed
                and config.initialization in ("mixed", "replicated")
            )
            else [int(lane) for lane in global_order]
        ),
        dtype=np.int64,
    )
    emitted = []
    for lane in order[: min(config.emit_lanes, config.lanes)]:
        source = population[int(lane)]
        if not config.terminal_measurement:
            final = source.copy()
            audit = {
                "success": False,
                "message": "terminal measurement explicitly skipped",
            }
        else:
            try:
                polished, audit = polish_pose_capacity_slsqp(source, final_side)
                source_state = capacity_state(source, final_side)
                polished_state = capacity_state(polished, final_side)
                measurement_anchor = (
                    polished
                    if polished_state.overlap_residual < source_state.overlap_residual
                    else source
                )
                measured, measured_audit = polish_measured_axis_capacity_slsqp(
                    measurement_anchor, final_side
                )
                candidates = [source, polished, measured]
                candidate_states = [
                    capacity_state(candidate, final_side) for candidate in candidates
                ]
                best_candidate = int(np.argmin([
                    state.overlap_residual for state in candidate_states
                ]))
                final = legalize_capacity(
                    candidates[best_candidate], final_side, iterations=1200
                )
                audit = dict(audit)
                audit["measured_axis"] = measured_audit
            except RuntimeError as error:
                final = source.copy()
                audit = {"success": False, "message": str(error)}
        state = capacity_state(final, final_side)
        emitted.append({
            "lane": int(lane),
            "sector": (
                (
                    int(lane) % (2 * len(OCCUPANCY_SECTORS))
                    if config.initialization == "replicated"
                    else int(lane)
                )
                if (
                    not resumed
                    and config.initialization in ("mixed", "replicated")
                    and (
                        config.initialization == "replicated"
                        or int(lane) < 2 * len(OCCUPANCY_SECTORS)
                    )
                )
                else (
                    "simplex"
                    if config.initialization == "simplex" and not resumed
                    else "continuous"
                )
            ),
            "posterior_probability": float(lane_probability[int(lane)]),
            "minimum_clearance": state.minimum_clearance,
            "overlap_residual": state.overlap_residual,
            "feasible": state.minimum_clearance >= -1.0e-8,
            "polish_audit": audit,
            "poses": final.tolist(),
        })
    emitted.sort(key=lambda item: (
        float(item["overlap_residual"]), -float(item["minimum_clearance"])
    ))
    return {
        "status": "floating_point_lifted_equilibrium_not_global_proof",
        "method": "night_vision_entropy_x_v3_first_arrival_x_bruun_dip_basis",
        "config": asdict(config),
        "target_side": final_side,
        "resumed_support": resumed,
        "feasible_emissions": sum(bool(item["feasible"]) for item in emitted),
        "best": emitted[0],
        "emissions": emitted,
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=1600)
    parser.add_argument("--start-side", type=float, default=5.0)
    parser.add_argument("--target-side", type=float, default=4.67)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit-lanes", type=int, default=8)
    parser.add_argument("--mutation-interval", type=int, default=20)
    parser.add_argument("--start-axis-temperature", type=float, default=0.18)
    parser.add_argument("--end-axis-temperature", type=float, default=0.0002)
    parser.add_argument("--mutation-translation", type=float, default=0.025)
    parser.add_argument("--mutation-phase", type=float, default=0.030)
    parser.add_argument("--skip-measurement", action="store_true")
    parser.add_argument("--resume-json", type=Path)
    parser.add_argument(
        "--initialization",
        choices=("mixed", "replicated", "simplex"),
        default="mixed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = LiftedConfig(
        lanes=args.lanes,
        sweeps=args.sweeps,
        start_side=args.start_side,
        target_side=args.target_side,
        seed=args.seed,
        emit_lanes=args.emit_lanes,
        mutation_interval=args.mutation_interval,
        start_axis_temperature=args.start_axis_temperature,
        end_axis_temperature=args.end_axis_temperature,
        mutation_translation=args.mutation_translation,
        mutation_phase=args.mutation_phase,
        terminal_measurement=not args.skip_measurement,
        initialization=args.initialization,
    )
    resume_poses = None
    if args.resume_json is not None:
        resume_payload = json.loads(args.resume_json.read_text())
        resume_poses = np.asarray(resume_payload["best"]["poses"], dtype=np.float64)
    result = solve_lifted_equilibrium(config, resume_poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(
            args.svg,
            np.asarray(result["best"]["poses"]),
            float(result["target_side"]),
        )
    print(json.dumps({
        "target_side": result["target_side"],
        "feasible_emissions": result["feasible_emissions"],
        "best_minimum_clearance": result["best"]["minimum_clearance"],
        "best_overlap_residual": result["best"]["overlap_residual"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
