"""Separating-axis capacity geometry for unit-square placement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SQUARE_COUNT = 17
PAIR_I, PAIR_J = np.triu_indices(SQUARE_COUNT, k=1)
QUARTER_TURN = 0.5 * math.pi


def wrap_square_phase(theta: np.ndarray) -> np.ndarray:
    return (theta + 0.25 * math.pi) % QUARTER_TURN - 0.25 * math.pi


def square_axes(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(theta)
    sine = np.sin(theta)
    u = np.stack((cosine, sine), axis=-1)
    return u, np.stack((-sine, cosine), axis=-1)


@dataclass
class CapacityState:
    boundary_clearance: np.ndarray
    pair_clearance: np.ndarray
    pair_axis: np.ndarray
    pair_center_i_gradient: np.ndarray
    pair_center_j_gradient: np.ndarray
    pair_theta_i_gradient: np.ndarray
    pair_theta_j_gradient: np.ndarray

    @property
    def minimum_clearance(self) -> float:
        return float(min(np.min(self.boundary_clearance), np.min(self.pair_clearance)))

    @property
    def overlap_residual(self) -> float:
        values = np.concatenate(
            (self.boundary_clearance.ravel(), self.pair_clearance.ravel())
        )
        return float(np.linalg.norm(np.minimum(values, 0.0)))


@dataclass
class PairWitnessState:
    """All four unoriented separating-axis witnesses for every square pair."""

    clearance: np.ndarray
    center_i_gradient: np.ndarray
    center_j_gradient: np.ndarray
    theta_i_gradient: np.ndarray
    theta_j_gradient: np.ndarray


def pair_witness_state(
    poses: np.ndarray,
    *,
    absolute_smoothing: float = 0.0,
) -> PairWitnessState:
    """Return every SAT branch and its pose derivative without ``argmax``.

    ``absolute_smoothing`` replaces ``abs(x)`` by ``sqrt(x*x + eps*eps)``
    during lifted relaxation.  The ordinary capacity audit keeps epsilon zero
    and remains the exact floating-point separating-axis test.
    """

    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (SQUARE_COUNT, 3):
        raise ValueError(f"poses must have shape ({SQUARE_COUNT}, 3)")
    ui, vi = square_axes(poses[:, 2])
    pair_ui, pair_vi = ui[PAIR_I], vi[PAIR_I]
    pair_uj, pair_vj = ui[PAIR_J], vi[PAIR_J]
    directions = np.stack((pair_ui, pair_vi, pair_uj, pair_vj), axis=1)
    zero = np.zeros_like(pair_ui)
    prime_i = np.stack((pair_vi, -pair_ui, zero, zero), axis=1)
    prime_j = np.stack((zero, zero, pair_vj, -pair_uj), axis=1)
    displacement = poses[PAIR_J, :2] - poses[PAIR_I, :2]
    projected = np.einsum("pd,pkd->pk", displacement, directions, optimize=True)
    direction_projection = np.einsum(
        "pkd,pmd->pkm", directions, directions, optimize=True
    )
    epsilon = max(float(absolute_smoothing), 0.0)
    if epsilon > 0.0:
        projected_absolute = np.sqrt(projected * projected + epsilon * epsilon)
        projection_sign = projected / projected_absolute
        direction_absolute = np.sqrt(
            direction_projection * direction_projection + epsilon * epsilon
        )
        direction_sign = direction_projection / direction_absolute
    else:
        projected_absolute = np.abs(projected)
        projection_sign = np.sign(projected)
        direction_absolute = np.abs(direction_projection)
        direction_sign = np.sign(direction_projection)
    clearance = projected_absolute - 0.5 * np.sum(direction_absolute, axis=2)

    center_j_gradient = projection_sign[:, :, None] * directions
    center_i_gradient = -center_j_gradient
    axis_prime_i = prime_i
    axis_prime_j = prime_j
    displacement_i = projection_sign * np.einsum(
        "pd,pkd->pk", displacement, axis_prime_i, optimize=True
    )
    displacement_j = projection_sign * np.einsum(
        "pd,pkd->pk", displacement, axis_prime_j, optimize=True
    )
    direction_term_i = (
        np.einsum("pmd,pkd->pkm", prime_i, directions, optimize=True)
        + np.einsum("pmd,pkd->pkm", directions, axis_prime_i, optimize=True)
    )
    direction_term_j = (
        np.einsum("pmd,pkd->pkm", prime_j, directions, optimize=True)
        + np.einsum("pmd,pkd->pkm", directions, axis_prime_j, optimize=True)
    )
    theta_i_gradient = displacement_i - 0.5 * np.sum(
        direction_sign * direction_term_i, axis=2
    )
    theta_j_gradient = displacement_j - 0.5 * np.sum(
        direction_sign * direction_term_j, axis=2
    )
    return PairWitnessState(
        clearance=clearance,
        center_i_gradient=center_i_gradient,
        center_j_gradient=center_j_gradient,
        theta_i_gradient=theta_i_gradient,
        theta_j_gradient=theta_j_gradient,
    )


def capacity_state(poses: np.ndarray, side: float) -> CapacityState:
    """Return all exact floating-point SAT capacity clearances and witnesses."""

    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (SQUARE_COUNT, 3):
        raise ValueError(f"poses must have shape ({SQUARE_COUNT}, 3)")
    cosine = np.cos(poses[:, 2])
    sine = np.sin(poses[:, 2])
    half_width = 0.5 * (np.abs(cosine) + np.abs(sine))
    boundary = np.column_stack(
        (
            poses[:, 0] - half_width,
            side - poses[:, 0] - half_width,
            poses[:, 1] - half_width,
            side - poses[:, 1] - half_width,
        )
    )

    witnesses = pair_witness_state(poses)
    candidates = witnesses.clearance
    face = np.argmax(candidates, axis=1)
    rows = np.arange(len(PAIR_I))
    pair_clearance = candidates[rows, face]
    grad_i = witnesses.center_i_gradient[rows, face]
    grad_j = witnesses.center_j_gradient[rows, face]
    grad_theta_i = witnesses.theta_i_gradient[rows, face]
    grad_theta_j = witnesses.theta_j_gradient[rows, face]
    pair_axis = grad_i.copy()
    return CapacityState(
        boundary_clearance=boundary,
        pair_clearance=pair_clearance,
        pair_axis=pair_axis,
        pair_center_i_gradient=grad_i,
        pair_center_j_gradient=grad_j,
        pair_theta_i_gradient=grad_theta_i,
        pair_theta_j_gradient=grad_theta_j,
    )


def capacity_loss_gradient(
    poses: np.ndarray,
    side: float,
    *,
    target_clearance: float = 2.0e-4,
    temperature: float = 0.012,
) -> tuple[float, np.ndarray, CapacityState]:
    """Smooth capacity residual and its active-witness gradient."""

    poses = np.asarray(poses, dtype=np.float64)
    state = capacity_state(poses, side)
    gradient = np.zeros_like(poses)

    z_boundary = np.clip(
        (target_clearance - state.boundary_clearance) / temperature, -60.0, 60.0
    )
    residual_boundary = temperature * (
        np.maximum(z_boundary, 0.0) + np.log1p(np.exp(-np.abs(z_boundary)))
    )
    gate_boundary = 1.0 / (1.0 + np.exp(-z_boundary))
    weight_boundary = -residual_boundary * gate_boundary
    gradient[:, 0] += weight_boundary[:, 0] - weight_boundary[:, 1]
    gradient[:, 1] += weight_boundary[:, 2] - weight_boundary[:, 3]
    cosine = np.cos(poses[:, 2])
    sine = np.sin(poses[:, 2])
    half_width_prime = 0.5 * (
        -np.sign(cosine) * sine + np.sign(sine) * cosine
    )
    gradient[:, 2] -= np.sum(weight_boundary, axis=1) * half_width_prime

    z_pair = np.clip(
        (target_clearance - state.pair_clearance) / temperature, -60.0, 60.0
    )
    residual_pair = temperature * (
        np.maximum(z_pair, 0.0) + np.log1p(np.exp(-np.abs(z_pair)))
    )
    gate_pair = 1.0 / (1.0 + np.exp(-z_pair))
    weight_pair = -residual_pair * gate_pair
    for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
        weight = weight_pair[pair]
        gradient[first, :2] += weight * state.pair_center_i_gradient[pair]
        gradient[second, :2] += weight * state.pair_center_j_gradient[pair]
        gradient[first, 2] += weight * state.pair_theta_i_gradient[pair]
        gradient[second, 2] += weight * state.pair_theta_j_gradient[pair]
    loss = 0.5 * float(
        np.sum(residual_boundary**2) + np.sum(residual_pair**2)
    )
    return loss, gradient, state


def square_corners(pose: np.ndarray) -> np.ndarray:
    u, v = square_axes(np.asarray(pose[2]))
    return np.asarray(
        [
            pose[:2] - 0.5 * u - 0.5 * v,
            pose[:2] + 0.5 * u - 0.5 * v,
            pose[:2] + 0.5 * u + 0.5 * v,
            pose[:2] - 0.5 * u + 0.5 * v,
        ]
    )
