#!/usr/bin/env python3
"""Exact small-alphabet control for the 17-body BFFT transport.

This module deliberately does *not* evolve a Euclidean pose chart.  Let
``P`` be a finite alphabet of poses for one unit square.  The transported
state is a strictly positive amplitude on the full product space

    X = P_0 x P_1 x ... x P_16.

For the binary control used here, ``|X| = 2**17``.  Every configuration is a
coordinate vector in ``R**|X|``; distinct configurations are therefore
exactly equidistant.  One imaginary-time step is

    psi <- D_V H_0 H_1 ... H_16 D_V psi,

where ``D_V`` is the diagonal exponential of the *physical intersection
area* and each ``H_i`` is the same positive two-point heat kernel acting on
particle axis ``i``.  No separating-axis clearance enters the evolution.

The product kernel is strictly positive when ``0 < mixing < 1``.  Its
normalized action is consequently a Birkhoff/Banach contraction on the
positive projective cone in Hilbert's metric.  The unique Perron fixed point
is relaxed before one terminal Euclidean measurement (the largest-amplitude
configuration).  At increasing inverse action it concentrates on the global
minimum of the supplied discrete pose alphabet rather than a local minimum.

This exact control validates the representation and contraction.  It is not
the production solver: a useful pose alphabet needs a tensor-train version of
the identical 17-axis operator rather than the binary dense vector used here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from floorplan_banach import _clip_convex, _polygon_area
from geometry import SQUARE_COUNT, capacity_state, square_corners


STATE_COUNT = 1 << SQUARE_COUNT


@dataclass(frozen=True)
class ConfigurationTransportConfig:
    side: float = 4.8
    seed: int = 0
    mixing: float = 0.08
    inverse_actions: tuple[float, ...] = (
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    )
    iterations_per_action: int = 160
    tolerance: float = 1.0e-13


def _fractional(value: np.ndarray) -> np.ndarray:
    return value - np.floor(value)


def binary_pose_alphabet(side: float, seed: int = 0) -> np.ndarray:
    """Return two deterministic, wall-feasible poses for each particle.

    The alphabet is intentionally generic and carries no published packing.
    It is only large enough to exercise all ``2**17`` joint configurations.
    """

    particle = np.arange(1, SQUARE_COUNT + 1, dtype=np.float64)[:, None]
    branch = np.arange(2, dtype=np.float64)[None, :]
    offset = 0.13750352375 * float(seed + 1)
    x_phase = _fractional(
        offset
        + particle * ((math.sqrt(5.0) - 1.0) / 2.0)
        + branch * (math.sqrt(3.0) - 1.0)
    )
    y_phase = _fractional(
        0.5 * offset
        + particle * (math.sqrt(2.0) - 1.0)
        + branch * (math.sqrt(7.0) - 2.0)
    )
    angle_phase = _fractional(
        particle * (math.sqrt(11.0) - 3.0)
        + branch * (math.sqrt(13.0) - 3.0)
        + offset
    )
    theta = (angle_phase - 0.5) * (0.5 * math.pi)
    half_extent = 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    x = half_extent + (side - 2.0 * half_extent) * x_phase
    y = half_extent + (side - 2.0 * half_extent) * y_phase
    return np.stack((x, y, theta), axis=2)


def pair_area_kernel(alphabet: np.ndarray) -> np.ndarray:
    """Exact physical intersection areas for every particle/branch pair."""

    poses = np.asarray(alphabet, dtype=np.float64)
    if poses.shape != (SQUARE_COUNT, 2, 3):
        raise ValueError("binary alphabet must have shape (17, 2, 3)")
    corners = [
        [square_corners(poses[particle, branch]) for branch in range(2)]
        for particle in range(SQUARE_COUNT)
    ]
    kernel = np.zeros((SQUARE_COUNT, SQUARE_COUNT, 2, 2), dtype=np.float64)
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            for first_branch in range(2):
                for second_branch in range(2):
                    area = _polygon_area(
                        _clip_convex(
                            corners[first][first_branch],
                            corners[second][second_branch],
                        )
                    )
                    kernel[first, second, first_branch, second_branch] = area
                    kernel[second, first, second_branch, first_branch] = area
    return kernel


def configuration_bits() -> np.ndarray:
    """The complete binary product basis, with one column per square."""

    index = np.arange(STATE_COUNT, dtype=np.uint32)[:, None]
    axis = np.arange(SQUARE_COUNT, dtype=np.uint32)[None, :]
    return ((index >> axis) & 1).astype(np.uint8)


def configuration_area_energy(kernel: np.ndarray) -> np.ndarray:
    """Evaluate physical overlap area on every one of the ``2**17`` states."""

    pair = np.asarray(kernel, dtype=np.float64)
    if pair.shape != (SQUARE_COUNT, SQUARE_COUNT, 2, 2):
        raise ValueError("pair kernel has the wrong shape")
    bits = configuration_bits()
    energy = np.zeros(STATE_COUNT, dtype=np.float64)
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            energy += pair[
                first,
                second,
                bits[:, first],
                bits[:, second],
            ]
    return energy


def apply_product_heat(amplitude: np.ndarray, mixing: float) -> np.ndarray:
    """Apply the tensor product of 17 identical positive heat kernels."""

    value = np.asarray(amplitude, dtype=np.float64).copy()
    if value.shape != (STATE_COUNT,):
        raise ValueError("amplitude must have length 2**17")
    eta = float(mixing)
    if not 0.0 < eta < 1.0:
        raise ValueError("mixing must lie strictly between zero and one")
    stay = 1.0 - eta
    for axis in range(SQUARE_COUNT):
        span = 1 << axis
        block = value.reshape(-1, 2, span)
        low = block[:, 0, :].copy()
        high = block[:, 1, :].copy()
        block[:, 0, :] = stay * low + eta * high
        block[:, 1, :] = eta * low + stay * high
    return value


def transfer_step(
    amplitude: np.ndarray,
    energy: np.ndarray,
    mixing: float,
    inverse_action: float,
) -> np.ndarray:
    """One positive Strang-split imaginary-time transfer and L1 gauge fix."""

    value = np.asarray(amplitude, dtype=np.float64)
    potential = np.asarray(energy, dtype=np.float64)
    if value.shape != (STATE_COUNT,) or potential.shape != (STATE_COUNT,):
        raise ValueError("amplitude and energy must have length 2**17")
    shifted = potential - float(np.min(potential))
    diagonal = np.exp(
        np.clip(-0.5 * float(inverse_action) * shifted, -700.0, 0.0)
    )
    transported = diagonal * value
    transported = apply_product_heat(transported, mixing)
    transported *= diagonal
    total = float(np.sum(transported))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("positive transfer lost its normalization")
    return transported / total


def hilbert_projective_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Hilbert distance on the interior of the finite positive cone."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("Hilbert distance requires equal strictly positive arrays")
    log_ratio = np.log(a) - np.log(b)
    return float(np.max(log_ratio) - np.min(log_ratio))


def fixed_point(
    initial: np.ndarray,
    energy: np.ndarray,
    mixing: float,
    inverse_action: float,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    """Relax the positive transfer to its unique projective fixed point."""

    amplitude = np.asarray(initial, dtype=np.float64).copy()
    amplitude = np.maximum(amplitude, np.finfo(np.float64).tiny)
    amplitude /= np.sum(amplitude)
    residual = math.inf
    completed = 0
    for iteration in range(max(int(iterations), 1)):
        updated = transfer_step(amplitude, energy, mixing, inverse_action)
        residual = float(np.sum(np.abs(updated - amplitude)))
        amplitude = updated
        completed = iteration + 1
        if residual <= float(tolerance):
            break
    return amplitude, {
        "iterations": completed,
        "l1_fixed_point_residual": residual,
        "inverse_action": float(inverse_action),
    }


def poses_from_state(alphabet: np.ndarray, state_index: int) -> np.ndarray:
    """One terminal measurement from a product-basis index to Euclidean poses."""

    branch = ((int(state_index) >> np.arange(SQUARE_COUNT)) & 1).astype(int)
    return np.asarray(alphabet, dtype=np.float64)[
        np.arange(SQUARE_COUNT), branch
    ]


def solve_configuration_transport(
    config: ConfigurationTransportConfig,
    *,
    alphabet: np.ndarray | None = None,
) -> dict:
    """Run the exact binary 17-body transport and terminal measurement."""

    poses = (
        binary_pose_alphabet(config.side, config.seed)
        if alphabet is None
        else np.asarray(alphabet, dtype=np.float64)
    )
    kernel = pair_area_kernel(poses)
    energy = configuration_area_energy(kernel)
    minimum_index = int(np.argmin(energy))
    minimum_energy = float(energy[minimum_index])

    amplitude = np.full(STATE_COUNT, 1.0 / STATE_COUNT, dtype=np.float64)
    stages = []
    for inverse_action in config.inverse_actions:
        amplitude, record = fixed_point(
            amplitude,
            energy,
            config.mixing,
            inverse_action,
            config.iterations_per_action,
            config.tolerance,
        )
        measurement = int(np.argmax(amplitude))
        record.update(
            {
                "measured_state": measurement,
                "measured_area_energy": float(energy[measurement]),
                "minimum_state_probability": float(amplitude[minimum_index]),
                "maximum_probability": float(amplitude[measurement]),
                "effective_state_count": float(
                    1.0 / np.sum(np.square(amplitude))
                ),
            }
        )
        stages.append(record)

    measured_index = int(np.argmax(amplitude))
    measured_poses = poses_from_state(poses, measured_index)
    audit = capacity_state(measured_poses, config.side)

    rng = np.random.default_rng(config.seed + 781)
    first = rng.uniform(0.5, 1.5, STATE_COUNT)
    second = rng.uniform(0.5, 1.5, STATE_COUNT)
    first /= np.sum(first)
    second /= np.sum(second)
    control_action = float(config.inverse_actions[0])
    distance_before = hilbert_projective_distance(first, second)
    image_first = transfer_step(
        first, energy, config.mixing, control_action
    )
    image_second = transfer_step(
        second, energy, config.mixing, control_action
    )
    distance_after = hilbert_projective_distance(image_first, image_second)

    return {
        "method": "exact_17_body_projective_transport_binary_control",
        "config": asdict(config),
        "transport_axes": SQUARE_COUNT,
        "product_basis_states": STATE_COUNT,
        "distinct_basis_distance": math.sqrt(2.0),
        "potential": "exact_pairwise_polygon_intersection_area",
        "minimum_discrete_area_energy": minimum_energy,
        "minimum_discrete_state": minimum_index,
        "measured_area_energy": float(energy[measured_index]),
        "measured_state": measured_index,
        "measured_matches_global_discrete_minimum": bool(
            abs(float(energy[measured_index]) - minimum_energy) <= 1.0e-13
        ),
        "projective_distance_before": distance_before,
        "projective_distance_after": distance_after,
        "observed_projective_ratio": distance_after / distance_before,
        "terminal_sat_audit": {
            "minimum_clearance": float(audit.minimum_clearance),
            "overlap_residual": float(audit.overlap_residual),
            "worst_penetration": max(-float(audit.minimum_clearance), 0.0),
        },
        "poses": measured_poses.tolist(),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=4.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixing", type=float, default=0.08)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = ConfigurationTransportConfig(
        side=args.side,
        seed=args.seed,
        mixing=args.mixing,
        iterations_per_action=args.iterations,
    )
    result = solve_configuration_transport(config)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.svg is not None:
        write_svg(args.svg, np.asarray(result["poses"]), config.side)


if __name__ == "__main__":
    main()
