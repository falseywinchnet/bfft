#!/usr/bin/env python3
"""Bruun/DIP packet transport over seventeen persistent square identities.

The exact occupation control proves the projective contraction but a small
global point alphabet has inadequate pose resolution.  This experiment gives
each of seventeen persistent transport identities its own one-particle
Bruun/DIP--Zak packet chart.  For ``K`` packets, the implicit basis contains
``K**17`` complete configurations without the ``17!`` labelled permutation
copies that defeated the earlier generic tensor-train ablation.

Each packet is a simultaneous ``(x, y, theta)`` displacement obtained by
restricting three physical phase carriers through the normalized intermediate
Bruun/DIP operator.  Evolution is a positive imaginary-time transfer:

* exact polygon intersection area supplies every two-particle diagonal gate;
* exact escaped polygon area supplies the one-particle wall gate;
* a strictly positive one-particle heat kernel transports every packet axis;
* an odd/even swap network applies every pair gate twice in Strang order; and
* a matrix-product state carries the full joint amplitude.

The uncompressed transfer is the same positive projective contraction derived
in ``TRANSPORT_FORMULATION.md``.  SVD restriction makes this implementation an
approximation, so discarded Schmidt weight and fixed-point distance are
reported.  No clearance or decoded Euclidean score enters evolution.  One
conditional Born mode measurement emits the terminal Euclidean chart, after
which SAT is used only as an independent feasibility audit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from floorplan_banach import _clip_convex, _polygon_area, physical_area_energy
from geometry import SQUARE_COUNT, capacity_state, square_corners, wrap_square_phase
from lifted_equilibrium import normalized_bruun_dip_basis
from reference_chart import REFERENCE_SIDE, reference_chart
from tensor_transport import (
    apply_one_site_gate,
    interaction_swap,
    measure_conditional_modes,
    mps_norm_squared,
    normalize_mps,
    pose_heat_kernel,
    uniform_mps,
)


@dataclass(frozen=True)
class SpectralPoseTransportConfig:
    side: float = REFERENCE_SIDE
    packet_count: int = 16
    dip_level: int = 2
    rank: int = 32
    seed: int = 0
    translation_radius: float = 0.08
    phase_radius: float = 0.08
    heat_scale: float = 0.12
    heat_floor: float = 1.0e-9
    inverse_actions: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    iterations_per_action: int = 4
    tolerance: float = 1.0e-8
    base_chart: str = "scaled_reference"


def scaled_reference_chart(side: float) -> np.ndarray:
    """Map boundary-support coordinates to a new side without pose descent."""

    chart = reference_chart()
    if float(side) <= 1.0:
        raise ValueError("container side must exceed one")
    scale = (float(side) - 1.0) / (REFERENCE_SIDE - 1.0)
    chart[:, :2] = 0.5 + scale * (chart[:, :2] - 0.5)
    return chart


def dip_packet_coordinates(packet_count: int, level: int) -> np.ndarray:
    """Restrict three pose-phase carriers through one normalized DIP rung."""

    count = int(packet_count)
    if count < 4 or count & (count - 1):
        raise ValueError("packet_count must be a power of two at least four")
    basis = normalized_bruun_dip_basis(count, int(level))
    phase = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    carriers = np.column_stack(
        (
            np.cos(phase),
            np.sin(phase),
            np.sin(2.0 * phase + math.pi / 8.0),
        )
    )
    packets = basis @ carriers
    packets -= np.mean(packets, axis=0, keepdims=True)
    covariance = packets.T @ packets / count
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    inverse_root = eigenvector @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalue, 1.0e-12))
    ) @ eigenvector.T
    packets = packets @ inverse_root
    maximum = np.max(np.abs(packets), axis=0)
    packets /= np.maximum(maximum, 1.0e-12)
    # The DC packet is the exact source chart.  The other kets retain the DIP
    # packet directions; orthogonality belongs to the ket basis, not to their
    # three-dimensional Euclidean readout.
    packets[0] = 0.0
    return packets


def spectral_pose_packets(
    base: np.ndarray,
    config: SpectralPoseTransportConfig,
) -> np.ndarray:
    """One DIP packet chart per persistent square identity."""

    chart = np.asarray(base, dtype=np.float64)
    if chart.shape != (SQUARE_COUNT, 3):
        raise ValueError("base chart must have shape (17, 3)")
    packet = dip_packet_coordinates(config.packet_count, config.dip_level)
    result = np.empty(
        (SQUARE_COUNT, config.packet_count, 3), dtype=np.float64
    )
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for square in range(SQUARE_COUNT):
        # A deterministic U(1) chart connection rotates the same spectral
        # packet family for every identity without changing its radius.
        angle = 2.0 * math.pi * ((square + 1) * golden + 0.137 * config.seed)
        cosine, sine = math.cos(angle), math.sin(angle)
        dx = cosine * packet[:, 0] - sine * packet[:, 1]
        dy = sine * packet[:, 0] + cosine * packet[:, 1]
        result[square, :, 0] = (
            chart[square, 0] + config.translation_radius * dx
        )
        result[square, :, 1] = (
            chart[square, 1] + config.translation_radius * dy
        )
        result[square, :, 2] = wrap_square_phase(
            chart[square, 2]
            + config.phase_radius
            * (
                math.cos(angle) * packet[:, 2]
                + math.sin(angle) * packet[:, 0]
            )
        )
    return result


def spectral_physical_energies(
    packets: np.ndarray,
    side: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact wall and pair polygon-area energies in packet coordinates."""

    poses = np.asarray(packets, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[0] != SQUARE_COUNT or poses.shape[2] != 3:
        raise ValueError("packets must have shape (17, K, 3)")
    count = poses.shape[1]
    corners = [
        [square_corners(poses[square, packet]) for packet in range(count)]
        for square in range(SQUARE_COUNT)
    ]
    container = np.asarray(
        ((0.0, 0.0), (side, 0.0), (side, side), (0.0, side)),
        dtype=np.float64,
    )
    wall = np.empty((SQUARE_COUNT, count), dtype=np.float64)
    for square in range(SQUARE_COUNT):
        for packet in range(count):
            contained = _polygon_area(
                _clip_convex(corners[square][packet], container)
            )
            wall[square, packet] = max(1.0 - contained, 0.0)

    pair = np.zeros(
        (SQUARE_COUNT, SQUARE_COUNT, count, count), dtype=np.float64
    )
    for first in range(SQUARE_COUNT):
        for second in range(first + 1, SQUARE_COUNT):
            for first_packet in range(count):
                for second_packet in range(count):
                    area = _polygon_area(
                        _clip_convex(
                            corners[first][first_packet],
                            corners[second][second_packet],
                        )
                    )
                    pair[first, second, first_packet, second_packet] = area
                    pair[second, first, second_packet, first_packet] = area
    return wall, pair


def spectral_heat_kernels(
    packets: np.ndarray,
    config: SpectralPoseTransportConfig,
) -> np.ndarray:
    return np.asarray(
        [
            pose_heat_kernel(
                packets[square], config.heat_scale, config.heat_floor
            )
            for square in range(SQUARE_COUNT)
        ]
    )


def mps_overlap(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    environment = np.ones((1, 1), dtype=np.float64)
    for a, b in zip(first, second):
        environment = np.einsum(
            "ij,iar,jas->rs", environment, a, b, optimize=True
        )
    return float(environment[0, 0])


def apply_pair_half(
    tensors: list[np.ndarray],
    order: list[int],
    pair_energy: np.ndarray,
    inverse_action: float,
    maximum_rank: int,
) -> dict:
    """Apply all pair potentials once while reversing physical identity order."""

    discarded = []
    encountered = []
    for layer in range(len(tensors)):
        for position in range(layer % 2, len(tensors) - 1, 2):
            first_identity = order[position]
            second_identity = order[position + 1]
            gate = np.exp(
                np.clip(
                    -0.5
                    * float(inverse_action)
                    * pair_energy[first_identity, second_identity],
                    -700.0,
                    0.0,
                )
            )
            left, right, record = interaction_swap(
                tensors[position],
                tensors[position + 1],
                gate,
                maximum_rank,
            )
            tensors[position] = left
            tensors[position + 1] = right
            order[position], order[position + 1] = (
                second_identity,
                first_identity,
            )
            encountered.append(tuple(sorted((first_identity, second_identity))))
            discarded.append(record["discarded_fraction"])
    pair_count = len(tensors) * (len(tensors) - 1) // 2
    if len(encountered) != pair_count or len(set(encountered)) != pair_count:
        raise AssertionError("identity swap network missed a pair")
    return {
        "maximum_discarded_fraction": float(max(discarded, default=0.0)),
        "sum_discarded_fraction": float(sum(discarded)),
        "maximum_bond_rank": int(max(tensor.shape[2] for tensor in tensors)),
    }


def spectral_transfer(
    tensors: list[np.ndarray],
    wall_energy: np.ndarray,
    pair_energy: np.ndarray,
    heat: np.ndarray,
    inverse_action: float,
    maximum_rank: int,
) -> dict:
    """One compressed positive transfer over all packet directions at once."""

    order = list(range(SQUARE_COUNT))
    for position, identity in enumerate(order):
        wall_gate = np.diag(
            np.exp(
                np.clip(
                    -0.5 * inverse_action * wall_energy[identity],
                    -700.0,
                    0.0,
                )
            )
        )
        tensors[position] = apply_one_site_gate(tensors[position], wall_gate)
    first = apply_pair_half(
        tensors, order, pair_energy, inverse_action, maximum_rank
    )

    for position, identity in enumerate(order):
        tensors[position] = apply_one_site_gate(
            tensors[position], heat[identity]
        )

    second = apply_pair_half(
        tensors, order, pair_energy, inverse_action, maximum_rank
    )
    if order != list(range(SQUARE_COUNT)):
        raise AssertionError("two pair networks did not restore identity order")
    for position, identity in enumerate(order):
        wall_gate = np.diag(
            np.exp(
                np.clip(
                    -0.5 * inverse_action * wall_energy[identity],
                    -700.0,
                    0.0,
                )
            )
        )
        tensors[position] = apply_one_site_gate(tensors[position], wall_gate)
    normalize_mps(tensors)
    return {
        "maximum_discarded_fraction": max(
            first["maximum_discarded_fraction"],
            second["maximum_discarded_fraction"],
        ),
        "sum_discarded_fraction": (
            first["sum_discarded_fraction"]
            + second["sum_discarded_fraction"]
        ),
        "maximum_bond_rank": max(
            first["maximum_bond_rank"], second["maximum_bond_rank"]
        ),
    }


def solve_spectral_pose_transport(
    config: SpectralPoseTransportConfig,
    *,
    base: np.ndarray | None = None,
) -> dict:
    if base is None:
        if config.base_chart != "scaled_reference":
            raise ValueError("only scaled_reference is currently implemented")
        source = scaled_reference_chart(config.side)
    else:
        source = np.asarray(base, dtype=np.float64)
    packets = spectral_pose_packets(source, config)
    wall, pair = spectral_physical_energies(packets, config.side)
    heat = spectral_heat_kernels(packets, config)
    tensors = uniform_mps(SQUARE_COUNT, config.packet_count)
    normalize_mps(tensors)
    stages = []
    for inverse_action in config.inverse_actions:
        records = []
        fixed_point_distance = math.inf
        for iteration in range(max(int(config.iterations_per_action), 1)):
            previous = [tensor.copy() for tensor in tensors]
            record = spectral_transfer(
                tensors,
                wall,
                pair,
                heat,
                inverse_action,
                config.rank,
            )
            overlap = abs(mps_overlap(previous, tensors))
            previous_norm = math.sqrt(max(mps_norm_squared(previous), 1.0e-300))
            current_norm = math.sqrt(max(mps_norm_squared(tensors), 1.0e-300))
            fidelity = min(overlap / (previous_norm * current_norm), 1.0)
            fixed_point_distance = math.sqrt(max(2.0 - 2.0 * fidelity, 0.0))
            record["fixed_point_distance"] = fixed_point_distance
            records.append(record)
            if fixed_point_distance <= config.tolerance:
                break
        indices, measurement = measure_conditional_modes(tensors)
        selected = packets[np.arange(SQUARE_COUNT), indices]
        area, _, overlap_area, wall_area = physical_area_energy(
            selected, config.side, 1.0
        )
        stages.append(
            {
                "inverse_action": float(inverse_action),
                "iterations": len(records),
                "fixed_point_distance": fixed_point_distance,
                "measured_area_energy": float(area),
                "measured_overlap_area": float(overlap_area),
                "measured_wall_area": float(wall_area),
                "maximum_discarded_fraction": float(
                    max(record["maximum_discarded_fraction"] for record in records)
                ),
                "sum_discarded_fraction": float(
                    sum(record["sum_discarded_fraction"] for record in records)
                ),
                "maximum_bond_rank": int(
                    max(record["maximum_bond_rank"] for record in records)
                ),
            }
        )

    indices, measurement = measure_conditional_modes(tensors)
    selected = packets[np.arange(SQUARE_COUNT), indices]
    area, _, overlap_area, wall_area = physical_area_energy(
        selected, config.side, 1.0
    )
    audit = capacity_state(selected, config.side)
    return {
        "method": "bruun_dip_packet_17_identity_transport",
        "config": asdict(config),
        "transport_identities": SQUARE_COUNT,
        "packets_per_identity": config.packet_count,
        "implicit_equidistant_configurations": str(
            config.packet_count ** SQUARE_COUNT
        ),
        "uncompressed_operator_is_projective_contraction": True,
        "compressed_operator_contraction_claimed": False,
        "physical_area_energy": float(area),
        "physical_overlap_area": float(overlap_area),
        "physical_wall_area": float(wall_area),
        "selected_packets": indices.tolist(),
        "terminal_sat_audit": {
            "minimum_clearance": float(audit.minimum_clearance),
            "overlap_residual": float(audit.overlap_residual),
            "worst_penetration": max(-float(audit.minimum_clearance), 0.0),
        },
        "measurement": measurement,
        "poses": selected.tolist(),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=REFERENCE_SIDE)
    parser.add_argument("--packets", type=int, default=16)
    parser.add_argument("--dip-level", type=int, default=2)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translation-radius", type=float, default=0.08)
    parser.add_argument("--phase-radius", type=float, default=0.08)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = SpectralPoseTransportConfig(
        side=args.side,
        packet_count=args.packets,
        dip_level=args.dip_level,
        rank=args.rank,
        seed=args.seed,
        translation_radius=args.translation_radius,
        phase_radius=args.phase_radius,
        iterations_per_action=args.iterations,
    )
    result = solve_spectral_pose_transport(config)
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
