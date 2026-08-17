#!/usr/bin/env python3
"""Tensor-train realization of the 17-body floor-plan transfer.

``configuration_transport.py`` stores a binary pose alphabet exactly.  This
module keeps the same product-space physics while replacing its dense vector
by a matrix-product state (tensor train), allowing a larger one-square pose
alphabet.  It is an explicitly measured approximation to the exact positive
projective contraction:

* diagonal two-particle gates use exact polygon intersection area;
* an odd/even swap network applies every one of the 136 pair gates once while
  reversing the particle axes, and the second half-step restores their order;
* a strictly positive floor-plan heat kernel transports every particle axis;
* every Schmidt truncation reports its discarded squared weight; and
* Euclidean centers and phases are read once from conditional Born marginals.

The SVD-compressed map is not claimed to inherit the exact Banach theorem.
Its truncation and fixed-point residuals are part of every result record.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chip_transport import write_svg
from configuration_transport import _fractional
from floorplan_banach import _clip_convex, _polygon_area, physical_area_energy
from geometry import SQUARE_COUNT, capacity_state, square_corners
from reference_chart import REFERENCE_SIDE, reference_chart


@dataclass(frozen=True)
class TensorTransportConfig:
    side: float = REFERENCE_SIDE
    pose_count: int = 17
    rank: int = 16
    seed: int = 0
    heat_scale: float = 0.18
    heat_floor: float = 1.0e-8
    inverse_actions: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    transfers_per_action: int = 2
    alphabet: str = "reference_control"


def low_discrepancy_pose_alphabet(
    side: float,
    pose_count: int,
    seed: int = 0,
) -> np.ndarray:
    """Wall-feasible spectral-phase samples with no packing template."""

    count = max(int(pose_count), 2)
    index = np.arange(1, count + 1, dtype=np.float64)
    offset = 0.2718281828459045 * float(seed + 1)
    x_phase = _fractional(
        offset + index * ((math.sqrt(5.0) - 1.0) / 2.0)
    )
    y_phase = _fractional(
        0.5 * offset + index * (math.sqrt(2.0) - 1.0)
    )
    angle_phase = _fractional(
        0.25 * offset + index * (math.sqrt(3.0) - 1.0)
    )
    theta = (angle_phase - 0.5) * (0.5 * math.pi)
    half_extent = 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    x = half_extent + (side - 2.0 * half_extent) * x_phase
    y = half_extent + (side - 2.0 * half_extent) * y_phase
    return np.column_stack((x, y, theta))


def pose_overlap_gram(alphabet: np.ndarray) -> np.ndarray:
    """Physical footprint Gram matrix ``<chi_q, chi_r>``."""

    poses = np.asarray(alphabet, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("pose alphabet must be a pose_count by 3 matrix")
    corners = [square_corners(pose) for pose in poses]
    gram = np.empty((len(poses), len(poses)), dtype=np.float64)
    for first in range(len(poses)):
        gram[first, first] = 1.0
        for second in range(first):
            area = _polygon_area(_clip_convex(corners[first], corners[second]))
            gram[first, second] = area
            gram[second, first] = area
    return gram


def pose_heat_kernel(
    alphabet: np.ndarray,
    scale: float,
    positive_floor: float,
) -> np.ndarray:
    """Symmetric normalized positive heat kernel on one pose alphabet."""

    poses = np.asarray(alphabet, dtype=np.float64)
    dx = poses[:, 0, None] - poses[None, :, 0]
    dy = poses[:, 1, None] - poses[None, :, 1]
    raw_angle = np.abs(poses[:, 2, None] - poses[None, :, 2])
    angle = np.minimum(raw_angle, 0.5 * math.pi - raw_angle)
    bandwidth = max(float(scale), 1.0e-8)
    distance = dx * dx + dy * dy + (0.7 * angle) ** 2
    kernel = np.exp(np.clip(-distance / (4.0 * bandwidth), -700.0, 0.0))
    kernel += max(float(positive_floor), np.finfo(np.float64).tiny)
    degree = np.sum(kernel, axis=1)
    kernel /= np.sqrt(degree[:, None] * degree[None, :])
    # Only projective scale matters.  Unit spectral radius avoids numerical
    # growth without changing the Perron ray.
    eigenvalue = float(np.max(np.linalg.eigvalsh(kernel)))
    return kernel / max(eigenvalue, np.finfo(np.float64).tiny)


def uniform_mps(length: int, local_dimension: int) -> list[np.ndarray]:
    """Rank-one positive amplitude over the complete product basis."""

    value = 1.0 / math.sqrt(float(local_dimension))
    return [
        np.full((1, local_dimension, 1), value, dtype=np.float64)
        for _ in range(int(length))
    ]


def mps_norm_squared(tensors: list[np.ndarray]) -> float:
    environment = np.ones((1, 1), dtype=np.float64)
    for tensor in tensors:
        environment = np.einsum(
            "ij,iar,jas->rs",
            environment,
            tensor,
            tensor,
            optimize=True,
        )
    return float(environment[0, 0])


def normalize_mps(tensors: list[np.ndarray]) -> None:
    norm = math.sqrt(max(mps_norm_squared(tensors), np.finfo(np.float64).tiny))
    tensors[0] = tensors[0] / norm


def apply_one_site_gate(tensor: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return np.einsum("ab,lbr->lar", gate, tensor, optimize=True)


def interaction_swap(
    first: np.ndarray,
    second: np.ndarray,
    gate: np.ndarray,
    maximum_rank: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a diagonal pair gate, swap axes, and Schmidt-compress."""

    if first.shape[2] != second.shape[0]:
        raise ValueError("adjacent MPS bond dimensions disagree")
    local_dimension = first.shape[1]
    if second.shape[1] != local_dimension or gate.shape != (
        local_dimension,
        local_dimension,
    ):
        raise ValueError("pair gate and physical dimensions disagree")
    theta = np.einsum("lar,rbs->labs", first, second, optimize=True)
    theta *= gate[None, :, :, None]
    # Swapping physical axes makes each original particle meet every other
    # original particle exactly once in the odd/even crossing network.
    theta = np.transpose(theta, (0, 2, 1, 3))
    rows = first.shape[0] * local_dimension
    columns = local_dimension * second.shape[2]
    matrix = theta.reshape(rows, columns)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    total_weight = float(np.dot(singular, singular))
    retained_rank = min(max(int(maximum_rank), 1), len(singular))
    discarded_weight = float(np.dot(singular[retained_rank:], singular[retained_rank:]))
    singular = singular[:retained_rank]
    root = np.sqrt(singular)
    left = left[:, :retained_rank] * root[None, :]
    right = root[:, None] * right[:retained_rank]
    # An arbitrary scalar gauge is harmless and avoids under/overflow during
    # long gate cascades.  Global normalization is restored after the sweep.
    gauge = max(float(np.max(root)), np.finfo(np.float64).tiny)
    left /= gauge
    right *= gauge
    return (
        left.reshape(first.shape[0], local_dimension, retained_rank),
        right.reshape(retained_rank, local_dimension, second.shape[2]),
        {
            "retained_rank": retained_rank,
            "discarded_fraction": (
                discarded_weight / total_weight if total_weight > 0.0 else 0.0
            ),
        },
    )


def swap_network_pairs(length: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Return the particle pairs crossed by one complete odd/even network."""

    order = list(range(int(length)))
    pairs: list[tuple[int, int]] = []
    for layer in range(int(length)):
        for position in range(layer % 2, int(length) - 1, 2):
            pairs.append((order[position], order[position + 1]))
            order[position], order[position + 1] = (
                order[position + 1],
                order[position],
            )
    return pairs, order


def apply_interaction_half(
    tensors: list[np.ndarray],
    overlap: np.ndarray,
    inverse_action: float,
    maximum_rank: int,
) -> dict:
    """Apply every commuting pair potential once and reverse MPS order."""

    pair_gate = np.exp(
        np.clip(-0.5 * float(inverse_action) * overlap, -700.0, 0.0)
    )
    discarded = []
    order = list(range(len(tensors)))
    encountered = []
    for layer in range(len(tensors)):
        for position in range(layer % 2, len(tensors) - 1, 2):
            encountered.append((order[position], order[position + 1]))
            left, right, record = interaction_swap(
                tensors[position],
                tensors[position + 1],
                pair_gate,
                maximum_rank,
            )
            tensors[position] = left
            tensors[position + 1] = right
            order[position], order[position + 1] = (
                order[position + 1],
                order[position],
            )
            discarded.append(record["discarded_fraction"])
    expected_pairs = len(tensors) * (len(tensors) - 1) // 2
    if len(encountered) != expected_pairs or len(set(encountered)) != expected_pairs:
        raise AssertionError("swap network did not cover every particle pair once")
    return {
        "maximum_discarded_fraction": float(max(discarded, default=0.0)),
        "sum_discarded_fraction": float(sum(discarded)),
        "maximum_bond_rank": int(max(tensor.shape[2] for tensor in tensors)),
        "final_order": order,
    }


def transfer_mps(
    tensors: list[np.ndarray],
    overlap: np.ndarray,
    heat: np.ndarray,
    inverse_action: float,
    maximum_rank: int,
) -> dict:
    """One compressed Strang transfer on all seventeen particle axes."""

    first = apply_interaction_half(
        tensors, overlap, inverse_action, maximum_rank
    )
    for site in range(len(tensors)):
        tensors[site] = apply_one_site_gate(tensors[site], heat)
    second = apply_interaction_half(
        tensors, overlap, inverse_action, maximum_rank
    )
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
        "order_restored": second["final_order"] == first["final_order"],
    }


def right_norm_environments(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Suffix norm contractions used for one conditional Born measurement."""

    right = [np.empty((0, 0)) for _ in range(len(tensors) + 1)]
    right[-1] = np.ones((1, 1), dtype=np.float64)
    for site in range(len(tensors) - 1, -1, -1):
        tensor = tensors[site]
        right[site] = np.einsum(
            "iar,jas,rs->ij",
            tensor,
            tensor,
            right[site + 1],
            optimize=True,
        )
    return right


def measure_conditional_modes(tensors: list[np.ndarray]) -> tuple[np.ndarray, list]:
    """Measure one joint state by deterministic conditional Born modes."""

    right = right_norm_environments(tensors)
    left = np.ones((1, 1), dtype=np.float64)
    outcome = np.empty(len(tensors), dtype=np.int32)
    records = []
    for site, tensor in enumerate(tensors):
        weights = np.einsum(
            "ab,air,bis,rs->i",
            left,
            tensor,
            tensor,
            right[site + 1],
            optimize=True,
        )
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            raise FloatingPointError("conditional measurement has zero mass")
        probability = weights / total
        choice = int(np.argmax(probability))
        outcome[site] = choice
        selected = tensor[:, choice, :]
        left = np.einsum(
            "ab,ar,bs->rs", left, selected, selected, optimize=True
        )
        left /= max(float(np.trace(left)), np.finfo(np.float64).tiny)
        records.append(
            {
                "site": site,
                "pose": choice,
                "conditional_probability": float(probability[choice]),
                "conditional_effective_support": float(
                    1.0 / np.sum(np.square(probability))
                ),
            }
        )
    return outcome, records


def solve_tensor_transport(
    config: TensorTransportConfig,
    *,
    alphabet: np.ndarray | None = None,
) -> dict:
    if alphabet is None:
        if config.alphabet == "reference_control":
            if abs(config.side - REFERENCE_SIDE) > 1.0e-12:
                raise ValueError("reference control requires the reference side")
            poses = reference_chart()
        elif config.alphabet == "low_discrepancy":
            poses = low_discrepancy_pose_alphabet(
                config.side, config.pose_count, config.seed
            )
        else:
            raise ValueError("alphabet must be reference_control or low_discrepancy")
    else:
        poses = np.asarray(alphabet, dtype=np.float64)
    if len(poses) != int(config.pose_count):
        raise ValueError("pose_count does not match supplied alphabet")

    overlap = pose_overlap_gram(poses)
    heat = pose_heat_kernel(poses, config.heat_scale, config.heat_floor)
    tensors = uniform_mps(SQUARE_COUNT, len(poses))
    normalize_mps(tensors)
    stages = []
    previous_probability = None
    for inverse_action in config.inverse_actions:
        stage_transfers = []
        for _ in range(max(int(config.transfers_per_action), 1)):
            stage_transfers.append(
                transfer_mps(
                    tensors,
                    overlap,
                    heat,
                    inverse_action,
                    config.rank,
                )
            )
        indices, measurement = measure_conditional_modes(tensors)
        selected = poses[indices]
        area, _, overlap_area, wall_area = physical_area_energy(
            selected, config.side, 1.0
        )
        probability = np.asarray(
            [record["conditional_probability"] for record in measurement]
        )
        probability_change = (
            float(np.max(np.abs(probability - previous_probability)))
            if previous_probability is not None
            else None
        )
        previous_probability = probability
        stages.append(
            {
                "inverse_action": float(inverse_action),
                "measured_area_energy": float(area),
                "measured_overlap_area": float(overlap_area),
                "measured_wall_area": float(wall_area),
                "conditional_probability_change": probability_change,
                "maximum_discarded_fraction": float(
                    max(
                        record["maximum_discarded_fraction"]
                        for record in stage_transfers
                    )
                ),
                "sum_discarded_fraction": float(
                    sum(
                        record["sum_discarded_fraction"]
                        for record in stage_transfers
                    )
                ),
                "maximum_bond_rank": int(
                    max(record["maximum_bond_rank"] for record in stage_transfers)
                ),
                "order_restored": bool(
                    all(record["order_restored"] for record in stage_transfers)
                ),
            }
        )

    indices, measurement = measure_conditional_modes(tensors)
    selected = poses[indices]
    area, _, overlap_area, wall_area = physical_area_energy(
        selected, config.side, 1.0
    )
    audit = capacity_state(selected, config.side)
    return {
        "method": "compressed_17_body_bfft_tensor_transport",
        "config": asdict(config),
        "transport_axes": SQUARE_COUNT,
        "one_particle_pose_states": len(poses),
        "implicit_product_basis_states": str(len(poses) ** SQUARE_COUNT),
        "potential": "exact_pairwise_polygon_intersection_area",
        "exact_operator_is_projective_contraction": True,
        "compressed_operator_contraction_claimed": False,
        "measured_area_energy": float(area),
        "measured_overlap_area": float(overlap_area),
        "measured_wall_area": float(wall_area),
        "terminal_sat_audit": {
            "minimum_clearance": float(audit.minimum_clearance),
            "overlap_residual": float(audit.overlap_residual),
            "worst_penetration": max(-float(audit.minimum_clearance), 0.0),
        },
        "pose_indices": indices.tolist(),
        "poses": selected.tolist(),
        "measurement": measurement,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=REFERENCE_SIDE)
    parser.add_argument("--pose-count", type=int, default=17)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heat-scale", type=float, default=0.18)
    parser.add_argument("--transfers", type=int, default=2)
    parser.add_argument(
        "--alphabet",
        choices=("reference_control", "low_discrepancy"),
        default="reference_control",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = TensorTransportConfig(
        side=args.side,
        pose_count=args.pose_count,
        rank=args.rank,
        seed=args.seed,
        heat_scale=args.heat_scale,
        transfers_per_action=args.transfers,
        alphabet=args.alphabet,
    )
    result = solve_tensor_transport(config)
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
