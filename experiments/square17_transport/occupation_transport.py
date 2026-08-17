#!/usr/bin/env python3
"""Exact symmetry-quotiented 17-square configuration transport.

The labelled tensor product contains 17! copies of every physical placement.
This module removes that redundancy.  Given ``M`` one-square floor-plan pose
modes, a basis state is an unordered 17-element occupied subset.  Distinct
subsets remain orthogonal/equidistant, while the exact state count is
``binomial(M, 17)`` rather than ``M**17``.

The kinetic operator is the random walk on the Johnson graph ``J(M, 17)``:
one occupied pose mode is exchanged with one unoccupied mode.  Its discounted
resolvent is evaluated by

    y[n+1] = b + gamma P y[n].

Because ``P`` is row stochastic, this is a Banach contraction in ``l_inf``
with factor ``gamma``.  The exact resolvent is strictly positive, and the
potential-sandwiched normalized transfer is consequently a projective
contraction.  The potential is the sum of physical polygon intersection
areas among occupied square footprints.  No clearance enters evolution.

This is still a finite pose-basis control.  It is exact within that basis and
can therefore distinguish a representation miss from a local optimizer miss.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from chip_transport import write_svg
from configuration_transport import hilbert_projective_distance
from floorplan_banach import physical_area_energy
from geometry import SQUARE_COUNT, capacity_state
from reference_chart import REFERENCE_SIDE, reference_chart
from tensor_transport import low_discrepancy_pose_alphabet, pose_overlap_gram


@dataclass(frozen=True)
class OccupationTransportConfig:
    side: float = REFERENCE_SIDE
    pose_count: int = 22
    seed: int = 0
    discount: float = 0.82
    resolvent_iterations: int = 80
    inverse_actions: tuple[float, ...] = (1.0, 8.0, 64.0)
    iterations_per_action: int = 40
    tolerance: float = 1.0e-13
    alphabet: str = "reference_control"


@dataclass(frozen=True)
class OccupationBasis:
    combinations: np.ndarray
    masks: np.ndarray
    transition: sparse.csr_matrix


def occupation_basis(pose_count: int) -> OccupationBasis:
    """Build the exact 17-particle subset basis and Johnson transition."""

    modes = int(pose_count)
    if modes < SQUARE_COUNT:
        raise ValueError("pose_count must be at least seventeen")
    if modes > 63:
        raise ValueError("the current exact bit-mask control supports at most 63 modes")
    combinations = np.asarray(
        list(itertools.combinations(range(modes), SQUARE_COUNT)),
        dtype=np.int16,
    )
    masks = np.zeros(len(combinations), dtype=np.uint64)
    for column in range(SQUARE_COUNT):
        masks |= np.left_shift(
            np.uint64(1), combinations[:, column].astype(np.uint64)
        )
    index = {int(mask): row for row, mask in enumerate(masks)}
    degree = SQUARE_COUNT * (modes - SQUARE_COUNT)
    if degree == 0:
        transition = sparse.identity(len(masks), format="csr", dtype=np.float64)
        return OccupationBasis(combinations, masks, transition)

    row = np.empty(len(masks) * degree, dtype=np.int32)
    column = np.empty_like(row)
    cursor = 0
    full_mask = (1 << modes) - 1
    for source, mask_value in enumerate(masks):
        mask = int(mask_value)
        occupied = mask
        empty = full_mask ^ mask
        while occupied:
            leave_bit = occupied & -occupied
            occupied ^= leave_bit
            available = empty
            while available:
                enter_bit = available & -available
                available ^= enter_bit
                row[cursor] = source
                column[cursor] = index[mask ^ leave_bit ^ enter_bit]
                cursor += 1
    data = np.full(cursor, 1.0 / degree, dtype=np.float64)
    transition = sparse.csr_matrix(
        (data, (row[:cursor], column[:cursor])),
        shape=(len(masks), len(masks)),
    )
    return OccupationBasis(combinations, masks, transition)


def occupation_area_energy(
    basis: OccupationBasis,
    overlap: np.ndarray,
) -> np.ndarray:
    """Physical pair-intersection energy of every occupied subset."""

    gram = np.asarray(overlap, dtype=np.float64)
    modes = gram.shape[0]
    if gram.shape != (modes, modes):
        raise ValueError("overlap must be a square matrix")
    occupancy = np.zeros((len(basis.combinations), modes), dtype=bool)
    rows = np.arange(len(occupancy))[:, None]
    occupancy[rows, basis.combinations] = True
    energy = np.zeros(len(occupancy), dtype=np.float64)
    for first in range(modes):
        for second in range(first):
            if gram[first, second] != 0.0:
                energy += (
                    gram[first, second]
                    * occupancy[:, first]
                    * occupancy[:, second]
                )
    return energy


def banach_resolvent(
    forcing: np.ndarray,
    transition: sparse.csr_matrix,
    discount: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    """Apply ``(I - gamma P)^-1`` by its literal Banach iteration."""

    source = np.asarray(forcing, dtype=np.float64)
    if source.shape != (transition.shape[0],):
        raise ValueError("forcing and transition dimensions disagree")
    gamma = float(discount)
    if not 0.0 < gamma < 1.0:
        raise ValueError("discount must lie strictly between zero and one")
    value = source.copy()
    difference = math.inf
    completed = 0
    for iteration in range(max(int(iterations), 1)):
        updated = source + gamma * (transition @ value)
        difference = float(np.max(np.abs(updated - value)))
        value = updated
        completed = iteration + 1
    residual = float(
        np.max(np.abs(value - source - gamma * (transition @ value)))
    )
    return value, {
        "iterations": completed,
        "last_iterate_difference_linf": difference,
        "fixed_point_residual_linf": residual,
        "banach_factor": gamma,
    }


def occupation_transfer_step(
    amplitude: np.ndarray,
    energy: np.ndarray,
    basis: OccupationBasis,
    config: OccupationTransportConfig,
    inverse_action: float,
) -> tuple[np.ndarray, dict]:
    """One exact potential--resolvent--potential projective transfer."""

    value = np.asarray(amplitude, dtype=np.float64)
    potential = np.asarray(energy, dtype=np.float64)
    shifted = potential - float(np.min(potential))
    diagonal = np.exp(
        np.clip(-0.5 * float(inverse_action) * shifted, -700.0, 0.0)
    )
    transported, record = banach_resolvent(
        diagonal * value,
        basis.transition,
        config.discount,
        config.resolvent_iterations,
    )
    transported *= diagonal
    total = float(np.sum(transported))
    if total <= 0.0 or not math.isfinite(total):
        raise FloatingPointError("occupation transfer lost positive mass")
    return transported / total, record


def fixed_point(
    initial: np.ndarray,
    energy: np.ndarray,
    basis: OccupationBasis,
    config: OccupationTransportConfig,
    inverse_action: float,
) -> tuple[np.ndarray, dict]:
    amplitude = np.asarray(initial, dtype=np.float64).copy()
    amplitude = np.maximum(amplitude, np.finfo(np.float64).tiny)
    amplitude /= np.sum(amplitude)
    residual = math.inf
    inner = None
    completed = 0
    for iteration in range(max(int(config.iterations_per_action), 1)):
        updated, inner = occupation_transfer_step(
            amplitude, energy, basis, config, inverse_action
        )
        residual = float(np.sum(np.abs(updated - amplitude)))
        amplitude = updated
        completed = iteration + 1
        if residual <= config.tolerance:
            break
    return amplitude, {
        "iterations": completed,
        "l1_fixed_point_residual": residual,
        "inverse_action": float(inverse_action),
        "resolvent": inner,
    }


def reference_control_alphabet(
    pose_count: int,
    seed: int,
) -> np.ndarray:
    """Known feasible modes plus deterministic distractors for calibration."""

    count = int(pose_count)
    if count < SQUARE_COUNT:
        raise ValueError("reference control needs at least seventeen modes")
    if count == SQUARE_COUNT:
        return reference_chart()
    distractors = low_discrepancy_pose_alphabet(
        REFERENCE_SIDE, count - SQUARE_COUNT, seed
    )
    return np.vstack((reference_chart(), distractors))


def solve_occupation_transport(
    config: OccupationTransportConfig,
    *,
    alphabet: np.ndarray | None = None,
) -> dict:
    if alphabet is None:
        if config.alphabet == "reference_control":
            if abs(config.side - REFERENCE_SIDE) > 1.0e-12:
                raise ValueError("reference control requires the reference side")
            poses = reference_control_alphabet(config.pose_count, config.seed)
        elif config.alphabet == "low_discrepancy":
            poses = low_discrepancy_pose_alphabet(
                config.side, config.pose_count, config.seed
            )
        else:
            raise ValueError("alphabet must be reference_control or low_discrepancy")
    else:
        poses = np.asarray(alphabet, dtype=np.float64)
    if len(poses) != config.pose_count:
        raise ValueError("pose_count does not match supplied alphabet")

    basis = occupation_basis(len(poses))
    overlap = pose_overlap_gram(poses)
    energy = occupation_area_energy(basis, overlap)
    minimum_index = int(np.argmin(energy))
    minimum_energy = float(energy[minimum_index])
    amplitude = np.full(len(energy), 1.0 / len(energy), dtype=np.float64)
    stages = []
    for inverse_action in config.inverse_actions:
        amplitude, record = fixed_point(
            amplitude, energy, basis, config, inverse_action
        )
        measured = int(np.argmax(amplitude))
        record.update(
            {
                "measured_state": measured,
                "measured_area_energy": float(energy[measured]),
                "global_minimum_probability": float(amplitude[minimum_index]),
                "maximum_probability": float(amplitude[measured]),
                "effective_state_count": float(
                    1.0 / np.sum(np.square(amplitude))
                ),
            }
        )
        stages.append(record)

    measured_index = int(np.argmax(amplitude))
    selected_indices = basis.combinations[measured_index]
    selected = poses[selected_indices]
    physical, _, overlap_area, wall_area = physical_area_energy(
        selected, config.side, 1.0
    )
    audit = capacity_state(selected, config.side)

    rng = np.random.default_rng(config.seed + 991)
    first = rng.uniform(0.5, 1.5, len(energy))
    second = rng.uniform(0.5, 1.5, len(energy))
    before = hilbert_projective_distance(first, second)
    image_first, _ = occupation_transfer_step(
        first, energy, basis, config, config.inverse_actions[0]
    )
    image_second, _ = occupation_transfer_step(
        second, energy, basis, config, config.inverse_actions[0]
    )
    after = hilbert_projective_distance(image_first, image_second)

    return {
        "method": "exact_symmetry_quotiented_17_body_transport",
        "config": asdict(config),
        "particles": SQUARE_COUNT,
        "pose_modes": len(poses),
        "eigenbasis_states": len(energy),
        "distinct_basis_distance": math.sqrt(2.0),
        "johnson_degree": int(SQUARE_COUNT * (len(poses) - SQUARE_COUNT)),
        "minimum_discrete_area_energy": minimum_energy,
        "minimum_discrete_state": minimum_index,
        "measured_area_energy": float(energy[measured_index]),
        "measured_state": measured_index,
        "measured_matches_global_discrete_minimum": bool(
            abs(float(energy[measured_index]) - minimum_energy) <= 1.0e-13
        ),
        "physical_area_energy": float(physical),
        "physical_overlap_area": float(overlap_area),
        "physical_wall_area": float(wall_area),
        "projective_distance_before": before,
        "projective_distance_after": after,
        "observed_projective_ratio": after / before,
        "selected_pose_modes": selected_indices.tolist(),
        "terminal_sat_audit": {
            "minimum_clearance": float(audit.minimum_clearance),
            "overlap_residual": float(audit.overlap_residual),
            "worst_penetration": max(-float(audit.minimum_clearance), 0.0),
        },
        "poses": selected.tolist(),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=float, default=REFERENCE_SIDE)
    parser.add_argument("--pose-count", type=int, default=22)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discount", type=float, default=0.82)
    parser.add_argument("--resolvent-iterations", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--alphabet",
        choices=("reference_control", "low_discrepancy"),
        default="reference_control",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    config = OccupationTransportConfig(
        side=args.side,
        pose_count=args.pose_count,
        seed=args.seed,
        discount=args.discount,
        resolvent_iterations=args.resolvent_iterations,
        iterations_per_action=args.iterations,
        alphabet=args.alphabet,
    )
    result = solve_occupation_transport(config)
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
