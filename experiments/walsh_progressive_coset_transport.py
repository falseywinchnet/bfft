#!/usr/bin/env python3
"""Progressive state-dependent transport into a Walsh half-coset.

The earlier syndrome-folding audit translated every point in a syndrome by
one fixed representative.  Here the h constraints are imposed one at a time.
At each index-two step, points in the rejected sibling are bijectively matched
to points in the retained child and their probability masses are folded.

Two matchings are audited:

* ``renyi`` minimizes the next-step target||proposal Renyi-2 mass exactly via
  a linear assignment; it is an information-theoretic lower target for this
  progressive deterministic architecture.
* ``geometric`` minimizes squared Euclidean displacement and is the simplest
  implementable state-dependent transport candidate.

Finite coefficient boxes use an even number of representatives per parity,
so every linear prefix fiber has exactly the same number of states.  Boundary
mass is reported separately.  The audit is evidence, not an asymptotic lattice
matching theorem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


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
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_progressive_coset_transport.json"


def balanced_coefficients(n: int, half_width: int) -> np.ndarray:
    values = range(-half_width, half_width)
    return np.asarray(list(itertools.product(values, repeat=n)), dtype=np.int16)


def transformed_bits(coefficients: np.ndarray, inverse_transform: np.ndarray) -> np.ndarray:
    return ((coefficients & 1).astype(np.uint8) @ inverse_transform) & 1


def renyi2_log2(target: np.ndarray, proposal: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    target = target / np.sum(target)
    proposal = proposal / np.sum(proposal)
    if np.any((target > 0.0) & (proposal <= 0.0)):
        return math.inf
    return math.log2(float(np.sum(target * target / proposal)))


def assignment_indices(
    mode: str,
    sibling_indices: np.ndarray,
    child_indices: np.ndarray,
    proposal: np.ndarray,
    target_mass: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    if len(sibling_indices) != len(child_indices):
        raise ValueError("balanced box did not produce equal sibling fibers")
    if mode == "geometric":
        differences = points[sibling_indices, None, :] - points[None, child_indices, :]
        cost = np.einsum("ijk,ijk->ij", differences, differences)
    elif mode == "renyi":
        incoming = proposal[sibling_indices, None]
        retained = proposal[None, child_indices]
        target = target_mass[None, child_indices]
        cost = target * target / np.maximum(retained + incoming, np.finfo(float).tiny)
    else:
        raise ValueError(f"unknown matching mode: {mode}")
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(rows))):
        raise RuntimeError("unexpected assignment row ordering")
    return columns


def progressive_fold(
    coefficients: np.ndarray,
    points: np.ndarray,
    target_mass: np.ndarray,
    bits: np.ndarray,
    target_prefix: np.ndarray,
    mode: str,
) -> dict[str, object]:
    proposal = target_mass.astype(np.float64).copy()
    proposal /= np.sum(proposal)
    active = np.ones(len(coefficients), dtype=bool)
    steps = []
    total_squared_displacement = 0.0

    for bit_index, bit_value in enumerate(target_prefix):
        child_mask = active & (bits[:, bit_index] == bit_value)
        sibling_mask = active & ~child_mask
        child = np.flatnonzero(child_mask)
        sibling = np.flatnonzero(sibling_mask)
        columns = assignment_indices(
            mode,
            sibling,
            child,
            proposal,
            target_mass,
            points,
        )
        destinations = child[columns]
        displacement = points[sibling] - points[destinations]
        squared_displacement = np.einsum("ij,ij->i", displacement, displacement)
        incoming_probability = proposal[sibling].copy()
        proposal[destinations] += incoming_probability
        proposal[sibling] = 0.0
        active = child_mask

        target = target_mass[active]
        target /= np.sum(target)
        current = proposal[active]
        current /= np.sum(current)
        action = renyi2_log2(target, current)
        weighted_displacement = float(incoming_probability @ squared_displacement)
        total_squared_displacement += weighted_displacement
        steps.append({
            "bit": bit_index,
            "active_states": int(np.sum(active)),
            "renyi2_log2": action,
            "renyi2_bits_per_dimension": action / coefficients.shape[1],
            "incoming_probability": float(np.sum(incoming_probability)),
            "incoming_weighted_squared_displacement": weighted_displacement,
            "maximum_squared_displacement": float(np.max(squared_displacement)),
        })

    final_target = target_mass[active]
    final_target /= np.sum(final_target)
    final_proposal = proposal[active]
    final_proposal /= np.sum(final_proposal)
    return {
        "mode": mode,
        "steps": steps,
        "final_renyi2_log2": renyi2_log2(final_target, final_proposal),
        "final_renyi2_bits_per_dimension": (
            renyi2_log2(final_target, final_proposal) / coefficients.shape[1]
        ),
        "total_incoming_weighted_squared_displacement": total_squared_displacement,
    }


def audit_dimension(
    n: int,
    *,
    half_width: int,
    h: int,
    seed: int,
    target_width: float = TARGET_WIDTH,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    inverse_basis = np.linalg.inv(basis)
    shortest, _ = shortest_vector_coefficients(basis)
    transform = random_gl2(n, rng)
    inverse_transform = _gf2_inverse(transform)
    coefficients = balanced_coefficients(n, half_width)
    points = coefficients @ inverse_basis
    norm2 = np.einsum("ij,ij->i", points, points)
    xi2 = 4.0 * n * target_width * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    target_mass = np.exp(-math.pi * norm2 / xi2)
    bits = transformed_bits(coefficients, inverse_transform)
    # Match the stationary-seed reduction: draw the address from the cold law.
    # In the asymptotic theorem leftover hashing makes it close to uniform, but
    # the distinction is material in these small finite dimensions.
    seed_index = int(rng.choice(len(coefficients), p=target_mass / np.sum(target_mass)))
    target_prefix = bits[seed_index, :h].copy()
    boundary = np.any(
        (coefficients == -half_width) | (coefficients == half_width - 1), axis=1
    )
    return {
        "dimension": n,
        "states": len(coefficients),
        "half_width": half_width,
        "h": h,
        "target_prefix": target_prefix.tolist(),
        "target_prefix_probability": float(
            np.sum(target_mass[np.all(bits[:, :h] == target_prefix, axis=1)])
            / np.sum(target_mass)
        ),
        "target_width": target_width,
        "required_action_budget_per_dimension": 0.5 - 2.0 * target_width,
        "full_gaussian_boundary_mass": float(
            np.sum(target_mass[boundary]) / np.sum(target_mass)
        ),
        "transports": [
            progressive_fold(
                coefficients,
                points,
                target_mass,
                bits,
                target_prefix,
                mode,
            )
            for mode in ("renyi", "geometric")
        ],
    }


def audit() -> dict[str, object]:
    rows = [
        audit_dimension(3, half_width=3, h=1, seed=11),
        audit_dimension(4, half_width=2, h=2, seed=11),
        audit_dimension(5, half_width=2, h=2, seed=11),
    ]
    return {
        "experiment": "walsh_progressive_coset_transport",
        "rows": rows,
        "interpretation": (
            "The Renyi assignment is the best next-step deterministic matching "
            "inside the finite support-chain architecture.  A gap between it and "
            "geometric matching measures the price of locality.  Small-dimensional "
            "prefix bias is reported and prevents an asymptotic no-go inference."
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
