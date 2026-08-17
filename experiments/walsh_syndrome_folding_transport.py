#!/usr/bin/env python3
"""Finite audit of cold full-lattice to fixed-half-coset syndrome folding.

The exact Gaussian combiner can cool samples while relaxing the h affine
parity constraints.  This audit separates that issue from temperature
transport.  Starting with a cold Gaussian on L*, map every parity-prefix
coset to one fixed prefix j by translating it with the shortest enumerated
representative of the required syndrome.

If z has target prefix j and c_delta has prefix delta, the folded mass is

    q_j(z) proportional to sum_delta rho_r(z-c_delta).

The target is rho_r(z) conditioned on prefix j.  Their exact finite-box
Renyi-2 divergence measures the remaining price of restoring the lost address
after cooling, without endpoint radial importance weights.
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

from experiments.walsh_coset_contraction_transport import renyi2_divergence
from experiments.walsh_hessian_noise_audit import (
    _binary_index,
    _gf2_inverse,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)
from experiments.walsh_radial_matched_filter import optimal_gaussian_target_width


T0 = 0.23147
SOURCE_WIDTH = 2.0 * T0
TARGET_WIDTH = optimal_gaussian_target_width(SOURCE_WIDTH)
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_syndrome_folding_transport.json"


def enumerated_coefficients(n: int, cutoff: int) -> np.ndarray:
    values = np.arange(-cutoff, cutoff + 1, dtype=np.int16)
    return np.asarray(list(itertools.product(values, repeat=n)), dtype=np.int16)


def parity_prefixes(
    coefficients: np.ndarray,
    inverse_transform: np.ndarray,
    h: int,
) -> np.ndarray:
    parity = (coefficients & 1).astype(np.uint8)
    coordinates = (parity @ inverse_transform) & 1
    return _binary_index(coordinates[:, :h])


def shortest_syndrome_representatives(
    n: int,
    h: int,
    inverse_basis: np.ndarray,
    inverse_transform: np.ndarray,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = enumerated_coefficients(n, cutoff)
    points = coefficients @ inverse_basis
    norm2 = np.einsum("ij,ij->i", points, points)
    syndromes = parity_prefixes(coefficients, inverse_transform, h)
    representatives = np.empty((1 << h, n), dtype=np.int16)
    representative_norm2 = np.empty(1 << h, dtype=np.float64)
    for syndrome in range(1 << h):
        indices = np.flatnonzero(syndromes == syndrome)
        if len(indices) == 0:
            raise RuntimeError("representative cutoff misses a syndrome")
        index = int(indices[np.argmin(norm2[indices])])
        representatives[syndrome] = coefficients[index]
        representative_norm2[syndrome] = norm2[index]
    return representatives, representative_norm2


def audit_dimension(
    n: int,
    *,
    target_width: float,
    chi: float,
    cutoff: int,
    representative_cutoff: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    inverse_basis = np.linalg.inv(basis)
    shortest, _ = shortest_vector_coefficients(basis)
    h = min(max(int(math.floor(chi * n)), 1), n - 1)
    transform = random_gl2(n, rng)
    inverse_transform = _gf2_inverse(transform)
    representatives, representative_norm2 = shortest_syndrome_representatives(
        n,
        h,
        inverse_basis,
        inverse_transform,
        representative_cutoff,
    )

    coefficients = enumerated_coefficients(n, cutoff)
    points = coefficients @ inverse_basis
    norm2 = np.einsum("ij,ij->i", points, points)
    syndromes = parity_prefixes(coefficients, inverse_transform, h)
    xi2 = 4.0 * n * target_width * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    target_mass_all = np.exp(-math.pi * norm2 / xi2)
    boundary = np.any(np.abs(coefficients) == cutoff, axis=1)

    reports = []
    for j in range(1 << h):
        indices = np.flatnonzero(syndromes == j)
        target_coefficients = coefficients[indices]
        target = target_mass_all[indices].astype(np.float64)
        target /= np.sum(target)

        # delta indexes the input syndrome relative to the desired output
        # syndrome. Translation by c_delta is a bijection from that input
        # coset to the fixed output coset.
        proposal_raw = np.zeros(len(indices), dtype=np.float64)
        for representative in representatives:
            preimage = target_coefficients - representative
            preimage_points = preimage @ inverse_basis
            preimage_norm2 = np.einsum(
                "ij,ij->i", preimage_points, preimage_points
            )
            proposal_raw += np.exp(-math.pi * preimage_norm2 / xi2)
        proposal = proposal_raw / np.sum(proposal_raw)
        log_d2 = renyi2_divergence(target, proposal)
        reports.append({
            "j": j,
            "points": int(len(indices)),
            "log2_renyi2_mass_per_dimension": log_d2 / (n * math.log(2.0)),
            "target_boundary_mass": float(np.sum(target[boundary[indices]])),
            "proposal_boundary_mass": float(np.sum(proposal[boundary[indices]])),
        })

    action = [row["log2_renyi2_mass_per_dimension"] for row in reports]
    budget = 0.5 - 2.0 * target_width
    return {
        "dimension": n,
        "h": h,
        "chi_realized": h / n,
        "cutoff": cutoff,
        "representative_cutoff": representative_cutoff,
        "target_width": target_width,
        "required_action_budget_per_dimension": budget,
        "representative_norm2": representative_norm2.tolist(),
        "summary": {
            "minimum_action": float(min(action)),
            "mean_action": float(np.mean(action)),
            "maximum_action": float(max(action)),
            "maximum_minus_budget": float(max(action) - budget),
            "maximum_target_boundary_mass": float(
                max(row["target_boundary_mass"] for row in reports)
            ),
            "maximum_proposal_boundary_mass": float(
                max(row["proposal_boundary_mass"] for row in reports)
            ),
        },
        "cosets": reports,
    }


def audit(
    dimensions: tuple[int, ...],
    *,
    target_width: float,
    chi: float,
    cutoff: int,
    representative_cutoff: int,
    seed: int,
) -> dict[str, object]:
    return {
        "experiment": "cold_gaussian_syndrome_folding",
        "source_width_before_exact_combine": SOURCE_WIDTH,
        "target_width_after_exact_combine": target_width,
        "dimensions": [
            audit_dimension(
                n,
                target_width=target_width,
                chi=chi,
                cutoff=cutoff,
                representative_cutoff=representative_cutoff,
                seed=seed,
            )
            for n in dimensions
        ],
        "interpretation": (
            "An action below 1/2-2r would make exact cooling followed by "
            "syndrome folding compatible with a 2^(n/2+o(n)) recovery."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(4, 5, 6))
    parser.add_argument("--target-width", type=float, default=TARGET_WIDTH)
    parser.add_argument("--chi", type=float, default=0.5)
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--representative-cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(
        tuple(args.dimensions),
        target_width=args.target_width,
        chi=args.chi,
        cutoff=args.cutoff,
        representative_cutoff=args.representative_cutoff,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
