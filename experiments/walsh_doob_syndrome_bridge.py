#!/usr/bin/env python3
"""Finite reversible-chain audit for the cold syndrome Doob bridge.

The state space is a truncated coefficient box in L*.  A lazy nearest-neighbor
Metropolis chain is reversible for the cold discrete Gaussian.  For each
parity-prefix target and time horizon, the script computes the exact backward
committor, the exact Doob-bridge endpoint law, and its Renyi-2 action relative
to the cold Gaussian conditioned on that prefix.

This does not assert rapid mixing for general lattices.  It tests the precise
bridge lemma and measures whether the most elementary local chain approaches
the required 0.02648-bit action budget at small dimension.
"""

from __future__ import annotations

import argparse
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
    _gf2_inverse,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)
from experiments.walsh_radial_matched_filter import optimal_gaussian_target_width
from experiments.walsh_syndrome_folding_transport import (
    enumerated_coefficients,
    parity_prefixes,
)


T0 = 0.23147
TARGET_WIDTH = optimal_gaussian_target_width(2.0 * T0)
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_doob_syndrome_bridge.json"


def metropolis_matrix(coefficients: np.ndarray, mass: np.ndarray):
    from scipy.sparse import coo_matrix

    coefficients = np.asarray(coefficients)
    mass = np.asarray(mass, dtype=np.float64)
    count, n = coefficients.shape
    lookup = {tuple(map(int, row)): i for i, row in enumerate(coefficients)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(count, dtype=np.float64)
    proposal_probability = 1.0 / (2.0 * n)
    for i, row in enumerate(coefficients):
        for coordinate in range(n):
            for direction in (-1, 1):
                neighbor = row.copy()
                neighbor[coordinate] += direction
                j = lookup.get(tuple(map(int, neighbor)))
                if j is None:
                    diagonal[i] += proposal_probability
                    continue
                acceptance = min(1.0, mass[j] / mass[i])
                probability = proposal_probability * acceptance
                rows.append(i)
                cols.append(j)
                data.append(probability)
                diagonal[i] += proposal_probability - probability
    rows.extend(range(count))
    cols.extend(range(count))
    data.extend(diagonal.tolist())
    return coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()


def apply_steps(transition, values: np.ndarray, steps: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    for _ in range(steps):
        result = transition @ result
    return result


def bridge_report(
    transition,
    stationary: np.ndarray,
    event: np.ndarray,
    steps: int,
) -> dict[str, float]:
    stationary = np.asarray(stationary, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    event_probability = float(np.sum(stationary[event]))
    h0 = apply_steps(transition, event.astype(np.float64), steps)
    positive = h0 > 0.0
    if not np.all(positive):
        return {
            "steps": steps,
            "zero_committor_fraction": float(np.mean(~positive)),
            "committor_log2_spread_per_dimension": math.inf,
            "renyi2_action_per_dimension": math.inf,
            "bridge_mass_error": math.inf,
        }
    ratio = h0 / event_probability
    reciprocal_evolved = apply_steps(transition, 1.0 / h0, steps)
    endpoint = stationary * reciprocal_evolved * event
    bridge_mass_error = abs(float(np.sum(endpoint)) - 1.0)
    endpoint /= np.sum(endpoint)
    target = stationary[event].copy()
    target /= np.sum(target)
    proposal = endpoint[event].copy()
    proposal /= np.sum(proposal)
    log_d2 = renyi2_divergence(target, proposal)
    return {
        "steps": steps,
        "zero_committor_fraction": 0.0,
        "committor_log2_min_ratio": float(np.log2(np.min(ratio))),
        "committor_log2_max_ratio": float(np.log2(np.max(ratio))),
        "committor_log2_spread": float(
            max(-np.log2(np.min(ratio)), np.log2(np.max(ratio)))
        ),
        "renyi2_log_mass": log_d2 / math.log(2.0),
        "bridge_mass_error": bridge_mass_error,
    }


def audit_dimension(
    n: int,
    *,
    target_width: float,
    chi: float,
    cutoff: int,
    steps: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    inverse_basis = np.linalg.inv(basis)
    shortest, _ = shortest_vector_coefficients(basis)
    coefficients = enumerated_coefficients(n, cutoff)
    points = coefficients @ inverse_basis
    norm2 = np.einsum("ij,ij->i", points, points)
    xi2 = 4.0 * n * target_width * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    mass = np.exp(-math.pi * norm2 / xi2)
    stationary = mass / np.sum(mass)
    transition = metropolis_matrix(coefficients, mass)
    row_error = float(np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1.0)))
    detailed_balance_error = 0.0
    coo = transition.tocoo()
    reverse = transition.T.tocsr()
    for i, j, probability in zip(coo.row, coo.col, coo.data):
        reverse_probability = reverse[i, j]
        detailed_balance_error = max(
            detailed_balance_error,
            abs(stationary[i] * probability - stationary[j] * reverse_probability),
        )

    h = min(max(int(math.floor(chi * n)), 1), n - 1)
    transform = random_gl2(n, rng)
    inverse_transform = _gf2_inverse(transform)
    syndromes = parity_prefixes(coefficients, inverse_transform, h)
    coset_reports = []
    for j in range(1 << h):
        event = syndromes == j
        reports = []
        for horizon in steps:
            row = bridge_report(transition, stationary, event, horizon)
            if math.isfinite(row["committor_log2_spread"]):
                row["committor_log2_spread_per_dimension"] = (
                    row["committor_log2_spread"] / n
                )
                row["renyi2_action_per_dimension"] = row["renyi2_log_mass"] / n
            reports.append(row)
        coset_reports.append({
            "j": j,
            "stationary_probability": float(np.sum(stationary[event])),
            "horizons": reports,
        })

    budget = 0.5 - 2.0 * target_width
    horizon_summary = []
    for horizon_index, horizon in enumerate(steps):
        actions = [
            row["horizons"][horizon_index]["renyi2_action_per_dimension"]
            for row in coset_reports
        ]
        spreads = [
            row["horizons"][horizon_index][
                "committor_log2_spread_per_dimension"
            ]
            for row in coset_reports
        ]
        horizon_summary.append({
            "steps": horizon,
            "maximum_action": float(max(actions)),
            "mean_action": float(np.mean(actions)),
            "maximum_committor_spread": float(max(spreads)),
            "maximum_action_minus_budget": float(max(actions) - budget),
        })
    return {
        "dimension": n,
        "states": int(len(coefficients)),
        "h": h,
        "target_width": target_width,
        "required_action_budget_per_dimension": budget,
        "row_stochastic_error": row_error,
        "detailed_balance_error": detailed_balance_error,
        "horizon_summary": horizon_summary,
        "cosets": coset_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--target-width", type=float, default=TARGET_WIDTH)
    parser.add_argument("--chi", type=float, default=0.5)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--steps", type=int, nargs="+", default=(4, 8, 16, 32, 64, 128))
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "experiment": "cold_gaussian_doob_syndrome_bridge",
        "dimensions": [
            audit_dimension(
                n,
                target_width=args.target_width,
                chi=args.chi,
                cutoff=args.cutoff,
                steps=tuple(args.steps),
                seed=args.seed,
            )
            for n in args.dimensions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
