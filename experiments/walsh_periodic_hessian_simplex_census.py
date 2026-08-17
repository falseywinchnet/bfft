#!/usr/bin/env python3
"""Census for a simplex-cancellation lattice with all-ones shortest parity.

The Gram matrix has unit diagonal and a common negative off-diagonal chosen
so that the sum of all basis columns has prescribed length `shortest_ratio`.
The all-ones coefficient vector is then the unique shortest pair in the
tested family, while every coordinate direction remains a length-one
competitor.  This stresses parity complexity rather than basis condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_adversarial_search import (
    half_grid_error,
    snapped_half_parity,
)
from experiments.walsh_periodic_hessian_branch_census import census_scale, sobol_starts
from experiments.walsh_periodic_hessian_descent import DEFAULT_R, coefficient_box


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_periodic_hessian_simplex_census.json"
)


def simplex_cancellation_basis(n: int, shortest_ratio: float) -> np.ndarray:
    if n < 2 or not 0.0 < shortest_ratio < 1.0:
        raise ValueError("expected n >= 2 and shortest_ratio in (0,1)")
    off_diagonal = (shortest_ratio * shortest_ratio - n) / (n * (n - 1))
    gram = np.full((n, n), off_diagonal, dtype=np.float64)
    np.fill_diagonal(gram, 1.0)
    # np.linalg.cholesky returns L with LL^T=G; columns of L^T have Gram G.
    return np.linalg.cholesky(gram).T


def audit_dimension(
    n: int,
    *,
    shortest_ratio: float,
    field_cutoff: int,
    probe_power: int,
    scales: list[float],
    seed: int,
) -> dict[str, object]:
    basis = simplex_cancellation_basis(n, shortest_ratio)
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(field_cutoff, 2)
    )
    target_parity = tuple(
        int(value) & 1 for value in np.asarray(shortest_coefficients)
    )
    dual_points = coefficient_box(n, field_cutoff) @ np.linalg.inv(basis)
    starts = sobol_starts(n, probe_power, seed + 104729 * n)
    levels = []
    for t in scales:
        branches, evaluations = census_scale(
            starts,
            basis=basis,
            dual_points=dual_points,
            shortest=shortest,
            shortest_coefficients=shortest_coefficients,
            t=t,
        )
        parities = [
            snapped_half_parity(branch["coefficient_point"]) for branch in branches
        ]
        hits = [index for index, parity in enumerate(parities) if parity == target_parity]
        levels.append({
            "t": t,
            "branch_count": len(branches),
            "field_gradient_evaluations": evaluations,
            "shortest_parity_present": bool(hits),
            "shortest_parity_probe_count": sum(
                int(branches[index]["merged_copies"]) for index in hits
            ),
            "shortest_parity_probe_fraction": sum(
                int(branches[index]["merged_copies"]) for index in hits
            ) / len(starts),
            "maximum_half_grid_error": max(
                (half_grid_error(branch["coefficient_point"]) for branch in branches),
                default=0.0,
            ),
            "distinct_snapped_parity_count": len(set(parities)),
        })
    return {
        "dimension": n,
        "shortest_ratio": shortest_ratio,
        "basis_condition_number": float(np.linalg.cond(basis)),
        "shortest_length": shortest,
        "shortest_coefficients": shortest_coefficients.tolist(),
        "target_parity": list(target_parity),
        "probe_count": len(starts),
        "levels": levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=6)
    parser.add_argument("--shortest-ratio", type=float, default=0.97)
    parser.add_argument("--field-cutoff", type=int, default=2)
    parser.add_argument("--probe-power", type=int, default=8)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[0.10, DEFAULT_R]
    )
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "experiment": "walsh_periodic_hessian_simplex_census",
        "rows": [
            audit_dimension(
                n,
                shortest_ratio=args.shortest_ratio,
                field_cutoff=args.field_cutoff,
                probe_power=args.probe_power,
                scales=list(args.scales),
                seed=args.seed,
            )
            for n in range(args.min_dimension, args.max_dimension + 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
