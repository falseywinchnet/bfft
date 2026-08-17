#!/usr/bin/env python3
"""Deterministic low-discrepancy census of periodic-Hessian branches.

The continuation audit can discover a shortest branch, but random birth
probes merely move cold-restart work between scales.  This experiment asks a
more structural question: after a dense common Sobol probe set has exposed
the local maxima at every scale, how many distinct branches are present, and
how many current branches are not reached by transporting the preceding
catalog?

The census is only a lower bound on the number of maxima.  Its useful output
is saturation: branch counts, basin masses, and uncarried-branch counts as the
probe budget doubles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)
from experiments.walsh_periodic_hessian_continuation import (
    branches_match,
    classify_branch,
    merge_branches,
    optimize_seed,
    spatial_width,
)
from experiments.walsh_periodic_hessian_descent import (
    DEFAULT_R,
    ascend_periodic_hessian,
    coefficient_box,
    periodic_hessian_spectral_data,
)


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_periodic_hessian_branch_census.json"
)


def sobol_starts(n: int, power: int, seed: int) -> np.ndarray:
    """A nested, deterministic low-discrepancy probe family on the torus."""
    return qmc.Sobol(d=n, scramble=True, seed=seed).random_base2(power) - 0.5


def census_scale(
    starts: np.ndarray,
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    shortest: float,
    shortest_coefficients: np.ndarray,
    t: float,
) -> tuple[list[dict[str, object]], int]:
    width = spatial_width(basis.shape[0], t, shortest)
    reference_score, reference_eigengap, _, _ = periodic_hessian_spectral_data(
        0.5 * shortest_coefficients, basis, dual_points, width
    )
    candidates = [
        optimize_seed(
            start,
            basis=basis,
            dual_points=dual_points,
            width=width,
            ancestor=index,
            birth_level=0,
            reference_score=reference_score,
            reference_eigengap=reference_eigengap,
        )
        for index, start in enumerate(starts)
    ]
    evaluations = sum(int(row["evaluations"]) for row in candidates)
    return merge_branches(candidates, basis, shortest), evaluations


def transport_catalog(
    branches: list[dict[str, object]],
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    shortest: float,
    t: float,
) -> list[dict[str, object]]:
    width = spatial_width(basis.shape[0], t, shortest)
    transported = []
    for branch in branches:
        result = ascend_periodic_hessian(
            np.asarray(branch["coefficient_point"]),
            basis,
            dual_points,
            width,
            max_iterations=120,
        )
        result.update({
            "ancestors": list(branch["ancestors"]),
            "birth_levels": list(branch["birth_levels"]),
            "merged_copies": int(branch["merged_copies"]),
        })
        transported.append(result)
    return merge_branches(transported, basis, shortest)


def is_carried(
    branch: dict[str, object],
    transported: list[dict[str, object]],
    basis: np.ndarray,
    shortest: float,
) -> bool:
    return any(
        branches_match(
            branch,
            old,
            basis,
            shortest,
            location_tolerance=2e-3,
            direction_alignment=0.99,
        )
        for old in transported
    )


def audit_dimension(
    n: int,
    *,
    cutoff: int,
    scales: list[float],
    probe_powers: list[int],
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(cutoff, 2)
    )
    dual_points = coefficient_box(n, cutoff) @ np.linalg.inv(basis)
    coefficients = coefficient_box(n, cutoff + 1)
    points = coefficients @ basis.T
    maximum_power = max(probe_powers)
    starts = sobol_starts(n, maximum_power, seed + 104729 * n)

    budgets = []
    for power in probe_powers:
        probe_count = 1 << power
        previous: list[dict[str, object]] = []
        levels = []
        for level, t in enumerate(scales):
            branches, evaluations = census_scale(
                starts[:probe_count],
                basis=basis,
                dual_points=dual_points,
                shortest=shortest,
                shortest_coefficients=shortest_coefficients,
                t=t,
            )
            classes = [
                classify_branch(
                    branch,
                    basis=basis,
                    coefficients=coefficients,
                    points=points,
                    shortest=shortest,
                )
                for branch in branches
            ]
            shortest_indices = [
                index for index, row in enumerate(classes) if row["shortest_edge"]
            ]
            transported = (
                transport_catalog(
                    previous,
                    basis=basis,
                    dual_points=dual_points,
                    shortest=shortest,
                    t=t,
                )
                if previous else []
            )
            carried = [
                is_carried(branch, transported, basis, shortest)
                for branch in branches
            ]
            masses = [int(branch["merged_copies"]) for branch in branches]
            shortest_mass = sum(masses[index] for index in shortest_indices)
            score_order = sorted(
                range(len(branches)),
                key=lambda index: float(branches[index]["score"]),
                reverse=True,
            )
            basin_order = sorted(
                range(len(branches)),
                key=lambda index: masses[index],
                reverse=True,
            )
            branch_details = [
                {
                    "edge": list(classes[index]["edge"]),
                    "edge_norm_over_shortest": classes[index][
                        "edge_norm_over_shortest"
                    ],
                    "shortest_edge": classes[index]["shortest_edge"],
                    "probe_basin_count": masses[index],
                    "probe_basin_fraction": masses[index] / probe_count,
                    "score_rank": score_order.index(index) + 1,
                    "basin_rank": basin_order.index(index) + 1,
                    "carried_from_previous_scale": carried[index],
                    "score": float(branches[index]["score"]),
                    "eigengap": float(branches[index]["eigengap"]),
                }
                for index in range(len(branches))
            ]
            levels.append({
                "level": level,
                "t": t,
                "probe_count": probe_count,
                "field_gradient_evaluations": evaluations,
                "branch_count": len(branches),
                "distinct_edge_count": len({row["edge"] for row in classes}),
                "shortest_branch_count": len(shortest_indices),
                "shortest_basin_probe_count": shortest_mass,
                "shortest_basin_fraction": shortest_mass / probe_count,
                "uncarried_branch_count": sum(not value for value in carried),
                "uncarried_shortest_branch_count": sum(
                    not carried[index] for index in shortest_indices
                ),
                "best_shortest_score_rank": min(
                    (score_order.index(index) + 1 for index in shortest_indices),
                    default=None,
                ),
                "best_shortest_basin_rank": min(
                    (basin_order.index(index) + 1 for index in shortest_indices),
                    default=None,
                ),
                "largest_branch_basin_fraction": (
                    max(masses) / probe_count if masses else 0.0
                ),
                "branches": branch_details,
            })
            previous = branches
        budgets.append({
            "probe_power": power,
            "probe_count": probe_count,
            "levels": levels,
        })
    return {
        "dimension": n,
        "basis_condition_number": float(np.linalg.cond(basis)),
        "shortest_length": shortest,
        "shortest_coefficients": shortest_coefficients.tolist(),
        "budgets": budgets,
    }


def audit(
    *,
    min_dimension: int,
    max_dimension: int,
    cutoff: int,
    scales: list[float],
    probe_powers: list[int],
    seed: int,
) -> dict[str, object]:
    return {
        "experiment": "walsh_periodic_hessian_branch_census",
        "warning": (
            "Each branch count is a lower bound from a finite Sobol census; "
            "an uncarried branch may be a true birth or a branch missed at the "
            "preceding scale."
        ),
        "rows": [
            audit_dimension(
                n,
                cutoff=cutoff,
                scales=scales,
                probe_powers=probe_powers,
                seed=seed,
            )
            for n in range(min_dimension, max_dimension + 1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.12, 0.15, 0.18, 0.21, DEFAULT_R],
    )
    parser.add_argument("--probe-powers", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.probe_powers != sorted(set(args.probe_powers)):
        raise ValueError("probe powers must be strictly increasing")
    if args.scales != sorted(args.scales):
        raise ValueError("scales must be increasing")
    report = audit(
        min_dimension=args.min_dimension,
        max_dimension=args.max_dimension,
        cutoff=args.cutoff,
        scales=list(args.scales),
        probe_powers=list(args.probe_powers),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
