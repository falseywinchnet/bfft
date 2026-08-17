#!/usr/bin/env python3
"""Adversarial search for failures of accessible-scale branch inheritance.

For each deterministic random lattice, enumerate a smooth and target-scale
Sobol branch catalog.  Rank lattices by the strongest observed failure:

1. a target shortest branch exists but is not reached by transporting the
   smooth catalog;
2. the smooth census misses every shortest branch;
3. the target shortest basin is small; or
4. the target shortest ridge has poor Hessian-score rank.

This is a falsification search, not evidence for a universal theorem.
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

from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_branch_census import (
    census_scale,
    is_carried,
    sobol_starts,
    transport_catalog,
)
from experiments.walsh_periodic_hessian_continuation import classify_branch
from experiments.walsh_periodic_hessian_descent import DEFAULT_R, coefficient_box


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out"
    / "walsh_periodic_hessian_adversarial_search.json"
)


def snapped_half_parity(coefficient_point: np.ndarray) -> tuple[int, ...]:
    """Recover the parity address of the nearest coefficient half-grid point."""
    doubled = 2.0 * np.asarray(coefficient_point, dtype=np.float64)
    return tuple(int(value) & 1 for value in np.rint(doubled).astype(np.int64))


def half_grid_error(coefficient_point: np.ndarray) -> float:
    doubled = 2.0 * np.asarray(coefficient_point, dtype=np.float64)
    return float(np.linalg.norm(doubled - np.rint(doubled)))


def adversarial_basis(
    n: int, rng: np.random.Generator, log_condition: float
) -> np.ndarray:
    """Rotated anisotropic basis with a prescribed condition envelope."""
    left, _ = np.linalg.qr(rng.standard_normal((n, n)))
    right, _ = np.linalg.qr(rng.standard_normal((n, n)))
    logs = rng.uniform(-0.5 * log_condition, 0.5 * log_condition, size=n)
    logs -= np.mean(logs)
    diagonal = np.exp(logs)
    return left @ np.diag(diagonal) @ right.T


def catalog_classes(
    branches: list[dict[str, object]],
    *,
    basis: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
) -> list[dict[str, object]]:
    return [
        classify_branch(
            branch,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
        )
        for branch in branches
    ]


def audit_candidate(
    n: int,
    candidate: int,
    *,
    log_condition: float,
    field_cutoff: int,
    probe_power: int,
    smooth_t: float,
    target_t: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1_000_003 * candidate + 1009 * n)
    basis = adversarial_basis(n, rng, log_condition)
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(field_cutoff + 2, 4)
    )
    # Normalize lambda_1 to remove a numerically irrelevant overall scale.
    basis = basis / shortest
    shortest = 1.0
    dual_points = coefficient_box(n, field_cutoff) @ np.linalg.inv(basis)
    coefficients = coefficient_box(n, field_cutoff + 2)
    points = coefficients @ basis.T
    starts = sobol_starts(n, probe_power, seed + 104729 * n)

    smooth, smooth_evaluations = census_scale(
        starts,
        basis=basis,
        dual_points=dual_points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        t=smooth_t,
    )
    target, target_evaluations = census_scale(
        starts,
        basis=basis,
        dual_points=dual_points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        t=target_t,
    )
    transported = transport_catalog(
        smooth,
        basis=basis,
        dual_points=dual_points,
        shortest=shortest,
        t=target_t,
    )
    smooth_classes = catalog_classes(
        smooth,
        basis=basis,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
    )
    target_classes = catalog_classes(
        target,
        basis=basis,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
    )
    transported_classes = catalog_classes(
        transported,
        basis=basis,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
    )
    smooth_shortest = [
        index for index, row in enumerate(smooth_classes) if row["shortest_edge"]
    ]
    target_parity = tuple(
        int(value) & 1 for value in np.asarray(shortest_coefficients)
    )
    smooth_snapped_parities = [
        snapped_half_parity(branch["coefficient_point"]) for branch in smooth
    ]
    smooth_parity_hits = [
        index
        for index, parity in enumerate(smooth_snapped_parities)
        if parity == target_parity
    ]
    target_shortest = [
        index for index, row in enumerate(target_classes) if row["shortest_edge"]
    ]
    carried = [
        is_carried(target[index], transported, basis, shortest)
        for index in target_shortest
    ]
    transported_shortest = [
        index
        for index, row in enumerate(transported_classes)
        if row["shortest_edge"]
    ]
    target_order = sorted(
        range(len(target)),
        key=lambda index: float(target[index]["score"]),
        reverse=True,
    )
    target_shortest_mass = sum(
        int(target[index]["merged_copies"]) for index in target_shortest
    )
    target_fraction = target_shortest_mass / len(starts)
    worst_rank = min(
        (target_order.index(index) + 1 for index in target_shortest),
        default=len(target) + 1,
    )
    failure = bool(target_shortest) and not bool(transported_shortest)
    return {
        "candidate": candidate,
        "basis": basis.tolist(),
        "basis_condition_number": float(np.linalg.cond(basis)),
        "shortest_coefficients": shortest_coefficients.tolist(),
        "smooth_branch_count": len(smooth),
        "smooth_shortest_present": bool(smooth_shortest),
        "smooth_shortest_basin_fraction": sum(
            int(smooth[index]["merged_copies"]) for index in smooth_shortest
        ) / len(starts),
        "smooth_target_parity_present": bool(smooth_parity_hits),
        "smooth_target_parity_probe_fraction": sum(
            int(smooth[index]["merged_copies"]) for index in smooth_parity_hits
        ) / len(starts),
        "maximum_smooth_half_grid_error": max(
            (half_grid_error(branch["coefficient_point"]) for branch in smooth),
            default=0.0,
        ),
        "target_branch_count": len(target),
        "target_shortest_present": bool(target_shortest),
        "target_shortest_basin_fraction": target_fraction,
        "target_best_shortest_score_rank": (
            worst_rank if target_shortest else None
        ),
        "target_shortest_carried_from_smooth": any(carried),
        "transported_branch_count": len(transported),
        "transported_shortest_present": bool(transported_shortest),
        "observed_inheritance_failure": failure,
        "field_gradient_evaluations": smooth_evaluations + target_evaluations,
        "adversarial_order_key": [
            int(failure),
            int(bool(target_shortest) and not bool(smooth_shortest)),
            -target_fraction,
            worst_rank,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--candidate-ids", type=int, nargs="+")
    parser.add_argument("--log-condition", type=float, default=1.6)
    parser.add_argument("--field-cutoff", type=int, default=2)
    parser.add_argument("--probe-power", type=int, default=6)
    parser.add_argument("--smooth-t", type=float, default=0.10)
    parser.add_argument("--target-t", type=float, default=DEFAULT_R)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate_ids = (
        args.candidate_ids
        if args.candidate_ids is not None
        else list(range(args.candidates))
    )
    rows = [
        audit_candidate(
            args.dimension,
            candidate,
            log_condition=args.log_condition,
            field_cutoff=args.field_cutoff,
            probe_power=args.probe_power,
            smooth_t=args.smooth_t,
            target_t=args.target_t,
            seed=args.seed,
        )
        for candidate in candidate_ids
    ]
    rows.sort(key=lambda row: tuple(row["adversarial_order_key"]), reverse=True)
    report = {
        "experiment": "walsh_periodic_hessian_adversarial_search",
        "dimension": args.dimension,
        "candidate_count": len(candidate_ids),
        "probe_count": 1 << args.probe_power,
        "smooth_t": args.smooth_t,
        "target_t": args.target_t,
        "observed_inheritance_failure_count": sum(
            int(row["observed_inheritance_failure"]) for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
