#!/usr/bin/env python3
"""Stress census for a unique shortest direction against a D_m shell.

The generic continuation fixtures are close to orthogonal and expose only a
few competing edge families.  Here the lattice is

    Z e_0  orthogonal-sum  ((1+epsilon)/sqrt(2)) D_(n-1).

The needle has unique shortest-vector pair of length one.  The D block has
m(m-1) root pairs, all of length 1+epsilon.  It is therefore a finite proxy
for the worst-case geometry in which many almost-short Voronoi directions
compete with one true shortest direction.
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
    ROOT / "experiments" / "out" / "walsh_periodic_hessian_stress_census.json"
)


def d_root_basis(m: int) -> np.ndarray:
    """Column basis of D_m={x in Z^m: sum(x)=0 mod 2}."""
    if m < 2:
        raise ValueError("D_m requires m >= 2")
    basis = np.zeros((m, m), dtype=np.float64)
    for column in range(m - 1):
        basis[column, column] = 1.0
        basis[column + 1, column] = -1.0
    basis[m - 2, m - 1] = 1.0
    basis[m - 1, m - 1] = 1.0
    return basis


def needle_d_basis(
    n: int, shell_ratio: float, tilt_amplitude: float = 0.0
) -> np.ndarray:
    if n < 3:
        raise ValueError("needle-D stress family requires n >= 3")
    if shell_ratio <= 1.0:
        raise ValueError("shell ratio must exceed one")
    roots = d_root_basis(n - 1)
    basis = np.zeros((n, n), dtype=np.float64)
    basis[0, 0] = 1.0
    # A deterministic irrational-looking tilt breaks the D_m symmetry while
    # keeping the shell and its many root directions recognizable.  A shell
    # point x is embedded as (a.x, scale*x), modulo the integral needle.
    angles = np.arange(1, n, dtype=np.float64) * math.sqrt(2.0)
    tilt = tilt_amplitude * np.sin(angles)
    basis[0, 1:] = tilt @ roots
    basis[1:, 1:] = (shell_ratio / math.sqrt(2.0)) * roots
    return basis


def branch_record(
    branch: dict[str, object],
    *,
    basis: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
    probe_count: int,
    carried: bool,
) -> dict[str, object]:
    classification = classify_branch(
        branch,
        basis=basis,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
    )
    edge = list(classification["edge"])
    return {
        "edge": edge,
        "edge_norm_over_shortest": classification["edge_norm_over_shortest"],
        "shortest_edge": classification["shortest_edge"],
        "needle_edge": bool(abs(edge[0]) == 1 and not any(edge[1:])),
        "probe_basin_count": int(branch["merged_copies"]),
        "probe_basin_fraction": int(branch["merged_copies"]) / probe_count,
        "carried_from_previous_scale": carried,
        "score": float(branch["score"]),
        "eigengap": float(branch["eigengap"]),
    }


def audit_dimension(
    n: int,
    *,
    shell_ratio: float,
    tilt_amplitude: float,
    field_cutoff: int,
    scales: list[float],
    probe_power: int,
    seed: int,
) -> dict[str, object]:
    basis = needle_d_basis(n, shell_ratio, tilt_amplitude)
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(field_cutoff + 1, 3)
    )
    dual_points = coefficient_box(n, field_cutoff) @ np.linalg.inv(basis)
    coefficients = coefficient_box(n, field_cutoff + 2)
    points = coefficients @ basis.T
    starts = sobol_starts(n, probe_power, seed + 104729 * n)
    previous: list[dict[str, object]] = []
    levels = []
    for level, t in enumerate(scales):
        branches, evaluations = census_scale(
            starts,
            basis=basis,
            dual_points=dual_points,
            shortest=shortest,
            shortest_coefficients=shortest_coefficients,
            t=t,
        )
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
        records = [
            branch_record(
                branch,
                basis=basis,
                coefficients=coefficients,
                points=points,
                shortest=shortest,
                probe_count=len(starts),
                carried=(
                    is_carried(branch, transported, basis, shortest)
                    if transported else False
                ),
            )
            for branch in branches
        ]
        needle = [record for record in records if record["needle_edge"]]
        score_order = sorted(records, key=lambda record: record["score"], reverse=True)
        basin_order = sorted(
            records, key=lambda record: record["probe_basin_count"], reverse=True
        )
        levels.append({
            "level": level,
            "t": t,
            "field_gradient_evaluations": evaluations,
            "branch_count": len(records),
            "theoretical_competing_d_root_pairs": (n - 1) * (n - 2),
            "needle_present": bool(needle),
            "needle_basin_fraction": sum(
                record["probe_basin_fraction"] for record in needle
            ),
            "needle_score_rank": min(
                (score_order.index(record) + 1 for record in needle), default=None
            ),
            "needle_basin_rank": min(
                (basin_order.index(record) + 1 for record in needle), default=None
            ),
            "needle_carried_from_previous_scale": any(
                record["carried_from_previous_scale"] for record in needle
            ),
            "uncarried_branch_count": sum(
                not record["carried_from_previous_scale"] for record in records
            ),
            "branches": records,
        })
        previous = branches
    return {
        "dimension": n,
        "shell_dimension": n - 1,
        "shell_ratio": shell_ratio,
        "tilt_amplitude": tilt_amplitude,
        "basis_condition_number": float(np.linalg.cond(basis)),
        "shortest_length": shortest,
        "shortest_coefficients": shortest_coefficients.tolist(),
        "probe_count": len(starts),
        "field_cutoff": field_cutoff,
        "levels": levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=6)
    parser.add_argument("--shell-ratio", type=float, default=1.03)
    parser.add_argument("--tilt-amplitude", type=float, default=0.0)
    parser.add_argument("--field-cutoff", type=int, default=2)
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.10, 0.12, 0.15, 0.18, 0.21, DEFAULT_R],
    )
    parser.add_argument("--probe-power", type=int, default=9)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "experiment": "walsh_periodic_hessian_stress_census",
        "family": "Z needle orthogonal-sum scaled D_(n-1) shell",
        "rows": [
            audit_dimension(
                n,
                shell_ratio=args.shell_ratio,
                tilt_amplitude=args.tilt_amplitude,
                field_cutoff=args.field_cutoff,
                scales=list(args.scales),
                probe_power=args.probe_power,
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
