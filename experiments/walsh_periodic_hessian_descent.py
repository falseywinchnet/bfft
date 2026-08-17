#!/usr/bin/env python3
"""Finite landscape audit for continuous periodic-Hessian descent.

The midpoint-Hessian algorithm evaluates the periodic Gaussian only at the
2^n half-lattice points.  The same full-dual DGS samples give an oracle at an
arbitrary real point.  This audit studies the periodic traceless Hessian

    T_s(z) = Hess F_s(z) - tr(Hess F_s(z)) I/n,

whose value at a shortest-vector midpoint has a rank-one leading term.  It
starts local ascent from random points of a fundamental parallelepiped and
measures how often the terminal nearest-neighbour edge is shortest.

This is a basin experiment, not a proof.  In particular, a finite success
fraction does not establish a dimension-uniform lower bound on the basin
measure for an arbitrary lattice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_periodic_hessian_descent.json"
DEFAULT_R = 0.23675858


def coefficient_box(n: int, cutoff: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(range(-cutoff, cutoff + 1), repeat=n)),
        dtype=np.int16,
    )


def wrap_coefficients(value: np.ndarray) -> np.ndarray:
    """Represent a torus point in the centered coefficient cell."""
    return np.asarray(value, dtype=np.float64) - np.floor(
        np.asarray(value, dtype=np.float64) + 0.5
    )


def periodic_hessian_spectral_data(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    spatial_width: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Leading value, eigengap, coefficient gradient, and eigenvector."""
    coefficient_point = wrap_coefficients(coefficient_point)
    z = basis @ coefficient_point
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    weight = np.exp(-math.pi * spatial_width * spatial_width * norm2)
    phase = 2.0 * math.pi * (dual_points @ z)
    cosine = np.cos(phase)
    sine = np.sin(phase)
    n = basis.shape[0]
    outer = np.einsum("i,i,ij,ik->jk", weight, cosine, dual_points, dual_points)
    trace_outer = float(np.dot(weight * cosine, norm2))
    matrix = -4.0 * math.pi * math.pi * (
        outer - (trace_outer / n) * np.eye(n)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    direction = eigenvectors[:, -1]
    score = float(eigenvalues[-1])
    eigengap = float(eigenvalues[-1] - eigenvalues[-2]) if n > 1 else score

    projection = dual_points @ direction
    directional_coefficient = -4.0 * math.pi * math.pi * (
        projection * projection - norm2 / n
    )
    gradient_z = np.sum(
        (
            -2.0
            * math.pi
            * weight
            * sine
            * directional_coefficient
        )[:, None]
        * dual_points,
        axis=0,
    )
    gradient_coefficients = basis.T @ gradient_z
    return score, eigengap, gradient_coefficients, direction


def periodic_hessian_value_and_gradient(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    spatial_width: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Leading eigenvalue, coefficient gradient, and leading eigenvector."""
    score, _eigengap, gradient, direction = periodic_hessian_spectral_data(
        coefficient_point, basis, dual_points, spatial_width
    )
    return score, gradient, direction


def ascend_periodic_hessian(
    start: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    spatial_width: float,
    *,
    max_iterations: int,
) -> dict[str, object]:
    evaluations = 0
    scores: list[float] = []
    eigengaps: list[float] = []

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        score, eigengap, gradient, _ = periodic_hessian_spectral_data(
            value, basis, dual_points, spatial_width
        )
        scores.append(score)
        eigengaps.append(eigengap)
        return -score, -gradient

    result = minimize(
        objective,
        np.asarray(start, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-9},
    )
    terminal = wrap_coefficients(result.x)
    score, eigengap, gradient, direction = periodic_hessian_spectral_data(
        terminal, basis, dual_points, spatial_width
    )
    return {
        "coefficient_point": terminal,
        "score": score,
        "eigengap": eigengap,
        "initial_score": scores[0],
        "minimum_evaluated_score": min(scores),
        "initial_eigengap": eigengaps[0],
        "minimum_evaluated_eigengap": min(eigengaps),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "leading_direction": direction,
        "evaluations": evaluations,
        "optimizer_success": bool(result.success),
    }


def nearest_edge(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    z = basis @ wrap_coefficients(coefficient_point)
    distance2 = np.sum((points - z[None, :]) ** 2, axis=1)
    nearest = np.argsort(distance2)[:2]
    edge_coefficients = (
        coefficients[nearest[1]].astype(np.int64)
        - coefficients[nearest[0]].astype(np.int64)
    )
    edge = basis @ edge_coefficients
    midpoint_error = abs(
        math.sqrt(float(distance2[nearest[0]]))
        - math.sqrt(float(distance2[nearest[1]]))
    )
    return edge_coefficients, edge, float(np.linalg.norm(edge)), midpoint_error


def audit_dimension(
    n: int,
    *,
    basis_mode: str,
    cutoff: int,
    starts: int,
    r: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n + (0 if basis_mode == "generic" else 1))
    if basis_mode == "generic":
        basis = generic_basis(n, rng)
    elif basis_mode == "orthogonal":
        basis = np.eye(n)
    else:
        raise ValueError(f"unknown basis mode {basis_mode!r}")

    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(cutoff, 2)
    )
    xi = math.sqrt(4.0 * n * r * math.log(2.0) / (math.pi * shortest * shortest))
    spatial_width = 1.0 / xi
    coefficients = coefficient_box(n, cutoff + 1)
    points = coefficients @ basis.T
    field_coefficients = coefficient_box(n, cutoff)
    dual_points = field_coefficients @ np.linalg.inv(basis)

    shortest_midpoint = 0.5 * shortest_coefficients
    shortest_score, shortest_eigengap, _, _ = periodic_hessian_spectral_data(
        shortest_midpoint, basis, dual_points, spatial_width
    )

    rows = []
    shortest_hits = 0
    for index in range(starts):
        start = rng.uniform(-0.5, 0.5, size=n)
        result = ascend_periodic_hessian(
            start,
            basis,
            dual_points,
            spatial_width,
            max_iterations=120,
        )
        edge_coefficients, edge, edge_norm, midpoint_error = nearest_edge(
            result["coefficient_point"], basis, coefficients, points
        )
        is_shortest = edge_norm <= shortest * (1.0 + 1e-7)
        shortest_hits += int(is_shortest)
        alignment = abs(float(
            np.dot(result["leading_direction"], edge) / max(edge_norm, 1e-300)
        ))
        rows.append({
            "start": index,
            "edge_coefficients": edge_coefficients.tolist(),
            "edge_norm_over_shortest": edge_norm / shortest,
            "shortest_edge": is_shortest,
            "midpoint_distance_mismatch": midpoint_error,
            "leading_direction_edge_alignment": alignment,
            "score_over_shortest_midpoint_score": (
                float(result["score"]) / max(shortest_score, 1e-300)
            ),
            "initial_score_over_shortest_midpoint_score": (
                float(result["initial_score"]) / max(shortest_score, 1e-300)
            ),
            "minimum_evaluated_score_over_shortest_midpoint_score": (
                float(result["minimum_evaluated_score"])
                / max(shortest_score, 1e-300)
            ),
            "eigengap_over_shortest_midpoint_eigengap": (
                float(result["eigengap"]) / max(shortest_eigengap, 1e-300)
            ),
            "initial_eigengap_over_shortest_midpoint_eigengap": (
                float(result["initial_eigengap"])
                / max(shortest_eigengap, 1e-300)
            ),
            "minimum_evaluated_eigengap_over_shortest_midpoint_eigengap": (
                float(result["minimum_evaluated_eigengap"])
                / max(shortest_eigengap, 1e-300)
            ),
            "gradient_norm": result["gradient_norm"],
            "evaluations": result["evaluations"],
            "optimizer_success": result["optimizer_success"],
        })

    hit_fraction = shortest_hits / starts
    hit_exponent = (
        -math.log2(hit_fraction) / n if hit_fraction > 0.0 else math.inf
    )
    return {
        "dimension": n,
        "basis_mode": basis_mode,
        "basis_condition_number": float(np.linalg.cond(basis)),
        "r": r,
        "xi": xi,
        "spatial_width": spatial_width,
        "shortest_length": shortest,
        "shortest_coefficients": shortest_coefficients.tolist(),
        "shortest_midpoint_score": shortest_score,
        "shortest_midpoint_eigengap": shortest_eigengap,
        "starts": starts,
        "shortest_basin_hits": shortest_hits,
        "shortest_basin_fraction": hit_fraction,
        "finite_basin_exponent_bits_per_dimension": hit_exponent,
        "available_query_exponent": 0.5 - 2.0 * r,
        "median_evaluations": float(np.median([row["evaluations"] for row in rows])),
        "rows": rows,
    }


def audit(
    *,
    max_dimension: int,
    cutoff: int,
    starts: int,
    r: float,
    seed: int,
) -> dict[str, object]:
    return {
        "experiment": "walsh_periodic_hessian_descent",
        "proof_target": (
            "Find a seed distribution and periodic-Hessian ascent whose total "
            "shortest-midpoint basin mass is at least "
            "2^(-(1/2-2r-o(1))n) for every lattice."
        ),
        "rows": [
            audit_dimension(
                n,
                basis_mode=mode,
                cutoff=cutoff,
                starts=starts,
                r=r,
                seed=seed,
            )
            for n in range(2, max_dimension + 1)
            for mode in ("orthogonal", "generic")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dimension", type=int, default=5)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(
        max_dimension=args.max_dimension,
        cutoff=args.cutoff,
        starts=args.starts,
        r=args.r,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
