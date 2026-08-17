#!/usr/bin/env python3
"""A samplable accessible-scale seed law with gap-free Hessian gradients.

Draw a Haar-uniform torus point, ascend the normalized soft maximum

    Phi_beta(T) = (sigma/beta) log tr exp(beta T/sigma),

and snap the terminal coefficient point to the half grid.  Unlike lambda_max,
Phi_beta is analytic through eigenvalue collisions.  This file constructs the
law and audits its parity capture; it does not assert the still-missing
worst-case lower bound on shortest-parity basin mass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_adversarial_search import (
    half_grid_error,
    snapped_half_parity,
)
from experiments.walsh_periodic_hessian_branch_census import sobol_starts
from experiments.walsh_periodic_hessian_continuation import spatial_width
from experiments.walsh_periodic_hessian_descent import (
    DEFAULT_R,
    coefficient_box,
    periodic_hessian_spectral_data,
    wrap_coefficients,
)
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_accessible_seed_law.json"
T0 = 0.23147


def accessible_scale(
    target_t: float, *, auxiliary_margin: float = 0.005
) -> float:
    """A rigorously controlled scale for the importance second moment.

    Hhan's theta-mass bound requires q=t*r/(2*r-t)>t0/2.  We choose
    q=t0/2+auxiliary_margin and solve for t.  The smaller root of the bare
    half-exponent equation does not satisfy this worst-case condition.
    """
    q = T0 / 2.0 + auxiliary_margin
    return 2.0 * q * target_t / (target_t + q)


def soft_hessian_value_and_gradient(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    spatial_width_value: float,
    *,
    beta: float,
    normalization: float,
) -> tuple[float, np.ndarray]:
    """Analytic soft leading response and coefficient gradient."""
    coefficient_point = wrap_coefficients(coefficient_point)
    z = basis @ coefficient_point
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    weight = np.exp(-math.pi * spatial_width_value**2 * norm2)
    phase = 2.0 * math.pi * (dual_points @ z)
    cosine = np.cos(phase)
    sine = np.sin(phase)
    n = basis.shape[0]
    outer = np.einsum("i,i,ij,ik->jk", weight, cosine, dual_points, dual_points)
    trace_outer = float(np.dot(weight * cosine, norm2))
    matrix = -4.0 * math.pi**2 * (
        outer - (trace_outer / n) * np.eye(n)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    logits = beta * eigenvalues / normalization
    probabilities = np.exp(logits - logsumexp(logits))
    density = (eigenvectors * probabilities[None, :]) @ eigenvectors.T
    value = normalization * float(logsumexp(logits)) / beta

    quadratic = np.einsum("ij,jk,ik->i", dual_points, density, dual_points)
    # tr(density)=1.
    contraction = quadratic - norm2 / n
    gradient_z = 8.0 * math.pi**3 * np.sum(
        (weight * sine * contraction)[:, None] * dual_points,
        axis=0,
    )
    return value, basis.T @ gradient_z


def ascend_soft_hessian(
    start: np.ndarray,
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    width: float,
    beta: float,
    normalization: float,
    max_iterations: int = 120,
) -> dict[str, object]:
    evaluations = 0

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        score, gradient = soft_hessian_value_and_gradient(
            value,
            basis,
            dual_points,
            width,
            beta=beta,
            normalization=normalization,
        )
        return -score, -gradient

    result = minimize(
        objective,
        np.asarray(start, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-9},
    )
    terminal = wrap_coefficients(result.x)
    return {
        "coefficient_point": terminal,
        "snapped_parity": snapped_half_parity(terminal),
        "half_grid_error": half_grid_error(terminal),
        "evaluations": evaluations,
        "optimizer_success": bool(result.success),
    }


def audit_dimension(
    n: int,
    *,
    shortest_ratio: float,
    cutoff: int,
    probe_power: int,
    beta: float,
    target_t: float,
    seed: int,
) -> dict[str, object]:
    basis = simplex_cancellation_basis(n, shortest_ratio)
    shortest, shortest_coefficients = shortest_vector_coefficients(basis, cutoff=2)
    target_parity = tuple(int(value) & 1 for value in shortest_coefficients)
    t = accessible_scale(target_t)
    width = spatial_width(n, t, shortest)
    dual_points = coefficient_box(n, cutoff) @ np.linalg.inv(basis)
    reference_score = periodic_hessian_spectral_data(
        0.5 * shortest_coefficients, basis, dual_points, width
    )[0]
    starts = sobol_starts(n, probe_power, seed + 104729 * n)
    rows = [
        ascend_soft_hessian(
            start,
            basis=basis,
            dual_points=dual_points,
            width=width,
            beta=beta,
            normalization=reference_score,
        )
        for start in starts
    ]
    hits = sum(row["snapped_parity"] == target_parity for row in rows)
    return {
        "dimension": n,
        "accessible_scale": t,
        "beta": beta,
        "probe_count": len(rows),
        "shortest_parity_hits": hits,
        "shortest_parity_fraction": hits / len(rows),
        "distinct_snapped_parities": len({row["snapped_parity"] for row in rows}),
        "maximum_half_grid_error": max(float(row["half_grid_error"]) for row in rows),
        "median_evaluations": float(np.median([row["evaluations"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=6)
    parser.add_argument("--shortest-ratio", type=float, default=0.97)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--probe-power", type=int, default=8)
    parser.add_argument("--beta", type=float, default=16.0)
    parser.add_argument("--target-t", type=float, default=DEFAULT_R)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "experiment": "walsh_accessible_seed_law",
        "law": "Haar torus seed -> normalized soft-Hessian ascent -> half-grid snap",
        "rows": [
            audit_dimension(
                n,
                shortest_ratio=args.shortest_ratio,
                cutoff=args.cutoff,
                probe_power=args.probe_power,
                beta=args.beta,
                target_t=args.target_t,
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
