#!/usr/bin/env python3
"""Finite-population audit of the accessible-scale empirical soft oracle.

Samples are drawn from the target-width dual DGS at `r` and reweighted to the
accessible scale `t<r`.  Since the target distribution is narrower,

    u_t(x) = exp(-pi (s_t^2-s_r^2) ||x||^2) <= 1.

The estimator is the ratio of the weighted trigonometric sum to the weighted
mass.  This audit compares its soft-Hessian value, gradient, and numerical
Hessian with the exact normalized finite-population field.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_accessible_seed_law import accessible_scale
from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)
from experiments.walsh_periodic_hessian_continuation import spatial_width
from experiments.walsh_periodic_hessian_descent import DEFAULT_R, coefficient_box


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_empirical_soft_oracle.json"


def normalized_gaussian_probabilities(
    dual_points: np.ndarray, width: float
) -> np.ndarray:
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    log_weight = -math.pi * width * width * norm2
    return np.exp(log_weight - logsumexp(log_weight))


def narrowing_importance_weights(
    dual_points: np.ndarray, source_width: float, target_width: float
) -> np.ndarray:
    if target_width < source_width:
        raise ValueError("narrowing requires target spatial width >= source width")
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    return np.exp(
        -math.pi * (target_width * target_width - source_width * source_width)
        * norm2
    )


def soft_oracle_from_measure(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    probabilities: np.ndarray,
    *,
    beta: float,
    normalization: float,
) -> tuple[float, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    z = basis @ np.asarray(coefficient_point, dtype=np.float64)
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    phase = 2.0 * math.pi * (dual_points @ z)
    cosine = np.cos(phase)
    sine = np.sin(phase)
    n = basis.shape[0]
    outer = np.einsum(
        "i,i,ij,ik->jk", probabilities, cosine, dual_points, dual_points
    )
    trace_outer = float(np.dot(probabilities * cosine, norm2))
    matrix = -4.0 * math.pi**2 * (
        outer - (trace_outer / n) * np.eye(n)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    logits = beta * eigenvalues / normalization
    spectral_probabilities = np.exp(logits - logsumexp(logits))
    density = (
        eigenvectors * spectral_probabilities[None, :]
    ) @ eigenvectors.T
    value = normalization * float(logsumexp(logits)) / beta
    quadratic = np.einsum("ij,jk,ik->i", dual_points, density, dual_points)
    contraction = quadratic - norm2 / n
    gradient_z = 8.0 * math.pi**3 * np.sum(
        (probabilities * sine * contraction)[:, None] * dual_points,
        axis=0,
    )
    return value, basis.T @ gradient_z


def traceless_hessian_from_measure(
    coefficient_point: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    z = basis @ np.asarray(coefficient_point, dtype=np.float64)
    norm2 = np.einsum("ij,ij->i", dual_points, dual_points)
    cosine = np.cos(2.0 * math.pi * (dual_points @ z))
    n = basis.shape[0]
    outer = np.einsum(
        "i,i,ij,ik->jk", probabilities, cosine, dual_points, dual_points
    )
    trace_outer = float(np.dot(probabilities * cosine, norm2))
    return -4.0 * math.pi**2 * (
        outer - (trace_outer / n) * np.eye(n)
    )


def numerical_hessian(
    coefficient_point: np.ndarray,
    oracle,
    step: float = 2e-5,
) -> np.ndarray:
    n = len(coefficient_point)
    hessian = np.empty((n, n), dtype=np.float64)
    for axis in range(n):
        delta = np.zeros(n)
        delta[axis] = step
        high = oracle(coefficient_point + delta)[1]
        low = oracle(coefficient_point - delta)[1]
        hessian[:, axis] = (high - low) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def audit_dimension(
    n: int,
    *,
    cutoff: int,
    samples: int,
    probes: int,
    beta: float,
    r: float,
    seed: int,
) -> dict[str, object]:
    basis_rng = np.random.default_rng(seed + 1009 * n)
    sample_rng = np.random.default_rng(seed + 65537 * n)
    probe_rng = np.random.default_rng(seed + 104729 * n)
    basis = generic_basis(n, basis_rng)
    shortest, shortest_coefficients = shortest_vector_coefficients(basis, cutoff=2)
    t = accessible_scale(r)
    source_width = spatial_width(n, r, shortest)
    target_width = spatial_width(n, t, shortest)
    dual_points = coefficient_box(n, cutoff) @ np.linalg.inv(basis)
    source_probability = normalized_gaussian_probabilities(dual_points, source_width)
    target_probability = normalized_gaussian_probabilities(dual_points, target_width)
    importance = narrowing_importance_weights(
        dual_points, source_width, target_width
    )

    exact_reweighted = source_probability * importance
    exact_reweighted /= np.sum(exact_reweighted)
    population_identity_error = float(
        np.max(np.abs(exact_reweighted - target_probability))
    )

    sampled_indices = sample_rng.choice(
        len(dual_points), size=samples, replace=True, p=source_probability
    )
    sampled_points = dual_points[sampled_indices]
    sampled_weights = importance[sampled_indices]
    sampled_probability = sampled_weights / np.sum(sampled_weights)
    effective_sample_size = float(
        np.sum(sampled_weights) ** 2 / np.sum(sampled_weights**2)
    )

    reference_matrix = traceless_hessian_from_measure(
        0.5 * shortest_coefficients,
        basis,
        dual_points,
        target_probability,
    )
    normalization = max(float(np.linalg.norm(reference_matrix, ord=2)), 1e-12)

    exact_oracle = lambda point: soft_oracle_from_measure(
        point,
        basis,
        dual_points,
        target_probability,
        beta=beta,
        normalization=normalization,
    )
    empirical_oracle = lambda point: soft_oracle_from_measure(
        point,
        basis,
        sampled_points,
        sampled_probability,
        beta=beta,
        normalization=normalization,
    )

    rows = []
    for index in range(probes):
        point = probe_rng.uniform(-0.5, 0.5, size=n)
        exact_value, exact_gradient = exact_oracle(point)
        empirical_value, empirical_gradient = empirical_oracle(point)
        exact_hessian = numerical_hessian(point, exact_oracle)
        empirical_hessian = numerical_hessian(point, empirical_oracle)
        rows.append({
            "probe": index,
            "value_absolute_error_over_normalization": (
                abs(empirical_value - exact_value) / normalization
            ),
            "gradient_error_over_normalization": (
                float(np.linalg.norm(empirical_gradient - exact_gradient))
                / normalization
            ),
            "hessian_error_over_normalization": (
                float(np.linalg.norm(
                    empirical_hessian - exact_hessian, ord=2
                )) / normalization
            ),
        })
    return {
        "dimension": n,
        "r": r,
        "t": t,
        "samples": samples,
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / samples,
        "population_importance_identity_error": population_identity_error,
        "maximum_importance_weight": float(np.max(importance)),
        "normalization": normalization,
        "maximum_value_error_over_normalization": max(
            row["value_absolute_error_over_normalization"] for row in rows
        ),
        "maximum_gradient_error_over_normalization": max(
            row["gradient_error_over_normalization"] for row in rows
        ),
        "maximum_hessian_error_over_normalization": max(
            row["hessian_error_over_normalization"] for row in rows
        ),
        "rms_value_error_over_normalization": float(np.sqrt(np.mean([
            row["value_absolute_error_over_normalization"] ** 2 for row in rows
        ]))),
        "rms_gradient_error_over_normalization": float(np.sqrt(np.mean([
            row["gradient_error_over_normalization"] ** 2 for row in rows
        ]))),
        "rms_hessian_error_over_normalization": float(np.sqrt(np.mean([
            row["hessian_error_over_normalization"] ** 2 for row in rows
        ]))),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = {
        "experiment": "walsh_empirical_soft_oracle",
        "rows": [
            audit_dimension(
                n,
                cutoff=args.cutoff,
                samples=args.samples,
                probes=args.probes,
                beta=args.beta,
                r=args.r,
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
