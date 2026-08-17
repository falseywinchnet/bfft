#!/usr/bin/env python3
"""Audit the directly samplable Gaussian-cell shortest-parity law."""

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

from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)
from experiments.walsh_periodic_hessian_adversarial_search import adversarial_basis
from experiments.walsh_periodic_hessian_simplex_census import simplex_cancellation_basis
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_spectral_parity_sieve import rectangular_basis


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_gaussian_cell_seed.json"
DELTA_QUERY = 0.02648284


def parity_rows(coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(coefficients, dtype=np.int64) & 1


def coefficient_gaussians(basis: np.ndarray, physical_gaussians: np.ndarray) -> np.ndarray:
    """Map row-wise physical standard Gaussians into coefficient space."""
    return np.linalg.solve(basis, physical_gaussians.T).T


def median_cell_scale(basis: np.ndarray, physical_gaussians: np.ndarray) -> float:
    coefficients = coefficient_gaussians(basis, physical_gaussians)
    maxima = np.max(np.abs(coefficients), axis=1)
    thresholds = np.divide(
        0.5,
        maxima,
        out=np.full_like(maxima, np.inf),
        where=maxima > 0.0,
    )
    return float(np.median(thresholds))


def gaussian_cell_audit(
    family: str,
    basis: np.ndarray,
    *,
    median_gaussians: np.ndarray,
    law_gaussians: np.ndarray,
    shortest_cutoff: int,
) -> dict[str, object]:
    n = basis.shape[0]
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=shortest_cutoff
    )
    target = parity_rows(shortest_coefficients)
    tau = median_cell_scale(basis, median_gaussians[:, :n])
    coefficient_samples = tau * coefficient_gaussians(basis, law_gaussians[:, :n])
    rounded = np.rint(coefficient_samples).astype(np.int64)
    zero = np.all(rounded == 0, axis=1)
    accepted = rounded[~zero]
    accepted_parity = parity_rows(accepted)
    hits = np.all(accepted_parity == target[None, :], axis=1)
    central_probability = float(np.mean(zero))
    accepted_count = int(len(accepted))
    empirical_hit = float(np.mean(hits)) if accepted_count else 0.0
    shift = math.exp(-(shortest * shortest) / (2.0 * tau * tau))
    certified_lower = min(
        1.0,
        2.0 * shift * central_probability / max(1.0 - central_probability, 1e-300),
    )
    exponent = (shortest * shortest) / (
        2.0 * tau * tau * n * math.log(2.0)
    )
    inverse = np.linalg.inv(basis)
    maximum_dual_row = float(np.max(np.linalg.norm(inverse, axis=1)))
    union_scale = 1.0 / (
        2.0 * maximum_dual_row * math.sqrt(2.0 * math.log(4.0 * n))
    )
    union_exponent = (shortest * shortest) / (
        2.0 * union_scale * union_scale * n * math.log(2.0)
    )
    return {
        "family": family,
        "dimension": n,
        "shortest_length": float(shortest),
        "shortest_coefficients": shortest_coefficients.tolist(),
        "shortest_parity": target.tolist(),
        "median_cell_scale": tau,
        "median_scale_times_sqrt_n_over_lambda": tau * math.sqrt(n) / shortest,
        "empirical_central_cell_probability": central_probability,
        "accepted_nonzero_samples": accepted_count,
        "empirical_chosen_shortest_parity_mass_after_zero_rejection": empirical_hit,
        "cell_shift_certified_mass_lower_bound": certified_lower,
        "cell_shift_exponent_per_dimension": exponent,
        "clears_delta_query_by_exact_median": exponent < DELTA_QUERY,
        "maximum_dual_basis_row_norm": maximum_dual_row,
        "dual_union_bound_scale": union_scale,
        "dual_union_bound_exponent_per_dimension": union_exponent,
        "clears_delta_query_by_dual_union_bound": union_exponent < DELTA_QUERY,
    }


def build_report(
    min_dimension: int,
    max_dimension: int,
    samples: int,
    shortest_cutoff: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    median_gaussians = rng.normal(size=(samples, max_dimension))
    law_gaussians = rng.normal(size=(samples, max_dimension))
    rows: list[dict[str, object]] = []
    for n in range(min_dimension, max_dimension + 1):
        fixtures = [
            ("rectangular", rectangular_basis(n, float(2**n))),
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
            ("generic", generic_basis(n, rng)),
            ("adversarial_rotated", adversarial_basis(n, rng, 2.0)),
        ]
        if n >= 3:
            fixtures.append(("needle_D_shell", needle_d_basis(n, 1.03)))
        for family, basis in fixtures:
            rows.append(gaussian_cell_audit(
                family,
                basis,
                median_gaussians=median_gaussians,
                law_gaussians=law_gaussians,
                shortest_cutoff=shortest_cutoff,
            ))
    return {
        "experiment": "walsh_gaussian_cell_seed",
        "proved_law": (
            "physical isotropic Gaussian at the coefficient central-cell median; "
            "round B^{-1}Y, reject only Z=0, and output Z mod 2"
        ),
        "delta_query": DELTA_QUERY,
        "samples_per_scale_and_law_audit": samples,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--shortest-cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.min_dimension,
        args.max_dimension,
        args.samples,
        args.shortest_cutoff,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
