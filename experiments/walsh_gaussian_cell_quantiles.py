#!/usr/bin/env python3
"""Test whether cold zero-parity rejection repairs Gaussian basis cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_gaussian_cell_seed import (
    coefficient_gaussians,
    parity_rows,
)
from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_spectral_parity_sieve import rectangular_basis


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_gaussian_cell_quantiles.json"
)


def central_probability_scale(
    basis: np.ndarray,
    physical_gaussians: np.ndarray,
    central_probability: float,
) -> float:
    """Empirical scale with the requested probability of rounding to zero."""
    if not 0.0 < central_probability < 1.0:
        raise ValueError("central probability must lie strictly between zero and one")
    coefficients = coefficient_gaussians(basis, physical_gaussians)
    threshold = 0.5 / np.max(np.abs(coefficients), axis=1)
    return float(np.quantile(threshold, 1.0 - central_probability))


def audit_fixture(
    family: str,
    basis: np.ndarray,
    *,
    scale_gaussians: np.ndarray,
    law_gaussians: np.ndarray,
    central_probabilities: tuple[float, ...],
    shortest_cutoff: int,
) -> dict[str, object]:
    n = basis.shape[0]
    shortest, coefficients = shortest_vector_coefficients(
        basis, cutoff=shortest_cutoff
    )
    target = parity_rows(coefficients)
    coefficient_law = coefficient_gaussians(basis, law_gaussians[:, :n])
    levels = []
    for requested in central_probabilities:
        tau = central_probability_scale(
            basis, scale_gaussians[:, :n], requested
        )
        rounded = np.rint(tau * coefficient_law).astype(np.int64)
        parity = parity_rows(rounded)
        exact_zero = np.all(rounded == 0, axis=1)
        zero_parity = np.all(parity == 0, axis=1)
        accepted = ~zero_parity
        hit = np.all(parity == target[None, :], axis=1)
        acceptance = float(np.mean(accepted))
        levels.append({
            "requested_central_cell_probability": requested,
            "scale": tau,
            "empirical_central_cell_probability": float(np.mean(exact_zero)),
            "empirical_zero_parity_probability": float(np.mean(zero_parity)),
            "nonzero_parity_acceptance_probability": acceptance,
            "cheap_trials_per_accepted_parity": (
                1.0 / acceptance if acceptance > 0.0 else float("inf")
            ),
            "chosen_shortest_parity_mass_after_zero_parity_rejection": (
                float(np.mean(hit[accepted])) if np.any(accepted) else 0.0
            ),
        })
    return {
        "family": family,
        "dimension": n,
        "shortest_length": shortest,
        "shortest_coefficients": coefficients.tolist(),
        "shortest_parity": target.tolist(),
        "levels": levels,
    }


def build_report(
    dimensions: tuple[int, ...],
    samples: int,
    central_probabilities: tuple[float, ...],
    shortest_cutoff: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    maximum = max(dimensions)
    scale_gaussians = rng.normal(size=(samples, maximum))
    law_gaussians = rng.normal(size=(samples, maximum))
    rows = []
    for n in dimensions:
        fixtures = [
            ("rectangular", rectangular_basis(n, float(2**n))),
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
        ]
        if n >= 3:
            fixtures.append(("needle_D_shell", needle_d_basis(n, 1.03)))
        for family, basis in fixtures:
            rows.append(audit_fixture(
                family,
                basis,
                scale_gaussians=scale_gaussians,
                law_gaussians=law_gaussians,
                central_probabilities=central_probabilities,
                shortest_cutoff=shortest_cutoff,
            ))
    return {
        "experiment": "walsh_gaussian_basis_cell_quantile_sweep",
        "interpretation": (
            "Cold rejection can amplify a shortest parity only when that "
            "parity is adjacent to the central basis parallelepiped.  Dense "
            "cancellation parities require simultaneous facet crossings."
        ),
        "samples_per_scale_and_law_audit": samples,
        "central_cell_probabilities": list(central_probabilities),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 6, 8))
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument(
        "--central-probabilities",
        type=float,
        nargs="+",
        default=(0.99, 0.9, 0.5, 0.1),
    )
    parser.add_argument("--shortest-cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802479)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.samples,
        tuple(args.central_probabilities),
        args.shortest_cutoff,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
