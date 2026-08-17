#!/usr/bin/env python3
"""Finite audit of causal label transport across the Voronoi tiling.

The earlier neighboring-ray experiment compares two facets of the origin
Voronoi cell.  That is not the first-arrival collision.  Here a point is drawn
uniformly modulo the lattice, displaced by a short isotropic physical step,
and the nearest-lattice labels before and after the step are compared.  An
infinitesimal label change crosses one translated Voronoi facet, so its label
difference is exactly the causal collision vector of that facet.

All nearest-point and shortest-vector calculations use finite coefficient
cubes.  The output is a falsification/target-selection audit, not a CVP oracle
or a worst-case completeness certificate.
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

from experiments.walsh_dense_core_transport import (
    first_exit_labels,
    parity_indices,
)
from experiments.walsh_hessian_noise_audit import generic_basis
from experiments.walsh_periodic_hessian_adversarial_search import adversarial_basis
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_spectral_parity_sieve import rectangular_basis
from experiments.walsh_voronoi_first_exit import coefficient_cube


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_voronoi_causal_transport.json"
)


def coefficient_cube_with_zero(n: int, cutoff: int) -> np.ndarray:
    nonzero = coefficient_cube(n, cutoff)
    return np.vstack([np.zeros((1, n), dtype=np.int64), nonzero])


def nearest_label_indices(
    points: np.ndarray,
    vectors: np.ndarray,
    *,
    batch: int = 128,
) -> np.ndarray:
    """Nearest enumerated vector for each physical row point."""
    sites = np.asarray(vectors, dtype=np.float64)
    site_norm2 = np.einsum("ij,ij->i", sites, sites)
    query = np.asarray(points, dtype=np.float64)
    winners = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), batch):
        stop = min(start + batch, len(query))
        block = query[start:stop]
        distances = (
            site_norm2[:, None]
            - 2.0 * sites @ block.T
            + np.einsum("ij,ij->i", block, block)[None, :]
        )
        winners[start:stop] = np.argmin(distances, axis=0)
    return winners


def gap_cap_parameters(second_length_ratio: float) -> dict[str, float]:
    """Universal winner cap implied by a gap above the shortest shell."""
    gamma = max(float(second_length_ratio), 1.0)
    if gamma <= math.sqrt(2.0):
        cosine = math.sqrt(max(0.0, 1.0 - gamma * gamma / 4.0))
    else:
        cosine = 0.0 if math.isinf(gamma) else 1.0 / gamma
    exponent = -0.5 * math.log2(max(1.0 - cosine * cosine, 1e-300))
    return {
        "nonshort_length_ratio": gamma,
        "winner_cap_cosine": cosine,
        "asymptotic_direct_mass_exponent": exponent,
    }


def audit_basis(
    family: str,
    basis: np.ndarray,
    coefficients: np.ndarray,
    displacements: np.ndarray,
    directions: np.ndarray,
    *,
    cutoff: int,
    step_ratios: tuple[float, ...],
) -> dict[str, object]:
    n = basis.shape[0]
    if any(float(ratio) <= 0.0 for ratio in step_ratios):
        raise ValueError("step ratios must be positive")
    labels = coefficient_cube_with_zero(n, cutoff)
    vectors = labels @ basis.T
    vector_norm2 = np.einsum("ij,ij->i", vectors, vectors)
    nonzero = np.any(labels != 0, axis=1)
    shortest2 = float(np.min(vector_norm2[nonzero]))
    shortest_length = math.sqrt(shortest2)
    shortest_vectors = nonzero & (
        vector_norm2 <= shortest2 * (1.0 + 1e-10)
    )
    shortest_parities = np.unique(parity_indices(labels[shortest_vectors]))
    shortest_parity_indicator = np.zeros(1 << n, dtype=bool)
    shortest_parity_indicator[shortest_parities] = True

    oriented_shortest = vectors[np.flatnonzero(shortest_vectors)[0]]
    projections = (
        vectors @ oriented_shortest / shortest2
    )[:, None] * oriented_shortest[None, :]
    noncollinear = np.linalg.norm(vectors - projections, axis=1) > (
        1e-9 * shortest_length
    )
    competitors = vector_norm2[nonzero & noncollinear]
    second_ratio = (
        math.sqrt(float(np.min(competitors)) / shortest2)
        if len(competitors) else math.inf
    )

    coefficient_points = np.asarray(coefficients[:, :n], dtype=np.float64)
    physical_points = coefficient_points @ basis.T
    unit_steps = np.asarray(displacements[:, :n], dtype=np.float64)
    unit_steps /= np.linalg.norm(unit_steps, axis=1)[:, None]
    base_indices = nearest_label_indices(physical_points, vectors)
    base_labels = labels[base_indices]

    rows = []
    for ratio in step_ratios:
        step_length = float(ratio) * shortest_length
        moved_points = physical_points + step_length * unit_steps
        moved_indices = nearest_label_indices(moved_points, vectors)
        moved_labels = labels[moved_indices]
        differences = moved_labels - base_labels
        changed = np.any(differences != 0, axis=1)
        difference_vectors = differences @ basis.T
        difference_norm2 = np.einsum(
            "ij,ij->i", difference_vectors, difference_vectors
        )
        shortest_crossing = changed & (
            difference_norm2 <= shortest2 * (1.0 + 1e-9)
        )
        difference_parities = parity_indices(differences)
        shortest_parity = changed & shortest_parity_indicator[
            difference_parities
        ]
        rows.append({
            "step_over_lambda": float(ratio),
            "label_change_mass": float(np.mean(changed)),
            "label_change_flux_per_lambda": float(
                np.mean(changed) / float(ratio)
            ),
            "shortest_vector_crossing_mass": float(
                np.mean(shortest_crossing)
            ),
            "shortest_vector_flux_per_lambda": float(
                np.mean(shortest_crossing) / float(ratio)
            ),
            "shortest_parity_crossing_mass": float(np.mean(shortest_parity)),
            "shortest_parity_flux_per_lambda": float(
                np.mean(shortest_parity) / float(ratio)
            ),
            "shortest_parity_given_label_change": (
                float(np.mean(shortest_parity[changed]))
                if np.any(changed) else 0.0
            ),
        })

    nonzero_vectors = vectors[nonzero]
    nonzero_norm2 = vector_norm2[nonzero]
    ray_winners = first_exit_labels(
        nonzero_vectors,
        nonzero_norm2,
        directions[:, :n],
    )
    ray_labels = labels[nonzero][ray_winners]
    ray_parities = parity_indices(ray_labels)

    return {
        "family": family,
        "dimension": n,
        "coefficient_cutoff": cutoff,
        "enumerated_candidate_labels": int(len(labels)),
        "shortest_length": shortest_length,
        "distinct_shortest_parities": int(len(shortest_parities)),
        "gap_cap_diagnostic": gap_cap_parameters(second_ratio),
        "direct_first_exit_shortest_parity_mass": float(np.mean(
            shortest_parity_indicator[ray_parities]
        )),
        "causal_steps": rows,
    }


def build_report(
    dimensions: tuple[int, ...],
    samples: int,
    cutoff: int,
    step_ratios: tuple[float, ...],
    random_replicates: int,
    adversarial_replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    maximum = max(dimensions)
    coefficients = rng.uniform(-0.5, 0.5, size=(samples, maximum))
    displacements = rng.normal(size=(samples, maximum))
    directions = rng.normal(size=(samples, maximum))
    rows = []
    for n in dimensions:
        fixtures: list[tuple[str, np.ndarray]] = [
            ("rectangular", rectangular_basis(n, float(2**n))),
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
        ]
        for replicate in range(random_replicates):
            fixtures.append((f"generic_{replicate}", generic_basis(n, rng)))
        for replicate in range(adversarial_replicates):
            fixtures.append((
                f"adversarial_{replicate}",
                adversarial_basis(n, rng, 2.0 + 2.0 * rng.random()),
            ))
        if n >= 3:
            fixtures.append(("needle_D_shell", needle_d_basis(n, 1.03)))
        for family, basis in fixtures:
            rows.append(audit_basis(
                family,
                basis,
                coefficients,
                displacements,
                directions,
                cutoff=cutoff,
                step_ratios=step_ratios,
            ))
    return {
        "experiment": "walsh_voronoi_causal_tiling_transport",
        "samples_per_fixture": samples,
        "warning": (
            "Nearest labels, shortest vectors, and first exits are minimized "
            "over a finite coefficient cube.  Finite-step flux divided by "
            "step length is diagnostic and does not certify the infinitesimal "
            "facet-area limit or a uniform robust scale."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--samples", type=int, default=16_384)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument(
        "--step-ratios", type=float, nargs="+", default=(0.01, 0.03, 0.1)
    )
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--adversarial-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260802484)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.samples,
        args.cutoff,
        tuple(args.step_ratios),
        args.random_replicates,
        args.adversarial_replicates,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
