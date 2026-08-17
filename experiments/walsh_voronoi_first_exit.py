#!/usr/bin/env python3
"""Finite audit of random-ray first exits from the origin Voronoi cell."""

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

from experiments.walsh_hessian_noise_audit import _integer_chunk, generic_basis
from experiments.walsh_periodic_hessian_adversarial_search import adversarial_basis
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_spectral_parity_sieve import rectangular_basis


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_voronoi_first_exit.json"


def coefficient_cube(n: int, cutoff: int) -> np.ndarray:
    count = (2 * cutoff + 1) ** n
    coefficients = _integer_chunk(0, count, n, cutoff).astype(np.int64)
    return coefficients[np.any(coefficients != 0, axis=1)]


def first_exit_audit(
    family: str,
    basis: np.ndarray,
    directions: np.ndarray,
    *,
    cutoff: int,
    direction_batch: int = 128,
) -> dict[str, object]:
    n = basis.shape[0]
    coefficients = coefficient_cube(n, cutoff)
    vectors = coefficients @ basis.T
    norm2 = np.einsum("ij,ij->i", vectors, vectors)
    shortest2 = float(np.min(norm2))
    shortest = norm2 <= shortest2 * (1.0 + 1e-10)
    shortest_parities = {
        tuple(map(int, row & 1)) for row in coefficients[shortest]
    }
    unit = directions[:, :n].copy()
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    winner_counts = np.zeros(len(coefficients), dtype=np.int64)
    boundary = np.empty(len(unit), dtype=np.float64)
    winners = np.empty(len(unit), dtype=np.int64)
    for start in range(0, len(unit), direction_batch):
        stop = min(start + direction_batch, len(unit))
        dot = vectors @ unit[start:stop].T
        times = np.divide(
            norm2[:, None],
            2.0 * dot,
            out=np.full_like(dot, np.inf),
            where=dot > 0.0,
        )
        local = np.argmin(times, axis=0)
        winners[start:stop] = local
        boundary[start:stop] = times[local, np.arange(stop - start)]
        winner_counts += np.bincount(local, minlength=len(coefficients))
    winning_shortest = shortest[winners]
    winning_parity = coefficients[winners] & 1
    shortest_parity_hit = np.asarray(
        [tuple(map(int, row)) in shortest_parities for row in winning_parity],
        dtype=bool,
    )
    active = np.flatnonzero(winner_counts)
    top = active[np.argsort(winner_counts[active])[::-1][:10]]
    return {
        "family": family,
        "dimension": n,
        "coefficient_cutoff": cutoff,
        "enumerated_nonzero_vectors": int(len(coefficients)),
        "shortest_length": math.sqrt(shortest2),
        "enumerated_shortest_vector_count": int(np.count_nonzero(shortest)),
        "distinct_shortest_parity_count": len(shortest_parities),
        "random_ray_shortest_vector_exit_mass": float(np.mean(winning_shortest)),
        "random_ray_shortest_parity_exit_mass": float(np.mean(shortest_parity_hit)),
        "active_voronoi_labels_in_audit": int(len(active)),
        "median_first_boundary_radius_over_lambda_half": float(
            np.median(boundary) / (0.5 * math.sqrt(shortest2))
        ),
        "top_exit_labels": [
            {
                "coefficients": coefficients[index].tolist(),
                "parity": (coefficients[index] & 1).tolist(),
                "length_over_shortest": float(
                    math.sqrt(norm2[index] / shortest2)
                ),
                "ray_mass": float(winner_counts[index] / len(unit)),
            }
            for index in top
        ],
    }


def build_report(
    dimensions: tuple[int, ...],
    samples: int,
    cutoff: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(samples, max(dimensions)))
    rows = []
    for n in dimensions:
        fixtures = [
            ("rectangular", rectangular_basis(n, float(2**n))),
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
            ("generic", generic_basis(n, rng)),
            ("adversarial_rotated", adversarial_basis(n, rng, 2.0)),
        ]
        if n >= 3:
            fixtures.append(("needle_D_shell", needle_d_basis(n, 1.03)))
        for family, basis in fixtures:
            rows.append(first_exit_audit(
                family,
                basis,
                directions,
                cutoff=cutoff,
            ))
    return {
        "experiment": "walsh_voronoi_random_ray_first_exit",
        "warning": (
            "Voronoi labels are minimized only over the displayed finite "
            "coefficient cube; this is a falsification audit, not a "
            "worst-case completeness certificate."
        ),
        "directions_per_fixture": samples,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 6))
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802480)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions), args.samples, args.cutoff, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
