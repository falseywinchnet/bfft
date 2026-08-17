#!/usr/bin/env python3
"""Exact Gram-factor GDL cover for the residual shortest-parity problem.

For ``v=Bz`` and a valid length upper bound ``d``, dual-coordinate Cauchy--
Schwarz gives the heterogeneous finite domains

    |z_i| <= R_i := floor(d ||row_i(B^{-1})||_2).

The energy ``||Bz||^2=z^T(B^TB)z`` is a pairwise factor graph on the nonzero
off-diagonal entries of the Gram matrix.  Variable elimination therefore
solves the bounded exact SVP in time polynomial in the largest induced table

    product_{i in bag} (2 R_i + 1).

The all-zero assignment is excluded by at most ``n`` pivot-domain runs.  A
backward GDL pass emits the minimizing coefficient vector, exactly as the
packet preimage does in the seventeen-square transport experiment.

This script audits the weighted min-fill table bound.  It uses floating-point
zero detection only for the fixture diagnostic; the theorem uses exact input
arithmetic to decide whether a Gram entry vanishes.
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

from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)
from experiments.walsh_periodic_hessian_adversarial_search import adversarial_basis
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_spectral_parity_sieve import (
    best_spectral_tube_bound,
    certificate_hamming_sieve_size,
    coefficient_hamming_radius,
    gf2_rank,
    rectangular_basis,
    short_dual_certificate_rows,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_gram_gdl_sieve.json"


def coefficient_domain_radii(basis: np.ndarray, d: float) -> np.ndarray:
    """Proved coordinate bounds for every vector of physical norm at most d."""
    inverse = np.linalg.inv(np.asarray(basis, dtype=np.float64))
    bounds = float(d) * np.linalg.norm(inverse, axis=1)
    return np.floor(bounds + 1e-9).astype(np.int64)


def gram_adjacency(
    basis: np.ndarray,
    *,
    relative_tolerance: float = 1e-12,
) -> dict[int, set[int]]:
    gram = np.asarray(basis, dtype=np.float64).T @ np.asarray(
        basis, dtype=np.float64
    )
    scale = max(float(np.max(np.abs(gram))), 1.0)
    threshold = float(relative_tolerance) * scale
    n = gram.shape[0]
    graph = {index: set() for index in range(n)}
    for first in range(n):
        for second in range(first):
            if abs(float(gram[first, second])) > threshold:
                graph[first].add(second)
                graph[second].add(first)
    return graph


def weighted_min_fill(
    adjacency: dict[int, set[int]],
    domain_sizes: np.ndarray,
) -> tuple[list[int], float, list[dict[str, object]]]:
    """A deterministic weighted min-fill elimination diagnostic."""
    graph = {vertex: set(neighbors) for vertex, neighbors in adjacency.items()}
    logs = np.log2(np.asarray(domain_sizes, dtype=np.float64))
    order: list[int] = []
    trace: list[dict[str, object]] = []
    maximum = 0.0
    while graph:
        def score(vertex: int) -> tuple[float, int, int, int]:
            neighbors = sorted(graph[vertex])
            bag = [vertex, *neighbors]
            entropy = float(np.sum(logs[bag]))
            fill = sum(
                second not in graph[first]
                for position, first in enumerate(neighbors)
                for second in neighbors[:position]
            )
            return entropy, int(fill), len(neighbors), vertex

        vertex = min(graph, key=score)
        neighbors = sorted(graph[vertex])
        bag = [vertex, *neighbors]
        entropy = float(np.sum(logs[bag]))
        maximum = max(maximum, entropy)
        trace.append({
            "variable": int(vertex),
            "remaining_neighbors": neighbors,
            "bag": bag,
            "bag_log2_entries": entropy,
            "fill_edges": int(score(vertex)[1]),
        })
        for position, first in enumerate(neighbors):
            for second in neighbors[:position]:
                graph[first].add(second)
                graph[second].add(first)
        for neighbor in neighbors:
            graph[neighbor].discard(vertex)
        del graph[vertex]
        order.append(vertex)
    return order, maximum, trace


def gram_gdl_bound(basis: np.ndarray, d: float) -> dict[str, object]:
    radii = coefficient_domain_radii(basis, d)
    domain_sizes = 2 * radii + 1
    graph = gram_adjacency(basis)
    order, width, trace = weighted_min_fill(graph, domain_sizes)
    return {
        "coefficient_domain_radii": radii.tolist(),
        "coefficient_domain_sizes": domain_sizes.tolist(),
        "gram_interaction_edges": int(sum(map(len, graph.values())) // 2),
        "weighted_min_fill_order": order,
        "maximum_table_log2_entries": float(width),
        "pivot_runs_to_exclude_zero": int(basis.shape[0]),
        "elimination_trace": trace,
    }


def audit_basis(
    family: str,
    basis: np.ndarray,
    *,
    shortest_cutoff: int,
    length_slack: float,
) -> dict[str, object]:
    shortest, coefficients = shortest_vector_coefficients(
        basis, cutoff=shortest_cutoff
    )
    n = basis.shape[0]
    d = (1.0 + length_slack / n) * shortest
    gdl = gram_gdl_bound(basis, d)
    _alpha2, hamming_radius = coefficient_hamming_radius(basis, d)
    rows = short_dual_certificate_rows(basis, d)
    rank = gf2_rank(rows, n)
    combined = certificate_hamming_sieve_size(n, hamming_radius, rows)
    spectral = best_spectral_tube_bound(basis, d)
    existing_log2 = min(
        math.log2(max(combined, 1)),
        float(spectral["parity_candidate_log2_bound"]),
    )
    gdl_log2 = float(gdl["maximum_table_log2_entries"])
    return {
        "family": family,
        "dimension": n,
        "shortest_length": float(shortest),
        "shortest_coefficients": coefficients.tolist(),
        "length_upper_bound": float(d),
        "verified_short_dual_rank": int(rank),
        "certificate_hamming_candidate_count": int(combined),
        "best_existing_cover_log2": float(existing_log2),
        "gram_gdl": gdl,
        "combined_cover_or_gdl_log2": float(min(existing_log2, gdl_log2)),
        "combined_cover_or_gdl_exponent": float(
            min(existing_log2, gdl_log2) / n
        ),
    }


def build_report(
    dimensions: tuple[int, ...],
    shortest_cutoff: int,
    length_slack: float,
    random_replicates: int,
    adversarial_replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
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
                shortest_cutoff=shortest_cutoff,
                length_slack=length_slack,
            ))
    return {
        "experiment": "walsh_gram_factor_gdl_sieve",
        "proved_reduction": (
            "bounded exact SVP by pair-factor min-sum elimination and one "
            "backward coefficient preimage"
        ),
        "diagnostic_warning": (
            "The theorem uses exact Gram zero tests and the optimal weighted "
            "treewidth.  Rows report a floating zero test and a min-fill upper "
            "bound for the displayed basis only."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 5, 6))
    parser.add_argument("--shortest-cutoff", type=int, default=2)
    parser.add_argument("--length-slack", type=float, default=1.0)
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--adversarial-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260802483)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.shortest_cutoff,
        args.length_slack,
        args.random_replicates,
        args.adversarial_replicates,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
