#!/usr/bin/env python3
"""Exact displayed-obtuse-superbase min-cut escape for SVP.

If ``b_1,...,b_n`` together with ``b_{n+1}=-sum_i b_i`` have nonpositive
pairwise inner products, every shortest vector is a nontrivial subset sum of
the superbase and its squared norm is the capacity of the corresponding cut
with weights ``-<b_i,b_j>``.  A global minimum cut therefore returns a
shortest vector in polynomial time.

The theorem is exact for rational input.  Fixture recognition below uses a
floating tolerance and is diagnostic.  The self-contained Stoer--Wagner
implementation is checked against exhaustive cuts in the regression test.
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
from experiments.walsh_spectral_parity_sieve import rectangular_basis


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_obtuse_superbase_sieve.json"
)


def displayed_superbase(basis: np.ndarray) -> np.ndarray:
    """Columns ``b_1,...,b_n,-sum_i b_i``."""
    value = np.asarray(basis, dtype=np.float64)
    return np.column_stack([value, -np.sum(value, axis=1)])


def selling_parameters(superbase: np.ndarray) -> np.ndarray:
    value = np.asarray(superbase, dtype=np.float64)
    return value.T @ value


def is_obtuse_superbase(
    superbase: np.ndarray,
    *,
    relative_tolerance: float = 1e-12,
) -> bool:
    gram = selling_parameters(superbase)
    scale = max(float(np.max(np.abs(gram))), 1.0)
    off_diagonal = gram - np.diag(np.diag(gram))
    return bool(np.max(off_diagonal) <= relative_tolerance * scale)


def stoer_wagner_min_cut(weights: np.ndarray) -> tuple[float, set[int]]:
    """Global minimum cut of a symmetric nonnegative weighted graph."""
    matrix = np.asarray(weights, dtype=np.float64).copy()
    n = matrix.shape[0]
    if matrix.shape != (n, n) or n < 2:
        raise ValueError("expected a square graph with at least two vertices")
    if np.max(np.abs(matrix - matrix.T)) > 1e-10:
        raise ValueError("cut weights must be symmetric")
    if np.min(matrix) < -1e-12:
        raise ValueError("cut weights must be nonnegative")
    np.fill_diagonal(matrix, 0.0)
    groups = [{index} for index in range(n)]
    best_weight = math.inf
    best_group: set[int] = set()

    while len(groups) > 1:
        size = len(groups)
        used = np.zeros(size, dtype=bool)
        connection = np.zeros(size, dtype=np.float64)
        previous = -1
        for phase_index in range(size):
            candidates = np.flatnonzero(~used)
            selected = int(candidates[np.argmax(connection[candidates])])
            used[selected] = True
            if phase_index == size - 1:
                cut_weight = float(connection[selected])
                if cut_weight < best_weight:
                    best_weight = cut_weight
                    best_group = set(groups[selected])
                if previous < 0:
                    break
                matrix[previous, :] += matrix[selected, :]
                matrix[:, previous] = matrix[previous, :]
                matrix = np.delete(np.delete(matrix, selected, axis=0), selected, axis=1)
                groups[previous].update(groups[selected])
                del groups[selected]
                break
            connection[~used] += matrix[selected, ~used]
            previous = selected
    return best_weight, best_group


def superbase_min_cut(basis: np.ndarray) -> dict[str, object] | None:
    superbase = displayed_superbase(basis)
    if not is_obtuse_superbase(superbase):
        return None
    gram = selling_parameters(superbase)
    weights = np.maximum(-gram, 0.0)
    np.fill_diagonal(weights, 0.0)
    cut_weight, cut = stoer_wagner_min_cut(weights)
    indicator = np.asarray(
        [int(index in cut) for index in range(superbase.shape[1])],
        dtype=np.int64,
    )
    vector = superbase @ indicator
    coefficient = indicator[:-1] - indicator[-1]
    return {
        "minimum_cut_weight": float(cut_weight),
        "cut_vertices": sorted(cut),
        "shortest_coefficients": coefficient.tolist(),
        "shortest_parity": (coefficient & 1).tolist(),
        "returned_squared_norm": float(np.dot(vector, vector)),
    }


def audit_basis(family: str, basis: np.ndarray, cutoff: int) -> dict[str, object]:
    shortest, coefficients = shortest_vector_coefficients(basis, cutoff=cutoff)
    result = superbase_min_cut(basis)
    return {
        "family": family,
        "dimension": int(basis.shape[0]),
        "displayed_superbase_is_obtuse": result is not None,
        "enumerated_shortest_length": float(shortest),
        "enumerated_shortest_coefficients": coefficients.tolist(),
        "minimum_cut": result,
        "minimum_cut_matches_enumerated_length": bool(
            result is not None
            and abs(float(result["returned_squared_norm"]) - shortest * shortest)
            <= 1e-8 * max(shortest * shortest, 1.0)
        ),
    }


def build_report(
    dimensions: tuple[int, ...],
    cutoff: int,
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
            rows.append(audit_basis(family, basis, cutoff))
    return {
        "experiment": "walsh_displayed_obtuse_superbase_min_cut_sieve",
        "warning": (
            "The theorem uses exact rational inner-product signs.  The fixture "
            "audit recognizes the displayed superbase with floating arithmetic "
            "and compares against finite-cube shortest-vector enumeration."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 5, 6))
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--adversarial-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260802485)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.cutoff,
        args.random_replicates,
        args.adversarial_replicates,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
