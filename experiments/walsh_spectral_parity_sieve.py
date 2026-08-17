#!/usr/bin/env python3
"""Lattice-adapted parity seeds from the coefficient ellipsoid.

The Haar counterexample for the accessible soft-Hessian law is caused by
irrelevant parity entropy, not by a failure of the midpoint recovery lemma.
For a basis ``B`` and a correct length upper bound ``d``, every shortest
coefficient vector ``z`` obeys

    ||z||_2 <= d ||B^{-1}||_op.

Hence its parity has Hamming weight at most the square of this quantity.  A
uniform seed on that Hamming ball is therefore an exact lattice-adapted seed
law.  This script audits that proved sieve and a deliberately separate
spectral-ray heuristic for the dense cancellation cases where the Hamming
bound is vacuous.

The ray heuristic is not used in the proved candidate-count bound.  It asks
whether the low-eigenvalue support chain of ``B.T @ B`` exposes a dense short
coefficient vector through one-dimensional rounding events.
"""

from __future__ import annotations

import argparse
import itertools
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


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_spectral_parity_sieve.json"
)


def parity(value: np.ndarray) -> tuple[int, ...]:
    return tuple(int(entry) & 1 for entry in np.asarray(value, dtype=np.int64))


def hamming_ball_size(n: int, radius: int, *, include_zero: bool = False) -> int:
    """Number of binary words of weight at most ``radius``."""
    radius = min(max(int(radius), 0), int(n))
    count = sum(math.comb(n, weight) for weight in range(radius + 1))
    return count if include_zero else max(count - 1, 0)


def gf2_rank(matrix: np.ndarray, width: int | None = None) -> int:
    value = np.asarray(matrix, dtype=np.int64) & 1
    if value.size == 0:
        return 0
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if width is not None and value.shape[1] != width:
        raise ValueError("unexpected GF(2) matrix width")
    value = value.astype(np.uint8, copy=True)
    rank = 0
    for column in range(value.shape[1]):
        pivots = np.flatnonzero(value[rank:, column])
        if not pivots.size:
            continue
        pivot = rank + int(pivots[0])
        value[[rank, pivot]] = value[[pivot, rank]]
        rows = np.flatnonzero(value[:, column])
        rows = rows[rows != rank]
        value[rows] ^= value[rank]
        rank += 1
        if rank == value.shape[0]:
            break
    return rank


def short_dual_certificate_rows(
    basis: np.ndarray,
    d: float,
    *,
    coefficient_cutoff: int = 1,
) -> np.ndarray:
    """Verified dual rows whose integer pairing with a shortest vector is zero.

    A row ``k`` represents ``y=B^{-T}k``.  If ``d||y||<1``, then for every
    vector of norm at most ``d`` the integer ``<y,Bz>=k.z`` must vanish.
    We enumerate only a tiny coefficient cube; completeness is unnecessary
    because every returned row is an independently valid certificate.
    """
    n = basis.shape[0]
    inverse = np.linalg.inv(basis)
    rows = []
    for entry in itertools.product(
        range(-coefficient_cutoff, coefficient_cutoff + 1), repeat=n
    ):
        if not any(entry):
            continue
        integer_row = np.asarray(entry, dtype=np.int64)
        dual_vector = integer_row @ inverse
        if d * float(np.linalg.norm(dual_vector)) < 1.0 - 1e-10:
            rows.append(integer_row & 1)
    if not rows:
        return np.zeros((0, n), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8)


def certificate_hamming_sieve_size(
    n: int,
    radius: int,
    certificate_rows: np.ndarray,
) -> int:
    """Exact finite count of nonzero words in the certificate Hamming sieve."""
    if n > 24:
        raise ValueError("exact diagnostic count is intentionally limited to n<=24")
    rows = np.asarray(certificate_rows, dtype=np.uint8).reshape(-1, n)
    count = 0
    for word in range(1, 1 << n):
        bits = np.asarray([(word >> i) & 1 for i in range(n)], dtype=np.uint8)
        if int(np.sum(bits)) > radius:
            continue
        if rows.size and np.any((rows @ bits) & 1):
            continue
        count += 1
    return count


def coefficient_hamming_radius(basis: np.ndarray, d: float) -> tuple[float, int]:
    """Return ``alpha^2`` and the proved shortest-parity weight bound."""
    inverse_norm = float(np.linalg.norm(np.linalg.inv(basis), ord=2))
    alpha_squared = float((d * inverse_norm) ** 2)
    # The exact statement uses floor(alpha_squared).  The tolerance only
    # prevents a floating representation of an integer from rounding down.
    radius = min(basis.shape[0], int(math.floor(alpha_squared + 1e-9)))
    return alpha_squared, radius


def _log2_sum_combinations(population: int, maximum: int) -> float:
    maximum = min(max(int(maximum), 0), int(population))
    return math.log2(sum(math.comb(population, index) for index in range(maximum + 1)))


def spectral_tube_bound(
    basis: np.ndarray,
    d: float,
    dimension: int,
) -> dict[str, float | int]:
    """Candidate-count certificate for a low-Gram-eigenspace tube.

    If ``U`` is the span of the first ``dimension`` Gram eigenvectors, every
    coefficient vector in the length-``d`` ellipsoid is within distance
    ``C=d/sqrt(lambda_(dimension+1))`` of ``U``.  Rounding cells met by the
    radius-``R`` ball in ``U`` form a hyperplane arrangement; an integer
    vector within distance ``C`` differs from one of those roundings on at
    most ``floor(4C^2)`` coordinates.
    """
    n = basis.shape[0]
    eigenvalues = np.linalg.eigvalsh(basis.T @ basis)
    radius = float(d / math.sqrt(float(eigenvalues[0])))
    if dimension >= n:
        tube = 0.0
    else:
        tube = float(d / math.sqrt(float(eigenvalues[dimension])))
    hyperplanes = n * (2 * int(math.ceil(radius)) + 2)
    # The leading factor covers lower-dimensional faces and every deterministic
    # choice at simultaneous half-integer ties, not only open chambers.
    arrangement_log2 = dimension + _log2_sum_combinations(
        hyperplanes, dimension
    )
    residual_support = min(n, int(math.floor(4.0 * tube * tube + 1e-9)))
    residual_magnitude = max(1, int(math.ceil(tube + 0.5)))
    residual_count = sum(
        math.comb(n, support) * (2 * residual_magnitude) ** support
        for support in range(residual_support + 1)
    )
    raw_log2 = arrangement_log2 + math.log2(residual_count)
    return {
        "subspace_dimension": int(dimension),
        "coefficient_ball_radius": radius,
        "orthogonal_tube_radius": tube,
        "rounding_hyperplane_count": hyperplanes,
        "residual_support_bound": residual_support,
        "raw_candidate_log2_bound": raw_log2,
        "parity_candidate_log2_bound": min(float(n), raw_log2),
    }


def best_spectral_tube_bound(basis: np.ndarray, d: float) -> dict[str, float | int]:
    rows = [spectral_tube_bound(basis, d, k) for k in range(basis.shape[0] + 1)]
    return min(rows, key=lambda row: float(row["raw_candidate_log2_bound"]))


def spectral_ray_parities(
    basis: np.ndarray,
    d: float,
    *,
    event_cap: int = 200_000,
) -> set[tuple[int, ...]]:
    """Round every cell met by each signed Gram-eigenvector ray.

    For a unit eigenvector ``q`` with eigenvalue ``lambda``, the coefficient
    ellipsoid permits the ray interval ``|a| <= d/sqrt(lambda)``.  Rounding
    changes only when one coordinate crosses a half integer, so the complete
    one-dimensional cell catalog is obtained from those breakpoints.  The
    event cap makes the diagnostic explicitly finite on enormous-bit inputs.
    """
    gram = basis.T @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    candidates: set[tuple[int, ...]] = set()
    event_count = 0
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= 0.0:
            continue
        direction = eigenvectors[:, index]
        limit = float(d / math.sqrt(float(eigenvalue)))
        breaks = {0.0, limit}
        for coordinate in np.abs(direction):
            if coordinate <= 1e-14:
                continue
            maximum_integer = int(math.floor(limit * coordinate - 0.5))
            if maximum_integer < 0:
                continue
            event_count += maximum_integer + 1
            if event_count > event_cap:
                return candidates
            for integer in range(maximum_integer + 1):
                breaks.add((integer + 0.5) / float(coordinate))
        ordered = sorted(value for value in breaks if 0.0 <= value <= limit)
        probes = set(ordered)
        probes.update(
            0.5 * (left + right) for left, right in zip(ordered, ordered[1:])
        )
        for sign in (-1.0, 1.0):
            for scale in probes:
                coefficients = np.rint(sign * scale * direction).astype(np.int64)
                if np.any(coefficients):
                    candidates.add(parity(coefficients))
    candidates.discard((0,) * basis.shape[0])
    return candidates


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
    d = (1.0 + length_slack / basis.shape[0]) * shortest
    target = parity(coefficients)
    alpha_squared, radius = coefficient_hamming_radius(basis, d)
    candidate_count = hamming_ball_size(basis.shape[0], radius)
    certificate_rows = short_dual_certificate_rows(basis, d)
    certificate_rank = gf2_rank(certificate_rows, basis.shape[0])
    combined_count = certificate_hamming_sieve_size(
        basis.shape[0], radius, certificate_rows
    )
    ray_candidates = spectral_ray_parities(basis, d)
    tube_bound = best_spectral_tube_bound(basis, d)
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(basis.T @ basis)
    tube_dimension = int(tube_bound["subspace_dimension"])
    if tube_dimension:
        low = gram_eigenvectors[:, :tube_dimension]
        projected = low @ (low.T @ coefficients)
    else:
        projected = np.zeros_like(coefficients, dtype=np.float64)
    rounded = np.rint(projected).astype(np.int64)
    residual = coefficients - rounded
    return {
        "family": family,
        "dimension": basis.shape[0],
        "condition_number": float(np.linalg.cond(basis)),
        "shortest_length": float(shortest),
        "length_upper_bound": float(d),
        "shortest_coefficients": coefficients.tolist(),
        "shortest_parity": list(target),
        "shortest_parity_weight": int(sum(target)),
        "coefficient_radius_squared": alpha_squared,
        "proved_hamming_radius": radius,
        "proved_candidate_count": candidate_count,
        "proved_candidate_log2": (
            math.log2(candidate_count) if candidate_count else -math.inf
        ),
        "proved_sieve_contains_shortest": sum(target) <= radius,
        "verified_short_dual_certificate_rank": certificate_rank,
        "certificate_hamming_candidate_count": combined_count,
        "certificate_hamming_candidate_log2": (
            math.log2(combined_count) if combined_count else -math.inf
        ),
        "certificate_sieve_contains_shortest": (
            sum(target) <= radius
            and not (
                certificate_rows.size
                and np.any((certificate_rows @ np.asarray(target, dtype=np.uint8)) & 1)
            )
        ),
        "best_spectral_tube": tube_bound,
        "shortest_residual_support_at_best_tube": int(np.count_nonzero(residual)),
        "shortest_obeys_best_tube_support_bound": (
            int(np.count_nonzero(residual))
            <= int(tube_bound["residual_support_bound"])
        ),
        "spectral_ray_candidate_count": len(ray_candidates),
        "spectral_ray_contains_shortest": target in ray_candidates,
    }


def rectangular_basis(n: int, long_scale: float) -> np.ndarray:
    diagonal = np.full(n, float(long_scale), dtype=np.float64)
    diagonal[0] = 1.0
    return np.diag(diagonal)


def build_rows(
    min_dimension: int,
    max_dimension: int,
    *,
    shortest_cutoff: int,
    length_slack: float,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
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
            rows.append(audit_basis(
                family,
                basis,
                shortest_cutoff=shortest_cutoff,
                length_slack=length_slack,
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=7)
    parser.add_argument("--shortest-cutoff", type=int, default=2)
    parser.add_argument(
        "--length-slack",
        type=float,
        default=1.0,
        help="use d=(1+length_slack/n) lambda_1",
    )
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_rows(
        args.min_dimension,
        args.max_dimension,
        shortest_cutoff=args.shortest_cutoff,
        length_slack=args.length_slack,
        seed=args.seed,
    )
    report = {
        "experiment": "walsh_spectral_parity_sieve",
        "proved_law": (
            "uniform nonzero parity satisfying verified short-dual syndromes "
            "and wt(theta)<=floor(d^2||B^{-1}||_op^2)"
        ),
        "spectral_ray_status": "diagnostic heuristic; not part of proved law",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
