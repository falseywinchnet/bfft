#!/usr/bin/env python3
"""Does information about a shortest parity live between boundary cells?

The one-ray law outputs the parity of the first enumerated Voronoi label.
This audit applies two inexpensive transports to that law:

* XOR two independent labels;
* XOR labels on two nearby rays on the same random great circle.

If either XOR is a shortest parity, one target-width Hessian evaluation at
that XOR has the rank-one recovery guarantee from Hhan's Lemma 3.4/3.6.
The nearby-ray construction is the discrete analogue of transporting a
segment label across a boundary rather than asking one cell to be correct.

All Voronoi calculations use a finite coefficient cube.  The script is a
falsification/target-selection audit, not an asymptotic lattice oracle.
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

from experiments.walsh_hessian_noise_audit import generic_basis
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
from experiments.walsh_voronoi_first_exit import coefficient_cube


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_dense_core_transport.json"


def parity_indices(coefficients: np.ndarray) -> np.ndarray:
    """Pack coefficient parities into little-endian integer labels."""
    bits = np.asarray(coefficients, dtype=np.int64) & 1
    powers = 1 << np.arange(bits.shape[1], dtype=np.int64)
    return bits @ powers


def packed_bits(labels: np.ndarray, width: int) -> np.ndarray:
    """Unpack little-endian integer labels into binary row vectors."""
    packed = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    shifts = np.arange(int(width), dtype=np.int64).reshape(1, -1)
    return ((packed >> shifts) & 1).astype(np.uint8)


def affine_rank(labels: np.ndarray, width: int) -> int:
    """Dimension of the affine GF(2) hull of a nonempty packed label set."""
    bits = packed_bits(np.unique(labels), width)
    if len(bits) <= 1:
        return 0
    return gf2_rank(bits[1:] ^ bits[0], width)


def in_affine_hull(
    labels: np.ndarray,
    queries: np.ndarray,
    width: int,
) -> np.ndarray:
    """Membership in the affine GF(2) hull of ``labels``."""
    active = packed_bits(np.unique(labels), width)
    if not len(active):
        return np.zeros(len(np.asarray(queries).reshape(-1)), dtype=bool)
    differences = active[1:] ^ active[0]
    rank = gf2_rank(differences, width)
    answers = []
    for query in packed_bits(queries, width):
        augmented = np.vstack([differences, query ^ active[0]])
        answers.append(gf2_rank(augmented, width) == rank)
    return np.asarray(answers, dtype=bool)


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform of a one-dimensional array."""
    out = np.asarray(values, dtype=np.float64).copy()
    width = 1
    while width < len(out):
        for start in range(0, len(out), 2 * width):
            left = out[start : start + width].copy()
            right = out[start + width : start + 2 * width].copy()
            out[start : start + width] = left + right
            out[start + width : start + 2 * width] = left - right
        width *= 2
    return out


def xor_square(probabilities: np.ndarray) -> np.ndarray:
    """Distribution of X xor Y for independent copies of ``probabilities``."""
    spectrum = fwht(probabilities)
    return fwht(spectrum * spectrum) / len(probabilities)


def first_exit_labels(
    vectors: np.ndarray,
    norm2: np.ndarray,
    directions: np.ndarray,
    *,
    batch: int = 128,
) -> np.ndarray:
    unit = np.asarray(directions, dtype=np.float64).copy()
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    winners = np.empty(len(unit), dtype=np.int64)
    for start in range(0, len(unit), batch):
        stop = min(start + batch, len(unit))
        dot = vectors @ unit[start:stop].T
        times = np.divide(
            norm2[:, None],
            2.0 * dot,
            out=np.full_like(dot, np.inf),
            where=dot > 0.0,
        )
        winners[start:stop] = np.argmin(times, axis=0)
    return winners


def nearby_directions(
    directions: np.ndarray,
    tangents: np.ndarray,
    angle: float,
) -> np.ndarray:
    unit = directions / np.linalg.norm(directions, axis=1)[:, None]
    tangent = tangents - np.einsum("ij,ij->i", tangents, unit)[:, None] * unit
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return math.cos(angle) * unit + math.sin(angle) * tangent


def audit_basis(
    family: str,
    basis: np.ndarray,
    directions: np.ndarray,
    tangents: np.ndarray,
    *,
    cutoff: int,
    angles: tuple[float, ...],
    length_slack: float,
) -> dict[str, object]:
    n = basis.shape[0]
    coefficients = coefficient_cube(n, cutoff)
    vectors = coefficients @ basis.T
    norm2 = np.einsum("ij,ij->i", vectors, vectors)
    shortest2 = float(np.min(norm2))
    shortest_mask = norm2 <= shortest2 * (1.0 + 1e-10)
    packed = parity_indices(coefficients)
    shortest_parities = np.unique(packed[shortest_mask])
    shortest_indicator = np.zeros(1 << n, dtype=bool)
    shortest_indicator[shortest_parities] = True

    base_winners = first_exit_labels(vectors, norm2, directions[:, :n])
    base_parities = packed[base_winners]
    histogram = np.bincount(base_parities, minlength=1 << n).astype(np.float64)
    probabilities = histogram / np.sum(histogram)
    xor2 = xor_square(probabilities)
    sampled_active = np.flatnonzero(histogram)
    sampled_affine_rank = affine_rank(sampled_active, n)

    local_rows = []
    for angle in angles:
        moved = nearby_directions(
            directions[:, :n], tangents[:, :n], float(angle)
        )
        moved_winners = first_exit_labels(vectors, norm2, moved)
        moved_parities = packed[moved_winners]
        changed = base_parities != moved_parities
        transported = np.bitwise_xor(base_parities, moved_parities)
        hits = shortest_indicator[transported]
        local_rows.append({
            "angle_radians": float(angle),
            "angle_degrees": float(math.degrees(angle)),
            "label_change_mass": float(np.mean(changed)),
            "label_change_flux_per_radian": float(
                np.mean(changed) / float(angle)
                if float(angle) > 0.0 else 0.0
            ),
            "unconditional_shortest_xor_mass": float(np.mean(hits)),
            "shortest_xor_flux_per_radian": float(
                np.mean(hits) / float(angle)
                if float(angle) > 0.0 else 0.0
            ),
            "shortest_xor_given_label_change": (
                float(np.mean(hits[changed])) if np.any(changed) else 0.0
            ),
            "distinct_transported_parities": int(len(np.unique(transported))),
        })

    shortest = math.sqrt(shortest2)
    d = (1.0 + length_slack / n) * shortest
    _alpha2, hamming_radius = coefficient_hamming_radius(basis, d)
    certificate_rows = short_dual_certificate_rows(basis, d)
    certificate_rank = gf2_rank(certificate_rows, n)
    combined_count = certificate_hamming_sieve_size(
        n, hamming_radius, certificate_rows
    )
    spectral = best_spectral_tube_bound(basis, d)
    cover_log2 = min(
        math.log2(max(combined_count, 1)),
        float(spectral["parity_candidate_log2_bound"]),
    )

    return {
        "family": family,
        "dimension": n,
        "condition_number": float(np.linalg.cond(basis)),
        "enumerated_shortest_vector_count": int(np.count_nonzero(shortest_mask)),
        "distinct_shortest_parity_count": int(len(shortest_parities)),
        "active_first_exit_parities": int(np.count_nonzero(histogram)),
        "sampled_active_affine_rank": int(sampled_affine_rank),
        "sampled_active_affine_cover_log2": int(sampled_affine_rank),
        "all_enumerated_shortest_parities_in_sampled_affine_hull": bool(
            np.all(in_affine_hull(sampled_active, shortest_parities, n))
        ),
        "single_ray_shortest_parity_mass": float(
            np.sum(probabilities[shortest_indicator])
        ),
        "independent_pair_shortest_xor_mass": float(
            np.sum(xor2[shortest_indicator])
        ),
        "independent_pair_boost": float(
            np.sum(xor2[shortest_indicator])
            / max(np.sum(probabilities[shortest_indicator]), 1e-300)
        ),
        "verified_short_dual_rank": int(certificate_rank),
        "certificate_hamming_candidate_count": int(combined_count),
        "best_proved_cover_log2": float(cover_log2),
        "best_proved_cover_exponent": float(cover_log2 / n),
        "best_spectral_tube": spectral,
        "nearby_ray_transports": local_rows,
    }


def build_report(
    dimensions: tuple[int, ...],
    samples: int,
    cutoff: int,
    angles: tuple[float, ...],
    random_replicates: int,
    adversarial_replicates: int,
    length_slack: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    maximum = max(dimensions)
    directions = rng.normal(size=(samples, maximum))
    tangents = rng.normal(size=(samples, maximum))
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
                directions,
                tangents,
                cutoff=cutoff,
                angles=angles,
                length_slack=length_slack,
            ))
    return {
        "experiment": "walsh_dense_core_boundary_transport",
        "warning": (
            "First-exit labels and shortest vectors are minimized over a finite "
            "coefficient cube.  Cover bounds are proved for the audited basis, "
            "but the geometric measurements are falsification evidence only. "
            "In particular, a sampled active-label affine rank is a lower bound "
            "on the full active rank and is not a certified parity cover. "
            "Finite-angle mass divided by angle is a flux diagnostic, not a "
            "certified infinitesimal derivative or robust-reach bound."
        ),
        "transport_identity": (
            "theta(w1-w2)=theta(w1) xor theta(w2); a shortest XOR may be "
            "passed to the target Hessian and BDD recovery lemma"
        ),
        "samples_per_fixture": samples,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 6))
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument(
        "--angles", type=float, nargs="+", default=(0.03, 0.1, 0.3)
    )
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--adversarial-replicates", type=int, default=1)
    parser.add_argument("--length-slack", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260802482)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.samples,
        args.cutoff,
        tuple(args.angles),
        args.random_replicates,
        args.adversarial_replicates,
        args.length_slack,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
