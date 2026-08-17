#!/usr/bin/env python3
"""Finite audit for the stationary-seed reduction in Walsh recovery.

For X drawn from a discrete Gaussian on a lattice Lambda, its coefficient
parity U has the exact collision identity

    Col(U) = (rho_{s/sqrt(2)}(Lambda) / rho_s(Lambda))**2.

A uniform full-rank binary hash H with h output bits therefore makes H(U)
close to uniform once the Gaussian is above sqrt(2) smoothing.  Choosing the
Walsh address j=H(U) after drawing X makes X an exact stationary seed for the
Gaussian conditioned on H(U)=j.  This module checks the two finite identities
used in that reduction; it does not assert a spectral-gap bound for a general
lattice walk.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_stationary_seed_reduction.json"


def coefficient_box(n: int, cutoff: int) -> np.ndarray:
    values = range(-cutoff, cutoff + 1)
    return np.asarray(list(itertools.product(values, repeat=n)), dtype=np.int16)


def gaussian_weights(coefficients: np.ndarray, basis: np.ndarray, s: float) -> np.ndarray:
    points = np.asarray(coefficients, dtype=np.float64) @ np.asarray(basis, dtype=np.float64)
    norm2 = np.einsum("ij,ij->i", points, points)
    return np.exp(-math.pi * norm2 / (s * s))


def parity_distribution(coefficients: np.ndarray, weights: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients)
    n = coefficients.shape[1]
    powers = 1 << np.arange(n, dtype=np.int64)
    labels = ((coefficients & 1).astype(np.int64) @ powers).astype(np.int64)
    masses = np.bincount(labels, weights=np.asarray(weights), minlength=1 << n)
    return masses / np.sum(masses)


def parity_collision(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(probabilities @ probabilities)


def theta_collision_ratio(
    coefficients: np.ndarray, basis: np.ndarray, s: float
) -> float:
    rho_s = float(np.sum(gaussian_weights(coefficients, basis, s)))
    rho_cold = float(
        np.sum(gaussian_weights(coefficients, basis, s / math.sqrt(2.0)))
    )
    return (rho_cold / rho_s) ** 2


def gf2_rank(matrix: np.ndarray) -> int:
    work = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    rows, columns = work.shape
    rank = 0
    for column in range(columns):
        pivot = next((r for r in range(rank, rows) if work[r, column]), None)
        if pivot is None:
            continue
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rows):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def full_rank_binary_hashes(n: int, h: int):
    for bits in itertools.product((0, 1), repeat=n * h):
        matrix = np.asarray(bits, dtype=np.uint8).reshape(h, n)
        if gf2_rank(matrix) == h:
            yield matrix


def hashed_distribution(probabilities: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.uint8)
    h, n = matrix.shape
    inputs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    outputs = (inputs @ matrix.T) & 1
    labels = outputs @ (1 << np.arange(h, dtype=np.int64))
    return np.bincount(labels, weights=probabilities, minlength=1 << h)


def chi_square_from_uniform(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(len(probabilities) * (probabilities @ probabilities) - 1.0)


def expected_full_rank_hash_chi_square(
    parity_collision_probability: float, n: int, h: int
) -> float:
    """Exact expectation over uniform surjective H:F_2^n -> F_2^h."""
    collision_kernel = ((1 << (n - h)) - 1) / ((1 << n) - 1)
    output_collision = (
        parity_collision_probability
        + (1.0 - parity_collision_probability) * collision_kernel
    )
    return (1 << h) * output_collision - 1.0


def smoothing_collision_upper_bound(n: int) -> float:
    """Col(U) bound when s > sqrt(2) eta_{1/2}(Lambda)."""
    return 2.25 * 2.0 ** (-n)


def gap_budget(target_width: float) -> float:
    return 0.5 - 2.0 * target_width


def rho_z(width: float, terms: int = 128) -> float:
    return 1.0 + 2.0 * sum(
        math.exp(-math.pi * k * k / (width * width))
        for k in range(1, terms + 1)
    )


def gram_schmidt_ratio_threshold(log2_budget: float) -> float:
    lo, hi = 1e-6, 4.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if math.log2(rho_z(mid)) < log2_budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def audit_dimension(n: int, h: int, cutoff: int, s: float) -> dict[str, object]:
    # A deterministic, mildly skew basis makes the check non-product while
    # keeping theta-tail truncation negligible at the selected cutoff.
    basis = np.eye(n)
    for row in range(n):
        basis[row, row] = 0.85 + 0.11 * row
        if row + 1 < n:
            basis[row, row + 1] = 0.17

    coefficients = coefficient_box(n, cutoff)
    probabilities = parity_distribution(
        coefficients, gaussian_weights(coefficients, basis, s)
    )
    collision = parity_collision(probabilities)
    theta_ratio = theta_collision_ratio(coefficients, basis, s)

    observed = []
    for matrix in full_rank_binary_hashes(n, h):
        observed.append(chi_square_from_uniform(hashed_distribution(probabilities, matrix)))
    predicted = expected_full_rank_hash_chi_square(collision, n, h)
    return {
        "dimension": n,
        "hash_bits": h,
        "cutoff": cutoff,
        "s": s,
        "parity_collision": collision,
        "theta_collision_ratio": theta_ratio,
        "collision_identity_absolute_error": abs(collision - theta_ratio),
        "full_rank_hash_count": len(observed),
        "mean_hash_chi_square": float(np.mean(observed)),
        "predicted_mean_hash_chi_square": predicted,
        "hash_expectation_absolute_error": abs(float(np.mean(observed)) - predicted),
        "maximum_hash_total_variation": float(
            max(
                0.5
                * np.sum(
                    np.abs(
                        hashed_distribution(probabilities, matrix) - 2.0 ** (-h)
                    )
                )
                for matrix in full_rank_binary_hashes(n, h)
            )
        ),
    }


def audit() -> dict[str, object]:
    target_width = 0.23675858
    budget = gap_budget(target_width)
    rows = [
        audit_dimension(n=2, h=1, cutoff=8, s=1.45),
        audit_dimension(n=3, h=1, cutoff=7, s=1.55),
        audit_dimension(n=4, h=2, cutoff=6, s=1.65),
    ]
    return {
        "experiment": "walsh_stationary_seed_reduction",
        "target_width": target_width,
        "gap_budget_bits_per_dimension": budget,
        "improved_spherical_g2_at_target_width": 0.02057231,
        "improved_spherical_g2_at_twice_target_width": 0.52039983,
        "uniform_gram_schmidt_ratio_sufficient_for_budget": (
            gram_schmidt_ratio_threshold(budget)
        ),
        "rows": rows,
        "interpretation": (
            "The parity collision and full-rank hash identities are exact up to "
            "theta-tail truncation.  They certify the seed/address reduction, not "
            "the still-missing basis-independent coset-walk spectral gap."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
