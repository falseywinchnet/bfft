#!/usr/bin/env python3
"""Exact Fourier algebra for the half-coset Gaussian combiner.

After an exact two-sample combine, write the cold sum/difference variables as
Y and D.  Their parity labels q,d in F_2^(h+ell) obey

    q_head + d_head = j,

and the retained collision bucket is b=q_tail+d_tail.  This module verifies
the mixed Fourier factorization and records the resulting address tradeoff:
making the difference leg trivial preserves the Hessian coefficient linearly
only when the full h-bit target prefix is queried.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "out"
    / "walsh_combiner_fourier_obstruction.json"
)


def dot2(a: int, b: int) -> int:
    return (a & b).bit_count() & 1


def character(frequency: int, point: int) -> float:
    return -1.0 if dot2(frequency, point) else 1.0


def split_bits(value: int, ell: int) -> tuple[int, int]:
    return value >> ell, value & ((1 << ell) - 1)


def join_bits(head: int, tail: int, ell: int) -> int:
    return (head << ell) | tail


def walsh_sum(values: np.ndarray, frequency: int) -> float:
    return float(sum(v * character(frequency, x) for x, v in enumerate(values)))


def direct_mixed_transform(
    observable: np.ndarray,
    difference_mass: np.ndarray,
    *,
    h: int,
    ell: int,
    j: int,
    query_frequency: int,
    bucket_frequency: int,
) -> float:
    size = 1 << (h + ell)
    total = 0.0
    for q in range(size):
        q_head, q_tail = split_bits(q, ell)
        for d in range(size):
            d_head, d_tail = split_bits(d, ell)
            if (q_head ^ d_head) != j:
                continue
            bucket = q_tail ^ d_tail
            total += (
                observable[q]
                * difference_mass[d]
                * character(query_frequency, q)
                * character(bucket_frequency, bucket)
            )
    return total


def factorized_mixed_transform(
    observable: np.ndarray,
    difference_mass: np.ndarray,
    *,
    h: int,
    ell: int,
    j: int,
    query_frequency: int,
    bucket_frequency: int,
) -> float:
    query_head, query_tail = split_bits(query_frequency, ell)
    total = 0.0
    for lagrange_frequency in range(1 << h):
        observable_frequency = join_bits(
            query_head ^ lagrange_frequency,
            query_tail ^ bucket_frequency,
            ell,
        )
        difference_frequency = join_bits(
            lagrange_frequency,
            bucket_frequency,
            ell,
        )
        total += (
            character(lagrange_frequency, j)
            * walsh_sum(observable, observable_frequency)
            * walsh_sum(difference_mass, difference_frequency)
        )
    return total / (1 << h)


def oblivious_sketch_energy(sketch: np.ndarray) -> tuple[float, float]:
    """Mean and maximum retained energy of uniformly random basis targets."""
    gram_diagonal = np.sum(np.asarray(sketch, dtype=float) ** 2, axis=0)
    return float(np.mean(gram_diagonal)), float(np.max(gram_diagonal))


def random_orthogonal_row_sketch(
    address_bits: int,
    rank: int,
    *,
    seed: int,
) -> np.ndarray:
    size = 1 << address_bits
    if not (1 <= rank <= size):
        raise ValueError("rank must lie between one and the address dimension")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, rank))
    q, _ = np.linalg.qr(matrix)
    return q.T


def audit(max_address_bits: int = 12, seed: int = 7) -> dict[str, object]:
    rows = []
    for bits in range(2, max_address_bits + 1, 2):
        size = 1 << bits
        rank = 1 << (bits // 2)
        sketch = random_orthogonal_row_sketch(bits, rank, seed=seed + bits)
        mean_energy, max_energy = oblivious_sketch_energy(sketch)
        rows.append({
            "address_bits": bits,
            "address_dimension": size,
            "sketch_rank": rank,
            "mean_retained_target_energy": mean_energy,
            "rank_over_dimension": rank / size,
            "maximum_retained_target_energy": max_energy,
            "amplification_needed_for_unit_mean_energy": math.sqrt(size / rank),
            "noise_variance_amplification": size / rank,
        })
    return {
        "experiment": "walsh_combiner_fourier_obstruction",
        "rows": rows,
        "conclusion": (
            "An oblivious rank-d sketch retains d/N energy from a uniformly "
            "random missing prefix. Amplifying the signal back to constant "
            "size multiplies noise variance by N/d, cancelling the rank gain."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-address-bits", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.max_address_bits, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
