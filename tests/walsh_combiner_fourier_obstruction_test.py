#!/usr/bin/env python3
"""Checks for the exact sum/difference Walsh factorization."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_combiner_fourier_obstruction import (
    direct_mixed_transform,
    factorized_mixed_transform,
    oblivious_sketch_energy,
    random_orthogonal_row_sketch,
)


def test_mixed_transform_factorization() -> None:
    h, ell = 2, 2
    rng = np.random.default_rng(19)
    observable = rng.normal(size=1 << (h + ell))
    difference_mass = rng.random(size=1 << (h + ell))
    for j in range(1 << h):
        for query in range(1 << (h + ell)):
            for bucket in range(1 << ell):
                direct = direct_mixed_transform(
                    observable,
                    difference_mass,
                    h=h,
                    ell=ell,
                    j=j,
                    query_frequency=query,
                    bucket_frequency=bucket,
                )
                factorized = factorized_mixed_transform(
                    observable,
                    difference_mass,
                    h=h,
                    ell=ell,
                    j=j,
                    query_frequency=query,
                    bucket_frequency=bucket,
                )
                assert math.isclose(direct, factorized, abs_tol=2e-11)


def test_oblivious_sketch_has_rank_over_dimension_mean_energy() -> None:
    bits, rank = 6, 8
    sketch = random_orthogonal_row_sketch(bits, rank, seed=23)
    mean_energy, _ = oblivious_sketch_energy(sketch)
    assert math.isclose(mean_energy, rank / (1 << bits), abs_tol=2e-15)


if __name__ == "__main__":
    test_mixed_transform_factorization()
    test_oblivious_sketch_has_rank_over_dimension_mean_energy()
    print("walsh combiner Fourier-obstruction tests passed")
