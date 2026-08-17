#!/usr/bin/env python3
"""Regression checks for the obtuse-superbase minimum-cut escape."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_obtuse_superbase_sieve import (
    displayed_superbase,
    is_obtuse_superbase,
    stoer_wagner_min_cut,
    superbase_min_cut,
)
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)


def brute_cut(weights: np.ndarray) -> float:
    n = len(weights)
    best = float("inf")
    for bits in itertools.product((0, 1), repeat=n - 1):
        side = np.asarray((0, *bits), dtype=bool)
        if not np.any(side):
            continue
        value = 0.0
        for first in range(n):
            for second in range(first):
                if side[first] != side[second]:
                    value += float(weights[first, second])
        best = min(best, value)
    return best


def main() -> None:
    rng = np.random.default_rng(485)
    for n in range(2, 8):
        weights = rng.uniform(0.0, 2.0, size=(n, n))
        weights = np.triu(weights, 1)
        weights += weights.T
        value, cut = stoer_wagner_min_cut(weights)
        assert cut and len(cut) < n
        assert abs(value - brute_cut(weights)) < 1e-10

    rectangular = np.diag([1.0, 8.0, 8.0])
    assert is_obtuse_superbase(displayed_superbase(rectangular))
    rectangular_result = superbase_min_cut(rectangular)
    assert rectangular_result is not None
    assert abs(rectangular_result["returned_squared_norm"] - 1.0) < 1e-12

    simplex = simplex_cancellation_basis(5, 0.97)
    assert is_obtuse_superbase(displayed_superbase(simplex))
    simplex_result = superbase_min_cut(simplex)
    assert simplex_result is not None
    assert abs(
        simplex_result["returned_squared_norm"] - 0.97**2
    ) < 1e-10
    assert simplex_result["shortest_parity"] == [1, 1, 1, 1, 1]

    acute = np.asarray([[1.0, 0.8], [0.0, 0.6]])
    assert not is_obtuse_superbase(displayed_superbase(acute))
    assert superbase_min_cut(acute) is None
    print("walsh obtuse-superbase sieve tests passed")


if __name__ == "__main__":
    main()
