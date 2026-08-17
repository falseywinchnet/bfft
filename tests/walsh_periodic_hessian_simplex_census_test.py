#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)


def test_all_ones_is_unique_shortest_pair() -> None:
    for n in range(3, 8):
        basis = simplex_cancellation_basis(n, 0.97)
        shortest, coefficients = shortest_vector_coefficients(basis, cutoff=2)
        assert abs(shortest - 0.97) < 1e-10
        assert np.all(np.abs(coefficients) == 1)
        assert len(set(int(value) for value in coefficients)) == 1


if __name__ == "__main__":
    test_all_ones_is_unique_shortest_pair()
    print("walsh periodic-Hessian simplex-census tests passed")
