#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import shortest_vector_coefficients
from experiments.walsh_periodic_hessian_stress_census import (
    d_root_basis,
    needle_d_basis,
)


def test_d_basis_has_even_coordinate_sum() -> None:
    basis = d_root_basis(5)
    assert abs(round(np.linalg.det(basis))) == 2
    assert np.all((np.sum(basis.astype(int), axis=0) & 1) == 0)


def test_needle_is_unique_shortest_pair() -> None:
    basis = needle_d_basis(6, 1.03)
    shortest, coefficients = shortest_vector_coefficients(basis, cutoff=3)
    assert abs(shortest - 1.0) < 1e-12
    assert abs(int(coefficients[0])) == 1
    assert not np.any(coefficients[1:])


def test_tilted_needle_remains_unique_shortest_pair() -> None:
    basis = needle_d_basis(6, 1.01, 0.12)
    shortest, coefficients = shortest_vector_coefficients(basis, cutoff=3)
    assert abs(shortest - 1.0) < 1e-12
    assert abs(int(coefficients[0])) == 1
    assert not np.any(coefficients[1:])


if __name__ == "__main__":
    test_d_basis_has_even_coordinate_sum()
    test_needle_is_unique_shortest_pair()
    test_tilted_needle_remains_unique_shortest_pair()
    print("walsh periodic-Hessian stress-census tests passed")
