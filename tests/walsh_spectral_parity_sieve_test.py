#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_spectral_parity_sieve import (
    audit_basis,
    certificate_hamming_sieve_size,
    coefficient_hamming_radius,
    gf2_rank,
    hamming_ball_size,
    rectangular_basis,
    short_dual_certificate_rows,
    spectral_tube_bound,
)


def test_hamming_ball_count() -> None:
    assert hamming_ball_size(5, 0) == 0
    assert hamming_ball_size(5, 1) == 5
    assert hamming_ball_size(5, 2) == 15
    assert hamming_ball_size(5, 5) == 31
    assert hamming_ball_size(5, 2, include_zero=True) == 16


def test_rectangular_obstruction_collapses_to_polynomial_parities() -> None:
    for n in range(3, 9):
        basis = rectangular_basis(n, float(2**n))
        alpha_squared, radius = coefficient_hamming_radius(
            basis, 1.0 + 1.0 / n
        )
        assert 1.0 < alpha_squared < 2.0
        assert radius == 1
        assert hamming_ball_size(n, radius) == n
        rows = short_dual_certificate_rows(basis, 1.0 + 1.0 / n)
        assert gf2_rank(rows, n) == n - 1
        assert certificate_hamming_sieve_size(n, radius, rows) == 1
        row = audit_basis(
            "rectangular",
            basis,
            shortest_cutoff=1,
            length_slack=1.0,
        )
        assert row["proved_sieve_contains_shortest"]
        assert row["certificate_sieve_contains_shortest"]
        assert row["certificate_hamming_candidate_count"] == 1
        assert row["spectral_ray_contains_shortest"]


def test_dense_cancellation_is_the_residual_not_a_false_certificate() -> None:
    for n in range(3, 8):
        row = audit_basis(
            "simplex_cancellation",
            simplex_cancellation_basis(n, 0.97),
            shortest_cutoff=2,
            length_slack=0.0,
        )
        assert row["shortest_parity_weight"] == n
        assert row["proved_hamming_radius"] == n
        assert row["proved_candidate_count"] == 2**n - 1
        assert row["verified_short_dual_certificate_rank"] == 0
        assert row["certificate_hamming_candidate_count"] == 2**n - 1
        assert row["shortest_obeys_best_tube_support_bound"]
        # One low Gram mode contains the all-ones cancellation direction;
        # the remaining coefficient ellipsoid has constant Euclidean width.
        one_mode = spectral_tube_bound(
            simplex_cancellation_basis(n, 0.97), 0.97, 1
        )
        assert one_mode["orthogonal_tube_radius"] < 1.0
        assert one_mode["residual_support_bound"] <= 3
        # The low Gram eigenvector is the all-ones cancellation direction.
        assert row["spectral_ray_contains_shortest"]


def test_every_enumerated_shortest_parity_obeys_the_sieve() -> None:
    rng = np.random.default_rng(17)
    for n in range(2, 6):
        for _ in range(8):
            basis = rng.normal(size=(n, n))
            while abs(float(np.linalg.det(basis))) < 0.05:
                basis = rng.normal(size=(n, n))
            row = audit_basis(
                "random",
                basis,
                shortest_cutoff=2,
                length_slack=0.0,
            )
            assert row["proved_sieve_contains_shortest"]
            assert row["shortest_obeys_best_tube_support_bound"]
            assert row["shortest_parity_weight"] <= math.floor(
                row["coefficient_radius_squared"] + 1e-8
            )


if __name__ == "__main__":
    test_hamming_ball_count()
    test_rectangular_obstruction_collapses_to_polynomial_parities()
    test_dense_cancellation_is_the_residual_not_a_false_certificate()
    test_every_enumerated_shortest_parity_obeys_the_sieve()
    print("walsh spectral parity-sieve tests passed")
