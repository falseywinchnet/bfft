import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_stationary_seed_reduction import (
    audit_dimension,
    expected_full_rank_hash_chi_square,
    gap_budget,
    gram_schmidt_ratio_threshold,
    rho_z,
    smoothing_collision_upper_bound,
)


def test_theta_and_hash_collision_identities() -> None:
    for n, h, cutoff, s in ((2, 1, 8, 1.45), (3, 1, 7, 1.55), (4, 2, 6, 1.65)):
        row = audit_dimension(n=n, h=h, cutoff=cutoff, s=s)
        assert row["collision_identity_absolute_error"] < 2e-12
        assert row["hash_expectation_absolute_error"] < 2e-12


def test_full_rank_hash_formula_at_uniform_and_point_mass() -> None:
    for n in range(2, 7):
        for h in range(1, n):
            assert abs(expected_full_rank_hash_chi_square(2.0 ** (-n), n, h)) < 1e-14
            assert math.isclose(
                expected_full_rank_hash_chi_square(1.0, n, h),
                (1 << h) - 1.0,
            )


def test_asymptotic_seed_and_gap_constants() -> None:
    n = 100
    h = n // 2
    collision = smoothing_collision_upper_bound(n)
    expected_chi_square = expected_full_rank_hash_chi_square(collision, n, h)
    assert expected_chi_square < 3.0 * 2.0 ** (-n / 2)
    budget = gap_budget(0.23675858)
    threshold = gram_schmidt_ratio_threshold(budget)
    assert 0.0264 < budget < 0.0266
    assert 0.8191 < threshold < 0.8193
    assert abs(math.log2(rho_z(threshold)) - budget) < 1e-12


if __name__ == "__main__":
    test_theta_and_hash_collision_identities()
    test_full_rank_hash_formula_at_uniform_and_point_mass()
    test_asymptotic_seed_and_gap_constants()
    print("walsh stationary-seed reduction tests passed")
