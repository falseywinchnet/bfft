#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_flatness_ou_kernel import audit_case, frontier_constants


def test_ou_identity_and_reversibility() -> None:
    for alpha in (0.25, 0.5, 0.75):
        row = audit_case(n=2, cutoff=3, s=1.8, alpha=alpha)
        assert row["maximum_quadratic_identity_error"] < 2e-14
        assert row["row_stochastic_error"] < 2e-14
        assert row["detailed_balance_error"] < 2e-14
        assert row["absolute_spectral_gap"] > 0.0


def test_frontier_exponents() -> None:
    row = frontier_constants()
    assert 0.0264 < row["gap_budget"] < 0.0266
    assert 0.4793 < row["cold_dgs_dual_mass_exponent"] < 0.4795
    assert 0.0203 < row["target_scale_flatness_slack"] < 0.0205
    assert row["superlattice_batch_shortfall_exponent"] > 0.45


if __name__ == "__main__":
    test_ou_identity_and_reversibility()
    test_frontier_exponents()
    print("walsh flatness-OU kernel tests passed")
