"""Regression checks for the recovery-theorem exponent ledger."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_recovery_theorem_ledger import (
    T0,
    build_report,
    iota,
    sparse_histogram_gl_exponent,
)


def test_zero_endpoint_importance_penalty() -> None:
    r = T0 + 1e-3
    assert math.isclose(iota(r, r), 0.0, abs_tol=1e-15)


def test_sparse_histogram_normalization_blocks_naive_gl() -> None:
    exponent = sparse_histogram_gl_exponent(
        0.22407354148,
        T0 * (1.0 + 1e-6),
        1.0,
    )
    assert exponent > 1.5


def test_direct_half_coset_transport_reaches_half_limit() -> None:
    epsilon = 1e-3
    report = build_report(epsilon)
    transported = report["candidates"][2]
    assert transported["conjectural"] is True
    assert transported["sample_exponent"] < 0.5
    assert transported["isolation_exponent"] < 0.0
    assert math.isclose(
        transported["overall_exponent"],
        0.5 + epsilon,
        abs_tol=1e-12,
    )


if __name__ == "__main__":
    test_zero_endpoint_importance_penalty()
    test_sparse_histogram_normalization_blocks_naive_gl()
    test_direct_half_coset_transport_reaches_half_limit()
    print("walsh recovery theorem ledger tests passed")
