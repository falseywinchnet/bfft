"""Regression tests for the shared-width midpoint-Hessian control variate."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_cross_scale_control_audit import audit_cross_scale


def test_proxy_preserving_control_minimizes_integrated_variance():
    report = audit_cross_scale(
        4, widths=(0.18, 0.205, 0.2222355, 0.24),
        cutoff=2, seed=19, chunk_size=10_000,
    )
    for affine in report["affine_reports"]:
        assert affine["variance_reduction"] >= 1.0 - 1e-9
        assert affine["proxy_preserving_control"]["coefficient_l1"] > 0.0
        assert affine["proxy_preserving_control"][
            "covariance_negative_eigenvalue_fraction"
        ] < 1e-8


if __name__ == "__main__":
    test_proxy_preserving_control_minimizes_integrated_variance()
    print("walsh cross-scale control tests passed")
