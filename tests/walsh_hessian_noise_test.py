"""Regression tests for the exact midpoint-Hessian Walsh covariance audit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import audit_dimension, fwht


def test_fwht_is_involution_up_to_scale():
    rng = np.random.default_rng(7)
    value = rng.standard_normal((16, 3))
    assert np.allclose(fwht(fwht(value)), 16.0 * value)


def test_exact_covariance_audit_is_psd_and_trace_removes_identity():
    report = audit_dimension(4, cutoff=2, seed=91, chunk_size=10_000)
    assert report["walsh_outputs"] == 8
    assert report["affine_reports"]
    for affine in report["affine_reports"]:
        variants = affine["variants"]
        for value in variants.values():
            assert value["covariance_negative_eigenvalue_fraction"] < 1e-9
            assert 0.0 <= value["covariance_effective_fraction"] <= 1.0 + 1e-12
            assert 0.0 <= value["kernel_off_origin_energy_fraction"] <= 1.0
        raw_variance = variants["raw_importance"]["mean_coefficient_variance"]
        trace_variance = variants["traceless_importance"][
            "mean_coefficient_variance"
        ]
        identity_variance = variants["identity_importance"][
            "mean_coefficient_variance"
        ]
        assert np.isclose(raw_variance, trace_variance + identity_variance,
                          rtol=1e-10, atol=1e-12)


def test_horvitz_thompson_does_not_reduce_second_moment():
    report = audit_dimension(4, cutoff=2, seed=101, chunk_size=10_000)
    for affine in report["affine_reports"]:
        variants = affine["variants"]
        ordinary = variants["traceless_importance"][
            "mean_coefficient_variance"
        ]
        sparse = variants["traceless_horvitz_thompson"][
            "mean_coefficient_variance"
        ]
        assert sparse >= ordinary * (1.0 - 1e-12)


if __name__ == "__main__":
    test_fwht_is_involution_up_to_scale()
    test_exact_covariance_audit_is_psd_and_trace_removes_identity()
    test_horvitz_thompson_does_not_reduce_second_moment()
    print("walsh hessian noise tests passed")
