#!/usr/bin/env python3
"""Unit checks for the spherical-code-to-SVP exponent transfer."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "walsh_spherical_bound_transfer.py"
SPEC = importlib.util.spec_from_file_location("walsh_spherical_bound_transfer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_classical_formula_matches_degree_zero_construction() -> None:
    for s in (0.1, 0.3, 0.5, 0.8):
        assert math.isclose(MOD.kl_closed(s), MOD.direct_kl(s), rel_tol=2e-14)


def test_one_row_contains_and_strictly_improves_classical_boundary() -> None:
    s = 0.5
    value, a, b = MOD.direct_row_cached(s)
    assert b > 0.0
    assert a > b
    assert 2.0 * MOD.gamma_row(a, b) >= s - 1e-12
    assert value < MOD.kl_closed(s) - 0.004


def test_reported_transfer_improves_paper_constant() -> None:
    report_path = ROOT / "experiments" / "out" / "walsh_spherical_bound_transfer.json"
    import json

    report = json.loads(report_path.read_text())
    baseline = report["baseline_KL"]["overall_exponent"]
    improved = report["moving_one_row"]["exact_nested_bound_check"][
        "overall_exponent"
    ]
    assert 0.6038 < baseline < 0.6039
    assert 0.6024 < improved < 0.6026
    assert baseline - improved > 0.0013


if __name__ == "__main__":
    test_classical_formula_matches_degree_zero_construction()
    test_one_row_contains_and_strictly_improves_classical_boundary()
    test_reported_transfer_improves_paper_constant()
    print("walsh spherical-bound transfer tests passed")
