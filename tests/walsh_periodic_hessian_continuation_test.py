#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_periodic_hessian_continuation import (
    branches_match,
    iota,
    merge_branches,
    run_continuation,
)
from experiments.walsh_periodic_hessian_descent import coefficient_box


def test_importance_exponent_is_zero_at_equal_scales() -> None:
    assert iota(0.2, 0.2) == 0.0
    assert 0.0 < iota(0.15, 0.23675858) < 0.11


def test_branch_merge_uses_torus_and_direction() -> None:
    left = {
        "coefficient_point": np.asarray([0.49, 0.1]),
        "leading_direction": np.asarray([1.0, 0.0]),
        "score": 2.0,
        "ancestors": [0],
        "birth_levels": [0],
        "merged_copies": 1,
    }
    right = {
        "coefficient_point": np.asarray([-0.51, 0.1]),
        "leading_direction": np.asarray([-1.0, 0.0]),
        "score": 1.0,
        "ancestors": [1],
        "birth_levels": [2],
        "merged_copies": 1,
    }
    assert branches_match(
        left,
        right,
        np.eye(2),
        1.0,
        location_tolerance=1e-6,
        direction_alignment=0.999,
    )
    merged = merge_branches([left, right], np.eye(2), 1.0)
    assert len(merged) == 1
    assert merged[0]["ancestors"] == [0, 1]
    assert merged[0]["birth_levels"] == [0, 2]


def test_orthogonal_continuation_retains_shortest_branch() -> None:
    basis = np.eye(2)
    field_coefficients = coefficient_box(2, 3)
    coefficients = coefficient_box(2, 3)
    points = coefficients.astype(float)
    report = run_continuation(
        basis=basis,
        dual_points=field_coefficients.astype(float),
        coefficients=coefficients,
        points=points,
        shortest=1.0,
        shortest_coefficients=np.asarray([1.0, 0.0]),
        ladder=np.asarray([0.15, 0.19, 0.23675858]),
        initial_starts=np.asarray([[0.28, 0.04], [-0.05, 0.31]]),
        birth_starts=[np.empty((0, 2)), np.empty((0, 2))],
        allow_births=False,
    )
    assert report["terminal_shortest_present"]
    assert report["terminal_shortest_inherited_from_initial_scale"]
    assert report["terminal_shortest_initial_ancestors"]
    final = report["levels"][-1]
    assert final["maximum_shortest_best_path_bottleneck_score_ratio"] > 0.0
    assert final["maximum_shortest_best_path_bottleneck_eigengap_ratio"] > 0.0


if __name__ == "__main__":
    test_importance_exponent_is_zero_at_equal_scales()
    test_branch_merge_uses_torus_and_direction()
    test_orthogonal_continuation_retains_shortest_branch()
    print("walsh periodic-Hessian continuation tests passed")
