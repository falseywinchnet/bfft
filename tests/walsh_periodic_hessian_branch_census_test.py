#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_periodic_hessian_branch_census import (
    census_scale,
    sobol_starts,
)
from experiments.walsh_periodic_hessian_descent import coefficient_box


def test_sobol_probe_family_is_nested() -> None:
    small = sobol_starts(3, 4, 17)
    large = sobol_starts(3, 5, 17)
    np.testing.assert_allclose(small, large[: len(small)])
    assert np.all(small >= -0.5)
    assert np.all(small < 0.5)


def test_orthogonal_census_finds_shortest_ridges() -> None:
    basis = np.eye(2)
    dual_points = coefficient_box(2, 3).astype(float)
    branches, evaluations = census_scale(
        sobol_starts(2, 4, 19),
        basis=basis,
        dual_points=dual_points,
        shortest=1.0,
        shortest_coefficients=np.asarray([1.0, 0.0]),
        t=0.2,
    )
    assert branches
    assert evaluations >= 16
    assert all(float(branch["score"]) > 0.0 for branch in branches)


if __name__ == "__main__":
    test_sobol_probe_family_is_nested()
    test_orthogonal_census_finds_shortest_ridges()
    print("walsh periodic-Hessian branch-census tests passed")
