#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_accessible_seed_law import (
    accessible_scale,
    soft_hessian_value_and_gradient,
)
from experiments.walsh_periodic_hessian_descent import coefficient_box


def test_accessible_scale_satisfies_mass_and_budget_conditions() -> None:
    r = 0.23675858
    t = accessible_scale(r)
    q = t * r / (2.0 * r - t)
    iota = 0.5 * np.log2(r * r / (t * (2.0 * r - t)))
    assert q > 0.23147 / 2.0
    assert 2.0 * t + iota < 0.5


def test_soft_gradient_and_half_grid_stationarity() -> None:
    basis = np.asarray([[1.0, 0.17], [0.11, 1.07]])
    dual = coefficient_box(2, 3) @ np.linalg.inv(basis)
    value = np.asarray([0.21, -0.16])
    kwargs = dict(beta=7.0, normalization=1.3)
    _, gradient = soft_hessian_value_and_gradient(value, basis, dual, 0.8, **kwargs)
    step = 1e-6
    numerical = np.empty(2)
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = step
        high = soft_hessian_value_and_gradient(
            value + delta, basis, dual, 0.8, **kwargs
        )[0]
        low = soft_hessian_value_and_gradient(
            value - delta, basis, dual, 0.8, **kwargs
        )[0]
        numerical[axis] = (high - low) / (2.0 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6)
    _, stationary = soft_hessian_value_and_gradient(
        np.asarray([0.5, 0.0]), basis, dual, 0.8, **kwargs
    )
    assert np.linalg.norm(stationary) < 1e-10


if __name__ == "__main__":
    test_accessible_scale_satisfies_mass_and_budget_conditions()
    test_soft_gradient_and_half_grid_stationarity()
    print("walsh accessible seed-law tests passed")
