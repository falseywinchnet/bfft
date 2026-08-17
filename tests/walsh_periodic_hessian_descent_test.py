#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_periodic_hessian_descent import (
    ascend_periodic_hessian,
    coefficient_box,
    nearest_edge,
    periodic_hessian_value_and_gradient,
    periodic_hessian_spectral_data,
)


def test_periodic_hessian_gradient() -> None:
    basis = np.asarray([[1.0, 0.17], [0.11, 1.07]])
    points = coefficient_box(2, 3) @ np.linalg.inv(basis)
    value = np.asarray([0.21, -0.16])
    width = 0.8
    score, gradient, _ = periodic_hessian_value_and_gradient(
        value, basis, points, width
    )
    assert score > 0.0
    step = 1e-6
    numerical = np.empty(2)
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = step
        high = periodic_hessian_value_and_gradient(
            value + delta, basis, points, width
        )[0]
        low = periodic_hessian_value_and_gradient(
            value - delta, basis, points, width
        )[0]
        numerical[axis] = (high - low) / (2.0 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6)
    spectral_score, eigengap, spectral_gradient, _ = (
        periodic_hessian_spectral_data(value, basis, points, width)
    )
    assert abs(spectral_score - score) < 1e-12
    assert eigengap >= 0.0
    np.testing.assert_allclose(spectral_gradient, gradient)


def test_orthogonal_ascent_reaches_a_shortest_edge() -> None:
    basis = np.eye(2)
    coefficients = coefficient_box(2, 3)
    points = coefficients @ basis.T
    result = ascend_periodic_hessian(
        np.asarray([0.29, 0.08]),
        basis,
        coefficients @ np.linalg.inv(basis),
        0.8,
        max_iterations=120,
    )
    _, edge, edge_norm, midpoint_error = nearest_edge(
        result["coefficient_point"], basis, coefficients, points
    )
    assert abs(edge_norm - 1.0) < 1e-7
    assert midpoint_error < 1e-5
    assert abs(float(np.dot(result["leading_direction"], edge))) > 0.999
    assert result["initial_score"] >= 0.0
    assert result["minimum_evaluated_eigengap"] >= 0.0


if __name__ == "__main__":
    test_periodic_hessian_gradient()
    test_orthogonal_ascent_reaches_a_shortest_edge()
    print("walsh periodic-Hessian descent tests passed")
