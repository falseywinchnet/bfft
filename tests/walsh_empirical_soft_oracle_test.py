#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_empirical_soft_oracle import (
    narrowing_importance_weights,
    normalized_gaussian_probabilities,
    soft_oracle_from_measure,
    traceless_hessian_from_measure,
)
from experiments.walsh_periodic_hessian_descent import coefficient_box


def test_population_importance_identity_and_bounded_weights() -> None:
    basis = np.asarray([[1.0, 0.17], [0.11, 1.07]])
    points = coefficient_box(2, 4) @ np.linalg.inv(basis)
    source_width = 0.7
    target_width = 1.1
    source = normalized_gaussian_probabilities(points, source_width)
    target = normalized_gaussian_probabilities(points, target_width)
    weight = narrowing_importance_weights(points, source_width, target_width)
    reweighted = source * weight
    reweighted /= np.sum(reweighted)
    np.testing.assert_allclose(reweighted, target, atol=2e-16, rtol=2e-15)
    assert np.max(weight) == 1.0
    assert np.min(weight) >= 0.0


def test_measure_soft_gradient() -> None:
    basis = np.asarray([[1.0, 0.17], [0.11, 1.07]])
    points = coefficient_box(2, 3) @ np.linalg.inv(basis)
    probabilities = normalized_gaussian_probabilities(points, 0.8)
    value = np.asarray([0.21, -0.16])
    kwargs = dict(beta=7.0, normalization=1.3)
    _, gradient = soft_oracle_from_measure(
        value, basis, points, probabilities, **kwargs
    )
    step = 1e-6
    numerical = np.empty(2)
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = step
        high = soft_oracle_from_measure(
            value + delta, basis, points, probabilities, **kwargs
        )[0]
        low = soft_oracle_from_measure(
            value - delta, basis, points, probabilities, **kwargs
        )[0]
        numerical[axis] = (high - low) / (2.0 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6)
    matrix = traceless_hessian_from_measure(
        value, basis, points, probabilities
    )
    assert abs(float(np.trace(matrix))) < 1e-12


if __name__ == "__main__":
    test_population_importance_identity_and_bounded_weights()
    test_measure_soft_gradient()
    print("walsh empirical soft-oracle tests passed")
