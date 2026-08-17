#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_periodic_hessian_adversarial_search import (
    adversarial_basis,
    half_grid_error,
    snapped_half_parity,
)


def test_adversarial_basis_is_reproducible_and_nonsingular() -> None:
    left = adversarial_basis(4, np.random.default_rng(17), 1.2)
    right = adversarial_basis(4, np.random.default_rng(17), 1.2)
    np.testing.assert_allclose(left, right)
    assert abs(np.linalg.det(left)) > 1e-8
    assert np.linalg.cond(left) > 1.0


def test_half_grid_snap_is_torus_parity() -> None:
    value = np.asarray([0.499999, -0.500001, 0.000002, -0.000003])
    assert snapped_half_parity(value) == (1, 1, 0, 0)
    assert half_grid_error(value) < 1e-4


if __name__ == "__main__":
    test_adversarial_basis_is_reproducible_and_nonsingular()
    test_half_grid_snap_is_torus_parity()
    print("walsh periodic-Hessian adversarial-search tests passed")
