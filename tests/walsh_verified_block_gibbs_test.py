#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_verified_block_gibbs import (
    audit_energy,
    heat_bath_transition,
)


def test_transition_is_stochastic_and_stationary() -> None:
    weights = np.asarray([0.05, 0.15, 0.3, 0.5], dtype=np.float64)
    transition = heat_bath_transition(weights, [(0,), (1,)])
    assert np.allclose(np.sum(transition, axis=1), 1.0)
    assert np.allclose(weights @ transition, weights)


def test_verified_gibbs_stationary_mass() -> None:
    row = audit_energy("needle", np.asarray([1.0] * 7 + [0.0]))
    assert row["stationary_minimum_mass"] >= 1.0 - 2.0 ** -6
    assert row["minimum_mass_from_uniform_by_step"][0] == 1.0 / 8.0


def main() -> None:
    test_transition_is_stochastic_and_stationary()
    test_verified_gibbs_stationary_mass()
    print("walsh verified block-Gibbs tests passed")


if __name__ == "__main__":
    main()
