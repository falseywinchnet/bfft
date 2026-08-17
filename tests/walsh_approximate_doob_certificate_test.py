#!/usr/bin/env python3
"""Checks the approximate-Doob likelihood and Renyi certificate."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_approximate_doob_certificate import (
    bridge_endpoint,
    certificate_budget,
    exact_backward_potentials,
    renyi2_log2,
)


def test_approximate_bridge_obeys_certificate() -> None:
    transition = np.array([
        [0.70, 0.20, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.20, 0.70],
    ])
    stationary = np.full(3, 1.0 / 3.0)
    event = np.array([False, False, True])
    exact = exact_backward_potentials(transition, event, horizon=5)
    approximate = [value.copy() for value in exact]
    for k in range(len(approximate) - 1):
        phase = np.array([-0.04, 0.02, 0.03]) * (k + 1)
        approximate[k] *= np.exp2(phase)

    endpoint = bridge_endpoint(stationary, transition, approximate)
    target = stationary * event
    actual = renyi2_log2(target[event], endpoint[event])
    certificate = certificate_budget(
        stationary,
        transition,
        event,
        exact,
        approximate,
    )
    assert np.sum(endpoint[~event]) < 1e-15
    assert actual <= certificate["total_log2_likelihood_budget"] + 1e-12


if __name__ == "__main__":
    test_approximate_bridge_obeys_certificate()
    print("walsh approximate-Doob certificate tests passed")
