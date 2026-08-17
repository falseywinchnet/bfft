#!/usr/bin/env python3
"""Checks for the finite cold-syndrome Doob bridge."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_doob_syndrome_bridge import (
    bridge_report,
    metropolis_matrix,
)
from experiments.walsh_syndrome_folding_transport import enumerated_coefficients


def test_metropolis_reversibility_and_bridge_bound() -> None:
    coefficients = enumerated_coefficients(2, 1)
    norm2 = np.einsum("ij,ij->i", coefficients, coefficients).astype(np.float64)
    mass = np.exp(-norm2)
    stationary = mass / np.sum(mass)
    transition = metropolis_matrix(coefficients, mass)
    assert np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1.0)) < 1e-14
    event = ((coefficients[:, 0] & 1) == 0)
    report = bridge_report(transition, stationary, event, steps=16)
    epsilon = report["committor_log2_spread"]
    assert report["bridge_mass_error"] < 1e-12
    assert report["renyi2_log_mass"] <= epsilon + 1e-12
    assert math.isfinite(report["renyi2_log_mass"])


if __name__ == "__main__":
    test_metropolis_reversibility_and_bridge_bound()
    print("walsh Doob syndrome-bridge tests passed")
