#!/usr/bin/env python3
"""Finite audit of a flatness-controlled lattice OU Metropolis kernel.

The proposal on a finite lattice state set A is

    q_x(y) proportional to exp(-pi ||y-alpha*x||^2 / tau^2),
    tau^2 = s^2(1-alpha^2).

The Gaussian quadratic identity leaves only the shifted proposal normalizers
in the Metropolis ratio.  On an infinite affine lattice those normalizers are
uniform whenever the dual theta mass at 1/tau is small.  The audit checks the
exact algebra, detailed balance, and small-dimensional spectral gaps.  It is
not a sampler for q_x: efficient shifted DGS at tau is the remaining
algorithmic obstruction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_flatness_ou_kernel.json"


def affine_box_points(n: int, cutoff: int, parity: int = 0) -> np.ndarray:
    rows = []
    for z in itertools.product(range(-cutoff, cutoff + 1), repeat=n):
        if (sum(z) & 1) == parity:
            rows.append(z)
    return np.asarray(rows, dtype=np.float64)


def ou_quadratic_error(x: np.ndarray, y: np.ndarray, s: float, alpha: float) -> float:
    tau2 = s * s * (1.0 - alpha * alpha)
    forward = np.dot(x, x) / (s * s) + np.dot(y - alpha * x, y - alpha * x) / tau2
    reverse = np.dot(y, y) / (s * s) + np.dot(x - alpha * y, x - alpha * y) / tau2
    return float(abs(forward - reverse))


def ou_metropolis_matrix(points: np.ndarray, s: float, alpha: float):
    points = np.asarray(points, dtype=np.float64)
    tau = s * math.sqrt(1.0 - alpha * alpha)
    target_mass = np.exp(-math.pi * np.einsum("ij,ij->i", points, points) / (s * s))
    stationary = target_mass / np.sum(target_mass)

    differences = points[None, :, :] - alpha * points[:, None, :]
    distance2 = np.einsum("ijk,ijk->ij", differences, differences)
    proposal_raw = np.exp(-math.pi * distance2 / (tau * tau))
    normalizers = np.sum(proposal_raw, axis=1)
    proposal = proposal_raw / normalizers[:, None]

    # The exact reverse/forward target-proposal ratio is Z_x/Z_y.
    ratio = normalizers[:, None] / normalizers[None, :]
    acceptance = np.minimum(1.0, ratio)
    transition = proposal * acceptance
    transition[np.diag_indices_from(transition)] += 1.0 - np.sum(transition, axis=1)
    return transition, stationary, normalizers, acceptance


def reversible_gap(transition: np.ndarray, stationary: np.ndarray) -> float:
    root = np.sqrt(stationary)
    symmetric = root[:, None] * transition / root[None, :]
    eigenvalues = np.linalg.eigvalsh(0.5 * (symmetric + symmetric.T))
    return float(1.0 - np.max(np.abs(eigenvalues[:-1])))


def audit_case(n: int, cutoff: int, s: float, alpha: float) -> dict[str, float | int]:
    coefficients = affine_box_points(n, cutoff, parity=1)
    basis = np.eye(n)
    for i in range(n - 1):
        basis[i, i + 1] = 0.19
    points = coefficients @ basis
    transition, stationary, normalizers, acceptance = ou_metropolis_matrix(points, s, alpha)
    detailed_balance = stationary[:, None] * transition
    return {
        "dimension": n,
        "states": len(points),
        "cutoff": cutoff,
        "s": s,
        "alpha": alpha,
        "proposal_width_ratio": math.sqrt(1.0 - alpha * alpha),
        "maximum_quadratic_identity_error": max(
            ou_quadratic_error(points[i], points[j], s, alpha)
            for i in range(len(points))
            for j in range(len(points))
        ),
        "row_stochastic_error": float(np.max(np.abs(np.sum(transition, axis=1) - 1.0))),
        "detailed_balance_error": float(
            np.max(np.abs(detailed_balance - detailed_balance.T))
        ),
        "normalizer_relative_spread": float(
            np.max(normalizers) / np.min(normalizers) - 1.0
        ),
        "minimum_off_diagonal_acceptance": float(
            np.min(acceptance[~np.eye(len(points), dtype=bool)])
        ),
        "absolute_spectral_gap": reversible_gap(transition, stationary),
    }


def frontier_constants(r: float = 0.23675858) -> dict[str, float]:
    g2_r = 0.02057231
    g2_2r = 0.52039983
    return {
        "target_width": r,
        "gap_budget": 0.5 - 2.0 * r,
        "cold_dgs_dual_mass_exponent": 0.5 - g2_r,
        "target_scale_flatness_slack": g2_2r - 0.5,
        "cold_points_from_one_superlattice_batch_exponent": g2_r,
        "required_effective_samples_exponent": 2.0 * r,
        "superlattice_batch_shortfall_exponent": 2.0 * r - g2_r,
    }


def audit() -> dict[str, object]:
    return {
        "experiment": "walsh_flatness_ou_kernel",
        "frontier": frontier_constants(),
        "cases": [
            audit_case(n=2, cutoff=4, s=1.7, alpha=0.35),
            audit_case(n=2, cutoff=4, s=1.7, alpha=0.70),
            audit_case(n=3, cutoff=2, s=1.9, alpha=0.50),
        ],
        "interpretation": (
            "The lattice OU Metropolis ratio is controlled solely by shifted "
            "theta normalizers and has healthy finite gaps when its proposal is "
            "available.  Its proposal width is strictly below the target width, "
            "so existing above-smoothing DGS does not implement the kernel."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
