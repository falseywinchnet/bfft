#!/usr/bin/env python3
"""Regression checks for causal Voronoi-tiling transport."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_voronoi_causal_transport import (
    audit_basis,
    coefficient_cube_with_zero,
    gap_cap_parameters,
    nearest_label_indices,
)


def main() -> None:
    gap = gap_cap_parameters(2.0)
    assert gap["winner_cap_cosine"] == 0.5
    assert abs(
        gap["asymptotic_direct_mass_exponent"]
        + 0.5 * math.log2(0.75)
    ) < 1e-12
    unit_gap = gap_cap_parameters(1.0)
    assert abs(unit_gap["winner_cap_cosine"] - math.sqrt(3.0) / 2.0) < 1e-12
    assert abs(unit_gap["asymptotic_direct_mass_exponent"] - 1.0) < 1e-12
    target_gap = gap_cap_parameters(5.2670066054020195)
    assert abs(
        target_gap["asymptotic_direct_mass_exponent"] - 0.02648284
    ) < 1e-10

    basis = np.diag([1.0, 8.0, 8.0])
    labels = coefficient_cube_with_zero(3, 1)
    vectors = labels @ basis.T
    points = np.asarray([
        [0.49, 0.0, 0.0],
        [0.51, 0.0, 0.0],
        [1.49, 0.0, 0.0],
    ])
    winners = labels[nearest_label_indices(points, vectors)]
    assert winners.tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 0]]

    rng = np.random.default_rng(484)
    samples = 4096
    coefficients = rng.uniform(-0.5, 0.5, size=(samples, 3))
    displacements = rng.normal(size=(samples, 3))
    directions = rng.normal(size=(samples, 3))
    row = audit_basis(
        "rectangular",
        basis,
        coefficients,
        displacements,
        directions,
        cutoff=1,
        step_ratios=(0.05,),
    )
    step = row["causal_steps"][0]
    assert row["gap_cap_diagnostic"][
        "asymptotic_direct_mass_exponent"
    ] < 0.02648284
    assert row["direct_first_exit_shortest_parity_mass"] > 0.7
    assert step["shortest_vector_crossing_mass"] > 0.0
    assert step["shortest_parity_crossing_mass"] >= step[
        "shortest_vector_crossing_mass"
    ]
    assert 0.0 <= step["shortest_parity_given_label_change"] <= 1.0
    print("walsh Voronoi causal-transport tests passed")


if __name__ == "__main__":
    main()
