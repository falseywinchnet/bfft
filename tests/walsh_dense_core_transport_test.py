#!/usr/bin/env python3
"""Regression checks for relational shortest-parity transport."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_dense_core_transport import (
    affine_rank,
    audit_basis,
    fwht,
    in_affine_hull,
    packed_bits,
    xor_square,
)
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)


def main() -> None:
    rng = np.random.default_rng(8317)

    signal = rng.normal(size=16)
    assert np.allclose(fwht(fwht(signal)), len(signal) * signal)

    probability = rng.random(16)
    probability /= probability.sum()
    expected = np.zeros(16)
    for left, p_left in enumerate(probability):
        for right, p_right in enumerate(probability):
            expected[left ^ right] += p_left * p_right
    assert np.allclose(xor_square(probability), expected)

    labels = np.asarray([0b0011, 0b0110, 0b1001, 0b1100])
    assert packed_bits(labels[:1], 4).tolist() == [[1, 1, 0, 0]]
    assert affine_rank(labels, 4) == 2
    assert in_affine_hull(labels, labels, 4).all()
    assert not in_affine_hull(labels, np.asarray([0]), 4)[0]

    directions = rng.normal(size=(256, 4))
    tangents = rng.normal(size=(256, 4))
    row = audit_basis(
        "simplex",
        simplex_cancellation_basis(4, 0.97),
        directions,
        tangents,
        cutoff=2,
        angles=(0.1,),
        length_slack=1.0,
    )
    assert 0.0 <= row["single_ray_shortest_parity_mass"] <= 1.0
    assert 0.0 <= row["independent_pair_shortest_xor_mass"] <= 1.0 + 1e-12
    transport = row["nearby_ray_transports"][0]
    assert 0.0 <= transport["label_change_mass"] <= 1.0
    assert 0.0 <= transport["unconditional_shortest_xor_mass"] <= 1.0
    assert transport["label_change_flux_per_radian"] >= 0.0
    assert transport["shortest_xor_flux_per_radian"] >= 0.0
    assert 0 <= row["sampled_active_affine_rank"] <= 4
    assert row["all_enumerated_shortest_parities_in_sampled_affine_hull"]
    assert row["best_proved_cover_log2"] <= 4.0
    print("walsh dense-core transport tests passed")


if __name__ == "__main__":
    main()
