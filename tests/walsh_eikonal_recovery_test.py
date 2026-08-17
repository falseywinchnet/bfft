"""Regression tests for causal matrix-Walsh frontier recovery."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_eikonal_recovery import (
    _parity_table,
    adaptive_causal_frontier,
    causal_frontier,
    certified_causal_frontier,
    descendant_energy,
    masked_descendant_energy,
    split_descendant_energy,
)
from experiments.walsh_hessian_noise_audit import fwht


def test_descendant_parseval_and_child_conservation():
    rng = np.random.default_rng(17)
    histogram = rng.standard_normal((16, 5))
    transformed = fwht(histogram)
    parity = _parity_table(16)
    for depth in range(5):
        for prefix in range(1 << depth):
            expected = float(np.sum(
                np.einsum("ij,ij->i", transformed, transformed)[
                    (np.arange(16) & ((1 << depth) - 1)) == prefix
                ]
            ))
            measured = descendant_energy(histogram, prefix, depth, parity)
            assert np.isclose(measured, expected, rtol=1e-12, atol=1e-10)
            if depth < 4:
                children = (
                    descendant_energy(histogram, prefix, depth + 1, parity)
                    + descendant_energy(
                        histogram, prefix | (1 << depth), depth + 1, parity
                    )
                )
                assert np.isclose(children, measured, rtol=1e-12, atol=1e-10)
    energy = np.einsum("ij,ij->i", transformed, transformed)
    for mask in range(16):
        for prefix in range(16):
            if prefix & ~mask:
                continue
            expected = float(np.sum(
                energy[(np.arange(16) & mask) == prefix]
            ))
            measured = masked_descendant_energy(
                histogram, prefix, mask, parity
            )
            assert np.isclose(measured, expected, rtol=1e-12, atol=1e-10)


def test_split_score_has_exact_child_conservation_for_fixed_panels():
    rng = np.random.default_rng(23)
    labels = rng.integers(0, 16, size=300)
    values = rng.standard_normal((300, 4))
    panel = np.arange(300, dtype=np.int8) & 1
    parity = _parity_table(16)
    for depth in range(4):
        for prefix in range(1 << depth):
            parent = split_descendant_energy(
                labels, values, panel, prefix, depth, 4, parity
            )
            children = split_descendant_energy(
                labels, values, panel, prefix, depth + 1, 4, parity
            ) + split_descendant_energy(
                labels, values, panel, prefix | (1 << depth), depth + 1,
                4, parity,
            )
            assert np.isclose(children, parent, rtol=1e-12, atol=1e-10)


def test_single_spike_arrives_through_width_one_front():
    rng = np.random.default_rng(29)
    theta = 11
    direction = rng.standard_normal(6)
    y = np.arange(16)
    sign = 1.0 - 2.0 * _parity_table(16)[y & theta]
    histogram = sign[:, None] * direction[None, :] / 16.0
    result = causal_frontier(
        lambda prefix, depth: descendant_energy(histogram, prefix, depth),
        ell=4, width=1, theta_star=theta,
    )
    assert result["target_retained_all_depths"]
    assert result["target_recovered"]
    assert result["visited_nodes"] == 8
    adaptive = adaptive_causal_frontier(
        lambda prefix, mask: masked_descendant_energy(
            histogram, prefix, mask
        ),
        ell=4, width=1, theta_star=theta,
    )
    assert adaptive["target_retained_all_depths"]
    assert adaptive["target_recovered"]
    assert sorted(adaptive["bit_order"]) == [0, 1, 2, 3]
    target_energy = descendant_energy(histogram, theta, 4)
    certified = certified_causal_frontier(
        lambda prefix, depth: descendant_energy(histogram, prefix, depth),
        ell=4, threshold=target_energy, theta_star=theta,
    )
    assert certified["target_retained_all_depths"]
    assert certified["target_recovered"]
    assert certified["maximum_frontier_width"] == 1


if __name__ == "__main__":
    test_descendant_parseval_and_child_conservation()
    test_split_score_has_exact_child_conservation_for_fixed_panels()
    test_single_spike_arrives_through_width_one_front()
    print("walsh eikonal recovery tests passed")
