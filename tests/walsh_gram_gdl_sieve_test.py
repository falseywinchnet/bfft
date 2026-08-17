#!/usr/bin/env python3
"""Regression checks for the exact Gram-factor GDL reduction."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_gram_gdl_sieve import (
    coefficient_domain_radii,
    gram_adjacency,
    gram_gdl_bound,
    weighted_min_fill,
)


def main() -> None:
    basis = np.diag([1.0, 8.0, 8.0])
    radii = coefficient_domain_radii(basis, 4.0 / 3.0)
    assert radii.tolist() == [1, 0, 0]
    graph = gram_adjacency(basis)
    assert all(not neighbors for neighbors in graph.values())
    bound = gram_gdl_bound(basis, 4.0 / 3.0)
    assert abs(bound["maximum_table_log2_entries"] - math.log2(3.0)) < 1e-12

    path = {0: {1}, 1: {0, 2}, 2: {1}}
    order, width, trace = weighted_min_fill(path, np.asarray([3, 5, 7]))
    assert sorted(order) == [0, 1, 2]
    assert len(trace) == 3
    assert width <= math.log2(35.0) + 1e-12

    dense = np.asarray([[1.0, 0.2], [0.1, 1.0]])
    assert gram_adjacency(dense) == {0: {1}, 1: {0}}
    print("walsh Gram-GDL sieve tests passed")


if __name__ == "__main__":
    main()
