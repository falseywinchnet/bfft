#!/usr/bin/env python3
"""Algebraic checks for cold Gaussian syndrome folding."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_syndrome_folding_transport import (
    parity_prefixes,
    shortest_syndrome_representatives,
)


def test_representatives_have_requested_syndromes() -> None:
    n, h = 4, 2
    inverse_basis = np.eye(n)
    inverse_transform = np.eye(n, dtype=np.uint8)
    representatives, norm2 = shortest_syndrome_representatives(
        n,
        h,
        inverse_basis,
        inverse_transform,
        cutoff=1,
    )
    observed = parity_prefixes(representatives, inverse_transform, h)
    assert np.array_equal(observed, np.arange(1 << h))
    assert math.isclose(norm2[0], 0.0)


if __name__ == "__main__":
    test_representatives_have_requested_syndromes()
    print("walsh syndrome-folding transport tests passed")
