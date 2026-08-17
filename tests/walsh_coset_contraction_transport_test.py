"""Small algebraic checks for projected-coset transport diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_coset_contraction_transport import (
    optimize_source_mixture,
    renyi2_divergence,
)


def test_renyi2_identity_and_mixture_optimization() -> None:
    target = np.array([0.7, 0.2, 0.1])
    source = np.array([0.2, 0.3, 0.5])
    transported = np.array([0.8, 0.2, 0.0])
    assert math.isclose(renyi2_divergence(target, target), 0.0, abs_tol=1e-15)
    epsilon, value = optimize_source_mixture(target, source, transported)
    assert 0.0 < epsilon < 1.0
    assert value < renyi2_divergence(target, source)


if __name__ == "__main__":
    test_renyi2_identity_and_mixture_optimization()
    print("walsh coset contraction transport tests passed")
