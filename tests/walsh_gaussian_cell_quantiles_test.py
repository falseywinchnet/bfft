#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_gaussian_cell_quantiles import (
    audit_fixture,
    central_probability_scale,
)


def test_requested_central_probabilities() -> None:
    rng = np.random.default_rng(17)
    scale = rng.normal(size=(100_000, 3))
    law = rng.normal(size=(100_000, 3))
    cold = central_probability_scale(np.eye(3), scale, 0.9)
    warm = central_probability_scale(np.eye(3), scale, 0.1)
    assert cold < warm
    row = audit_fixture(
        "identity",
        np.eye(3),
        scale_gaussians=scale,
        law_gaussians=law,
        central_probabilities=(0.9, 0.5),
        shortest_cutoff=1,
    )
    for requested, level in zip((0.9, 0.5), row["levels"]):
        assert abs(level["empirical_central_cell_probability"] - requested) < 0.01
        assert level["chosen_shortest_parity_mass_after_zero_parity_rejection"] > 0.2


def main() -> None:
    test_requested_central_probabilities()
    print("walsh Gaussian-cell quantile tests passed")


if __name__ == "__main__":
    main()
