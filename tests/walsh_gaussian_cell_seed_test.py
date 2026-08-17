#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_gaussian_cell_seed import (
    coefficient_gaussians,
    gaussian_cell_audit,
    median_cell_scale,
)


def test_coefficient_gaussian_map() -> None:
    basis = np.diag([2.0, 3.0])
    samples = np.asarray([[2.0, 3.0], [-4.0, 6.0]])
    assert np.allclose(
        coefficient_gaussians(basis, samples),
        np.asarray([[1.0, 1.0], [-2.0, 2.0]]),
    )


def test_median_scale_and_shortest_mass_on_identity() -> None:
    rng = np.random.default_rng(13)
    median_samples = rng.normal(size=(80_000, 3))
    law_samples = rng.normal(size=(80_000, 3))
    tau = median_cell_scale(np.eye(3), median_samples)
    assert 0.25 < tau < 0.5
    row = gaussian_cell_audit(
        "identity",
        np.eye(3),
        median_gaussians=median_samples,
        law_gaussians=law_samples,
        shortest_cutoff=1,
    )
    assert 0.47 < row["empirical_central_cell_probability"] < 0.53
    # The identity lattice has three distinct shortest parity classes.  This
    # audit fixes one representative, as does the cell-shift theorem.
    assert row["empirical_chosen_shortest_parity_mass_after_zero_rejection"] > 0.2
    assert row["cell_shift_certified_mass_lower_bound"] > 0.01


def main() -> None:
    test_coefficient_gaussian_map()
    test_median_scale_and_shortest_mass_on_identity()
    print("walsh Gaussian-cell seed tests passed")


if __name__ == "__main__":
    main()
