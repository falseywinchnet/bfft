#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_voronoi_first_exit import first_exit_audit


def test_rectangular_first_exit_finds_short_axis() -> None:
    rng = np.random.default_rng(19)
    directions = rng.normal(size=(20_000, 3))
    row = first_exit_audit(
        "rectangular",
        np.diag([1.0, 8.0, 8.0]),
        directions,
        cutoff=1,
    )
    assert row["random_ray_shortest_vector_exit_mass"] > 0.85
    assert row["random_ray_shortest_parity_exit_mass"] > 0.85


def test_identity_every_exit_is_shortest() -> None:
    rng = np.random.default_rng(23)
    directions = rng.normal(size=(10_000, 3))
    row = first_exit_audit("identity", np.eye(3), directions, cutoff=1)
    assert row["random_ray_shortest_vector_exit_mass"] == 1.0
    assert row["random_ray_shortest_parity_exit_mass"] == 1.0


def main() -> None:
    test_rectangular_first_exit_finds_short_axis()
    test_identity_every_exit_is_shortest()
    print("walsh Voronoi first-exit tests passed")


if __name__ == "__main__":
    main()
