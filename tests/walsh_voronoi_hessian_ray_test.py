#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_voronoi_hessian_ray import ray_audit


def test_identity_boundary_exposes_axis_label() -> None:
    rng = np.random.default_rng(29)
    directions = rng.normal(size=(64, 3))
    row = ray_audit(
        "identity",
        np.eye(3),
        directions,
        lattice_cutoff=1,
        field_cutoff=2,
        grid_fractions=np.asarray([0.8, 1.0, 1.2]),
        absolute_radius_ratios=np.linspace(0.4, 2.5, 15),
        radial_candidates=2,
        r=0.23675858,
        center_iterations=40,
    )
    assert row["ideal_shortest_exit_mass"] == 1.0
    assert row["median_exact_boundary_label_alignment"] > 0.99
    assert row["optimistic_best_grid_shortest_success_mass"] > 0.95
    assert row["boundary_initialized_centering_shortest_success_mass"] > 0.95
    assert row["absolute_radial_catalog_shortest_success_mass"] > 0.95


def main() -> None:
    test_identity_boundary_exposes_axis_label()
    print("walsh Voronoi Hessian-ray tests passed")


if __name__ == "__main__":
    main()
