#!/usr/bin/env python3
"""Can the periodic Hessian expose an ideal Voronoi first-exit label?"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import generic_basis
from experiments.walsh_periodic_hessian_descent import (
    DEFAULT_R,
    ascend_periodic_hessian,
    coefficient_box,
    nearest_edge,
    periodic_hessian_spectral_data,
)
from experiments.walsh_periodic_hessian_adversarial_search import adversarial_basis
from experiments.walsh_periodic_hessian_simplex_census import (
    simplex_cancellation_basis,
)
from experiments.walsh_periodic_hessian_stress_census import needle_d_basis
from experiments.walsh_voronoi_first_exit import coefficient_cube


DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_voronoi_hessian_ray.json"


def ray_audit(
    family: str,
    basis: np.ndarray,
    directions: np.ndarray,
    *,
    lattice_cutoff: int,
    field_cutoff: int,
    grid_fractions: np.ndarray,
    absolute_radius_ratios: np.ndarray,
    radial_candidates: int,
    r: float,
    center_iterations: int,
) -> dict[str, object]:
    n = basis.shape[0]
    coefficients = coefficient_cube(n, lattice_cutoff)
    vectors = coefficients @ basis.T
    norm2 = np.einsum("ij,ij->i", vectors, vectors)
    shortest2 = float(np.min(norm2))
    shortest = norm2 <= shortest2 * (1.0 + 1e-10)
    xi = math.sqrt(4.0 * n * r * math.log(2.0) / (math.pi * shortest2))
    spatial_width = 1.0 / xi
    dual_points = coefficient_box(n, field_cutoff) @ np.linalg.inv(basis)
    edge_coefficients = coefficient_box(n, lattice_cutoff + 1)
    edge_points = edge_coefficients @ basis.T
    unit = directions[:, :n].copy()
    unit /= np.linalg.norm(unit, axis=1)[:, None]

    exact_alignments = []
    selected_alignments = []
    best_grid_alignments = []
    shortest_exit = []
    selected_shortest_success = []
    best_grid_shortest_success = []
    exact_boundary_shortest_success = []
    shortest_exact_alignments = []
    selected_fractions = []
    centered_shortest_success = []
    centered_winner_parity_success = []
    centering_evaluations = []
    absolute_catalog_shortest_success = []
    absolute_score_catalog_shortest_success = []
    absolute_union_catalog_shortest_success = []
    absolute_catalog_evaluations = []
    for u in unit:
        dot = vectors @ u
        times = np.divide(
            norm2,
            2.0 * dot,
            out=np.full_like(dot, np.inf),
            where=dot > 0.0,
        )
        winner = int(np.argmin(times))
        boundary = float(times[winner])
        label = vectors[winner]
        label_unit = label / np.linalg.norm(label)
        is_shortest = bool(shortest[winner])
        shortest_exit.append(is_shortest)

        scores = []
        gaps = []
        alignments = []
        for fraction in grid_fractions:
            physical = (fraction * boundary) * u
            coefficient_point = np.linalg.solve(basis, physical)
            score, gap, _gradient, direction = periodic_hessian_spectral_data(
                coefficient_point,
                basis,
                dual_points,
                spatial_width,
            )
            scores.append(score)
            gaps.append(gap)
            alignments.append(abs(float(np.dot(direction, label_unit))))
        exact_index = int(np.argmin(np.abs(grid_fractions - 1.0)))
        selected = int(np.argmax(gaps))
        best = int(np.argmax(alignments))
        exact_alignments.append(alignments[exact_index])
        exact_boundary_shortest_success.append(
            is_shortest and alignments[exact_index] >= 0.95
        )
        if is_shortest:
            shortest_exact_alignments.append(alignments[exact_index])
        selected_alignments.append(alignments[selected])
        best_grid_alignments.append(alignments[best])
        selected_fractions.append(float(grid_fractions[selected]))
        selected_shortest_success.append(
            is_shortest and alignments[selected] >= 0.95
        )
        best_grid_shortest_success.append(
            is_shortest and alignments[best] >= 0.95
        )

        boundary_coefficients = np.linalg.solve(basis, boundary * u)
        centered = ascend_periodic_hessian(
            boundary_coefficients,
            basis,
            dual_points,
            spatial_width,
            max_iterations=center_iterations,
        )
        edge_coeff, _edge, edge_norm, _midpoint_error = nearest_edge(
            centered["coefficient_point"],
            basis,
            edge_coefficients,
            edge_points,
        )
        centered_shortest_success.append(
            edge_norm <= math.sqrt(shortest2) * (1.0 + 1e-7)
        )
        centered_winner_parity_success.append(bool(np.all(
            (edge_coeff.astype(np.int64) & 1)
            == (coefficients[winner].astype(np.int64) & 1)
        )))
        centering_evaluations.append(int(centered["evaluations"]))

        absolute_gaps = []
        absolute_scores = []
        absolute_starts = []
        for ratio in absolute_radius_ratios:
            physical = (ratio * math.sqrt(shortest2)) * u
            coefficient_point = np.linalg.solve(basis, physical)
            score, gap, _gradient, _direction = periodic_hessian_spectral_data(
                coefficient_point,
                basis,
                dual_points,
                spatial_width,
            )
            absolute_scores.append(score)
            absolute_gaps.append(gap)
            absolute_starts.append(coefficient_point)

        def separated_top(values: list[float]) -> list[int]:
            selected: list[int] = []
            for index in np.argsort(values)[::-1]:
                if all(abs(int(index) - old) > 2 for old in selected):
                    selected.append(int(index))
                if len(selected) == radial_candidates:
                    break
            return selected

        gap_chosen = separated_top(absolute_gaps)
        score_chosen = separated_top(absolute_scores)
        chosen = list(dict.fromkeys(gap_chosen + score_chosen))
        gap_catalog_hit = False
        score_catalog_hit = False
        catalog_evaluations = len(absolute_radius_ratios)
        for index in chosen:
            catalog_centered = ascend_periodic_hessian(
                absolute_starts[index],
                basis,
                dual_points,
                spatial_width,
                max_iterations=center_iterations,
            )
            _coeff, _edge, edge_norm, _error = nearest_edge(
                catalog_centered["coefficient_point"],
                basis,
                edge_coefficients,
                edge_points,
            )
            catalog_evaluations += int(catalog_centered["evaluations"])
            hit = edge_norm <= math.sqrt(shortest2) * (1.0 + 1e-7)
            if index in gap_chosen:
                gap_catalog_hit |= hit
            if index in score_chosen:
                score_catalog_hit |= hit
        absolute_catalog_shortest_success.append(gap_catalog_hit)
        absolute_score_catalog_shortest_success.append(score_catalog_hit)
        absolute_union_catalog_shortest_success.append(
            gap_catalog_hit or score_catalog_hit
        )
        absolute_catalog_evaluations.append(catalog_evaluations)

    return {
        "family": family,
        "dimension": n,
        "directions": len(unit),
        "r": r,
        "lattice_cutoff": lattice_cutoff,
        "field_cutoff": field_cutoff,
        "grid_fractions": grid_fractions.tolist(),
        "ideal_shortest_exit_mass": float(np.mean(shortest_exit)),
        "median_exact_boundary_label_alignment": float(np.median(exact_alignments)),
        "minimum_exact_boundary_label_alignment": float(np.min(exact_alignments)),
        "median_shortest_exit_exact_boundary_alignment": (
            float(np.median(shortest_exact_alignments))
            if shortest_exact_alignments else 0.0
        ),
        "exact_boundary_shortest_success_mass": float(
            np.mean(exact_boundary_shortest_success)
        ),
        "median_max_eigengap_label_alignment": float(np.median(selected_alignments)),
        "median_best_grid_label_alignment": float(np.median(best_grid_alignments)),
        "max_eigengap_shortest_success_mass": float(
            np.mean(selected_shortest_success)
        ),
        "optimistic_best_grid_shortest_success_mass": float(
            np.mean(best_grid_shortest_success)
        ),
        "median_max_eigengap_grid_fraction": float(np.median(selected_fractions)),
        "boundary_initialized_centering_shortest_success_mass": float(
            np.mean(centered_shortest_success)
        ),
        "boundary_initialized_centering_winner_parity_mass": float(
            np.mean(centered_winner_parity_success)
        ),
        "median_centering_field_evaluations": float(
            np.median(centering_evaluations)
        ),
        "absolute_radius_ratios": absolute_radius_ratios.tolist(),
        "absolute_radial_candidates": radial_candidates,
        "absolute_radial_catalog_shortest_success_mass": float(
            np.mean(absolute_catalog_shortest_success)
        ),
        "absolute_score_catalog_shortest_success_mass": float(
            np.mean(absolute_score_catalog_shortest_success)
        ),
        "absolute_union_catalog_shortest_success_mass": float(
            np.mean(absolute_union_catalog_shortest_success)
        ),
        "median_absolute_catalog_field_evaluations": float(
            np.median(absolute_catalog_evaluations)
        ),
    }


def build_report(
    dimensions: tuple[int, ...],
    samples: int,
    lattice_cutoff: int,
    field_cutoff: int,
    grid_points: int,
    absolute_grid_points: int,
    maximum_radius_ratio: float,
    radial_candidates: int,
    random_replicates: int,
    adversarial_replicates: int,
    r: float,
    seed: int,
    center_iterations: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(samples, max(dimensions)))
    grid = np.linspace(0.6, 1.3, grid_points)
    grid = np.unique(np.concatenate((grid, np.asarray([1.0]))))
    absolute_grid = np.linspace(0.4, maximum_radius_ratio, absolute_grid_points)
    rows = []
    for n in dimensions:
        fixtures = [
            ("simplex_cancellation", simplex_cancellation_basis(n, 0.97)),
        ]
        for replicate in range(random_replicates):
            label = "generic" if random_replicates == 1 else f"generic_{replicate}"
            fixtures.append((label, generic_basis(n, rng)))
        for replicate in range(adversarial_replicates):
            fixtures.append((
                f"adversarial_{replicate}",
                adversarial_basis(n, rng, 2.0 + 2.0 * rng.random()),
            ))
        if n >= 3:
            fixtures.append(("needle_D_shell", needle_d_basis(n, 1.03)))
        for family, basis in fixtures:
            rows.append(ray_audit(
                family,
                basis,
                directions,
                lattice_cutoff=lattice_cutoff,
                field_cutoff=field_cutoff,
                grid_fractions=grid,
                absolute_radius_ratios=absolute_grid,
                radial_candidates=radial_candidates,
                r=r,
                center_iterations=center_iterations,
            ))
    return {
        "experiment": "walsh_voronoi_first_exit_hessian_ray",
        "warning": (
            "Ideal-boundary diagnostics use finite lattice enumeration and "
            "a boundary-centered grid.  The separately reported absolute "
            "radial catalog uses only the length scale and no boundary time."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--lattice-cutoff", type=int, default=2)
    parser.add_argument("--field-cutoff", type=int, default=2)
    parser.add_argument("--grid-points", type=int, default=15)
    parser.add_argument("--absolute-grid-points", type=int, default=41)
    parser.add_argument("--maximum-radius-ratio", type=float, default=3.0)
    parser.add_argument("--radial-candidates", type=int, default=3)
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--adversarial-replicates", type=int, default=0)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--center-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=260802481)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        tuple(args.dimensions),
        args.samples,
        args.lattice_cutoff,
        args.field_cutoff,
        args.grid_points,
        args.absolute_grid_points,
        args.maximum_radius_ratio,
        args.radial_candidates,
        args.random_replicates,
        args.adversarial_replicates,
        args.r,
        args.seed,
        args.center_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
