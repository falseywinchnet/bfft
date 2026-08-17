#!/usr/bin/env python3
"""Exact finite-cube audit of projected affine-coset Gaussian contraction.

For each random affine parity coset, contract every wide-source lattice point
toward the origin by sqrt(r/R), project to the nearest enumerated point in the
same coset, and push the source mass through this deterministic map.  Because
the pushforward can miss target support, mix it with epsilon of the original
source and optimize epsilon for Renyi-2 divergence from the target Gaussian.

This is the simplest support-moving eikonal proposal.  Unlike a radial weight,
it can create additional mass on cold, low-action points.  The experiment is
finite and diagnostic; a useful asymptotic theorem would still need to replace
the truncated nearest-neighbour lookup by an efficient BDD-backed operation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_hessian_noise_audit import (
    _binary_index,
    _gf2_inverse,
    generic_basis,
    random_gl2,
    shortest_vector_coefficients,
)
from experiments.walsh_radial_matched_filter import (
    gaussian_importance_exponent,
    optimal_gaussian_target_width,
)


T0 = 0.23147
DEFAULT_OUTPUT = ROOT / "experiments" / "out" / "walsh_coset_contraction_transport.json"


def renyi2_divergence(target: np.ndarray, proposal: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if np.any(target < 0.0) or np.any(proposal < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if not math.isclose(float(np.sum(target)), 1.0, abs_tol=1e-10):
        raise ValueError("target must sum to one")
    if not math.isclose(float(np.sum(proposal)), 1.0, abs_tol=1e-10):
        raise ValueError("proposal must sum to one")
    if np.any((target > 0.0) & (proposal == 0.0)):
        return math.inf
    return math.log(float(np.sum(target * target / proposal)))


def optimize_source_mixture(
    target: np.ndarray,
    source: np.ndarray,
    transported: np.ndarray,
    *,
    iterations: int = 96,
) -> tuple[float, float]:
    """Return epsilon and minimum log Renyi-2 mass for eps*source+(1-eps)*q."""
    def objective(epsilon: float) -> float:
        proposal = epsilon * source + (1.0 - epsilon) * transported
        return renyi2_divergence(target, proposal)

    # The unlogged Renyi mass is a convex function of epsilon.  Golden-section
    # search avoids the large diagnostic grid while retaining boundary checks.
    left, right = 0.0, 1.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    candidates = ((0.0, objective(0.0)), (1.0, objective(1.0)), (c, fc), (d, fd))
    epsilon, value = min(candidates, key=lambda item: item[1])
    return float(epsilon), float(value)


def _enumerated_coefficients(n: int, cutoff: int) -> np.ndarray:
    values = np.arange(-cutoff, cutoff + 1, dtype=np.int16)
    return np.asarray(list(itertools.product(values, repeat=n)), dtype=np.int16)


def _soft_contraction_step(
    points: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    contraction: float,
    tree,
    neighbours: int,
) -> tuple[np.ndarray, float, float]:
    query_k = min(max(neighbours, 1), len(points))
    distances, destination = tree.query(contraction * points, k=query_k)
    if query_k == 1:
        distances = distances[:, None]
        destination = destination[:, None]
    positive_distance = distances[distances > 1e-14]
    reference_bandwidth = (
        float(np.median(positive_distance))
        if len(positive_distance)
        else 1.0
    )
    best = (math.inf, 0.0, current)
    for factor in np.geomspace(0.025, 4.0, 21):
        bandwidth = factor * reference_bandwidth
        logits = -(distances * distances) / (2.0 * bandwidth * bandwidth)
        logits -= np.max(logits, axis=1, keepdims=True)
        local_weight = np.exp(logits)
        local_weight /= np.sum(local_weight, axis=1, keepdims=True)
        proposal = np.bincount(
            destination.ravel(),
            weights=(current[:, None] * local_weight).ravel(),
            minlength=len(points),
        ).astype(np.float64)
        proposal /= np.sum(proposal)
        value = renyi2_divergence(target, proposal)
        if value < best[0]:
            best = (value, float(factor), proposal)
    return best[2], best[1], best[0]


def audit_dimension(
    n: int,
    *,
    source_width: float,
    target_width: float,
    chi: float,
    cutoff: int,
    seed: int,
    neighbours: int,
    ladder_steps: tuple[int, ...],
) -> dict[str, object]:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed + 1009 * n)
    basis = generic_basis(n, rng)
    shortest, _ = shortest_vector_coefficients(basis)
    inverse_basis = np.linalg.inv(basis)
    coefficients = _enumerated_coefficients(n, cutoff)
    points = coefficients @ inverse_basis
    norm2 = np.einsum("ij,ij->i", points, points)
    h = min(max(int(math.floor(chi * n)), 1), n - 1)
    transform = random_gl2(n, rng)
    inverse_transform = _gf2_inverse(transform)
    parity = (coefficients & 1).astype(np.uint8)
    coordinates = (parity @ inverse_transform) & 1
    groups = _binary_index(coordinates[:, :h])
    cosets = 1 << h
    xi_R2 = 4.0 * n * source_width * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    xi_r2 = 4.0 * n * target_width * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    source_mass = np.exp(-math.pi * norm2 / xi_R2)
    target_mass = np.exp(-math.pi * norm2 / xi_r2)
    contraction = math.sqrt(target_width / source_width)
    reports = []
    for j in range(cosets):
        indices = np.flatnonzero(groups == j)
        local_points = points[indices]
        source = source_mass[indices]
        target = target_mass[indices]
        source /= np.sum(source)
        target /= np.sum(target)
        tree = cKDTree(local_points)
        query_k = min(max(neighbours, 1), len(indices))
        distances, destination = tree.query(
            contraction * local_points,
            k=query_k,
        )
        if query_k == 1:
            distances = distances[:, None]
            destination = destination[:, None]
        transported = np.bincount(
            destination[:, 0],
            weights=source,
            minlength=len(indices),
        ).astype(np.float64)
        transported /= np.sum(transported)
        hard_epsilon, hard_log_d2 = optimize_source_mixture(
            target,
            source,
            transported,
        )
        soft, bandwidth_factor, _ = _soft_contraction_step(
            local_points,
            source,
            target,
            contraction,
            tree,
            neighbours,
        )
        epsilon, mixed_log_d2 = optimize_source_mixture(
            target,
            source,
            soft,
        )
        best_transport = epsilon * source + (1.0 - epsilon) * soft
        baseline_log_d2 = renyi2_divergence(target, source)
        ladder_reports = []
        local_norm2 = norm2[indices]
        for steps in ladder_steps:
            widths = np.geomspace(source_width, target_width, steps + 1)
            proposal = source.copy()
            factors = []
            for previous, next_width in zip(widths[:-1], widths[1:]):
                intermediate = np.exp(
                    -math.pi * local_norm2
                    / (
                        4.0 * n * next_width * math.log(2.0)
                        / (math.pi * shortest * shortest)
                    )
                )
                intermediate /= np.sum(intermediate)
                proposal, factor, _ = _soft_contraction_step(
                    local_points,
                    proposal,
                    intermediate,
                    math.sqrt(next_width / previous),
                    tree,
                    neighbours,
                )
                factors.append(factor)
            ladder_epsilon, ladder_log_d2 = optimize_source_mixture(
                target,
                source,
                proposal,
            )
            final_proposal = (
                ladder_epsilon * source + (1.0 - ladder_epsilon) * proposal
            )
            ladder_reports.append({
                "steps": steps,
                "log2_renyi2_mass_per_dimension": ladder_log_d2
                / (n * math.log(2.0)),
                "action_saved_per_dimension": (
                    baseline_log_d2 - ladder_log_d2
                ) / (n * math.log(2.0)),
                "source_mixture": ladder_epsilon,
                "mean_bandwidth_factor": float(np.mean(factors)),
                "image_fraction": float(
                    np.count_nonzero(final_proposal) / len(indices)
                ),
            })
        boundary = np.any(np.abs(coefficients[indices]) == cutoff, axis=1)
        reports.append({
            "j": j,
            "points": int(len(indices)),
            "target_boundary_mass": float(np.sum(target[boundary])),
            "source_boundary_mass": float(np.sum(source[boundary])),
            "hard_transport_image_fraction": float(
                np.count_nonzero(transported) / len(indices)
            ),
            "soft_transport_image_fraction": float(
                np.count_nonzero(best_transport) / len(indices)
            ),
            "optimal_bandwidth_factor": bandwidth_factor,
            "optimal_source_mixture": epsilon,
            "hard_transport_log2_renyi2_mass_per_dimension": hard_log_d2 / (
                n * math.log(2.0)
            ),
            "baseline_log2_renyi2_mass_per_dimension": baseline_log_d2 / (
                n * math.log(2.0)
            ),
            "transport_log2_renyi2_mass_per_dimension": mixed_log_d2 / (
                n * math.log(2.0)
            ),
            "finite_n_action_saved_per_dimension": (
                baseline_log_d2 - mixed_log_d2
            ) / (n * math.log(2.0)),
            "ladder_reports": ladder_reports,
        })
    keys = (
        "target_boundary_mass",
        "source_boundary_mass",
        "hard_transport_image_fraction",
        "soft_transport_image_fraction",
        "optimal_bandwidth_factor",
        "optimal_source_mixture",
        "hard_transport_log2_renyi2_mass_per_dimension",
        "baseline_log2_renyi2_mass_per_dimension",
        "transport_log2_renyi2_mass_per_dimension",
        "finite_n_action_saved_per_dimension",
    )
    summary = {
        key: {
            "minimum": float(min(row[key] for row in reports)),
            "mean": float(np.mean([row[key] for row in reports])),
            "maximum": float(max(row[key] for row in reports)),
        }
        for key in keys
    }
    return {
        "dimension": n,
        "cutoff": cutoff,
        "enumerated_points": int(len(points)),
        "h": h,
        "cosets": cosets,
        "source_width": source_width,
        "target_width": target_width,
        "contraction": contraction,
        "summary": summary,
        "coset_reports": reports,
        "ladder_summary": {
            str(steps): {
                key: {
                    "minimum": float(min(
                        next(item for item in row["ladder_reports"] if item["steps"] == steps)[key]
                        for row in reports
                    )),
                    "mean": float(np.mean([
                        next(item for item in row["ladder_reports"] if item["steps"] == steps)[key]
                        for row in reports
                    ])),
                    "maximum": float(max(
                        next(item for item in row["ladder_reports"] if item["steps"] == steps)[key]
                        for row in reports
                    )),
                }
                for key in (
                    "log2_renyi2_mass_per_dimension",
                    "action_saved_per_dimension",
                    "source_mixture",
                    "mean_bandwidth_factor",
                    "image_fraction",
                )
            }
            for steps in ladder_steps
        },
    }


def audit(
    dimensions: tuple[int, ...],
    *,
    source_width: float,
    target_width: float | None,
    chi: float,
    cutoff: int,
    seed: int,
    neighbours: int,
    ladder_steps: tuple[int, ...],
) -> dict[str, object]:
    r = (
        optimal_gaussian_target_width(source_width)
        if target_width is None
        else target_width
    )
    radial_total = gaussian_importance_exponent(r, source_width)
    radial_action = radial_total - 2.0 * r
    allowed_action = 0.5 - 2.0 * r
    return {
        "experiment": "projected_affine_coset_gaussian_contraction",
        "source_width": source_width,
        "target_width": r,
        "asymptotic_ledger": {
            "radial_total_sample_exponent": radial_total,
            "radial_renyi2_action_exponent": radial_action,
            "maximum_action_for_half_exponent": allowed_action,
            "required_action_saving": radial_action - allowed_action,
        },
        "dimensions": [
            audit_dimension(
                n,
                source_width=source_width,
                target_width=r,
                chi=chi,
                cutoff=cutoff,
                seed=seed,
                neighbours=neighbours,
                ladder_steps=ladder_steps,
            )
            for n in dimensions
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=(4, 5, 6, 7))
    parser.add_argument("--source-width", type=float, default=2.0 * T0)
    parser.add_argument("--target-width", type=float)
    parser.add_argument("--chi", type=float, default=0.5)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--neighbours", type=int, default=16)
    parser.add_argument("--ladder-steps", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(
        tuple(args.dimensions),
        source_width=args.source_width,
        target_width=args.target_width,
        chi=args.chi,
        cutoff=args.cutoff,
        seed=args.seed,
        neighbours=args.neighbours,
        ladder_steps=tuple(args.ladder_steps),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
