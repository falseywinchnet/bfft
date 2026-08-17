#!/usr/bin/env python3
"""Scale continuation and branch nucleation for periodic-Hessian recovery.

This experiment attacks the branch-budget target in WALSH_REVERSE_PATHS.md.
For one lattice it:

* discovers local maxima of the periodic traceless-Hessian field at a smooth
  scale t_start;
* transports every distinct branch through a scale ladder to t_target;
* probes for newly born branches after every transport step;
* merges duplicate torus locations with aligned eigendirections; and
* compares against cold target-scale restarts with the same number of field
  and gradient evaluations.

The field is evaluated by a truncated dual Fourier series, hence every
finite audit remains exactly periodic.  This is an experimental topology
census, not an asymptotic branch-count theorem.
"""

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

from experiments.walsh_hessian_noise_audit import (
    generic_basis,
    shortest_vector_coefficients,
)
from experiments.walsh_periodic_hessian_descent import (
    DEFAULT_R,
    ascend_periodic_hessian,
    coefficient_box,
    nearest_edge,
    periodic_hessian_spectral_data,
    wrap_coefficients,
)


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "out" / "walsh_periodic_hessian_continuation.json"
)


def iota(target_t: float, source_t: float) -> float:
    """Renyi-2 importance exponent from source_t samples to target_t."""
    if not (0.0 < target_t <= source_t):
        raise ValueError("expected 0 < target_t <= source_t")
    return 0.5 * math.log2(
        source_t * source_t / (target_t * (2.0 * source_t - target_t))
    )


def spatial_width(n: int, t: float, shortest: float) -> float:
    xi_squared = 4.0 * n * t * math.log(2.0) / (
        math.pi * shortest * shortest
    )
    return 1.0 / math.sqrt(xi_squared)


def canonical_edge(coefficients: np.ndarray) -> tuple[int, ...]:
    value = np.asarray(coefficients, dtype=np.int64)
    nonzero = np.flatnonzero(value)
    if not nonzero.size:
        return tuple(int(x) for x in value)
    if value[int(nonzero[0])] < 0:
        value = -value
    return tuple(int(x) for x in value)


def branches_match(
    left: dict[str, object],
    right: dict[str, object],
    basis: np.ndarray,
    shortest: float,
    *,
    location_tolerance: float,
    direction_alignment: float,
) -> bool:
    difference = wrap_coefficients(
        np.asarray(left["coefficient_point"])
        - np.asarray(right["coefficient_point"])
    )
    location_distance = float(np.linalg.norm(basis @ difference))
    alignment = abs(float(np.dot(
        np.asarray(left["leading_direction"]),
        np.asarray(right["leading_direction"]),
    )))
    return (
        location_distance <= location_tolerance * shortest
        and alignment >= direction_alignment
    )


def merge_branches(
    candidates: list[dict[str, object]],
    basis: np.ndarray,
    shortest: float,
    *,
    location_tolerance: float = 2e-4,
    direction_alignment: float = 0.995,
) -> list[dict[str, object]]:
    """Merge converged copies while preserving all branch ancestry."""
    ordered = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
    merged: list[dict[str, object]] = []
    for candidate in ordered:
        for incumbent in merged:
            if branches_match(
                incumbent,
                candidate,
                basis,
                shortest,
                location_tolerance=location_tolerance,
                direction_alignment=direction_alignment,
            ):
                incumbent["ancestors"] = sorted(set(incumbent["ancestors"]) | set(
                    candidate["ancestors"]
                ))
                incumbent["birth_levels"] = sorted(set(incumbent["birth_levels"]) | set(
                    candidate["birth_levels"]
                ))
                incumbent["merged_copies"] = int(incumbent["merged_copies"]) + int(
                    candidate["merged_copies"]
                )
                for key in (
                    "best_path_bottleneck_score_ratio",
                    "best_path_bottleneck_eigengap_ratio",
                ):
                    if key in candidate:
                        incumbent[key] = max(
                            float(incumbent.get(key, -math.inf)),
                            float(candidate[key]),
                        )
                break
        else:
            merged.append(candidate)
    return merged


def optimize_seed(
    start: np.ndarray,
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    width: float,
    ancestor: int,
    birth_level: int,
    reference_score: float,
    reference_eigengap: float,
    inherited_score_bottleneck: float = math.inf,
    inherited_eigengap_bottleneck: float = math.inf,
) -> dict[str, object]:
    result = ascend_periodic_hessian(
        start,
        basis,
        dual_points,
        width,
        max_iterations=120,
    )
    segment_score_ratio = float(result["minimum_evaluated_score"]) / max(
        reference_score, 1e-300
    )
    segment_eigengap_ratio = float(
        result["minimum_evaluated_eigengap"]
    ) / max(reference_eigengap, 1e-300)
    result.update({
        "ancestors": [ancestor],
        "birth_levels": [birth_level],
        "merged_copies": 1,
        "segment_bottleneck_score_ratio": segment_score_ratio,
        "segment_bottleneck_eigengap_ratio": segment_eigengap_ratio,
        "best_path_bottleneck_score_ratio": min(
            inherited_score_bottleneck, segment_score_ratio
        ),
        "best_path_bottleneck_eigengap_ratio": min(
            inherited_eigengap_bottleneck, segment_eigengap_ratio
        ),
    })
    return result


def classify_branch(
    branch: dict[str, object],
    *,
    basis: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
) -> dict[str, object]:
    edge_coefficients, edge, edge_norm, midpoint_error = nearest_edge(
        np.asarray(branch["coefficient_point"]),
        basis,
        coefficients,
        points,
    )
    return {
        "edge": canonical_edge(edge_coefficients),
        "edge_norm_over_shortest": edge_norm / shortest,
        "shortest_edge": edge_norm <= shortest * (1.0 + 1e-7),
        "midpoint_distance_mismatch": midpoint_error,
        "leading_direction_edge_alignment": abs(float(
            np.dot(np.asarray(branch["leading_direction"]), edge)
            / max(edge_norm, 1e-300)
        )),
    }


def level_summary(
    level: int,
    t: float,
    branches: list[dict[str, object]],
    *,
    basis: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
    continued_evaluations: int,
    birth_evaluations: int,
    new_unique_branches: int,
) -> dict[str, object]:
    classes = [
        classify_branch(
            branch,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
        )
        for branch in branches
    ]
    edges = {row["edge"] for row in classes}
    shortest_indices = [
        index for index, row in enumerate(classes) if row["shortest_edge"]
    ]
    shortest_birth = sorted({
        birth
        for index in shortest_indices
        for birth in branches[index]["birth_levels"]
    })
    score_bottlenecks = [
        float(branch["best_path_bottleneck_score_ratio"])
        for branch in branches
    ]
    eigengap_bottlenecks = [
        float(branch["best_path_bottleneck_eigengap_ratio"])
        for branch in branches
    ]
    return {
        "level": level,
        "t": t,
        "branch_count": len(branches),
        "distinct_nearest_edges": len(edges),
        "shortest_branch_count": len(shortest_indices),
        "shortest_branch_present": bool(shortest_indices),
        "shortest_branch_birth_levels": shortest_birth,
        "new_unique_branches_from_probes": new_unique_branches,
        "continued_evaluations": continued_evaluations,
        "birth_probe_evaluations": birth_evaluations,
        "maximum_score": max((float(row["score"]) for row in branches), default=0.0),
        "maximum_shortest_score": max(
            (float(branches[index]["score"]) for index in shortest_indices),
            default=None,
        ),
        "median_best_path_bottleneck_score_ratio": (
            float(np.median(score_bottlenecks)) if score_bottlenecks else None
        ),
        "minimum_best_path_bottleneck_score_ratio": (
            min(score_bottlenecks) if score_bottlenecks else None
        ),
        "maximum_shortest_best_path_bottleneck_score_ratio": max(
            (score_bottlenecks[index] for index in shortest_indices),
            default=None,
        ),
        "median_best_path_bottleneck_eigengap_ratio": (
            float(np.median(eigengap_bottlenecks))
            if eigengap_bottlenecks else None
        ),
        "minimum_best_path_bottleneck_eigengap_ratio": (
            min(eigengap_bottlenecks) if eigengap_bottlenecks else None
        ),
        "maximum_shortest_best_path_bottleneck_eigengap_ratio": max(
            (eigengap_bottlenecks[index] for index in shortest_indices),
            default=None,
        ),
    }


def run_continuation(
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
    shortest_coefficients: np.ndarray,
    ladder: np.ndarray,
    initial_starts: np.ndarray,
    birth_starts: list[np.ndarray],
    allow_births: bool,
) -> dict[str, object]:
    next_ancestor = 0
    initial_width = spatial_width(
        basis.shape[0], float(ladder[0]), shortest
    )
    initial_reference_score, initial_reference_eigengap, _, _ = (
        periodic_hessian_spectral_data(
            0.5 * shortest_coefficients,
            basis,
            dual_points,
            initial_width,
        )
    )
    branches = [
        optimize_seed(
            start,
            basis=basis,
            dual_points=dual_points,
            width=initial_width,
            ancestor=index,
            birth_level=0,
            reference_score=initial_reference_score,
            reference_eigengap=initial_reference_eigengap,
        )
        for index, start in enumerate(initial_starts)
    ]
    next_ancestor += len(initial_starts)
    initial_evaluations = sum(int(row["evaluations"]) for row in branches)
    branches = merge_branches(branches, basis, shortest)
    summaries = [
        level_summary(
            0,
            float(ladder[0]),
            branches,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
            continued_evaluations=0,
            birth_evaluations=initial_evaluations,
            new_unique_branches=len(branches),
        )
    ]
    total_evaluations = initial_evaluations

    for level, t in enumerate(ladder[1:], start=1):
        width = spatial_width(basis.shape[0], float(t), shortest)
        reference_score, reference_eigengap, _, _ = (
            periodic_hessian_spectral_data(
                0.5 * shortest_coefficients,
                basis,
                dual_points,
                width,
            )
        )
        transported = []
        for branch in branches:
            result = optimize_seed(
                np.asarray(branch["coefficient_point"]),
                basis=basis,
                dual_points=dual_points,
                width=width,
                ancestor=int(branch["ancestors"][0]),
                birth_level=int(branch["birth_levels"][0]),
                reference_score=reference_score,
                reference_eigengap=reference_eigengap,
                inherited_score_bottleneck=float(
                    branch["best_path_bottleneck_score_ratio"]
                ),
                inherited_eigengap_bottleneck=float(
                    branch["best_path_bottleneck_eigengap_ratio"]
                ),
            )
            result.update({
                "ancestors": list(branch["ancestors"]),
                "birth_levels": list(branch["birth_levels"]),
                "merged_copies": int(branch["merged_copies"]),
            })
            transported.append(result)
        continued_evaluations = sum(int(row["evaluations"]) for row in transported)
        branches = merge_branches(transported, basis, shortest)
        before_births = len(branches)

        born: list[dict[str, object]] = []
        if allow_births:
            for start in birth_starts[level - 1]:
                born.append(optimize_seed(
                    start,
                    basis=basis,
                    dual_points=dual_points,
                    width=width,
                    ancestor=next_ancestor,
                    birth_level=level,
                    reference_score=reference_score,
                    reference_eigengap=reference_eigengap,
                ))
                next_ancestor += 1
        birth_evaluations = sum(int(row["evaluations"]) for row in born)
        branches = merge_branches(branches + born, basis, shortest)
        new_unique = max(len(branches) - before_births, 0)
        total_evaluations += continued_evaluations + birth_evaluations
        summaries.append(level_summary(
            level,
            float(t),
            branches,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
            continued_evaluations=continued_evaluations,
            birth_evaluations=birth_evaluations,
            new_unique_branches=new_unique,
        ))

    terminal_classes = [
        classify_branch(
            branch,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
        )
        for branch in branches
    ]
    return {
        "allow_births": allow_births,
        "total_field_gradient_evaluations": total_evaluations,
        "terminal_branch_count": len(branches),
        "terminal_distinct_edges": len({row["edge"] for row in terminal_classes}),
        "terminal_shortest_branch_count": sum(
            int(row["shortest_edge"]) for row in terminal_classes
        ),
        "terminal_shortest_present": any(
            row["shortest_edge"] for row in terminal_classes
        ),
        "terminal_shortest_inherited_from_initial_scale": any(
            row["shortest_edge"] and 0 in branch["birth_levels"]
            for row, branch in zip(terminal_classes, branches)
        ),
        "terminal_shortest_initial_ancestors": sorted({
            ancestor
            for row, branch in zip(terminal_classes, branches)
            if row["shortest_edge"]
            for ancestor in branch["ancestors"]
            if ancestor < len(initial_starts)
        }),
        "levels": summaries,
    }


def cold_equal_evaluation_baseline(
    *,
    evaluation_budget: int,
    starts: np.ndarray,
    basis: np.ndarray,
    dual_points: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
    shortest_coefficients: np.ndarray,
    target_t: float,
) -> dict[str, object]:
    width = spatial_width(basis.shape[0], target_t, shortest)
    reference_score, reference_eigengap, _, _ = periodic_hessian_spectral_data(
        0.5 * shortest_coefficients, basis, dual_points, width
    )
    candidates: list[dict[str, object]] = []
    used = 0
    attempted = 0
    shortest_hits = 0
    shortest_hit_indices: list[int] = []
    while used < evaluation_budget and attempted < len(starts):
        result = optimize_seed(
            starts[attempted],
            basis=basis,
            dual_points=dual_points,
            width=width,
            ancestor=attempted,
            birth_level=0,
            reference_score=reference_score,
            reference_eigengap=reference_eigengap,
        )
        attempted += 1
        used += int(result["evaluations"])
        classification = classify_branch(
            result,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
        )
        shortest_hits += int(classification["shortest_edge"])
        if classification["shortest_edge"]:
            shortest_hit_indices.append(attempted - 1)
        candidates.append(result)
    branches = merge_branches(candidates, basis, shortest)
    classes = [
        classify_branch(
            branch,
            basis=basis,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
        )
        for branch in branches
    ]
    return {
        "field_gradient_evaluation_budget": evaluation_budget,
        "field_gradient_evaluations_used": used,
        "cold_starts_attempted": attempted,
        "cold_shortest_hits": shortest_hits,
        "cold_shortest_hit_indices": shortest_hit_indices,
        "cold_shortest_present": shortest_hits > 0,
        "cold_unique_branch_count": len(branches),
        "cold_distinct_edges": len({row["edge"] for row in classes}),
    }


def audit_schedule(
    *,
    basis: np.ndarray,
    dual_points: np.ndarray,
    coefficients: np.ndarray,
    points: np.ndarray,
    shortest: float,
    shortest_coefficients: np.ndarray,
    start_t: float,
    target_t: float,
    levels: int,
    initial_probe_count: int,
    birth_probe_count: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    n = basis.shape[0]
    ladder = np.linspace(start_t, target_t, levels)
    initial_starts = rng.uniform(-0.5, 0.5, size=(initial_probe_count, n))
    birth_starts = [
        rng.uniform(-0.5, 0.5, size=(birth_probe_count, n))
        for _ in range(levels - 1)
    ]
    transported_only = run_continuation(
        basis=basis,
        dual_points=dual_points,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        ladder=ladder,
        initial_starts=initial_starts,
        birth_starts=birth_starts,
        allow_births=False,
    )
    nucleating = run_continuation(
        basis=basis,
        dual_points=dual_points,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        ladder=ladder,
        initial_starts=initial_starts,
        birth_starts=birth_starts,
        allow_births=True,
    )
    cold_starts = rng.uniform(-0.5, 0.5, size=(4096, n))
    cold = cold_equal_evaluation_baseline(
        evaluation_budget=int(nucleating["total_field_gradient_evaluations"]),
        starts=cold_starts,
        basis=basis,
        dual_points=dual_points,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        target_t=target_t,
    )
    paired_initial_cold = cold_equal_evaluation_baseline(
        evaluation_budget=10**12,
        starts=initial_starts,
        basis=basis,
        dual_points=dual_points,
        coefficients=coefficients,
        points=points,
        shortest=shortest,
        shortest_coefficients=shortest_coefficients,
        target_t=target_t,
    )
    initial_ancestors = set(nucleating["terminal_shortest_initial_ancestors"])
    paired_cold_hits = set(paired_initial_cold["cold_shortest_hit_indices"])
    importance = iota(start_t, target_t)
    return {
        "start_t": start_t,
        "target_t": target_t,
        "levels": levels,
        "ladder": ladder.tolist(),
        "source_to_start_importance_exponent": importance,
        "start_field_sample_exponent": 2.0 * start_t + importance,
        "within_half_exponent": 2.0 * start_t + importance <= 0.5,
        "initial_probe_count": initial_probe_count,
        "birth_probe_count_per_level": birth_probe_count,
        "transported_only": transported_only,
        "nucleating": nucleating,
        "paired_initial_cold": paired_initial_cold,
        "initial_seeds_rescued_by_continuation": sorted(
            initial_ancestors - paired_cold_hits
        ),
        "initial_seeds_lost_by_continuation": sorted(
            paired_cold_hits - initial_ancestors
        ),
        "cold_equal_evaluation_baseline": cold,
    }


def audit_dimension(
    n: int,
    *,
    cutoff: int,
    start_scales: list[float],
    target_t: float,
    levels: int,
    initial_probe_count: int,
    birth_probe_count: int,
    seed: int,
    replicate: int,
) -> dict[str, object]:
    replicate_seed = seed + 1_000_003 * replicate
    basis_rng = np.random.default_rng(replicate_seed + 1009 * n)
    basis = generic_basis(n, basis_rng)
    shortest, shortest_coefficients = shortest_vector_coefficients(
        basis, cutoff=max(cutoff, 2)
    )
    field_coefficients = coefficient_box(n, cutoff)
    dual_points = field_coefficients @ np.linalg.inv(basis)
    coefficients = coefficient_box(n, cutoff + 1)
    points = coefficients @ basis.T
    schedules = []
    for index, start_t in enumerate(start_scales):
        schedules.append(audit_schedule(
            basis=basis,
            dual_points=dual_points,
            coefficients=coefficients,
            points=points,
            shortest=shortest,
            shortest_coefficients=shortest_coefficients,
            start_t=start_t,
            target_t=target_t,
            levels=levels,
            initial_probe_count=initial_probe_count,
            birth_probe_count=birth_probe_count,
            rng=np.random.default_rng(replicate_seed + 104729 * n + index),
        ))
    return {
        "dimension": n,
        "replicate": replicate,
        "basis_condition_number": float(np.linalg.cond(basis)),
        "shortest_length": shortest,
        "shortest_coefficients": shortest_coefficients.tolist(),
        "target_query_exponent": 0.5 - 2.0 * target_t,
        "schedules": schedules,
    }


def audit(
    *,
    min_dimension: int,
    max_dimension: int,
    cutoff: int,
    start_scales: list[float],
    target_t: float,
    levels: int,
    initial_probe_count: int,
    birth_probe_count: int,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    return {
        "experiment": "walsh_periodic_hessian_continuation",
        "interpretation": (
            "Continuation succeeds only if it retains a shortest branch with "
            "materially fewer field evaluations or branches than equal-cost "
            "cold restarts. Birth probes test whether branches appear after the "
            "smooth initial scale."
        ),
        "rows": [
            audit_dimension(
                n,
                cutoff=cutoff,
                start_scales=start_scales,
                target_t=target_t,
                levels=levels,
                initial_probe_count=initial_probe_count,
                birth_probe_count=birth_probe_count,
                seed=seed,
                replicate=replicate,
            )
            for n in range(min_dimension, max_dimension + 1)
            for replicate in range(replicates)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dimension", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=5)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument(
        "--start-scales",
        type=float,
        nargs="+",
        default=[0.12, 0.15, 0.18, 0.22, 0.23147],
    )
    parser.add_argument("--target-t", type=float, default=DEFAULT_R)
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--initial-probes", type=int, default=24)
    parser.add_argument("--birth-probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260802478)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(not 0.0 < value <= args.target_t for value in args.start_scales):
        raise ValueError("every start scale must lie in (0,target_t]")
    report = audit(
        min_dimension=args.min_dimension,
        max_dimension=args.max_dimension,
        cutoff=args.cutoff,
        start_scales=list(args.start_scales),
        target_t=args.target_t,
        levels=args.levels,
        initial_probe_count=args.initial_probes,
        birth_probe_count=args.birth_probes,
        seed=args.seed,
        replicates=args.replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
