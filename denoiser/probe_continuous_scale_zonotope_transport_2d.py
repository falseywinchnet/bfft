"""Measure scale-lineage set coverage before and after positive push-forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .causal_scale_transport_2d import causal_scale_transport_observation_2d
from .continuous_scale_zonotope_transport_2d import (
    continuous_scale_zonotope_transport_state_2d,
)
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _coverage(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def run(size: int, selected: tuple[str, ...]) -> dict[str, Any]:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        cases = (
            ("clean", truth),
            (
                "mixed replacement + uniform 0.25",
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=9100,
                ),
            ),
        )
        for condition, observation in cases:
            _base, _residual, base_diagnostic = (
                causal_scale_transport_observation_2d(observation))
            shared_initial = np.asarray(
                base_diagnostic["readouts"]["phase_susceptibility"])
            refinement_rows = []
            for refinement in (0, 1):
                state = continuous_scale_zonotope_transport_state_2d(
                    observation,
                    initial_posterior=shared_initial,
                    trace_refinement=refinement,
                )
                posterior = np.asarray(state["posterior_after_erosion"])
                truth_transfer = truth - posterior
                pushed_base = np.asarray(state["pushed_posterior_base"])
                pushed_truth_transfer = truth - pushed_base
                contraction = state["contraction"]
                refinement_rows.append({
                    "trace_refinement": refinement,
                    "generator_count": int(state["generator"].shape[1]),
                    "mean_coefficient_width": contraction[
                        "mean_coefficient_width"],
                    "contracted_coefficient_fraction": contraction[
                        "contracted_coefficient_fraction"],
                    "truth_value_coverage_before_push": _coverage(
                        truth_transfer,
                        state["transfer_enclosure_lower"],
                        state["transfer_enclosure_upper"],
                    ),
                    "truth_value_coverage_after_push": _coverage(
                        pushed_truth_transfer,
                        state["pushed_transfer_enclosure_lower"],
                        state["pushed_transfer_enclosure_upper"],
                    ),
                    "lineage_recomposition_error": state[
                        "full_lineage_recomposition_error"],
                    "pushforward_linearity_error": state[
                        "pushforward_center_linearity_error"],
                    "evolved_flux_reconstruction_error": state[
                        "pushed_flux_patterns"][
                            "reconstruction_maximum_error"],
                    "raw_witness_exclusion_fraction": state[
                        "raw_witness_exclusion_fraction"],
                    "feasible_outer_component": contraction[
                        "feasible_outer_component"],
                })
            rows.append({
                "source": source,
                "condition": condition,
                "refinements": refinement_rows,
                "coverage_change_refinement_1_minus_0_before_push": float(
                    refinement_rows[1]["truth_value_coverage_before_push"]
                    - refinement_rows[0]["truth_value_coverage_before_push"]
                ),
                "coverage_change_refinement_1_minus_0_after_push": float(
                    refinement_rows[1]["truth_value_coverage_after_push"]
                    - refinement_rows[0]["truth_value_coverage_after_push"]
                ),
            })
    return {
        "purpose": (
            "audit exact continuous-scale ancestry generators, safe interval "
            "contraction, positive zonotope push-forward, evolved-graph flux "
            "re-expression, and nested trace behavior"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f"{row['source']} | {row['condition']}")
        for refinement in row["refinements"]:
            print(
                "refinement", refinement["trace_refinement"],
                "generators", refinement["generator_count"],
                "width", round(refinement["mean_coefficient_width"], 4),
                "coverage", (
                    round(refinement["truth_value_coverage_before_push"], 4),
                    round(refinement["truth_value_coverage_after_push"], 4),
                ),
                "flux error", refinement[
                    "evolved_flux_reconstruction_error"],
            )


if __name__ == "__main__":
    main()
