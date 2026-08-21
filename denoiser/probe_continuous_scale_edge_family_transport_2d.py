"""Audit local scale-edge set coverage and refinement under frozen push."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .causal_scale_transport_2d import causal_scale_transport_observation_2d
from .continuous_scale_edge_family_transport_2d import (
    continuous_scale_edge_family_transport_state_2d,
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
            initial = np.asarray(
                base_diagnostic["readouts"]["phase_susceptibility"])
            refinement_rows = []
            for refinement in (0, 1):
                state = continuous_scale_edge_family_transport_state_2d(
                    observation,
                    initial_posterior=initial,
                    trace_refinement=refinement,
                )
                posterior = np.asarray(state["posterior_after_erosion"])
                pushed_base = np.asarray(state["pushed_posterior_base"])
                contraction = state["contraction"]
                identity_contains = (
                    (truth - posterior >= state["transfer_enclosure_lower"])
                    & (truth - posterior <= state["transfer_enclosure_upper"])
                )
                pushed_contains = (
                    (truth - pushed_base
                     >= state["pushed_transfer_enclosure_lower"])
                    & (truth - pushed_base
                       <= state["pushed_transfer_enclosure_upper"])
                )
                refinement_rows.append({
                    "trace_refinement": refinement,
                    "lineage_count": len(state["lineage_labels"]),
                    "edge_variable_count": state["representation"][
                        "edge_variable_count"],
                    "zero_variable_count": state["representation"][
                        "zero_variable_count"],
                    "mean_coefficient_width": contraction[
                        "mean_coefficient_width"],
                    "contracted_coefficient_fraction": contraction[
                        "contracted_coefficient_fraction"],
                    "truth_value_coverage_before_push": _coverage(
                        truth - posterior,
                        state["transfer_enclosure_lower"],
                        state["transfer_enclosure_upper"],
                    ),
                    "truth_value_coverage_after_push": _coverage(
                        truth - pushed_base,
                        state["pushed_transfer_enclosure_lower"],
                        state["pushed_transfer_enclosure_upper"],
                    ),
                    "truth_value_coverage_any_branch": float(np.mean(
                        identity_contains | pushed_contains)),
                    "pushed_only_truth_coverage_fraction": float(np.mean(
                        pushed_contains & ~identity_contains)),
                    "mean_transfer_enclosure_width": float(np.mean(
                        state["transfer_enclosure_upper"]
                        - state["transfer_enclosure_lower"])),
                    "mean_pushed_enclosure_width": float(np.mean(
                        state["pushed_transfer_enclosure_upper"]
                        - state["pushed_transfer_enclosure_lower"])),
                    "lineage_recomposition_error": state[
                        "full_lineage_recomposition_error"],
                    "evolved_edge_flux_reconstruction_error": state[
                        "evolved_edge_response_flux"][
                            "reconstruction_maximum_error"],
                    "evolved_zero_flux_reconstruction_error": state[
                        "evolved_zero_response_flux"][
                            "reconstruction_maximum_error"],
                    "feasible_outer_component": contraction[
                        "feasible_outer_component"],
                })
                del state
            rows.append({
                "source": source,
                "condition": condition,
                "refinements": refinement_rows,
            })
    return {
        "purpose": (
            "measure local continuous-scale Selling-edge family coverage, "
            "safe coefficient contraction, factorized positive push-forward, "
            "and evolved-graph flux reconstruction"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
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
                "lineages", refinement["lineage_count"],
                "variables", refinement["edge_variable_count"],
                "width", round(refinement["mean_coefficient_width"], 4),
                "coverage", (
                    round(refinement["truth_value_coverage_before_push"], 4),
                    round(refinement["truth_value_coverage_after_push"], 4),
                ),
                "enclosure", (
                    round(refinement["mean_transfer_enclosure_width"], 4),
                    round(refinement["mean_pushed_enclosure_width"], 4),
                ),
            )


if __name__ == "__main__":
    main()
