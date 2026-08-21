"""Audit truth coverage and contraction of the two edge-flux zonotopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .zonotopic_edge_flux_2d import zonotopic_edge_flux_state_2d


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
            state = zonotopic_edge_flux_state_2d(observation)
            posterior = np.asarray(state["posterior_after_erosion"])
            truth_transfer = truth - posterior
            component_rows = []
            mixture_lower = np.full_like(truth, np.inf)
            mixture_upper = np.full_like(truth, -np.inf)
            for component in state["components"]:
                lower = np.asarray(component["transfer_lower"])
                upper = np.asarray(component["transfer_upper"])
                mixture_lower = np.minimum(mixture_lower, lower)
                mixture_upper = np.maximum(mixture_upper, upper)
                midpoint = posterior + np.asarray(
                    component["midpoint_transfer"])
                contraction = component["contraction"]
                representation = component["representation"]
                component_rows.append({
                    "name": component["name"],
                    "feasible_outer_component": contraction[
                        "feasible_outer_component"],
                    "mean_coefficient_width": contraction[
                        "mean_coefficient_width"],
                    "contracted_coefficient_fraction": contraction[
                        "contracted_coefficient_fraction"],
                    "point_coefficient_fraction": contraction[
                        "point_coefficient_fraction"],
                    "full_proposal_in_contracted_box": component[
                        "full_proposal_in_contracted_box"],
                    "zero_transfer_in_contracted_box": component[
                        "zero_transfer_in_contracted_box"],
                    "edge_count": representation["edge_count"],
                    "zero_mode_count": representation["zero_mode_count"],
                    "proposal_reconstruction_maximum_error": representation[
                        "proposal_reconstruction_maximum_error"],
                    "truth_value_coverage_fraction": float(np.mean(
                        (truth_transfer >= lower)
                        & (truth_transfer <= upper)
                    )),
                    "midpoint_metrics_for_audit_only": metrics(
                        midpoint, truth),
                })
            rows.append({
                "source": source,
                "condition": condition,
                "eroded_posterior_metrics": metrics(posterior, truth),
                "components": component_rows,
                "mixture_outer_value_coverage_fraction": float(np.mean(
                    (truth_transfer >= mixture_lower)
                    & (truth_transfer <= mixture_upper)
                )),
                "observation_recomposition_error": state[
                    "observation_recomposition_error"],
                "raw_witness_exclusion_fraction": state[
                    "raw_witness_exclusion_fraction"],
            })
    return {
        "purpose": (
            "measure whether separate phase and curvature edge-flux "
            "zonotopes retain truth while bounded residual evidence contracts "
            "their generator intervals"
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
        for component in row["components"]:
            print(
                component["name"],
                "feasible", component["feasible_outer_component"],
                "width", round(component["mean_coefficient_width"], 4),
                "contracted", round(
                    component["contracted_coefficient_fraction"], 4),
                "truth coverage", round(
                    component["truth_value_coverage_fraction"], 4),
                "midpoint", (
                    round(component["midpoint_metrics_for_audit_only"]["mse"], 6),
                    round(component["midpoint_metrics_for_audit_only"][
                        "edge_retention"], 4),
                ),
            )
        print("mixture truth coverage", round(
            row["mixture_outer_value_coverage_fraction"], 4))


if __name__ == "__main__":
    main()
