"""Audit the joint value/first-jet constrained scale-edge zonotope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .causal_scale_transport_2d import causal_scale_transport_observation_2d
from .joint_value_jet_zonotope_contractor_2d import (
    contract_joint_value_jet_scale_edge_state_2d,
)
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def run(size: int, selected: tuple[str, ...]) -> dict[str, Any]:
    catalogue = sources(size)
    rows: list[dict[str, Any]] = []
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
            state = contract_joint_value_jet_scale_edge_state_2d(
                observation,
                initial_posterior=initial,
            )
            posterior = np.asarray(state["posterior_after_erosion"])
            pushed_base = np.asarray(state["pushed_posterior_base"])
            constrained = state["constrained_zonotope"]
            truth_transfer = (truth - posterior).reshape(-1)
            truth_coordinate = np.asarray(
                constrained["coordinate_operator"] @ truth_transfer).ravel()
            coordinate_compatible = (
                (truth_coordinate >= constrained["constraint_lower"])
                & (truth_coordinate <= constrained["constraint_upper"])
            )
            edge_count = constrained["constraint_block_edge_count"]
            residual_compatible = coordinate_compatible[:edge_count]
            posterior_compatible = coordinate_compatible[edge_count:]
            parent_contains = (
                (truth - posterior >= state["transfer_enclosure_lower"])
                & (truth - posterior <= state["transfer_enclosure_upper"])
            )
            pushed_contains = (
                (truth - pushed_base
                 >= state["pushed_transfer_enclosure_lower"])
                & (truth - pushed_base
                   <= state["pushed_transfer_enclosure_upper"])
            )
            witness = state["joint_witness"]
            rows.append({
                "source": source,
                "condition": condition,
                "variable_count": int(state["generator"].shape[1]),
                "target_edge_count": witness["target_edge_count"],
                "joint_constraint_count": witness["joint_constraint_count"],
                "target_exclusion_error": witness["target_exclusion_error"],
                "fold_zero_edge_count": witness["fold_zero_edge_count"],
                "fold_one_edge_count": witness["fold_one_edge_count"],
                "orientation_edge_counts": witness["orientation_edge_counts"],
                "mean_value_interval_width": witness[
                    "mean_value_interval_width"],
                "mean_jet_interval_width": witness[
                    "mean_jet_interval_width"],
                "mean_coefficient_width_before_joint": state[
                    "mean_coefficient_width_before_joint"],
                "mean_coefficient_width_after_joint": state[
                    "mean_coefficient_width_after_joint"],
                "additional_contracted_coefficient_fraction": state[
                    "additional_contracted_coefficient_fraction"],
                "zero_transfer_feasible": constrained[
                    "zero_transfer_feasible"],
                "full_transfer_violation_fraction": constrained[
                    "full_transfer_violation_fraction"],
                "mean_constraint_support_width_ratio": constrained[
                    "mean_constraint_support_width_ratio"],
                "median_constraint_support_width_ratio": constrained[
                    "median_constraint_support_width_ratio"],
                "truth_residual_normal_compatibility": float(np.mean(
                    residual_compatible)),
                "truth_posterior_normal_compatibility": float(np.mean(
                    posterior_compatible)),
                "truth_joint_edge_compatibility": float(np.mean(
                    residual_compatible & posterior_compatible)),
                "truth_value_coverage_identity_outer_shadow": float(np.mean(
                    parent_contains)),
                "truth_value_coverage_positive_push_outer_shadow": float(
                    np.mean(pushed_contains)),
                "truth_value_coverage_any_outer_shadow": float(np.mean(
                    parent_contains | pushed_contains)),
                "joint_contractor_feasible": state["joint_contraction"][
                    "feasible_outer_component"],
                "lineage_recomposition_error": state[
                    "full_lineage_recomposition_error"],
            })
            del state
    return {
        "purpose": (
            "measure the actual sparse value/first-jet slab intersection, "
            "its axis-aligned coefficient shadow, retrospective truth "
            "compatibility, and retained identity/pushed branch coverage"
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
        print(
            row["source"], "|", row["condition"],
            "coefficient widths", (
                round(row["mean_coefficient_width_before_joint"], 5),
                round(row["mean_coefficient_width_after_joint"], 5),
            ),
            "constraint ratio", round(
                row["mean_constraint_support_width_ratio"], 5),
            "full rejected", round(
                row["full_transfer_violation_fraction"], 5),
            "truth joint", round(
                row["truth_joint_edge_compatibility"], 5),
            "outer coverage", (
                round(row["truth_value_coverage_identity_outer_shadow"], 5),
                round(row["truth_value_coverage_positive_push_outer_shadow"], 5),
            ),
        )


if __name__ == "__main__":
    main()
