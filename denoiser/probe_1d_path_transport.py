"""Broad gate for complete-history collision and fidelity transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    denoise_cross_predictive_transport,
    lineage_branch_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "accepted_equilibrium",
    "information_lineage_collision_mean",
    "path_collision_mean",
    "path_collision_median",
    "path_affinity_mean",
    "path_affinity_median",
    "path_fidelity_mean",
    "path_fidelity_median",
    "transport_fidelity_mean",
    "transport_fidelity_median",
    "transport_plan_history_mean",
    "self_consistent_transport_mean",
    "distributed_transport_mean",
    "action_contracting_transport_mean",
    "two_history_action_transport_mean",
    "path_fidelity_participation_section",
    "transport_history_participation_mean",
    "transport_history_participation_median",
    "global_characteristic_section",
    "posterior_characteristic_section",
)


def run(size: int, seeds: int) -> dict:
    clean_rows: list[dict] = []
    noisy_rows: list[dict] = []

    def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        accepted = denoise_cross_predictive_transport(value)[0]
        lineage, diagnostic = lineage_branch_readout_forms(value)
        return {
            "accepted_equilibrium": accepted,
            "information_lineage_collision_mean": lineage["collision_mean"],
            "path_collision_mean": lineage["path_collision_mean"],
            "path_collision_median": lineage["path_collision_median"],
            "path_affinity_mean": lineage["path_affinity_mean"],
            "path_affinity_median": lineage["path_affinity_median"],
            "path_fidelity_mean": lineage["path_fidelity_mean"],
            "path_fidelity_median": lineage["path_fidelity_median"],
            "transport_fidelity_mean": lineage[
                "transport_fidelity_mean"],
            "transport_fidelity_median": lineage[
                "transport_fidelity_median"],
            "transport_plan_history_mean": lineage[
                "transport_plan_history_mean"],
            "self_consistent_transport_mean": lineage[
                "self_consistent_transport_mean"],
            "distributed_transport_mean": lineage[
                "distributed_transport_mean"],
            "action_contracting_transport_mean": lineage[
                "action_contracting_transport_mean"],
            "two_history_action_transport_mean": lineage[
                "two_history_action_transport_mean"],
            "path_fidelity_participation_section": lineage[
                "path_fidelity_participation_section"],
            "transport_history_participation_mean": lineage[
                "transport_history_participation_mean"],
            "transport_history_participation_median": lineage[
                "transport_history_participation_median"],
            "global_characteristic_section": lineage[
                "hj_global_characteristic_section"],
            "posterior_characteristic_section": lineage[
                "posterior_characteristic_section"],
        }, diagnostic

    def row(
        preset: str,
        truth: np.ndarray,
        value: np.ndarray,
        *,
        condition: str | None = None,
        seed: int | None = None,
    ) -> dict:
        forms, diagnostic = evaluate(value)
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            "path_fidelity": diagnostic["path_fidelity_geodesic"][
                "mean_fidelity"],
            "path_collision_population": diagnostic[
                "global_path_collision"]["mean_terminal_population"],
            "lineage_population": diagnostic["mean_lineage_population"],
            "barycentric_participation": diagnostic[
                "path_fidelity_participation"][
                    "mean_barycentric_participation"],
            "transport_edge_fidelity": diagnostic[
                "transport_plan_fidelity"]["mean_edge_fidelity"],
            "transport_vertex_survival": diagnostic[
                "transport_plan_fidelity"]["mean_vertex_survival"],
        }
        if condition is not None:
            result["condition"] = condition
        if seed is not None:
            result["seed"] = seed
        return result

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_rows.append(row(preset, truth, truth))
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=13100 + seed,
                )
                noisy_rows.append(row(
                    preset,
                    truth,
                    observation,
                    condition=condition,
                    seed=seed,
                ))

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([entry[method][key] for entry in rows]))
                for key in rows[0][method]
            }
            for method in METHODS
        }

    def uncertainty(rows: list[dict]) -> dict:
        return {
            key: float(np.mean([entry[key] for entry in rows]))
            for key in (
                "path_fidelity",
                "path_collision_population",
                "lineage_population",
                "barycentric_participation",
                "transport_edge_fidelity",
                "transport_vertex_survival",
            )
        }

    return {
        "purpose": (
            "broad falsification of complete-history collision, posterior "
            "characteristics, and fidelity-governed path transport"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "clean_uncertainty": uncertainty(clean_rows),
        "noisy_uncertainty": uncertainty(noisy_rows),
        "by_condition": {
            condition: summarize([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "uncertainty_by_condition": {
            condition: uncertainty([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "clean_rows": clean_rows,
        "rows": noisy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
        "uncertainty": {
            "clean": result["clean_uncertainty"],
            "noisy": result["noisy_uncertainty"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
