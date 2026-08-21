"""Broad gate for target-free contextual energy-distance root participation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    denoise_cross_predictive_transport,
    lineage_branch_readout_forms,
    nested_midpoint_lineage_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "accepted_equilibrium",
    "information_lineage_median",
    "information_lineage_collision_mean",
    "information_coupled_phase_mean",
    "information_coupled_phase_median",
    "information_coupled_phase_collision_mean",
    "information_coupled_phase_collision_median",
    "information_coupled_phase_coverage_mean",
    "information_coupled_phase_coverage_median",
    "information_coupled_phase_bundle_coverage_mean",
    "information_coupled_phase_bundle_coverage_median",
    "information_global_characteristic_section",
    "information_posterior_characteristic_section",
    "information_path_collision_mean",
    "information_path_collision_median",
    "information_path_affinity_mean",
    "information_path_affinity_median",
    "information_path_fidelity_mean",
    "information_path_fidelity_median",
    "nested_context_mean",
    "nested_context_collision_mean",
    "energy_root_mean",
    "energy_root_median",
    "energy_root_collision_mean",
    "transport_energy_root_mean",
    "transport_energy_root_median",
    "transport_energy_root_collision_mean",
    "energy_root_quantile",
    "transport_energy_root_quantile",
    "hilbert_value_jet_mean",
    "hilbert_value_jet_collision",
    "phase_sasaki_mean",
    "phase_sasaki_collision",
    "phase_sasaki_energy_root_mean",
    "phase_sasaki_energy_root_collision",
    "phase_collision_mean",
    "phase_collision_median",
    "coupled_phase_mean",
    "coupled_phase_median",
    "coupled_phase_collision_mean",
    "coupled_phase_collision_median",
    "coupled_phase_coverage_mean",
    "coupled_phase_coverage_median",
    "coupled_phase_bundle_coverage_mean",
    "coupled_phase_bundle_coverage_median",
    "global_characteristic_section",
    "posterior_characteristic_section",
    "path_collision_mean",
    "path_collision_median",
    "path_affinity_mean",
    "path_affinity_median",
    "path_fidelity_mean",
    "path_fidelity_median",
    "distinct_residual_path_restore_mean",
    "distinct_residual_path_restore_median",
)


def run(size: int, seeds: int) -> dict:
    clean_rows = []
    noisy_rows = []

    def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        accepted = denoise_cross_predictive_transport(value)[0]
        lineage = lineage_branch_readout_forms(value)[0]
        nested, diagnostic = nested_midpoint_lineage_readout_forms(value)
        robust = lineage["collision_mean"]
        residual = value - robust
        residual_path = nested_midpoint_lineage_readout_forms(residual)[0]
        return {
            "accepted_equilibrium": accepted,
            "information_lineage_median": lineage["median"],
            "information_lineage_collision_mean": lineage["collision_mean"],
            "information_coupled_phase_mean": lineage[
                "hj_coupled_phase_mean"],
            "information_coupled_phase_median": lineage[
                "hj_coupled_phase_median"],
            "information_coupled_phase_collision_mean": lineage[
                "hj_coupled_phase_collision_mean"],
            "information_coupled_phase_collision_median": lineage[
                "hj_coupled_phase_collision_median"],
            "information_coupled_phase_coverage_mean": lineage[
                "hj_coupled_phase_coverage_mean"],
            "information_coupled_phase_coverage_median": lineage[
                "hj_coupled_phase_coverage_median"],
            "information_coupled_phase_bundle_coverage_mean": lineage[
                "hj_coupled_phase_bundle_coverage_mean"],
            "information_coupled_phase_bundle_coverage_median": lineage[
                "hj_coupled_phase_bundle_coverage_median"],
            "information_global_characteristic_section": lineage[
                "hj_global_characteristic_section"],
            "information_posterior_characteristic_section": lineage[
                "posterior_characteristic_section"],
            "information_path_collision_mean": lineage[
                "path_collision_mean"],
            "information_path_collision_median": lineage[
                "path_collision_median"],
            "information_path_affinity_mean": lineage[
                "path_affinity_mean"],
            "information_path_affinity_median": lineage[
                "path_affinity_median"],
            "information_path_fidelity_mean": lineage[
                "path_fidelity_mean"],
            "information_path_fidelity_median": lineage[
                "path_fidelity_median"],
            "nested_context_mean": nested["mean"],
            "nested_context_collision_mean": nested["collision_mean"],
            "energy_root_mean": nested["energy_root_mean"],
            "energy_root_median": nested["energy_root_median"],
            "energy_root_collision_mean": nested[
                "energy_root_collision_mean"],
            "transport_energy_root_mean": nested[
                "transport_energy_root_mean"],
            "transport_energy_root_median": nested[
                "transport_energy_root_median"],
            "transport_energy_root_collision_mean": nested[
                "transport_energy_root_collision_mean"],
            "energy_root_quantile": nested["energy_root_quantile"],
            "transport_energy_root_quantile": nested[
                "transport_energy_root_quantile"],
            "hilbert_value_jet_mean": nested["hilbert_value_jet_mean"],
            "hilbert_value_jet_collision": nested[
                "hilbert_value_jet_collision"],
            "phase_sasaki_mean": nested["phase_sasaki_mean"],
            "phase_sasaki_collision": nested["phase_sasaki_collision"],
            "phase_sasaki_energy_root_mean": nested[
                "phase_sasaki_energy_root_mean"],
            "phase_sasaki_energy_root_collision": nested[
                "phase_sasaki_energy_root_collision"],
            "phase_collision_mean": nested["phase_collision_mean"],
            "phase_collision_median": nested["phase_collision_median"],
            "coupled_phase_mean": nested["coupled_phase_mean"],
            "coupled_phase_median": nested["coupled_phase_median"],
            "coupled_phase_collision_mean": nested[
                "coupled_phase_collision_mean"],
            "coupled_phase_collision_median": nested[
                "coupled_phase_collision_median"],
            "coupled_phase_coverage_mean": nested[
                "coupled_phase_coverage_mean"],
            "coupled_phase_coverage_median": nested[
                "coupled_phase_coverage_median"],
            "coupled_phase_bundle_coverage_mean": nested[
                "coupled_phase_bundle_coverage_mean"],
            "coupled_phase_bundle_coverage_median": nested[
                "coupled_phase_bundle_coverage_median"],
            "global_characteristic_section": nested[
                "global_characteristic_section"],
            "posterior_characteristic_section": nested[
                "posterior_characteristic_section"],
            "path_collision_mean": nested["path_collision_mean"],
            "path_collision_median": nested["path_collision_median"],
            "path_affinity_mean": nested["path_affinity_mean"],
            "path_affinity_median": nested["path_affinity_median"],
            "path_fidelity_mean": nested["path_fidelity_mean"],
            "path_fidelity_median": nested["path_fidelity_median"],
            "distinct_residual_path_restore_mean": (
                robust + residual_path["path_collision_mean"]),
            "distinct_residual_path_restore_median": (
                robust + residual_path["path_collision_median"]),
        }, diagnostic

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        values, diagnostic = evaluate(truth)
        clean_rows.append({
            "preset": preset,
            **{name: metrics(value, truth) for name, value in values.items()},
            "mean_energy_root_authority": diagnostic[
                "energy_root_participation"]["mean_authority"],
            "phase_orientation_variation": diagnostic[
                "phase_sasaki_collision"]["phase_orientation_variation"],
            "mean_phase_anisotropy": diagnostic[
                "phase_sasaki_collision"]["mean_phase_anisotropy"],
            "mean_phase_root_authority": diagnostic[
                "phase_sasaki_collision"]["mean_root_authority"],
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=10100 + seed,
                )
                values, diagnostic = evaluate(observation)
                noisy_rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    **{
                        name: metrics(value, truth)
                        for name, value in values.items()
                    },
                    "mean_energy_root_authority": diagnostic[
                        "energy_root_participation"]["mean_authority"],
                    "phase_orientation_variation": diagnostic[
                        "phase_sasaki_collision"][
                            "phase_orientation_variation"],
                    "mean_phase_anisotropy": diagnostic[
                        "phase_sasaki_collision"]["mean_phase_anisotropy"],
                    "mean_phase_root_authority": diagnostic[
                        "phase_sasaki_collision"]["mean_root_authority"],
                })

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([row[method][key] for row in rows]))
                for key in rows[0][method]
            }
            for method in METHODS
        }

    return {
        "purpose": (
            "broad falsification of energy-distance root participation on a "
            "target-free nested contextual law"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "by_condition": {
            condition: summarize([
                row for row in noisy_rows if row["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "mean_clean_energy_root_authority": float(np.mean([
            row["mean_energy_root_authority"] for row in clean_rows
        ])),
        "mean_noisy_energy_root_authority": float(np.mean([
            row["mean_energy_root_authority"] for row in noisy_rows
        ])),
        "phase_behavior": {
            "clean_orientation_variation": float(np.mean([
                row["phase_orientation_variation"] for row in clean_rows
            ])),
            "noisy_orientation_variation": float(np.mean([
                row["phase_orientation_variation"] for row in noisy_rows
            ])),
            "clean_anisotropy": float(np.mean([
                row["mean_phase_anisotropy"] for row in clean_rows
            ])),
            "noisy_anisotropy": float(np.mean([
                row["mean_phase_anisotropy"] for row in noisy_rows
            ])),
            "clean_root_authority": float(np.mean([
                row["mean_phase_root_authority"] for row in clean_rows
            ])),
            "noisy_root_authority": float(np.mean([
                row["mean_phase_root_authority"] for row in noisy_rows
            ])),
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
        "authority": {
            "clean": result["mean_clean_energy_root_authority"],
            "noisy": result["mean_noisy_energy_root_authority"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
