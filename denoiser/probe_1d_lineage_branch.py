"""Gate positive jet-bundle lineage against local scalar readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    denoise_cross_predictive_transport,
    denoise_lineage_branch_transport,
    continuous_curvature_lineage_readout_forms,
    curvature_consensus_lineage_readout_forms,
    independent_side_collision_readout_forms,
    lineage_branch_readout_forms,
    nested_midpoint_lineage_readout_forms,
    paired_side_collision_lineage_readout_forms,
    relation_scale_readout_forms,
    root_context_collision_lineage_readout_forms,
    symmetric_second_jet_lineage_readout_forms,
    symmetric_second_jet_curvature_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(
    size: int,
    seeds: int,
    *,
    preset_names: tuple[str, ...] = PRESET_NAMES,
    condition_names=None,
    include_equilibrium: bool = True,
) -> dict:
    methods = (
        "local_mean", "local_median", "accepted_equilibrium",
        "lineage_mean", "lineage_median", "lineage_maximum",
        "lineage_collision_mean", "lineage_collision_median",
        "lineage_oriented_collision_section",
        "lineage_symmetric_parent_mean", "lineage_symmetric_parent_median",
        "lineage_hj_joint_mean", "lineage_hj_joint_median",
        "lineage_hj_joint_collision_mean",
        "lineage_hj_joint_collision_median",
        "lineage_joint_w1_value_jet",
        "lineage_joint_information_field",
        "lineage_energy_root_mean", "lineage_energy_root_median",
        "lineage_energy_root_collision_mean",
        "paired_side_lineage_mean", "paired_side_lineage_median",
        "paired_side_lineage_collision_mean",
        "nested_midpoint_lineage_mean", "nested_midpoint_lineage_median",
        "nested_midpoint_lineage_collision_mean",
        "nested_midpoint_energy_root_mean",
        "nested_midpoint_energy_root_median",
        "nested_midpoint_energy_root_collision_mean",
        "nested_midpoint_root_mean", "nested_midpoint_root_median",
        "nested_midpoint_root_collision_mean",
        "root_context_lineage_mean", "root_context_lineage_median",
        "root_context_lineage_collision_mean",
        "root_context_hj_mean", "root_context_hj_median",
        "root_context_hj_collision_mean",
        "root_context_hj_collision_median",
        "root_context_markov_collision_mean",
        "root_context_ancestry_geodesic_mean",
        "root_context_ancestry_geodesic_median",
        "root_context_ancestry_geodesic_simplex_mean",
        "independent_side_collision_mean",
        "independent_side_collision_median",
        "independent_side_joint_collision_mean",
        "independent_side_joint_collision_median",
        "symmetric_second_jet_mean", "symmetric_second_jet_median",
        "symmetric_second_jet_collision_mean",
        "symmetric_second_jet_hj_collision_mean",
        "curvature_bundle_mean", "curvature_bundle_median",
        "curvature_bundle_collision_mean",
        "curvature_bundle_hj_collision_mean",
        "curvature_consensus_mean", "curvature_consensus_median",
        "curvature_consensus_collision_mean",
        "curvature_consensus_hj_collision_mean",
        "continuous_curvature_mean", "continuous_curvature_median",
        "continuous_curvature_collision_mean",
        "continuous_curvature_energy_root_mean",
        "continuous_curvature_energy_root_collision_mean",
        "continuous_curvature_authority_mean",
        "continuous_curvature_authority_median",
        "continuous_curvature_authority_collision_mean",
    ) + (("lineage_equilibrium",) if include_equilibrium else ())
    clean_rows = []
    rows = []

    def evaluate(value):
        local = relation_scale_readout_forms(value)[0]
        accepted = denoise_cross_predictive_transport(value)[0]
        lineage, diagnostic = lineage_branch_readout_forms(
            value, include_experimental=True)
        paired_side, paired_side_diagnostic = (
            paired_side_collision_lineage_readout_forms(value))
        nested_midpoint, nested_midpoint_diagnostic = (
            nested_midpoint_lineage_readout_forms(value))
        root_context, root_context_diagnostic = (
            root_context_collision_lineage_readout_forms(value))
        root_context_markov, root_context_markov_diagnostic = (
            root_context_collision_lineage_readout_forms(
                value, transition_normalization="markov"))
        independent_side, independent_side_diagnostic = (
            independent_side_collision_readout_forms(value))
        second_jet, second_jet_diagnostic = (
            symmetric_second_jet_lineage_readout_forms(value))
        curvature_bundle, curvature_bundle_diagnostic = (
            symmetric_second_jet_curvature_readout_forms(value))
        curvature_consensus, curvature_consensus_diagnostic = (
            curvature_consensus_lineage_readout_forms(value))
        continuous_curvature, continuous_curvature_diagnostic = (
            continuous_curvature_lineage_readout_forms(value))
        if include_equilibrium:
            equilibrium, equilibrium_diagnostic = (
                denoise_lineage_branch_transport(value))
        else:
            equilibrium = None
            equilibrium_diagnostic = None
        values = {
            "local_mean": local["mean"],
            "local_median": local["median"],
            "accepted_equilibrium": accepted,
            "lineage_mean": lineage["mean"],
            "lineage_median": lineage["median"],
            "lineage_maximum": lineage["maximum_branch"],
            "lineage_collision_mean": lineage["collision_mean"],
            "lineage_collision_median": lineage["collision_median"],
            "lineage_oriented_collision_section": lineage[
                "oriented_collision_section"],
            "lineage_symmetric_parent_mean": lineage[
                "symmetric_parent_mean"],
            "lineage_symmetric_parent_median": lineage[
                "symmetric_parent_median"],
            "lineage_hj_joint_mean": lineage["hj_joint_mean"],
            "lineage_hj_joint_median": lineage["hj_joint_median"],
            "lineage_hj_joint_collision_mean": lineage[
                "hj_joint_collision_mean"],
            "lineage_hj_joint_collision_median": lineage[
                "hj_joint_collision_median"],
            "lineage_joint_w1_value_jet": lineage["joint_w1_value_jet"],
            "lineage_joint_information_field": lineage[
                "joint_information_field"],
            "lineage_energy_root_mean": lineage["energy_root_mean"],
            "lineage_energy_root_median": lineage["energy_root_median"],
            "lineage_energy_root_collision_mean": lineage[
                "energy_root_collision_mean"],
            "paired_side_lineage_mean": paired_side["mean"],
            "paired_side_lineage_median": paired_side["median"],
            "paired_side_lineage_collision_mean": paired_side[
                "collision_mean"],
            "nested_midpoint_lineage_mean": nested_midpoint["mean"],
            "nested_midpoint_lineage_median": nested_midpoint["median"],
            "nested_midpoint_lineage_collision_mean": nested_midpoint[
                "collision_mean"],
            "nested_midpoint_energy_root_mean": nested_midpoint[
                "energy_root_mean"],
            "nested_midpoint_energy_root_median": nested_midpoint[
                "energy_root_median"],
            "nested_midpoint_energy_root_collision_mean": nested_midpoint[
                "energy_root_collision_mean"],
            # The observation and the independently transported contextual
            # law are two causal parents. Their L2 collision barycenter has
            # equal source multiplicity; this coefficient is structural, not
            # a fitted denoising strength.
            "nested_midpoint_root_mean": 0.5 * (
                value + nested_midpoint["mean"]),
            "nested_midpoint_root_median": 0.5 * (
                value + nested_midpoint["median"]),
            "nested_midpoint_root_collision_mean": 0.5 * (
                value + nested_midpoint["collision_mean"]),
            "root_context_lineage_mean": root_context["mean"],
            "root_context_lineage_median": root_context["median"],
            "root_context_lineage_collision_mean": root_context[
                "collision_mean"],
            "root_context_hj_mean": root_context["hj_mean"],
            "root_context_hj_median": root_context["hj_median"],
            "root_context_hj_collision_mean": root_context[
                "hj_collision_mean"],
            "root_context_hj_collision_median": root_context[
                "hj_collision_median"],
            "root_context_markov_collision_mean": root_context_markov[
                "collision_mean"],
            "root_context_ancestry_geodesic_mean": root_context[
                "ancestry_geodesic_mean"],
            "root_context_ancestry_geodesic_median": root_context[
                "ancestry_geodesic_median"],
            "root_context_ancestry_geodesic_simplex_mean": root_context[
                "ancestry_geodesic_simplex_mean"],
            "independent_side_collision_mean": independent_side["mean"],
            "independent_side_collision_median": independent_side["median"],
            "independent_side_joint_collision_mean": independent_side[
                "joint_mean"],
            "independent_side_joint_collision_median": independent_side[
                "joint_median"],
            "symmetric_second_jet_mean": second_jet["mean"],
            "symmetric_second_jet_median": second_jet["median"],
            "symmetric_second_jet_collision_mean": second_jet[
                "collision_mean"],
            "symmetric_second_jet_hj_collision_mean": second_jet[
                "hj_collision_mean"],
            "curvature_bundle_mean": curvature_bundle["mean"],
            "curvature_bundle_median": curvature_bundle["median"],
            "curvature_bundle_collision_mean": curvature_bundle[
                "collision_mean"],
            "curvature_bundle_hj_collision_mean": curvature_bundle[
                "hj_collision_mean"],
            "curvature_consensus_mean": curvature_consensus["mean"],
            "curvature_consensus_median": curvature_consensus["median"],
            "curvature_consensus_collision_mean": curvature_consensus[
                "collision_mean"],
            "curvature_consensus_hj_collision_mean": curvature_consensus[
                "hj_collision_mean"],
            "continuous_curvature_mean": continuous_curvature["mean"],
            "continuous_curvature_median": continuous_curvature["median"],
            "continuous_curvature_collision_mean": continuous_curvature[
                "collision_mean"],
            "continuous_curvature_energy_root_mean": continuous_curvature[
                "energy_root_mean"],
            "continuous_curvature_energy_root_collision_mean": (
                continuous_curvature["energy_root_collision_mean"]),
            "continuous_curvature_authority_mean": continuous_curvature[
                "authority_curvature_mean"],
            "continuous_curvature_authority_median": continuous_curvature[
                "authority_curvature_median"],
            "continuous_curvature_authority_collision_mean": (
                continuous_curvature["authority_curvature_collision_mean"]),
        }
        if equilibrium is not None:
            values["lineage_equilibrium"] = equilibrium
        return {
            "values": values,
            "diagnostic": diagnostic,
            "paired_side_diagnostic": paired_side_diagnostic,
            "nested_midpoint_diagnostic": nested_midpoint_diagnostic,
            "root_context_diagnostic": root_context_diagnostic,
            "root_context_markov_diagnostic": root_context_markov_diagnostic,
            "independent_side_diagnostic": independent_side_diagnostic,
            "second_jet_diagnostic": second_jet_diagnostic,
            "curvature_bundle_diagnostic": curvature_bundle_diagnostic,
            "curvature_consensus_diagnostic": curvature_consensus_diagnostic,
            "continuous_curvature_diagnostic": continuous_curvature_diagnostic,
            "equilibrium_diagnostic": equilibrium_diagnostic,
        }

    selected_conditions = tuple(
        condition for condition in CONDITIONS
        if condition_names is None or condition[0] in condition_names
    )
    for preset in preset_names:
        truth = compose_series(size, PRESETS[preset])[1]
        clean = evaluate(truth)
        clean_rows.append({
            "preset": preset,
            **{name: metrics(value, truth)
               for name, value in clean["values"].items()},
            "lineage": clean["diagnostic"],
            "paired_side_lineage": clean["paired_side_diagnostic"],
            "nested_midpoint_lineage": clean[
                "nested_midpoint_diagnostic"],
            "root_context_lineage": clean["root_context_diagnostic"],
            **({"equilibrium": clean["equilibrium_diagnostic"]}
               if include_equilibrium else {}),
        })
        for condition, kind, amount, density in selected_conditions:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=10100 + seed)
                result = evaluate(observation)
                rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    **{name: metrics(value, truth)
                       for name, value in result["values"].items()},
                    "lineage": result["diagnostic"],
                    "paired_side_lineage": result[
                        "paired_side_diagnostic"],
                    "nested_midpoint_lineage": result[
                        "nested_midpoint_diagnostic"],
                    "root_context_lineage": result[
                        "root_context_diagnostic"],
                    **({"equilibrium": result["equilibrium_diagnostic"]}
                       if include_equilibrium else {}),
                })

    def summarize(selected):
        return {
            method: {
                key: float(np.mean([row[method][key] for row in selected]))
                for key in selected[0][method]
            }
            for method in methods
        }

    output = {
        "purpose": (
            "test bidirectional positive mass transport on the 1-D jet bundle"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition, *_rest in selected_conditions
        },
        "mean_clean_lineage_population": float(np.mean([
            row["lineage"]["mean_lineage_population"]
            for row in clean_rows
        ])),
        "mean_noisy_lineage_population": float(np.mean([
            row["lineage"]["mean_lineage_population"] for row in rows
        ])),
        "clean_rows": clean_rows,
        "rows": rows,
    }
    if include_equilibrium:
        output["equilibrium_ceiling_hits"] = int(sum(
            row["equilibrium"]["continuation_ceiling_hit"] for row in rows
        ))
        output["mean_equilibrium_continuations"] = float(np.mean([
            row["equilibrium"]["accepted_continuations"] for row in rows
        ]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--presets",
        default=",".join(PRESET_NAMES),
        help="comma-separated diagnostic source names",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="optional comma-separated corruption-control names",
    )
    parser.add_argument(
        "--skip-equilibrium",
        action="store_true",
        help="measure the one-pass lineage marginal without continuation",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preset_names = tuple(
        name.strip() for name in args.presets.split(",") if name.strip())
    condition_names = tuple(
        name.strip() for name in args.conditions.split(",") if name.strip())
    result = run(
        args.size,
        args.seeds,
        preset_names=preset_names,
        condition_names=condition_names or None,
        include_equilibrium=not args.skip_equilibrium,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
        "heavy": {
            condition: result["by_condition"][condition]
            for condition in (
                "replacement 0.25", "replacement 0.40",
                "mixed 0.25", "mixed 0.40",
            )
            if condition in result["by_condition"]
        },
        "mean_clean_lineage_population": result[
            "mean_clean_lineage_population"],
        "mean_noisy_lineage_population": result[
            "mean_noisy_lineage_population"],
        **({
            "equilibrium_ceiling_hits": result["equilibrium_ceiling_hits"],
            "mean_equilibrium_continuations": result[
                "mean_equilibrium_continuations"],
        } if "equilibrium_ceiling_hits" in result else {}),
    }, indent=2))


if __name__ == "__main__":
    main()
