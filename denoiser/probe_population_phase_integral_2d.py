"""Gate measure-level population-phase integration under nested refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_phase_refinement_readouts_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_continuous_tangent_convergence import CONDITIONS as CORE_CONDITIONS
from .run_2d_denoiser_battery import (
    CONDITIONS as CORRUPTION_CONDITIONS,
    metrics,
    sources,
)
from .sample_series import corrupt


CONDITIONS = (CORE_CONDITIONS[0],) + CORRUPTION_CONDITIONS


def run(
    size: int,
    selected_sources: tuple[str, ...],
    selected_conditions: tuple[str, ...],
    phase_counts: tuple[int, ...],
    angular_count: int,
    quantile_count: int,
) -> dict:
    catalogue = sources(size)
    condition_catalogue = {row[0]: row for row in CONDITIONS}
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition in selected_conditions:
            label, kind, amount, density = condition_catalogue[condition]
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=11107)
            )
            fmmt = denoise_fmmt(observation)[0]
            count_rows = []
            fields = {}
            mean_fields = {}
            jet_fields = {}
            maximum_fields = {}
            phase_maximum_fields = {}
            joint_median_fields = {}
            hj_fields = {}
            hj_barycenter_fields = {}
            hj_w1_fields = {}
            hj_ancestry_fields = {}
            hj_ancestry_w1_fields = {}
            hj_simplex_fields = {}
            hj_simplex_w1_fields = {}
            mode_fields = {}
            refinement = causal_information_phase_refinement_readouts_2d(
                observation,
                phase_counts=phase_counts,
                angular_count=angular_count,
                quantile_count=quantile_count,
            )
            for count in phase_counts:
                forms, diagnostic = refinement[count]
                fields[count] = forms["causal_collision_median"]
                mean_fields[count] = forms["causal_collision_mean"]
                jet_fields[count] = forms["causal_collision_jet_potential"]
                maximum_fields[count] = forms["causal_collision_maximum"]
                phase_maximum_fields[count] = forms[
                    "causal_phase_average_collision_maximum"]
                joint_median_fields[count] = forms[
                    "causal_collision_joint_median"]
                hj_fields[count] = forms[
                    "causal_phase_average_hj_collision_section"]
                hj_barycenter_fields[count] = forms[
                    "causal_phase_average_hj_collision_barycenter"]
                hj_w1_fields[count] = forms[
                    "causal_phase_average_hj_collision_w1_barycenter"]
                hj_ancestry_fields[count] = forms[
                    "causal_phase_average_hj_ancestry_collision_barycenter"]
                hj_ancestry_w1_fields[count] = forms[
                    "causal_phase_average_hj_ancestry_collision_w1_barycenter"]
                hj_simplex_fields[count] = forms[
                    "causal_phase_average_hj_simplex_collision_barycenter"]
                hj_simplex_w1_fields[count] = forms[
                    "causal_phase_average_hj_simplex_collision_w1_barycenter"]
                mode_fields[count] = forms[
                    "causal_phase_average_collision_mode"]
                count_rows.append({
                    "phase_count": int(count),
                    "local_median": metrics(forms["local_median"], truth),
                    "causal_collision_median": metrics(
                        forms["causal_collision_median"], truth),
                    "causal_collision_mean": metrics(
                        forms["causal_collision_mean"], truth),
                    "causal_collision_jet_potential": metrics(
                        forms["causal_collision_jet_potential"], truth),
                    "causal_collision_maximum": metrics(
                        forms["causal_collision_maximum"], truth),
                    "causal_phase_average_collision_maximum": metrics(
                        forms["causal_phase_average_collision_maximum"], truth),
                    "causal_collision_joint_median": metrics(
                        forms["causal_collision_joint_median"], truth),
                    "causal_phase_average_hj_collision_section": metrics(
                        forms["causal_phase_average_hj_collision_section"], truth),
                    "causal_phase_average_hj_collision_barycenter": metrics(
                        forms[
                            "causal_phase_average_hj_collision_barycenter"],
                        truth),
                    "causal_phase_average_hj_collision_w1_barycenter": metrics(
                        forms[
                            "causal_phase_average_hj_collision_w1_barycenter"],
                        truth),
                    "causal_phase_average_hj_ancestry_collision_barycenter": (
                        metrics(forms[
                            "causal_phase_average_hj_ancestry_collision_barycenter"
                        ], truth)
                    ),
                    "causal_phase_average_hj_ancestry_collision_w1_barycenter": (
                        metrics(forms[
                            "causal_phase_average_hj_ancestry_collision_w1_barycenter"
                        ], truth)
                    ),
                    "causal_phase_average_hj_simplex_collision_barycenter": (
                        metrics(forms[
                            "causal_phase_average_hj_simplex_collision_barycenter"
                        ], truth)
                    ),
                    "causal_phase_average_hj_simplex_collision_w1_barycenter": (
                        metrics(forms[
                            "causal_phase_average_hj_simplex_collision_w1_barycenter"
                        ], truth)
                    ),
                    "causal_phase_average_collision_mode": metrics(
                        forms["causal_phase_average_collision_mode"], truth),
                    "integrated_fmmt": metrics(fmmt, truth),
                    **diagnostic,
                })
            differences = []
            for previous, current in zip(phase_counts, phase_counts[1:]):
                delta = fields[current] - fields[previous]
                mean_delta = mean_fields[current] - mean_fields[previous]
                jet_delta = jet_fields[current] - jet_fields[previous]
                maximum_delta = (
                    maximum_fields[current] - maximum_fields[previous])
                phase_maximum_delta = (
                    phase_maximum_fields[current]
                    - phase_maximum_fields[previous])
                joint_median_delta = (
                    joint_median_fields[current] - joint_median_fields[previous])
                hj_delta = hj_fields[current] - hj_fields[previous]
                hj_barycenter_delta = (
                    hj_barycenter_fields[current]
                    - hj_barycenter_fields[previous])
                hj_w1_delta = hj_w1_fields[current] - hj_w1_fields[previous]
                hj_ancestry_delta = (
                    hj_ancestry_fields[current] - hj_ancestry_fields[previous])
                hj_ancestry_w1_delta = (
                    hj_ancestry_w1_fields[current]
                    - hj_ancestry_w1_fields[previous])
                hj_simplex_delta = (
                    hj_simplex_fields[current] - hj_simplex_fields[previous])
                hj_simplex_w1_delta = (
                    hj_simplex_w1_fields[current]
                    - hj_simplex_w1_fields[previous])
                mode_delta = mode_fields[current] - mode_fields[previous]
                differences.append({
                    "transition": f"{previous}->{current}",
                    "rms": float(np.sqrt(np.mean(delta * delta))),
                    "maximum": float(np.max(np.abs(delta))),
                    "collision_mean_rms": float(np.sqrt(np.mean(
                        mean_delta * mean_delta))),
                    "collision_mean_maximum": float(np.max(np.abs(
                        mean_delta))),
                    "collision_jet_rms": float(np.sqrt(np.mean(
                        jet_delta * jet_delta))),
                    "collision_jet_maximum": float(np.max(np.abs(
                        jet_delta))),
                    "collision_maximum_rms": float(np.sqrt(np.mean(
                        maximum_delta * maximum_delta))),
                    "collision_maximum_maximum": float(np.max(np.abs(
                        maximum_delta))),
                    "phase_collision_maximum_rms": float(np.sqrt(np.mean(
                        phase_maximum_delta * phase_maximum_delta))),
                    "phase_collision_maximum_maximum": float(np.max(np.abs(
                        phase_maximum_delta))),
                    "collision_joint_median_rms": float(np.sqrt(np.mean(
                        joint_median_delta * joint_median_delta))),
                    "collision_joint_median_maximum": float(np.max(np.abs(
                        joint_median_delta))),
                    "phase_hj_collision_rms": float(np.sqrt(np.mean(
                        hj_delta * hj_delta))),
                    "phase_hj_collision_maximum": float(np.max(np.abs(
                        hj_delta))),
                    "phase_hj_collision_barycenter_rms": float(np.sqrt(
                        np.mean(hj_barycenter_delta * hj_barycenter_delta))),
                    "phase_hj_collision_barycenter_maximum": float(np.max(
                        np.abs(hj_barycenter_delta))),
                    "phase_hj_collision_w1_rms": float(np.sqrt(
                        np.mean(hj_w1_delta * hj_w1_delta))),
                    "phase_hj_collision_w1_maximum": float(np.max(
                        np.abs(hj_w1_delta))),
                    "phase_hj_ancestry_collision_rms": float(np.sqrt(
                        np.mean(hj_ancestry_delta * hj_ancestry_delta))),
                    "phase_hj_ancestry_collision_maximum": float(np.max(
                        np.abs(hj_ancestry_delta))),
                    "phase_hj_ancestry_collision_w1_rms": float(np.sqrt(
                        np.mean(hj_ancestry_w1_delta * hj_ancestry_w1_delta))),
                    "phase_hj_ancestry_collision_w1_maximum": float(np.max(
                        np.abs(hj_ancestry_w1_delta))),
                    "phase_hj_simplex_collision_rms": float(np.sqrt(
                        np.mean(hj_simplex_delta * hj_simplex_delta))),
                    "phase_hj_simplex_collision_maximum": float(np.max(
                        np.abs(hj_simplex_delta))),
                    "phase_hj_simplex_collision_w1_rms": float(np.sqrt(
                        np.mean(hj_simplex_w1_delta * hj_simplex_w1_delta))),
                    "phase_hj_simplex_collision_w1_maximum": float(np.max(
                        np.abs(hj_simplex_w1_delta))),
                    "phase_collision_mode_rms": float(np.sqrt(np.mean(
                        mode_delta * mode_delta))),
                    "phase_collision_mode_maximum": float(np.max(np.abs(
                        mode_delta))),
                })
            rows.append({
                "source": source,
                "condition": label,
                "counts": count_rows,
                "refinement_difference": differences,
            })

    summary = {}
    for count in phase_counts:
        records = [
            item for row in rows for item in row["counts"]
            if item["phase_count"] == count
        ]
        summary[str(count)] = {
            method: {
                key: float(np.mean([record[method][key] for record in records]))
                for key in records[0][method]
            }
            for method in (
                "local_median", "causal_collision_median",
                "causal_collision_mean", "causal_collision_jet_potential",
                "causal_collision_maximum",
                "causal_phase_average_collision_maximum",
                "causal_collision_joint_median",
                "causal_phase_average_hj_collision_section",
                "causal_phase_average_hj_collision_barycenter",
                "causal_phase_average_hj_collision_w1_barycenter",
                "causal_phase_average_hj_ancestry_collision_barycenter",
                "causal_phase_average_hj_ancestry_collision_w1_barycenter",
                "causal_phase_average_hj_simplex_collision_barycenter",
                "causal_phase_average_hj_simplex_collision_w1_barycenter",
                "causal_phase_average_collision_mode",
                "integrated_fmmt")
        }
    differences = [
        item for row in rows for item in row["refinement_difference"]
    ]
    return {
        "purpose": (
            "integrate population realization as a numerical fibre before "
            "scalar projection; condition names remain outside the solver"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "conditions": list(selected_conditions),
        "phase_counts": list(phase_counts),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "summary": summary,
        "mean_refinement_rms": (
            float(np.mean([item["rms"] for item in differences]))
            if differences else None
        ),
        "maximum_refinement_rms": (
            float(np.max([item["rms"] for item in differences]))
            if differences else None
        ),
        "mean_collision_mean_refinement_rms": (
            float(np.mean([
                item["collision_mean_rms"] for item in differences]))
            if differences else None
        ),
        "mean_collision_jet_refinement_rms": (
            float(np.mean([
                item["collision_jet_rms"] for item in differences]))
            if differences else None
        ),
        "mean_collision_maximum_refinement_rms": (
            float(np.mean([
                item["collision_maximum_rms"] for item in differences]))
            if differences else None
        ),
        "mean_phase_collision_maximum_refinement_rms": (
            float(np.mean([
                item["phase_collision_maximum_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_collision_joint_median_refinement_rms": (
            float(np.mean([
                item["collision_joint_median_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_collision_refinement_rms": (
            float(np.mean([
                item["phase_hj_collision_rms"] for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_collision_barycenter_refinement_rms": (
            float(np.mean([
                item["phase_hj_collision_barycenter_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_collision_w1_refinement_rms": (
            float(np.mean([
                item["phase_hj_collision_w1_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_ancestry_collision_refinement_rms": (
            float(np.mean([
                item["phase_hj_ancestry_collision_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_ancestry_collision_w1_refinement_rms": (
            float(np.mean([
                item["phase_hj_ancestry_collision_w1_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_simplex_collision_refinement_rms": (
            float(np.mean([
                item["phase_hj_simplex_collision_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_hj_simplex_collision_w1_refinement_rms": (
            float(np.mean([
                item["phase_hj_simplex_collision_w1_rms"]
                for item in differences]))
            if differences else None
        ),
        "mean_phase_collision_mode_refinement_rms": (
            float(np.mean([
                item["phase_collision_mode_rms"] for item in differences]))
            if differences else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--sources", default="geometric interfaces")
    parser.add_argument("--conditions", default="clean,mixed 0.25")
    parser.add_argument(
        "--condition-catalog",
        choices=("selected", "full"),
        default="selected",
        help=(
            "full runs clean plus every external corruption control; labels "
            "never enter the solver"
        ),
    )
    parser.add_argument("--phase-counts", default="1,2,4,8")
    parser.add_argument("--angular-count", type=int, default=4)
    parser.add_argument("--quantile-count", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected_conditions = (
        tuple(row[0] for row in CONDITIONS)
        if args.condition_catalog == "full"
        else tuple(
            value.strip() for value in args.conditions.split(",")
            if value.strip())
    )
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        selected_conditions,
        tuple(int(value) for value in args.phase_counts.split(",") if value),
        args.angular_count,
        args.quantile_count,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "mean_refinement_rms": result["mean_refinement_rms"],
        "maximum_refinement_rms": result["maximum_refinement_rms"],
    }, indent=2))


if __name__ == "__main__":
    main()
