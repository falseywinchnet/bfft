"""Gate the 2-D Hopf--Lax information-lineage branch law across phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_lineage_readouts_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_continuous_tangent_convergence import CONDITIONS as MINIMAL_CONDITIONS
from .run_2d_denoiser_battery import (
    CONDITIONS as CORRUPTION_CONDITIONS,
    metrics,
    sources,
)
from .sample_series import corrupt


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    quantile_count: int,
    phases: tuple[float, ...],
    conditions=MINIMAL_CONDITIONS,
) -> dict:
    catalogue = sources(size)
    methods = (
        "local_median", "local_maximum", "causal_mean", "causal_median",
        "causal_maximum", "causal_collision_median",
        "causal_collision_mean", "causal_collision_maximum",
        "causal_cross_lineage_median", "integrated_fmmt",
    )
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in conditions:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=11107)
            )
            fmmt = denoise_fmmt(observation)[0]
            phase_rows = []
            causal_medians = []
            for phase in phases:
                forms, diagnostic = causal_information_lineage_readouts_2d(
                    observation,
                    angular_count=angular_count,
                    quantile_count=quantile_count,
                    population_phase=phase,
                )
                causal_medians.append(forms["causal_median"])
                phase_rows.append({
                    "phase": float(phase),
                    **{name: metrics(value, truth)
                       for name, value in forms.items()},
                    "integrated_fmmt": metrics(fmmt, truth),
                    "continuous_roots": diagnostic["continuous_root_count"],
                    "raster_roots": diagnostic["raster_root_count"],
                    "mean_branch_population": diagnostic[
                        "mean_branch_collision_population"],
                    "mean_bundle_anisotropy": diagnostic[
                        "mean_bundle_anisotropy"],
                    "mass_maximum_error": diagnostic["mass_maximum_error"],
                    "initial_implied_support": diagnostic[
                        "initial_implied_support"],
                    "causal_implied_support": diagnostic[
                        "causal_implied_support"],
                    "causal_measure_relative_rms": diagnostic[
                        "causal_measure_relative_rms"],
                })
            reference = causal_medians[0]
            phase_difference = [
                {
                    "phase": float(phase),
                    "rms": float(np.sqrt(np.mean((field - reference) ** 2))),
                    "maximum": float(np.max(np.abs(field - reference))),
                }
                for phase, field in zip(phases[1:], causal_medians[1:])
            ]
            rows.append({
                "source": source,
                "condition": condition,
                "phases": phase_rows,
                "phase_difference": phase_difference,
            })

    phase_records = [phase for row in rows for phase in row["phases"]]
    corrupted = [
        phase for row in rows if row["condition"] != "clean"
        for phase in row["phases"]
    ]

    def summarize(records):
        return {
            method: {
                key: float(np.mean([record[method][key] for record in records]))
                for key in records[0][method]
            }
            for method in methods
        }

    differences = [
        difference for row in rows for difference in row["phase_difference"]
    ]
    return {
        "purpose": (
            "test joint-information branch mass on exact Hopf--Lax parents "
            "before any 2-D continuation"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "phases": list(phases),
        "conditions": [condition[0] for condition in conditions],
        "summary": summarize(phase_records),
        "summary_corrupted": summarize(corrupted),
        "mean_phase_rms": (
            float(np.mean([row["rms"] for row in differences]))
            if differences else None
        ),
        "maximum_phase_rms": (
            float(np.max([row["rms"] for row in differences]))
            if differences else None
        ),
        "maximum_phase_point_difference": (
            float(np.max([row["maximum"] for row in differences]))
            if differences else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--angular-count", type=int, default=4)
    parser.add_argument("--quantile-count", type=int, default=16)
    parser.add_argument("--phases", default="0,0.125,0.25,0.375")
    parser.add_argument(
        "--condition-catalog", choices=("minimal", "full"), default="minimal",
        help="external stress-test catalogue; never passed to the solver",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.angular_count,
        args.quantile_count,
        tuple(float(value) for value in args.phases.split(",") if value),
        (
            MINIMAL_CONDITIONS
            if args.condition_catalog == "minimal"
            else (MINIMAL_CONDITIONS[0],) + CORRUPTION_CONDITIONS
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "summary_corrupted": result["summary_corrupted"],
        "mean_phase_rms": result["mean_phase_rms"],
        "maximum_phase_rms": result["maximum_phase_rms"],
    }, indent=2))


if __name__ == "__main__":
    main()
