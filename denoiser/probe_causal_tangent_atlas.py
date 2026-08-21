"""Population-phase gate for the continuous predictive Hopf--Lax atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_tangent_atlas_2d import continuous_tangent_causal_atlas_2d
from .fmmt_certified import denoise_fmmt
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    quantile_count: int,
    phases: tuple[float, ...],
) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=27031)
            )
            phase_rows = []
            fields = []
            for phase in phases:
                field, diagnostic = continuous_tangent_causal_atlas_2d(
                    observation,
                    angular_count=angular_count,
                    quantile_count=quantile_count,
                    population_phase=phase,
                )
                fields.append(field)
                phase_rows.append({
                    "phase": phase,
                    "metrics": metrics(field, truth),
                    "realized_germs": int(
                        diagnostic["population"]["realized_cells"]),
                    "horizontal_implied_support": diagnostic[
                        "horizontal_implied_support"],
                    "causal_jet_implied_support": diagnostic[
                        "causal_jet_implied_support"],
                    "mean_collision_population": diagnostic[
                        "mean_collision_population"],
                    "maximum_collision_population": diagnostic[
                        "maximum_collision_population"],
                    "population_quantization_error": diagnostic[
                        "population"]["quantization_error"],
                    "centers": np.asarray(diagnostic["centers"]).tolist(),
                    "front_pushes": int(diagnostic["forest"]["front_pushes"]),
                })
            reference = fields[0]
            phase_difference = []
            for phase, field in zip(phases[1:], fields[1:]):
                delta = field - reference
                phase_difference.append({
                    "phase": phase,
                    "rms_from_first_phase": float(np.sqrt(np.mean(delta * delta))),
                    "maximum_from_first_phase": float(np.max(np.abs(delta))),
                })
            rows.append({
                "source": source,
                "condition": condition,
                "phases": phase_rows,
                "phase_difference": phase_difference,
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
            })

    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    summary = {}
    for condition, _kind, _amount, _density in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        phase_metrics = [
            phase_row["metrics"]
            for row in selected for phase_row in row["phases"]
        ]
        phase_rms = [
            difference["rms_from_first_phase"]
            for row in selected for difference in row["phase_difference"]
        ]
        summary[condition] = {
            "causal_atlas": {
                metric: float(np.mean([
                    record[metric] for record in phase_metrics
                ]))
                for metric in metric_names
            },
            "integrated_fmmt": {
                metric: float(np.mean([
                    row["integrated_fmmt"][metric] for row in selected
                ]))
                for metric in metric_names
            },
            "mean_realized_germs": float(np.mean([
                phase_row["realized_germs"]
                for row in selected for phase_row in row["phases"]
            ])),
            "minimum_realized_germs": int(np.min([
                phase_row["realized_germs"]
                for row in selected for phase_row in row["phases"]
            ])),
            "maximum_realized_germs": int(np.max([
                phase_row["realized_germs"]
                for row in selected for phase_row in row["phases"]
            ])),
            "mean_phase_rms": float(np.mean(phase_rms)),
            "maximum_phase_rms": float(np.max(phase_rms)),
            "maximum_phase_point_difference": float(np.max([
                difference["maximum_from_first_phase"]
                for row in selected for difference in row["phase_difference"]
            ])),
        }
    return {
        "purpose": (
            "test whether continuous predictive information volume and its "
            "shared-label Hopf--Lax ancestry define a germ-phase-stable affine "
            "atlas before residual transport"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "population_phases": list(phases),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--angular-count", type=int, default=16)
    parser.add_argument("--quantile-count", type=int, default=32)
    parser.add_argument("--phases", default="0,0.125,0.25,0.375")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(
        value.strip() for value in args.sources.split(",") if value.strip())
    phases = tuple(
        float(value) for value in args.phases.split(",") if value.strip())
    result = run(
        args.size,
        selected,
        args.angular_count,
        args.quantile_count,
        phases,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
