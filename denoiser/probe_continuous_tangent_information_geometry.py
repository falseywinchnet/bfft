"""Probe support volume extracted from the continuous joint tangent law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_population_2d,
    continuous_tangent_jet_field_2d,
)
from .fused_transport_geometry import (
    predictive_directional_jet_sasaki_geometry,
    predictive_horizontal_wasserstein_geometry,
    predictive_jet_horizontal_wasserstein_geometry,
    predictive_lineage_jet_geometry,
    predictive_wasserstein_geometry,
    weighted_empirical_quantiles,
)
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    _source_influence_and_lineage,
)


def _geometry_summary(geometry: dict) -> dict[str, float]:
    xx = np.asarray(geometry["metric_xx"], dtype=np.float64)
    xy = np.asarray(geometry["metric_xy"], dtype=np.float64)
    yy = np.asarray(geometry["metric_yy"], dtype=np.float64)
    trace = xx + yy
    coherence = np.hypot(xx - yy, 2.0 * xy) / np.maximum(
        trace, np.finfo(float).tiny)
    return {
        "implied_support": float(geometry["implied_support"]),
        "information_trace_mean": float(geometry["information_trace_mean"]),
        "metric_coherence_mean": float(np.mean(coherence)),
        "metric_coherence_p90": float(np.quantile(coherence, 0.90)),
        "metric_determinant_max_error": float(np.max(np.abs(
            np.asarray(geometry["metric_determinant"]) - 1.0))),
    }


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    quantile_counts: tuple[int, ...],
) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=23017)
            )
            population, diagnostic = continuous_tangent_joint_population_2d(
                observation, angular_count=angular_count)
            gradient_x, gradient_y, jet_diagnostic = (
                continuous_tangent_jet_field_2d(population))
            prior_population = {**population, "mass": population["prior_mass"]}
            prior_gradient_x, prior_gradient_y, prior_jet_diagnostic = (
                continuous_tangent_jet_field_2d(prior_population))
            _influence, source_lineage = _source_influence_and_lineage(
                population["prior_mass"],
                population["source_identity"],
                population["source_coefficient"],
            )
            source_lineage = source_lineage.reshape(
                observation.shape + (observation.size,))
            resolutions = {}
            for count in quantile_counts:
                quantiles = weighted_empirical_quantiles(
                    population["signal"], population["mass"], count)
                prior_quantiles = weighted_empirical_quantiles(
                    population["signal"], population["prior_mass"], count)
                ordinary = predictive_wasserstein_geometry(quantiles)
                translated = predictive_horizontal_wasserstein_geometry(
                    quantiles)
                jet_horizontal = predictive_jet_horizontal_wasserstein_geometry(
                    population["signal"],
                    population["mass"],
                    gradient_x,
                    gradient_y,
                    quantile_count=count,
                )
                prior_translated = predictive_horizontal_wasserstein_geometry(
                    prior_quantiles)
                prior_jet_horizontal = (
                    predictive_jet_horizontal_wasserstein_geometry(
                        population["signal"],
                        population["prior_mass"],
                        prior_gradient_x,
                        prior_gradient_y,
                        quantile_count=count,
                    ))
                prior_sasaki = predictive_directional_jet_sasaki_geometry(
                    population["signal"],
                    population["prior_mass"],
                    population["directional_derivative"],
                    population["tangent"],
                    quantile_count=count,
                )
                lineage_jet = predictive_lineage_jet_geometry(
                    source_lineage,
                    prior_gradient_x,
                    prior_gradient_y,
                    quantile_count=count,
                )
                resolutions[str(count)] = {
                    "ordinary": _geometry_summary(ordinary),
                    "translation_quotient": _geometry_summary(translated),
                    "jet_horizontal": _geometry_summary(jet_horizontal),
                    "prior_translation_quotient": _geometry_summary(
                        prior_translated),
                    "prior_jet_horizontal": _geometry_summary(
                        prior_jet_horizontal),
                    "prior_directional_sasaki": {
                        **_geometry_summary(prior_sasaki),
                        "horizontal_signal_trace_mean": prior_sasaki[
                            "horizontal_signal_trace_mean"],
                        "vertical_jet_trace_mean": prior_sasaki[
                            "vertical_jet_trace_mean"],
                    },
                    "lineage_jet": _geometry_summary(lineage_jet),
                }
            rows.append({
                "source": source,
                "condition": condition,
                "resolutions": resolutions,
                "jet": jet_diagnostic,
                "prior_jet": prior_jet_diagnostic,
                "joint_target_identity_excluded": diagnostic[
                    "target_identity_excluded"],
            })

    geometries = (
        "ordinary",
        "translation_quotient",
        "jet_horizontal",
        "prior_translation_quotient",
        "prior_jet_horizontal",
        "prior_directional_sasaki",
        "lineage_jet",
    )
    clean = {
        (row["source"], count, geometry):
        row["resolutions"][count][geometry]["implied_support"]
        for row in rows if row["condition"] == "clean"
        for count in map(str, quantile_counts)
        for geometry in geometries
    }
    condition_summary = {}
    for condition, _kind, _amount, _density in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        condition_summary[condition] = {
            str(count): {
                geometry: {
                    "mean_implied_support": float(np.mean([
                        row["resolutions"][str(count)][geometry][
                            "implied_support"]
                        for row in condition_rows
                    ])),
                    "mean_ratio_to_source_clean": float(np.mean([
                        row["resolutions"][str(count)][geometry][
                            "implied_support"]
                        / clean[(row["source"], str(count), geometry)]
                        for row in condition_rows
                    ])),
                }
                for geometry in geometries
            }
            for count in quantile_counts
        }

    refinement = {}
    for previous, current in zip(quantile_counts, quantile_counts[1:]):
        key = f"{previous}->{current}"
        refinement[key] = {
            geometry: {
                "mean_relative_support_change": float(np.mean([
                    abs(
                        row["resolutions"][str(current)][geometry][
                            "implied_support"]
                        - row["resolutions"][str(previous)][geometry][
                            "implied_support"]
                    ) / max(
                        row["resolutions"][str(current)][geometry][
                            "implied_support"],
                        np.finfo(float).tiny,
                    )
                    for row in rows
                ])),
                "maximum_relative_support_change": float(np.max([
                    abs(
                        row["resolutions"][str(current)][geometry][
                            "implied_support"]
                        - row["resolutions"][str(previous)][geometry][
                            "implied_support"]
                    ) / max(
                        row["resolutions"][str(current)][geometry][
                            "implied_support"],
                        np.finfo(float).tiny,
                    )
                    for row in rows
                ])),
            }
            for geometry in geometries
        }
    return {
        "purpose": (
            "test whether posterior jet transport converts the continuous "
            "joint law into a refinement-stable, corruption-resistant support "
            "volume before any eikonal population is emitted"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "quantile_counts": list(quantile_counts),
        "condition_summary": condition_summary,
        "refinement": refinement,
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
    parser.add_argument("--quantile-counts", default="8,16,32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(
        value.strip() for value in args.sources.split(",") if value.strip())
    counts = tuple(
        int(value) for value in args.quantile_counts.split(",") if value.strip())
    result = run(args.size, selected, args.angular_count, counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "condition_summary": result["condition_summary"],
        "refinement": result["refinement"],
    }, indent=2))


if __name__ == "__main__":
    main()
