"""Measure the first direct eikonal compressed-sensing observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compressed_eikonal_observer_2d import (
    compressed_eikonal_observation_2d,
    cross_measured_eikonal_observation_2d,
    phase_ordered_cross_observation_2d,
    phase_resolved_eikonal_observation_2d,
    phase_union_eikonal_observation_2d,
    pursue_compressed_eikonal_scene_2d,
    screened_selling_posterior_observation_2d,
)
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(size: int, selected: tuple[str, ...]) -> dict:
    rows = []
    catalogue = sources(size)
    rng = np.random.default_rng(173)
    cases = []
    for source in selected:
        truth = catalogue[source]
        cases.extend((
            (source, "clean", truth, truth),
            (
                source,
                "mixed replacement + uniform 0.25",
                truth,
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=271828,
                ),
            ),
        ))
    cases.extend((
        ("null", "zero-mean Gaussian", np.zeros((size, size)),
         rng.normal(size=(size, size))),
        ("null", "uniform", np.full((size, size), 0.5),
         rng.random((size, size))),
    ))

    for source, condition, truth, observation in cases:
        first, first_residual, first_diagnostic = (
            compressed_eikonal_observation_2d(observation))
        cross_prior, cross_residual, cross_diagnostic = (
            cross_measured_eikonal_observation_2d(observation))
        phase_prior, phase_residual, phase_diagnostic = (
            phase_resolved_eikonal_observation_2d(observation))
        ordered_prior, ordered_residual, ordered_diagnostic = (
            phase_ordered_cross_observation_2d(observation))
        union_prior, union_residual, union_diagnostic = (
            phase_union_eikonal_observation_2d(observation))
        posterior, posterior_residual, posterior_diagnostic = (
            screened_selling_posterior_observation_2d(observation))
        global_cross_prior = cross_diagnostic["readouts"][
            "global_cross_prior"]
        local_cross_prior = cross_diagnostic["readouts"][
            "local_cross_prior"]
        pursuit, pursuit_diagnostic = pursue_compressed_eikonal_scene_2d(
            observation)
        rows.append({
            "source": source,
            "condition": condition,
            "observation": metrics(observation, truth),
            "first_explanation": metrics(first, truth),
            "transported_cross_prior": metrics(cross_prior, truth),
            "phase_resolved_prior": metrics(phase_prior, truth),
            "phase_ordered_cross_prior": metrics(ordered_prior, truth),
            "phase_union_prior": metrics(union_prior, truth),
            "screened_selling_posterior": metrics(posterior, truth),
            "local_cross_prior": metrics(local_cross_prior, truth),
            "global_cross_prior": metrics(global_cross_prior, truth),
            "stopped_pursuit": metrics(pursuit, truth),
            "first": {
                key: value
                for key, value in first_diagnostic.items()
                if key not in {"forest", "geometry", "centers"}
            },
            "pursuit": {
                key: value
                for key, value in pursuit_diagnostic.items()
                if key not in {"explained_scene", "unexplained_scene"}
            },
            "cross_measure": {
                key: value
                for key, value in cross_diagnostic.items()
                if key not in {"charts", "readouts"}
            },
            "cross_charts": [
                {
                    key: value
                    for key, value in chart.items()
                    if key not in {"forest", "geometry", "centers"}
                }
                for chart in cross_diagnostic["charts"]
            ],
            "phase_measure": {
                key: value
                for key, value in phase_diagnostic.items()
                if key not in {
                    "charts", "local_phase_order", "local_phase_authority"
                }
            },
            "phase_charts": [
                {
                    key: value
                    for key, value in chart.items()
                    if key not in {
                        "forest", "geometry", "centers", "pixel_phase"
                    }
                }
                for chart in phase_diagnostic["charts"]
            ],
            "phase_ordered": {
                key: value
                for key, value in ordered_diagnostic.items()
                if key not in {
                    "readouts", "local_phase_order", "local_phase_authority"
                }
            },
            "phase_union": {
                key: value
                for key, value in union_diagnostic.items()
                if key not in {
                    "local_phase_authority", "cross_barycenter",
                    "covector_diagnostics",
                }
            },
            "screened_posterior": {
                key: value
                for key, value in posterior_diagnostic.items()
                if key not in {
                    "support_prior", "support_prior_diagnostic",
                    "residual_component", "residual_component_diagnostic",
                    "posterior_correction", "unresolved_before_posterior",
                    "local_phase_order", "local_phase_authority",
                    "posterior_authority",
                }
            },
            "first_residual_rms": float(np.sqrt(np.mean(
                first_residual * first_residual))),
            "cross_residual_rms": float(np.sqrt(np.mean(
                cross_residual * cross_residual))),
            "phase_residual_rms": float(np.sqrt(np.mean(
                phase_residual * phase_residual))),
            "phase_ordered_residual_rms": float(np.sqrt(np.mean(
                ordered_residual * ordered_residual))),
            "phase_union_residual_rms": float(np.sqrt(np.mean(
                union_residual * union_residual))),
            "posterior_residual_rms": float(np.sqrt(np.mean(
                posterior_residual * posterior_residual))),
        })
    return {
        "purpose": (
            "test direct compressed scene explanation from V3 transport "
            "support before introducing a posterior or residual smoother"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
        "theory_gate": (
            "structure must show explanation yield materially above null "
            "fields; self-measured support bias is measured, not ignored"
        ),
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
    print(json.dumps({
        "rows": [
            {
                "source": row["source"],
                "condition": row["condition"],
                "yield": row["first"][
                    "dimension_corrected_explanation_yield"],
                "compression_ratio": row["first"]["compression_ratio"],
                "accepted_observations": row["pursuit"][
                    "accepted_observations"],
                "first_mse": row["first_explanation"]["mse"],
                "local_cross_mse": row["local_cross_prior"]["mse"],
                "transported_cross_mse": row[
                    "transported_cross_prior"]["mse"],
                "phase_resolved_mse": row["phase_resolved_prior"]["mse"],
                "phase_ordered_cross_mse": row[
                    "phase_ordered_cross_prior"]["mse"],
                "phase_union_mse": row["phase_union_prior"]["mse"],
                "screened_selling_mse": row[
                    "screened_selling_posterior"]["mse"],
                "global_cross_mse": row["global_cross_prior"]["mse"],
                "mean_local_cross_gain": row["cross_measure"][
                    "mean_local_cross_gain"],
                "mean_absolute_phase": row["phase_measure"][
                    "mean_absolute_phase"],
                "phase_order": row["phase_ordered"]["phase_order"],
                "phase_union_order": row["phase_union"][
                    "phase_union_order"],
            }
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
