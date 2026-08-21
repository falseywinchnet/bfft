"""Nested gauge-phase integral of the predictive causal affine atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_tangent_atlas_2d import continuous_tangent_causal_atlas_2d
from .continuous_tangent_transport_2d import continuous_tangent_joint_measure_2d
from .fmmt_certified import denoise_fmmt
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    quantile_count: int,
    phase_counts: tuple[int, ...],
) -> dict:
    maximum_count = max(phase_counts)
    if any(
        count < 2 or maximum_count % count != 0
        for count in phase_counts
    ):
        raise ValueError("phase counts must be nested divisors of the maximum")
    phases = np.arange(maximum_count, dtype=np.float64) / maximum_count
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=29047)
            )
            phase_fields = []
            phase_records = []
            for phase in phases:
                field, diagnostic = continuous_tangent_causal_atlas_2d(
                    observation,
                    angular_count=angular_count,
                    quantile_count=quantile_count,
                    population_phase=float(phase),
                )
                phase_fields.append(field)
                phase_records.append({
                    "phase": float(phase),
                    "realized_germs": int(
                        diagnostic["population"]["realized_cells"]),
                    "horizontal_implied_support": diagnostic[
                        "horizontal_implied_support"],
                    "causal_jet_implied_support": diagnostic[
                        "causal_jet_implied_support"],
                })
            fields = np.stack(phase_fields, axis=0)
            integrated = {}
            integrated_fields = {}
            for count in phase_counts:
                selected = fields[::maximum_count // count]
                mean = np.mean(selected, axis=0)
                integrated_fields[count] = mean
                integrated[str(count)] = {
                    "metrics": metrics(mean, truth),
                    "mean_phase_variance": float(np.mean(
                        (selected - mean[None, ...]) ** 2)),
                }
            difference = {}
            for previous, current in zip(phase_counts, phase_counts[1:]):
                delta = integrated_fields[current] - integrated_fields[previous]
                difference[f"{previous}->{current}"] = {
                    "rms": float(np.sqrt(np.mean(delta * delta))),
                    "maximum": float(np.max(np.abs(delta))),
                }
            tangent, _ = continuous_tangent_joint_measure_2d(
                observation,
                barycenter="median",
                angular_count=angular_count,
            )
            rows.append({
                "source": source,
                "condition": condition,
                "phase_realizations": phase_records,
                "integrated": integrated,
                "successive_difference": difference,
                "continuous_tangent": metrics(tangent, truth),
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
            })

    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    summary = {
        str(count): {
            metric: float(np.mean([
                row["integrated"][str(count)]["metrics"][metric]
                for row in rows
            ]))
            for metric in metric_names
        }
        for count in phase_counts
    }
    for baseline in ("continuous_tangent", "integrated_fmmt"):
        summary[baseline] = {
            metric: float(np.mean([
                row[baseline][metric] for row in rows
            ]))
            for metric in metric_names
        }
    convergence = {
        pair: {
            "mean_rms": float(np.mean([
                row["successive_difference"][pair]["rms"] for row in rows
            ])),
            "maximum_rms": float(np.max([
                row["successive_difference"][pair]["rms"] for row in rows
            ])),
            "maximum_point_difference": float(np.max([
                row["successive_difference"][pair]["maximum"] for row in rows
            ])),
        }
        for pair in rows[0]["successive_difference"]
    }
    return {
        "purpose": (
            "test the population phase as a periodic gauge by nested "
            "quadrature instead of selecting one raster germ realization"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "phase_counts": list(phase_counts),
        "summary": summary,
        "phase_convergence": convergence,
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
    parser.add_argument("--phase-counts", default="4,8,16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(
        value.strip() for value in args.sources.split(",") if value.strip())
    counts = tuple(
        int(value) for value in args.phase_counts.split(",") if value.strip())
    result = run(
        args.size,
        selected,
        args.angular_count,
        args.quantile_count,
        counts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "phase_convergence": result["phase_convergence"],
    }, indent=2))


if __name__ == "__main__":
    main()
