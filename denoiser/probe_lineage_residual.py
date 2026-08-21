"""Matched gate for global-lane versus source-lineage residual priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    joint_characteristic_measure_2d,
    lineage_joint_characteristic_measure_2d,
    transported_lineage_joint_characteristic_measure_2d,
)


CONDITIONS = (
    ("clean", None, 0.0, 0.0),
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(size: int, selected_sources: tuple[str, ...]) -> dict:
    catalogue = sources(size)
    unknown = sorted(set(selected_sources) - set(catalogue))
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=19000)
            )
            global_median = joint_characteristic_measure_2d(
                observation, barycenter="median")[0]
            local_mean, diagnostic = lineage_joint_characteristic_measure_2d(
                observation, barycenter="mean")
            local_median = lineage_joint_characteristic_measure_2d(
                observation, barycenter="median")[0]
            transported_median, transported_diagnostic = (
                transported_lineage_joint_characteristic_measure_2d(
                    observation, barycenter="median")
            )
            rows.append({
                "source": source,
                "condition": condition,
                "global_lane_median": metrics(global_median, truth),
                "lineage_mean": metrics(local_mean, truth),
                "lineage_median": metrics(local_median, truth),
                "transported_lineage_median": metrics(
                    transported_median, truth),
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
                "mean_source_population": diagnostic[
                    "mean_residual_source_population"],
                "mean_collision_population": diagnostic[
                    "mean_residual_source_collision_population"],
                "accepted_lineage_transports": transported_diagnostic[
                    "accepted_lineage_transports"],
                "lineage_transport_ceiling_hit": transported_diagnostic[
                    "lineage_transport_ceiling_hit"],
            })
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    methods = (
        "global_lane_median", "lineage_mean", "lineage_median",
        "transported_lineage_median", "integrated_fmmt",
    )
    corrupted = [row for row in rows if row["condition"] != "clean"]
    summary = {
        method: {
            metric: float(np.mean([row[method][metric] for row in corrupted]))
            for metric in metric_names
        }
        for method in methods
    }
    return {
        "purpose": "test source-lineage residual transport against global lane pooling",
        "size": int(size),
        "sources": list(selected_sources),
        "summary_corrupted": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument(
        "--sources",
        default="cameraman,tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    result = run(args.size, selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary_corrupted"], indent=2))


if __name__ == "__main__":
    main()
