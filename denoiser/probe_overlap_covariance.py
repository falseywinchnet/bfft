"""Focused falsification gate for lineage-overlap covariance authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    denoise_joint_lineage_covariance_transport_2d,
    joint_characteristic_measure_2d,
)


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(
    size: int,
    selected_sources: tuple[str, ...],
    ceiling: int,
    *,
    clean_only: bool = False,
) -> dict:
    catalogue = sources(size)
    unknown = sorted(set(selected_sources) - set(catalogue))
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        cases = [("clean", truth)]
        if not clean_only:
            cases.extend((
                name,
                corrupt(
                    truth, kind, amount=amount, density=density, seed=19000),
            ) for name, kind, amount, density in CONDITIONS)
        for condition, observation in cases:
            initial = joint_characteristic_measure_2d(
                observation, barycenter="median")[0]
            continued, diagnostic = (
                denoise_joint_lineage_covariance_transport_2d(
                    observation, maximum_continuations=ceiling)
            )
            empirical = denoise_fmmt(observation)[0]
            records = diagnostic["continuations"]
            rows.append({
                "source": source,
                "condition": condition,
                "joint_median": metrics(initial, truth),
                "overlap_covariance": metrics(continued, truth),
                "integrated_fmmt": metrics(empirical, truth),
                "accepted_continuations": diagnostic[
                    "accepted_continuations"],
                "ceiling_hit": diagnostic["continuation_ceiling_hit"],
                "initial_lineage_covariance": (
                    records[0]["mean_lineage_covariance"] if records else None),
                "initial_positive_authority_fraction": (
                    records[0]["positive_authority_fraction"]
                    if records else None),
            })
    corrupted = [row for row in rows if row["condition"] != "clean"]
    summarized = corrupted if corrupted else rows
    methods = ("joint_median", "overlap_covariance", "integrated_fmmt")
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    summary = {
        method: {
            metric: float(np.mean([row[method][metric] for row in summarized]))
            for metric in metric_names
        }
        for method in methods
    }
    return {
        "purpose": (
            "falsify lineage-local cross-fitted covariance before a broad gate"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "continuation_ceiling": int(ceiling),
        "summary_corrupted": summary,
        "ceiling_hits": int(sum(row["ceiling_hit"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair",
        help="comma-separated source names",
    )
    parser.add_argument("--continuations", type=int, default=8)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    result = run(
        args.size, selected, args.continuations, clean_only=args.clean_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary_corrupted"],
        "ceiling_hits": result["ceiling_hits"],
        "steps": [
            [row["source"], row["condition"], row["accepted_continuations"]]
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
