"""Matched gate for convergent tangent lineage-covariance continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_measure_2d,
    continuous_tangent_jet_projection_2d,
    denoise_continuous_tangent_lineage_covariance_2d,
)
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    denoise_joint_lineage_covariance_transport_2d,
)


CONDITIONS = (
    ("clean", None, 0.0, 0.0),
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    ceiling: int,
) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=19000)
            )
            continuous_initial = continuous_tangent_joint_measure_2d(
                observation,
                barycenter="median",
                angular_count=angular_count,
            )[0]
            continuous, diagnostic = (
                denoise_continuous_tangent_lineage_covariance_2d(
                    observation,
                    angular_count=angular_count,
                    maximum_continuations=ceiling,
                )
            )
            jet_projection = continuous_tangent_jet_projection_2d(
                observation, angular_count=angular_count)[0]
            crystalline = denoise_joint_lineage_covariance_transport_2d(
                observation, maximum_continuations=ceiling)[0]
            rows.append({
                "source": source,
                "condition": condition,
                "continuous_initial": metrics(continuous_initial, truth),
                "continuous_covariance": metrics(continuous, truth),
                "continuous_jet_projection": metrics(jet_projection, truth),
                "crystalline_covariance": metrics(crystalline, truth),
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
                "accepted_continuations": diagnostic["accepted_continuations"],
                "ceiling_hit": diagnostic["continuation_ceiling_hit"],
            })
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    methods = (
        "continuous_initial", "continuous_covariance",
        "continuous_jet_projection", "crystalline_covariance",
        "integrated_fmmt",
    )
    summary = {
        method: {
            metric: float(np.mean([row[method][metric] for row in rows]))
            for metric in metric_names
        }
        for method in methods
    }
    return {
        "purpose": (
            "test convergent common-scale tangents under fixed lineage covariance law"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "continuation_ceiling": int(ceiling),
        "summary": summary,
        "ceiling_hits": int(sum(row["ceiling_hit"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources", default="geometric interfaces,woven chirps")
    parser.add_argument("--angular-count", type=int, default=16)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    result = run(
        args.size, selected, args.angular_count, args.continuations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "ceiling_hits": result["ceiling_hits"],
        "steps": [
            [row["source"], row["condition"], row["accepted_continuations"]]
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
