"""Convergence gate for common-radius target-free tangent quadrature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_measure_2d,
)
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


CONDITIONS = (
    ("clean", None, 0.0, 0.0),
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_counts: tuple[int, ...],
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
            fields = {}
            count_metrics = {}
            proposal_counts = {}
            for count in angular_counts:
                field, diagnostic = continuous_tangent_joint_measure_2d(
                    observation,
                    barycenter="median",
                    angular_count=count,
                )
                fields[count] = field
                count_metrics[str(count)] = metrics(field, truth)
                proposal_counts[str(count)] = diagnostic["proposal"][
                    "proposal_count"]
            difference = {}
            for previous, current in zip(angular_counts, angular_counts[1:]):
                delta = fields[current] - fields[previous]
                difference[f"{previous}->{current}"] = {
                    "rms": float(np.sqrt(np.mean(delta * delta))),
                    "maximum": float(np.max(np.abs(delta))),
                }
            rows.append({
                "source": source,
                "condition": condition,
                "angular_counts": count_metrics,
                "proposal_counts": proposal_counts,
                "successive_field_difference": difference,
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
            })
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    summary = {
        str(count): {
            metric: float(np.mean([
                row["angular_counts"][str(count)][metric] for row in rows
            ]))
            for metric in metric_names
        }
        for count in angular_counts
    }
    summary["integrated_fmmt"] = {
        metric: float(np.mean([row["integrated_fmmt"][metric] for row in rows]))
        for metric in metric_names
    }
    differences = {
        pair: {
            "mean_rms": float(np.mean([
                row["successive_field_difference"][pair]["rms"]
                for row in rows
            ])),
            "maximum": float(np.max([
                row["successive_field_difference"][pair]["maximum"]
                for row in rows
            ])),
        }
        for pair in rows[0]["successive_field_difference"]
    }
    return {
        "purpose": "test common-physical-scale projective tangent convergence",
        "size": int(size),
        "sources": list(selected_sources),
        "angular_counts": list(angular_counts),
        "summary": summary,
        "successive_field_difference": differences,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="cameraman,tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--angular-counts", default="4,8,16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    counts = tuple(int(value) for value in args.angular_counts.split(",") if value.strip())
    result = run(args.size, selected, counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "field_difference": result["successive_field_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()
