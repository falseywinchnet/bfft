"""Diverse falsification gate for strict direction-lane cross-fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_measure_2d,
    denoise_crossfit_characteristic_transport_2d,
)
from .cross_predictive_transport_2d import denoise_cross_predictive_transport_2d
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    joint_characteristic_measure_2d,
    witnessed_characteristic_measure_2d,
)


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def _mean(rows: list[dict], method: str, metric: str) -> float:
    return float(np.mean([row[method][metric] for row in rows]))


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    for source, truth in sources(size).items():
        strict_mean = crossfit_characteristic_measure_2d(
            truth, barycenter="mean")[0]
        strict_median = crossfit_characteristic_measure_2d(
            truth, barycenter="median")[0]
        continued, diagnostic = denoise_crossfit_characteristic_transport_2d(
            truth)
        witnessed_mean = witnessed_characteristic_measure_2d(
            truth, barycenter="mean")[0]
        witnessed_median = witnessed_characteristic_measure_2d(
            truth, barycenter="median")[0]
        joint_mean = joint_characteristic_measure_2d(
            truth, barycenter="mean")[0]
        joint_median = joint_characteristic_measure_2d(
            truth, barycenter="median")[0]
        clean_rows.append({
            "source": source,
            "strict_mean": metrics(strict_mean, truth),
            "strict_median": metrics(strict_median, truth),
            "strict_continued": metrics(continued, truth),
            "witnessed_mean": metrics(witnessed_mean, truth),
            "witnessed_median": metrics(witnessed_median, truth),
            "joint_mean": metrics(joint_mean, truth),
            "joint_median": metrics(joint_median, truth),
            "accepted_continuations": diagnostic["accepted_continuations"],
            "ceiling_hit": diagnostic["continuation_ceiling_hit"],
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=19000 + seed)
                strict_mean = crossfit_characteristic_measure_2d(
                    observation, barycenter="mean")[0]
                strict_median = crossfit_characteristic_measure_2d(
                    observation, barycenter="median")[0]
                continued, diagnostic = (
                    denoise_crossfit_characteristic_transport_2d(observation)
                )
                witnessed_mean = witnessed_characteristic_measure_2d(
                    observation, barycenter="mean")[0]
                witnessed_median = witnessed_characteristic_measure_2d(
                    observation, barycenter="median")[0]
                joint_mean = joint_characteristic_measure_2d(
                    observation, barycenter="mean")[0]
                joint_median = joint_characteristic_measure_2d(
                    observation, barycenter="median")[0]
                characteristic = denoise_cross_predictive_transport_2d(
                    observation)[0]
                empirical = denoise_fmmt(observation)[0]
                rows.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "observation": metrics(observation, truth),
                    "strict_mean": metrics(strict_mean, truth),
                    "strict_median": metrics(strict_median, truth),
                    "strict_continued": metrics(continued, truth),
                    "witnessed_mean": metrics(witnessed_mean, truth),
                    "witnessed_median": metrics(witnessed_median, truth),
                    "joint_mean": metrics(joint_mean, truth),
                    "joint_median": metrics(joint_median, truth),
                    "four_direction": metrics(characteristic, truth),
                    "integrated_fmmt": metrics(empirical, truth),
                    "accepted_continuations": diagnostic[
                        "accepted_continuations"],
                    "ceiling_hit": diagnostic["continuation_ceiling_hit"],
                })
    methods = (
        "observation", "strict_mean", "strict_median", "strict_continued",
        "witnessed_mean", "witnessed_median", "joint_mean", "joint_median",
        "four_direction", "integrated_fmmt",
    )
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias",
    )
    summary = {
        method: {
            metric: _mean(rows, method, metric) for metric in metric_names
        }
        for method in methods
    }
    return {
        "purpose": (
            "falsify strict target-independent direction-lane cross-fitting "
            "before combining it with source-measure continuation"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "summary": summary,
        "continuation_summary": {
            "mean_accepted": float(np.mean([
                row["accepted_continuations"] for row in rows])),
            "maximum_accepted": int(max(
                row["accepted_continuations"] for row in rows)),
            "ceiling_hits": int(sum(row["ceiling_hit"] for row in rows)),
        },
        "clean_rows": clean_rows,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "continuation": result["continuation_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
