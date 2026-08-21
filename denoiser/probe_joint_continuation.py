"""Focused gate for self-similar joint residual continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import (
    denoise_joint_source_authority_transport_2d,
    joint_characteristic_measure_2d,
)


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    for source, truth in sources(size).items():
        median = joint_characteristic_measure_2d(
            truth, barycenter="median")[0]
        authority, authority_diagnostic = (
            denoise_joint_source_authority_transport_2d(truth)
        )
        clean_rows.append({
            "source": source,
            "joint_median": metrics(median, truth),
            "joint_authority": metrics(authority, truth),
            "authority_accepted_continuations": authority_diagnostic[
                "accepted_continuations"],
            "authority_ceiling_hit": authority_diagnostic[
                "continuation_ceiling_hit"],
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=19000 + seed)
                median = joint_characteristic_measure_2d(
                    observation, barycenter="median")[0]
                authority, authority_diagnostic = (
                    denoise_joint_source_authority_transport_2d(observation)
                )
                empirical = denoise_fmmt(observation)[0]
                rows.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "joint_median": metrics(median, truth),
                    "joint_authority": metrics(authority, truth),
                    "integrated_fmmt": metrics(empirical, truth),
                    "authority_accepted_continuations": authority_diagnostic[
                        "accepted_continuations"],
                    "authority_ceiling_hit": authority_diagnostic[
                        "continuation_ceiling_hit"],
                })
    methods = (
        "joint_median", "joint_authority", "integrated_fmmt")
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias",
    )
    summary = {
        method: {
            metric: float(np.mean([row[method][metric] for row in rows]))
            for metric in metric_names
        }
        for method in methods
    }
    return {
        "purpose": "test same-law joint residual continuation at covariance equilibrium",
        "size": int(size),
        "seeds": int(seeds),
        "summary": summary,
        "continuation_summary": {
            "authority_mean_accepted": float(np.mean([
                row["authority_accepted_continuations"] for row in rows])),
            "authority_maximum_accepted": int(max(
                row["authority_accepted_continuations"] for row in rows)),
            "authority_ceiling_hits": int(sum(
                row["authority_ceiling_hit"] for row in rows)),
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
