"""Diverse gate for continuous source-measure residual continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .continuous_source_transport import denoise_continuous_source_transport
from .cross_predictive_transport_2d import denoise_cross_predictive_transport_2d
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


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
        clean, clean_diagnostic = denoise_continuous_source_transport(truth)
        clean_rows.append({
            "source": source,
            "continuous_source": metrics(clean, truth),
            "accepted_continuations": clean_diagnostic[
                "accepted_continuations"],
            "ceiling_hit": clean_diagnostic["continuation_ceiling_hit"],
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=18000 + seed)
                continuous, diagnostic = denoise_continuous_source_transport(
                    observation)
                characteristic = denoise_cross_predictive_transport_2d(
                    observation)[0]
                empirical = denoise_fmmt(observation)[0]
                rows.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "observation": metrics(observation, truth),
                    "continuous_source": metrics(continuous, truth),
                    "four_direction": metrics(characteristic, truth),
                    "integrated_fmmt": metrics(empirical, truth),
                    "accepted_continuations": diagnostic[
                        "accepted_continuations"],
                    "ceiling_hit": diagnostic["continuation_ceiling_hit"],
                })
    methods = (
        "observation", "continuous_source", "four_direction", "integrated_fmmt")
    summary = {
        method: {
            metric: _mean(rows, method, metric)
            for metric in (
                "mse", "ssim", "variance_ratio", "central_range_ratio",
                "edge_retention", "mean_bias")
        }
        for method in methods
    }
    return {
        "purpose": (
            "gate continuous source ancestry and held-out error continuation "
            "before any authoritative 2-D battery"
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
        "gates": {
            "reaches_equilibrium": not any(row["ceiling_hit"] for row in rows),
            "beats_four_direction_mse": (
                summary["continuous_source"]["mse"]
                <= summary["four_direction"]["mse"]),
            "beats_four_direction_ssim": (
                summary["continuous_source"]["ssim"]
                >= summary["four_direction"]["ssim"]),
            "retains_half_variance": (
                summary["continuous_source"]["variance_ratio"] >= 0.5),
            "retains_half_range": (
                summary["continuous_source"]["central_range_ratio"] >= 0.5),
        },
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
        "gates": result["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
