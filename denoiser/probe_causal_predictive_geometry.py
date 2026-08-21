"""Measure corruption and quantile-refinement stability of causal laws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_predictive_geometry import causal_parity_predictive_geometry
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def _record(observation: np.ndarray, count: int) -> dict[str, float]:
    _particles, diagnostic = causal_parity_predictive_geometry(
        observation, quantile_count=count)
    geometry = diagnostic["predictive_geometry"]
    horizontal = diagnostic["horizontal_predictive_geometry"]
    return {
        "ordinary_implied_support": float(geometry["implied_support"]),
        "ordinary_information_trace_mean": float(
            geometry["information_trace_mean"]),
        "horizontal_implied_support": float(horizontal["implied_support"]),
        "horizontal_information_trace_mean": float(
            horizontal["information_trace_mean"]),
        "ordinary_metric_determinant_max_error": float(np.max(np.abs(
            geometry["metric_determinant"] - 1.0))),
        "horizontal_metric_determinant_max_error": float(np.max(np.abs(
            horizontal["metric_determinant"] - 1.0))),
    }


def run(size: int, seeds: int) -> dict:
    counts = (16, 32)
    records = []
    for source, truth in sources(size).items():
        clean = {count: _record(truth, count) for count in counts}
        records.append({
            "source": source,
            "condition": "clean",
            "seed": None,
            "resolutions": clean,
            "support_ratio_to_clean": {
                count: {"ordinary": 1.0, "horizontal": 1.0}
                for count in counts
            },
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=14000 + seed)
                measured = {
                    count: _record(observation, count) for count in counts}
                records.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "resolutions": measured,
                    "support_ratio_to_clean": {
                        count: {
                            geometry: (
                                measured[count][f"{geometry}_implied_support"]
                                / clean[count][f"{geometry}_implied_support"])
                            for geometry in ("ordinary", "horizontal")
                        }
                        for count in counts
                    },
                })
    noisy = [row for row in records if row["condition"] != "clean"]
    ratios = {
        geometry: {
            count: [
                row["support_ratio_to_clean"][count][geometry]
                for row in noisy
            ]
            for count in counts
        }
        for geometry in ("ordinary", "horizontal")
    }
    refinement = {"ordinary": [], "horizontal": []}
    for row in records:
        for geometry in refinement:
            low = row["resolutions"][16][f"{geometry}_implied_support"]
            high = row["resolutions"][32][f"{geometry}_implied_support"]
            refinement[geometry].append(abs(high - low) / max(high, 1e-30))
    return {
        "purpose": (
            "test whether exact ancestry transport creates a corruption-stable "
            "predictive law before horizontal Wasserstein transport"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "quantile_resolutions": list(counts),
        "summary": {
            "mean_population_ratio_to_clean": {
                geometry: {
                    count: float(np.mean(value))
                    for count, value in by_count.items()
                }
                for geometry, by_count in ratios.items()
            },
            "maximum_population_ratio_to_clean": {
                geometry: {
                    count: float(np.max(value))
                    for count, value in by_count.items()
                }
                for geometry, by_count in ratios.items()
            },
            "mean_relative_16_to_32_change": {
                geometry: float(np.mean(value))
                for geometry, value in refinement.items()
            },
            "maximum_relative_16_to_32_change": {
                geometry: float(np.max(value))
                for geometry, value in refinement.items()
            },
        },
        "records": records,
        "verdict_rule": (
            "retain only if quantile refinement is convergent and corruption "
            "does not recreate the false population explosion of the raw "
            "leave-one-out predictive seed"
        ),
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
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
