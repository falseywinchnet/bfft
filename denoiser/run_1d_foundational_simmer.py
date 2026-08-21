"""Matched 1-D falsification sweep for smoothing versus relation transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from .affine_relation_transport import RelationResolution, denoise_affine_relations
from .sample_series import PRESETS, compose_series, corrupt
from .transport_support import TransportResolution, denoise_1d


PRESET_NAMES = (
    "mixed transport stress",
    "smooth geometry",
    "step + carrier",
    "oscillatory composite",
)


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(estimate) - np.asarray(truth)
    truth_difference = np.diff(truth)
    estimate_difference = np.diff(estimate)
    truth_curvature = np.diff(truth, n=2)
    estimate_curvature = np.diff(estimate, n=2)
    return {
        "mse": float(np.mean(error * error)),
        "first_difference_mse": float(np.mean(
            (estimate_difference - truth_difference) ** 2)),
        "second_difference_mse": float(np.mean(
            (estimate_curvature - truth_curvature) ** 2)),
        "total_variation_ratio": float(
            np.sum(np.abs(estimate_difference))
            / max(float(np.sum(np.abs(truth_difference))), np.finfo(float).tiny)),
    }


def run(size: int, seeds: int) -> dict:
    support_resolution = TransportResolution(
        scale_samples=5, histogram_bins=32, maximum_steps=2048)
    relation_resolution = RelationResolution(
        maximum_lag=max(1, int(round(np.sqrt(size)))), scale_quadrature=12)

    def legacy(y):
        return denoise_1d(
            y, support_resolution, provisional_sigma=2.0,
            action_budget_multiplier=8.0, continuation_rounds=4)[0]

    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "observation": lambda y: y,
        "Gaussian sigma=2 control": lambda y: ndimage.gaussian_filter1d(
            y, 2.0, mode="reflect"),
        "median width=5 control": lambda y: ndimage.median_filter(
            y, size=5, mode="reflect"),
        "legacy Gaussian+support flow": legacy,
        "affine relation pushforward": lambda y: denoise_affine_relations(
            y, relation_resolution)[0],
    }
    rows = []
    clean_distortion = []
    horizon_sweep = []
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        for method, estimator in methods.items():
            clean_distortion.append({
                "preset": preset,
                "method": method,
                **metrics(estimator(truth.copy()), truth),
            })
        for lag in sorted({2, 4, 8, 16, int(round(np.sqrt(size)))}):
            relation = lambda value, lag=lag: denoise_affine_relations(
                value,
                RelationResolution(maximum_lag=lag, scale_quadrature=12),
            )[0]
            horizon_sweep.append({
                "preset": preset,
                "maximum_lag": lag,
                "condition": "clean",
                **metrics(relation(truth.copy()), truth),
            })
            noisy_metrics = []
            for seed in range(seeds):
                observation = corrupt(
                    truth, "mixed replacement + uniform",
                    amount=0.15, density=0.25, seed=7000 + seed)
                noisy_metrics.append(metrics(relation(observation), truth))
            horizon_sweep.append({
                "preset": preset,
                "maximum_lag": lag,
                "condition": "replacement density 0.25",
                **{
                    key: float(np.mean([record[key] for record in noisy_metrics]))
                    for key in noisy_metrics[0]
                },
            })
        for density in (0.10, 0.25, 0.40):
            for seed in range(seeds):
                observation = corrupt(
                    truth, "mixed replacement + uniform",
                    amount=0.15, density=density, seed=7000 + seed)
                for method, estimator in methods.items():
                    rows.append({
                        "preset": preset,
                        "density": density,
                        "seed": seed,
                        "method": method,
                        **metrics(estimator(observation), truth),
                    })
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in metrics(np.zeros(4), np.zeros(4))
        }
    return {
        "status": "falsification sweep; no method is promoted by this file",
        "corruption": "mixed replacement + uniform",
        "size": size,
        "seeds": seeds,
        "relation_resolution": {
            "maximum_lag": relation_resolution.maximum_lag,
            "scale_quadrature": relation_resolution.scale_quadrature,
        },
        "summary": summary,
        "clean_distortion": clean_distortion,
        "horizon_sweep": horizon_sweep,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    report = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
