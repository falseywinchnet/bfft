"""Broad 1-D structure/corruption battery for relation-scale transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from .affine_relation_transport import denoise_affine_relations
from .cross_predictive_transport import denoise_cross_predictive_transport
from .sample_series import PRESETS, compose_series, corrupt
from .transport_support import TransportResolution, denoise_1d


PRESET_NAMES = (
    "mixed transport stress",
    "smooth geometry",
    "step + carrier",
    "chirp packet",
    "pulses + drift",
    "oscillatory composite",
)

# These names are diagnostic labels only.  The candidate receives only the
# resulting observation and has no corruption switch.
CONDITIONS = (
    ("uniform 0.08", "uniform additive", 0.08, 0.25),
    ("uniform 0.15", "uniform additive", 0.15, 0.25),
    ("Gaussian 0.15", "Gaussian additive", 0.15, 0.25),
    ("Laplace 0.12", "Laplace additive", 0.12, 0.25),
    ("multiplicative 0.15", "multiplicative", 0.15, 0.25),
    ("replacement 0.10", "random-value replacement", 0.15, 0.10),
    ("replacement 0.25", "random-value replacement", 0.15, 0.25),
    ("replacement 0.40", "random-value replacement", 0.15, 0.40),
    ("salt-pepper 0.10", "salt and pepper", 0.15, 0.10),
    ("salt-pepper 0.25", "salt and pepper", 0.15, 0.25),
    ("mixed 0.10", "mixed replacement + uniform", 0.15, 0.10),
    ("mixed 0.25", "mixed replacement + uniform", 0.15, 0.25),
    ("mixed 0.40", "mixed replacement + uniform", 0.15, 0.40),
)


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = estimate - truth
    truth_d1 = np.diff(truth)
    estimate_d1 = np.diff(estimate)
    truth_d2 = np.diff(truth, n=2)
    estimate_d2 = np.diff(estimate, n=2)
    truth_tv = float(np.sum(np.abs(truth_d1)))
    truth_variance = float(np.var(truth))
    truth_range = float(np.quantile(truth, 0.95) - np.quantile(truth, 0.05))
    return {
        "mse": float(np.mean(error * error)),
        "first_difference_mse": float(np.mean((estimate_d1 - truth_d1) ** 2)),
        "second_difference_mse": float(np.mean((estimate_d2 - truth_d2) ** 2)),
        "total_variation_ratio": float(
            np.sum(np.abs(estimate_d1)) / max(truth_tv, np.finfo(float).tiny)),
        "variance_ratio": float(
            np.var(estimate) / max(truth_variance, np.finfo(float).tiny)),
        "central_range_ratio": float(
            (np.quantile(estimate, 0.95) - np.quantile(estimate, 0.05))
            / max(truth_range, np.finfo(float).tiny)),
        "mean_bias": float(np.mean(estimate) - np.mean(truth)),
    }


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = tuple(metrics(np.zeros(8), np.zeros(8)))
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run(size: int, seeds: int) -> dict:
    if size < 32 or seeds < 1:
        raise ValueError("battery needs size >= 32 and at least one seed")
    support_resolution = TransportResolution(
        scale_samples=5, histogram_bins=32, maximum_steps=2048)

    def legacy(value: np.ndarray) -> np.ndarray:
        return denoise_1d(
            value,
            support_resolution,
            provisional_sigma=2.0,
            action_budget_multiplier=8.0,
            continuation_rounds=4,
        )[0]

    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "observation": lambda value: value.copy(),
        "Gaussian sigma=2 control": lambda value: ndimage.gaussian_filter1d(
            value, 2.0, mode="reflect"),
        "median width=5 control": lambda value: ndimage.median_filter(
            value, size=5, mode="reflect"),
        "legacy Gaussian+support flow": legacy,
        "fixed-horizon affine relation": lambda value: denoise_affine_relations(value)[0],
        "full-scale cross-predictive transport": lambda value: (
            denoise_cross_predictive_transport(value)[0]),
    }

    clean_rows = []
    noisy_rows = []
    continuation_records = []
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        for method, estimator in methods.items():
            clean_rows.append({
                "preset": preset,
                "method": method,
                **metrics(estimator(truth), truth),
            })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=8100 + seed,
                )
                for method, estimator in methods.items():
                    estimate = estimator(observation)
                    noisy_rows.append({
                        "preset": preset,
                        "condition": condition,
                        "seed": seed,
                        "method": method,
                        **metrics(estimate, truth),
                    })
                _estimate, diagnostic = denoise_cross_predictive_transport(observation)
                continuation_records.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    "accepted_continuations": diagnostic["accepted_continuations"],
                    "continuation_ceiling_hit": diagnostic["continuation_ceiling_hit"],
                    "final_residual_action": diagnostic["final_residual_action"],
                })

    clean_summary = {}
    noisy_summary = {}
    by_condition = {}
    by_preset = {}
    for method in methods:
        clean_summary[method] = _mean_metrics([
            row for row in clean_rows if row["method"] == method])
        noisy_summary[method] = _mean_metrics([
            row for row in noisy_rows if row["method"] == method])
    for condition, _kind, _amount, _density in CONDITIONS:
        by_condition[condition] = {
            method: _mean_metrics([
                row for row in noisy_rows
                if row["condition"] == condition and row["method"] == method
            ])
            for method in methods
        }
    for preset in PRESET_NAMES:
        by_preset[preset] = {
            method: _mean_metrics([
                row for row in noisy_rows
                if row["preset"] == preset and row["method"] == method
            ])
            for method in methods
        }

    candidate = "full-scale cross-predictive transport"
    comparison_methods = [name for name in methods if name != "observation"]
    case_keys = sorted({
        (row["preset"], row["condition"], row["seed"])
        for row in noisy_rows
    })
    mse_wins = {method: 0 for method in methods}
    for key in case_keys:
        case = [
            row for row in noisy_rows
            if (row["preset"], row["condition"], row["seed"]) == key
        ]
        winner = min(case, key=lambda row: row["mse"])["method"]
        mse_wins[winner] += 1

    candidate_clean = clean_summary[candidate]
    candidate_noisy = noisy_summary[candidate]
    gates = {
        "all_runs_reached_intrinsic_equilibrium": not any(
            record["continuation_ceiling_hit"] for record in continuation_records),
        "clean_mean_mse_below_2e-4": candidate_clean["mse"] < 2.0e-4,
        "clean_mean_tv_between_0.9_and_1.1": (
            0.9 < candidate_clean["total_variation_ratio"] < 1.1),
        "noisy_mean_mse_below_each_engineering_control": all(
            candidate_noisy["mse"] < noisy_summary[method]["mse"]
            for method in comparison_methods if method != candidate),
        "noisy_mean_variance_not_deflated_below_half_truth": (
            candidate_noisy["variance_ratio"] > 0.5),
    }
    return {
        "status": "broad falsification battery; runtime receives no truth or noise label",
        "size": int(size),
        "seeds": int(seeds),
        "presets": list(PRESET_NAMES),
        "conditions": [condition for condition, *_rest in CONDITIONS],
        "clean_summary": clean_summary,
        "noisy_summary": noisy_summary,
        "by_condition": by_condition,
        "by_preset": by_preset,
        "mse_case_wins": mse_wins,
        "gates": gates,
        "continuation_summary": {
            "mean_accepted": float(np.mean([
                record["accepted_continuations"]
                for record in continuation_records
            ])),
            "maximum_accepted": int(max(
                record["accepted_continuations"]
                for record in continuation_records
            )),
            "ceiling_hits": int(sum(
                record["continuation_ceiling_hit"]
                for record in continuation_records
            )),
        },
        "clean_rows": clean_rows,
        "noisy_rows": noisy_rows,
        "continuation_records": continuation_records,
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
    print(json.dumps({
        "clean": report["clean_summary"]["full-scale cross-predictive transport"],
        "noisy": report["noisy_summary"]["full-scale cross-predictive transport"],
        "wins": report["mse_case_wins"],
        "gates": report["gates"],
        "continuation": report["continuation_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
