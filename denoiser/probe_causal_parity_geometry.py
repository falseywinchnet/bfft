"""Falsify parity value readout and measure shared-label ancestry geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_parity_transport import denoise_causal_parity_transport
from .cross_predictive_transport_2d import denoise_cross_predictive_transport_2d
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def _population_record(
    population: np.ndarray,
    truth: np.ndarray,
    clean_population: np.ndarray,
) -> dict[str, float]:
    gradient_y, gradient_x = np.gradient(truth)
    edge = np.hypot(gradient_x, gradient_y)
    edge_mask = edge >= np.quantile(edge, 0.90)
    flat_mask = edge <= np.quantile(edge, 0.50)
    flat_population = max(float(np.mean(population[flat_mask])), 1e-30)
    centered_population = population.ravel() - float(np.mean(population))
    centered_clean = clean_population.ravel() - float(np.mean(clean_population))
    denominator = float(np.linalg.norm(centered_population) *
                        np.linalg.norm(centered_clean))
    correlation = (
        1.0 if denominator == 0.0
        else float(np.dot(centered_population, centered_clean) / denominator)
    )
    return {
        "mean": float(np.mean(population)),
        "p90": float(np.quantile(population, 0.90)),
        "maximum": float(np.max(population)),
        "relative_l1_to_clean": float(
            np.mean(np.abs(population - clean_population))
            / max(float(np.mean(clean_population)), 1e-30)),
        "correlation_to_clean": correlation,
        "edge_to_flat_ratio": float(
            np.mean(population[edge_mask]) / flat_population),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    for source, truth in sources(size).items():
        clean_estimate, clean_diagnostic = denoise_causal_parity_transport(truth)
        clean_population = clean_diagnostic["collision_population"]
        clean_rows.append({
            "source": source,
            **metrics(clean_estimate, truth),
            "population": _population_record(
                clean_population, truth, clean_population),
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=12000 + seed,
                )
                parity, diagnostic = denoise_causal_parity_transport(observation)
                characteristic = denoise_cross_predictive_transport_2d(
                    observation)[0]
                empirical = denoise_fmmt(observation)[0]
                rows.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "parity_readout": metrics(parity, truth),
                    "four_direction_readout": metrics(characteristic, truth),
                    "integrated_fmmt": metrics(empirical, truth),
                    "population": _population_record(
                        diagnostic["collision_population"],
                        truth,
                        clean_population,
                    ),
                    "metric_determinant_max_error": diagnostic[
                        "metric_determinant_max_error"],
                    "lanes": diagnostic["lanes"],
                })
    method_names = (
        "parity_readout", "four_direction_readout", "integrated_fmmt")
    summary = {
        method: {
            metric: _mean([row[method] for row in rows], metric)
            for metric in (
                "mse", "ssim", "variance_ratio", "central_range_ratio",
                "edge_retention", "mean_bias")
        }
        for method in method_names
    }
    population_summary = {
        key: _mean([row["population"] for row in rows], key)
        for key in (
            "mean", "p90", "maximum", "relative_l1_to_clean",
            "correlation_to_clean", "edge_to_flat_ratio")
    }
    return {
        "purpose": (
            "test exact shared-label ancestry as population geometry and "
            "falsify its direct value barycenter"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "sources": list(sources(size)),
        "conditions": [condition for condition, *_ in CONDITIONS],
        "summary": summary,
        "population_summary": population_summary,
        "clean_rows": clean_rows,
        "rows": rows,
        "verdict_rule": (
            "direct readout is rejected if it loses both MSE and SSIM to the "
            "four-direction seed; ancestry geometry is retained only if its "
            "population remains correlated under corruption and distinguishes "
            "interfaces from flat regions"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "population_summary": result["population_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
