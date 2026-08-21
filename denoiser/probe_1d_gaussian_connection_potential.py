"""Broad gate for the continuous Gaussian connection-potential transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    gaussian_connection_potential_readout_forms,
    lineage_branch_transport_1d,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "fused_collision_mean",
    "gaussian_potential_mean",
    "gaussian_potential_collision_mean",
    "gaussian_potential_path_collision_mean",
)


def _baseline_collision(value: np.ndarray) -> np.ndarray:
    law, _diagnostic = lineage_branch_transport_1d(value)
    prediction = law["prediction"]
    reference_mass = law["reference_mass"]
    reference = (
        np.broadcast_to(reference_mass, prediction.shape)
        if reference_mass.ndim == 1 else reference_mass)
    mass = law["mass"]
    collision = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision /= np.sum(collision, axis=1, keepdims=True)
    return np.sum(collision * prediction, axis=1)


def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    candidate, diagnostic = gaussian_connection_potential_readout_forms(value)
    return {
        "fused_collision_mean": _baseline_collision(value),
        "gaussian_potential_mean": candidate["mean"],
        "gaussian_potential_collision_mean": candidate["collision_mean"],
        "gaussian_potential_path_collision_mean": candidate[
            "path_collision_mean"],
    }, diagnostic


def run(size: int, seeds: int) -> dict:
    clean_rows: list[dict] = []
    noisy_rows: list[dict] = []

    def row(
        preset: str,
        truth: np.ndarray,
        value: np.ndarray,
        *,
        condition: str | None = None,
        seed: int | None = None,
    ) -> dict:
        forms, diagnostic = evaluate(value)
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            "mean_connection_authority": diagnostic.get(
                "mean_connection_authority", 0.0),
            "mean_connection_covariance_trace": diagnostic.get(
                "mean_connection_covariance_trace", 0.0),
            "mean_lineage_population": diagnostic[
                "mean_lineage_population"],
        }
        if condition is not None:
            result["condition"] = condition
        if seed is not None:
            result["seed"] = seed
        return result

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_rows.append(row(preset, truth, truth))
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=19100 + seed,
                )
                noisy_rows.append(row(
                    preset,
                    truth,
                    observation,
                    condition=condition,
                    seed=seed,
                ))

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([entry[method][key] for entry in rows]))
                for key in rows[0][method]
            }
            for method in METHODS
        }

    return {
        "purpose": (
            "falsify the exact Gaussian Newton potential of the inferred "
            "connection law as a continuous transport estimator"
        ),
        "connection_conductance": "erf(r / sqrt(2)) / r",
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "by_condition": {
            condition: summarize([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "clean_rows": clean_rows,
        "rows": noisy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
