"""Measure the compact endpoint point estimator across unknown corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .lifted_endpoint_action_transport_2d import (
    denoise_lifted_endpoint_action_transport_2d,
)
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first) - np.asarray(second)) ** 2))


def _gradient_mse(first: np.ndarray, second: np.ndarray) -> float:
    first_gradient = np.concatenate((
        np.diff(first, axis=0).reshape(-1),
        np.diff(first, axis=1).reshape(-1),
    ))
    second_gradient = np.concatenate((
        np.diff(second, axis=0).reshape(-1),
        np.diff(second, axis=1).reshape(-1),
    ))
    return _mse(first_gradient, second_gradient)


def run(size: int, selected: tuple[str, ...]) -> dict[str, Any]:
    catalogue = sources(size)
    corruption_cases = (
        ("clean", None, 0.0, 0.0),
        ("Gaussian additive 0.10", "Gaussian additive", 0.10, 0.0),
        ("uniform additive 0.10", "uniform additive", 0.10, 0.0),
        ("salt and pepper 0.20", "salt and pepper", 0.0, 0.20),
        (
            "mixed replacement + uniform 0.25",
            "mixed replacement + uniform", 0.10, 0.25,
        ),
    )
    rows = []
    for source in selected:
        truth = catalogue[source]
        for condition, kind, amount, density in corruption_cases:
            observation = (
                truth.copy()
                if kind is None
                else corrupt(
                    truth, kind, amount=amount, density=density, seed=9100)
            )
            started = perf_counter()
            estimate, diagnostic = denoise_lifted_endpoint_action_transport_2d(
                observation, return_temporal_ablation=True)
            elapsed = perf_counter() - started
            initial = np.asarray(diagnostic["lifted"]["initial_posterior"])
            first_cycle = np.asarray(diagnostic["first_cycle_estimate"])
            unstopped = np.asarray(
                diagnostic["unstopped_second_cycle_estimate"])
            observation_mse = _mse(observation, truth)
            initial_mse = _mse(initial, truth)
            first_cycle_mse = _mse(first_cycle, truth)
            unstopped_mse = _mse(unstopped, truth)
            estimate_mse = _mse(estimate, truth)
            rows.append({
                "source": source,
                "condition": condition,
                "elapsed_seconds": elapsed,
                "observation_mse": observation_mse,
                "initial_posterior_mse": initial_mse,
                "first_cycle_estimate_mse": first_cycle_mse,
                "unstopped_second_cycle_mse": unstopped_mse,
                "endpoint_estimate_mse": estimate_mse,
                "endpoint_improvement_over_observation_fraction": (
                    float((observation_mse - estimate_mse) / observation_mse)
                    if observation_mse > 0.0 else None
                ),
                "endpoint_improvement_over_initial_fraction": (
                    float((initial_mse - estimate_mse) / initial_mse)
                    if initial_mse > 0.0 else None
                ),
                "second_cycle_improvement_over_first_fraction": (
                    float((first_cycle_mse - estimate_mse) / first_cycle_mse)
                    if first_cycle_mse > 0.0 else None
                ),
                "stopping_improvement_over_unstopped_fraction": (
                    float((unstopped_mse - estimate_mse) / unstopped_mse)
                    if unstopped_mse > 0.0 else None
                ),
                "observation_gradient_mse": _gradient_mse(observation, truth),
                "initial_gradient_mse": _gradient_mse(initial, truth),
                "first_cycle_gradient_mse": _gradient_mse(first_cycle, truth),
                "unstopped_second_cycle_gradient_mse": _gradient_mse(
                    unstopped, truth),
                "endpoint_gradient_mse": _gradient_mse(estimate, truth),
                "mean_first_cycle_fine_endpoint": diagnostic[
                    "mean_first_cycle_fine_endpoint"],
                "mean_first_cycle_coarse_endpoint": diagnostic[
                    "mean_first_cycle_coarse_endpoint"],
                "mean_fine_endpoint": diagnostic["mean_fine_endpoint"],
                "mean_coarse_endpoint": diagnostic["mean_coarse_endpoint"],
                "mean_absolute_transfer": diagnostic["mean_absolute_transfer"],
                "mean_observation_phase_authority": diagnostic[
                    "observation_phase"]["mean_authority"],
                "mean_transported_scale_support": diagnostic[
                    "mean_transported_scale_support"],
                "observation_recomposition_error": diagnostic[
                    "observation_recomposition_error"],
                "mean_between_cycle_action_variance": diagnostic[
                    "endpoint_action_evidence"]["causal_temporal_fusion"][
                        "mean_between_cycle_action_variance"],
                "mean_temporal_continuation_authority": diagnostic[
                    "endpoint_action_evidence"]["causal_temporal_fusion"][
                        "mean_continuation_authority"],
                "action_evidence": diagnostic["endpoint_action_evidence"],
            })
    corrupted = [row for row in rows if row["observation_mse"] > 0.0]
    clean = [row for row in rows if row["condition"] == "clean"]
    additive = [
        row for row in rows
        if row["condition"].startswith(("Gaussian", "uniform"))
    ]
    replacement = [
        row for row in rows
        if row["condition"].startswith(("salt", "mixed"))
    ]

    def second_better(group: list[dict[str, Any]]) -> int:
        return sum(
            row["endpoint_estimate_mse"] < row["first_cycle_estimate_mse"]
            for row in group
        )

    return {
        "purpose": (
            "ablate one versus two conservative cycles of the retained "
            "four-coordinate support/rejection action measure across "
            "corruption laws not supplied to the estimator"
        ),
        "size": int(size),
        "sources": list(selected),
        "summary": {
            "final_improves_observation_count": sum(
                row["endpoint_estimate_mse"] < row["observation_mse"]
                for row in corrupted
            ),
            "corrupted_case_count": len(corrupted),
            "final_improves_initial_count": sum(
                row["endpoint_estimate_mse"] < row["initial_posterior_mse"]
                for row in corrupted
            ),
            "second_cycle_improves_first_clean_count": second_better(clean),
            "clean_case_count": len(clean),
            "second_cycle_improves_first_additive_count": second_better(
                additive),
            "additive_case_count": len(additive),
            "second_cycle_improves_first_replacement_count": second_better(
                replacement),
            "replacement_case_count": len(replacement),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(
            row["source"], "|", row["condition"],
            "mse", (
                round(row["observation_mse"], 7),
                round(row["initial_posterior_mse"], 7),
                round(row["first_cycle_estimate_mse"], 7),
                round(row["unstopped_second_cycle_mse"], 7),
                round(row["endpoint_estimate_mse"], 7),
            ),
            "gradient", (
                round(row["observation_gradient_mse"], 7),
                round(row["endpoint_gradient_mse"], 7),
            ),
            "endpoints", (
                round(row["mean_fine_endpoint"], 4),
                round(row["mean_coarse_endpoint"], 4),
            ),
            "seconds", round(row["elapsed_seconds"], 3),
        )


if __name__ == "__main__":
    main()
