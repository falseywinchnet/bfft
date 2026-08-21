"""Test complete residual moments inside the causal HJ branch geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_phase_integrated_readouts_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_nuisance_geometry_2d import (
    poisson_observation,
    row_correlated_signal_dependent_observation,
)
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


READOUT = "causal_phase_average_hj_simplex_collision_barycenter"


def displacement(
    estimate: np.ndarray,
    observation: np.ndarray,
) -> dict[str, float]:
    """Measure estimator motion independently of unavailable truth."""
    difference = np.asarray(estimate) - np.asarray(observation)
    absolute = np.abs(difference)
    return {
        "observation_displacement_mse": float(np.mean(difference * difference)),
        "observation_displacement_rms": float(np.sqrt(np.mean(
            difference * difference))),
        "observation_displacement_maximum": float(np.max(absolute)),
        "fraction_moved_over_one_8bit_level": float(np.mean(
            absolute > 1.0 / 255.0)),
    }


def run(
    size: int,
    phase_count: int,
    selected: tuple[str, ...] = (
        "cameraman", "tapered hair", "woven chirps"),
) -> dict[str, object]:
    catalogue = sources(size)
    cases = (
        ("clean", None),
        ("Gaussian 0.10", "gaussian"),
        ("replacement 0.25", "replacement"),
        ("mixed 0.25", "mixed"),
        ("Poisson exposure 16", "poisson"),
        ("row-correlated signal-dependent 0.15", "row"),
    )
    rows = []
    for source in selected:
        truth = catalogue[source]
        for condition, kind in cases:
            rng = np.random.default_rng(271828)
            if kind is None:
                observation = truth
            elif kind == "gaussian":
                observation = corrupt(
                    truth, "Gaussian additive", amount=0.10, density=0.25,
                    seed=271828)
            elif kind == "replacement":
                observation = corrupt(
                    truth, "random-value replacement", amount=0.10,
                    density=0.25, seed=271828)
            elif kind == "mixed":
                observation = corrupt(
                    truth, "mixed replacement + uniform", amount=0.10,
                    density=0.25, seed=271828)
            elif kind == "poisson":
                observation = poisson_observation(truth, 16.0, rng)
            else:
                observation = row_correlated_signal_dependent_observation(
                    truth, 0.15, rng)

            central, central_diagnostic = (
                causal_information_phase_integrated_readouts_2d(
                    observation,
                    angular_count=4,
                    quantile_count=16,
                    phase_count=phase_count,
                    complete_residual_moment=False,
                ))
            complete, complete_diagnostic = (
                causal_information_phase_integrated_readouts_2d(
                    observation,
                    angular_count=4,
                    quantile_count=16,
                    phase_count=phase_count,
                    complete_residual_moment=True,
                ))
            fmmt = denoise_fmmt(observation)[0]
            central_image = central[READOUT]
            complete_image = complete[READOUT]
            pair_difference = complete_image - central_image
            rows.append({
                "source": source,
                "condition": condition,
                "observation": metrics(observation, truth),
                "central-residual HJ simplex": {
                    **metrics(central_image, truth),
                    **displacement(central_image, observation),
                },
                "complete-residual HJ simplex": {
                    **metrics(complete_image, truth),
                    **displacement(complete_image, observation),
                },
                "integrated FMMT control": {
                    **metrics(fmmt, truth),
                    **displacement(fmmt, observation),
                },
                "complete_vs_central": {
                    "mse": float(np.mean(pair_difference * pair_difference)),
                    "rms": float(np.sqrt(np.mean(
                        pair_difference * pair_difference))),
                    "maximum": float(np.max(np.abs(pair_difference))),
                    "fraction_over_one_8bit_level": float(np.mean(
                        np.abs(pair_difference) > 1.0 / 255.0)),
                },
                "central_phase_mass_rms": central_diagnostic[
                    "phase_mass_mean_rms"],
                "complete_phase_mass_rms": complete_diagnostic[
                    "phase_mass_mean_rms"],
            })

    methods = (
        "observation",
        "central-residual HJ simplex",
        "complete-residual HJ simplex",
        "integrated FMMT control",
    )

    def summarize(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
        names = ("mse", "ssim", "variance_ratio", "central_range_ratio",
                 "edge_retention", "mean_bias",
                 "observation_displacement_mse",
                 "observation_displacement_rms",
                 "observation_displacement_maximum",
                 "fraction_moved_over_one_8bit_level")
        return {
            method: {
                name: (
                    0.0
                    if method == "observation" and name.startswith(
                        ("observation_displacement", "fraction_moved"))
                    else float(np.mean([
                        row[method][name] for row in records]))
                )
                for name in names
            }
            for method in methods
        }

    result = {
        "purpose": (
            "test whether zero-referenced complete residual moments improve "
            "the causal HJ branch metric without a neural or noise-class model"),
        "size": size,
        "phase_count": phase_count,
        "sources": list(selected),
        "summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition, _kind in cases
        },
        "case_wins": {
            "mse": int(sum(
                row["complete-residual HJ simplex"]["mse"]
                < row["central-residual HJ simplex"]["mse"]
                for row in rows)),
            "ssim": int(sum(
                row["complete-residual HJ simplex"]["ssim"]
                > row["central-residual HJ simplex"]["ssim"]
                for row in rows)),
            "edge_retention": int(sum(
                row["complete-residual HJ simplex"]["edge_retention"]
                > row["central-residual HJ simplex"]["edge_retention"]
                for row in rows)),
        },
        "complete_vs_central": {
            name: float(np.mean([
                row["complete_vs_central"][name] for row in rows]))
            for name in (
                "mse", "rms", "maximum", "fraction_over_one_8bit_level")
        },
        "rows": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--phase-count", type=int, default=2)
    parser.add_argument(
        "--sources",
        default="cameraman,tapered hair,woven chirps",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        args.phase_count,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "case_wins": result["case_wins"],
        "by_condition": result["by_condition"],
    }, indent=2))


if __name__ == "__main__":
    main()
