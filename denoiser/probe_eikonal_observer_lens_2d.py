"""Gate the virtual eikonal lens and its first transported-jet uplift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .eikonal_observer_lens_2d import (
    eikonal_jet_prediction_offset_2d,
    eikonal_lens_analysis_2d,
    eikonal_lens_synthesis_2d,
    smooth_eikonal_lens_detail_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_complete_moment_lineage_2d import displacement
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(size: int, selected: tuple[str, ...]) -> dict:
    rows = []
    catalogue = sources(size)
    for source in selected:
        truth = catalogue[source]
        for condition, observation in (
            ("clean", truth),
            (
                "mixed replacement + uniform 0.25",
                corrupt(
                    truth, "mixed replacement + uniform",
                    amount=0.10, density=0.25, seed=271828),
            ),
        ):
            law, transport = causal_information_lineage_law_2d(
                observation,
                angular_count=4,
                quantile_count=16,
                complete_residual_moment=True,
            )
            forest = transport["forest"]
            scalar_coarse, scalar_detail, scalar_analysis = (
                eikonal_lens_analysis_2d(observation, forest))
            scalar_smoothed, scalar_smoothing = (
                smooth_eikonal_lens_detail_2d(scalar_detail, forest))
            scalar = eikonal_lens_synthesis_2d(
                scalar_coarse, scalar_smoothed, forest)
            scalar_exact = eikonal_lens_synthesis_2d(
                scalar_coarse, scalar_detail, forest)

            offset, jet = eikonal_jet_prediction_offset_2d(law, forest)
            jet_coarse, jet_detail, jet_analysis = eikonal_lens_analysis_2d(
                observation, forest, prediction_offset=offset)
            jet_smoothed, jet_smoothing = smooth_eikonal_lens_detail_2d(
                jet_detail, forest)
            jet_estimate = eikonal_lens_synthesis_2d(
                jet_coarse, jet_smoothed, forest, prediction_offset=offset)
            jet_exact = eikonal_lens_synthesis_2d(
                jet_coarse, jet_detail, forest, prediction_offset=offset)
            estimates = {
                "observation": observation,
                "scalar observer lens": scalar,
                "jet observer lens": jet_estimate,
                "integrated FMMT control": denoise_fmmt(observation)[0],
            }
            rows.append({
                "source": source,
                "condition": condition,
                **{
                    method: {
                        **metrics(estimate, truth),
                        **displacement(estimate, observation),
                    }
                    for method, estimate in estimates.items()
                },
                "scalar_analysis": scalar_analysis,
                "scalar_smoothing": scalar_smoothing,
                "jet_analysis": jet_analysis,
                "jet_smoothing": jet_smoothing,
                "jet": jet,
                "scalar_exact_inverse_maximum_error": float(np.max(np.abs(
                    scalar_exact - observation))),
                "jet_exact_inverse_maximum_error": float(np.max(np.abs(
                    jet_exact - observation))),
            })

    methods = (
        "observation", "scalar observer lens", "jet observer lens",
        "integrated FMMT control")
    names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "observation_displacement_rms")

    def summarize(records: list[dict]) -> dict:
        return {
            method: {
                name: float(np.mean([row[method][name] for row in records]))
                for name in names
            }
            for method in methods
        }

    return {
        "purpose": (
            "test an invertible virtual lens that absorbs structure through "
            "reverse eikonal lifting, smooths only observer-space detail, and "
            "renders the scene by exact forward inversion"
        ),
        "size": size,
        "sources": list(selected),
        "summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition in ("clean", "mixed replacement + uniform 0.25")
        },
        "jet_vs_scalar_case_wins": {
            "mse": int(sum(
                row["jet observer lens"]["mse"]
                < row["scalar observer lens"]["mse"] for row in rows)),
            "ssim": int(sum(
                row["jet observer lens"]["ssim"]
                > row["scalar observer lens"]["ssim"] for row in rows)),
            "edge_retention": int(sum(
                row["jet observer lens"]["edge_retention"]
                > row["scalar observer lens"]["edge_retention"]
                for row in rows)),
        },
        "maximum_exact_inverse_error": float(max(
            max(row["scalar_exact_inverse_maximum_error"],
                row["jet_exact_inverse_maximum_error"])
            for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
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
    print(json.dumps({
        "summary": result["summary"],
        "jet_vs_scalar_case_wins": result["jet_vs_scalar_case_wins"],
        "maximum_exact_inverse_error": result[
            "maximum_exact_inverse_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
