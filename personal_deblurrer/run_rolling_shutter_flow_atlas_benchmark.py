#!/usr/bin/env python3
"""Measure accelerated rolling-shutter layered motion in the flow atlas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import map_coordinates

from denoiser.run_2d_denoiser_battery import metrics, sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus


def _rolling_shutter_layer_pair(
    background: np.ndarray,
    base_displacement: float,
    row_acceleration: float,
    *,
    exposure_extent: float,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate an observable row-dependent relative displacement pair."""
    height, width = background.shape
    yy, xx = np.mgrid[:height, :width]
    normalized_y = (yy - 0.5 * (height - 1)) / max(height - 1, 1)
    mask = (
        ((xx - 0.5 * width) / (0.22 * width)) ** 2
        + ((yy - 0.5 * height) / (0.28 * height)) ** 2
        < 1.0
    ).astype(np.float64)
    foreground = np.clip(0.1 + 0.8 * (1.0 - background), 0.0, 1.0)
    truth = mask * foreground + (1.0 - mask) * background
    observations = []
    times = np.linspace(-0.5, 0.5, 7, dtype=np.float64)
    for capture, sign in enumerate((-1.0, 1.0)):
        observation = np.zeros_like(background)
        for exposure_time in times:
            offset = (
                sign * base_displacement
                * (1.0 + row_acceleration * normalized_y)
                + exposure_time * exposure_extent
                * (1.0 + 0.5 * row_acceleration * normalized_y)
            )
            coordinates = (yy, xx - offset)
            moved_mask = map_coordinates(
                mask,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            moved_foreground = map_coordinates(
                foreground,
                coordinates,
                order=1,
                mode="reflect",
                prefilter=False,
            )
            observation += (
                moved_mask * moved_foreground
                + (1.0 - moved_mask) * background
            )
        observation /= len(times)
        rng = np.random.default_rng(seed + capture)
        observations.append(np.clip(
            observation + rng.normal(0.0, noise_sigma, observation.shape),
            0.0,
            1.0,
        ))
    return truth, observations


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(
    rows: list[dict[str, object]], case: str, method: str,
) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["case"] == case and row["method"] == method
    ]
    keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    return {
        key: float(np.mean([float(row[key]) for row in selected]))
        for key in keys
    }


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {
        "moderate_row_acceleration": 0.6,
        "strong_row_acceleration": 1.2,
    }
    methods = (
        "best_capture", "unregistered_average", "dense_common_gauge",
        "raw_fourier_circle_atlas", "unified_flow_atlas",
    )
    base_displacement = 3.0
    exposure_extent = 1.0
    duty_cycle = exposure_extent / (2.0 * base_displacement)
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    gains = []
    deltas = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, acceleration in cases.items():
            truth, observations = _rolling_shutter_layer_pair(
                background,
                base_displacement,
                acceleration,
                exposure_extent=exposure_extent,
                noise_sigma=0.002,
                seed=51000 + 31 * source_index,
            )
            result = deblur_flow_fiber_consensus(
                observations[0],
                observations[1],
                duty_cycle=duty_cycle,
                passes=passes,
            )
            assert result.fiber_solution is not None
            images = {
                "best_capture": min(
                    observations,
                    key=lambda image: float(np.mean((image - truth) ** 2)),
                ),
                "unregistered_average": np.mean(observations, axis=0),
                "dense_common_gauge": result.common_image,
                "raw_fourier_circle_atlas": result.fiber_solution.image,
                "unified_flow_atlas": result.image,
            }
            scored = {}
            for method, image in images.items():
                score = _score(image, truth)
                scored[method] = score
                rows.append({
                    "source": source,
                    "case": case,
                    "method": method,
                    **score,
                })
            gain = (
                scored["unified_flow_atlas"]["psnr"]
                - scored["unregistered_average"]["psnr"]
            )
            delta = (
                scored["unified_flow_atlas"]["psnr"]
                - scored["dense_common_gauge"]["psnr"]
            )
            gains.append(gain)
            deltas.append(delta)
            charts = result.diagnostics[
                "fourier_circle_atlas_transport"]["chart_records"]
            chart_x = np.asarray([
                record["translation_xy"][0] for record in charts
            ], dtype=np.float64)
            audits[f"{source}/{case}"] = {
                "base_displacement_each_side_pixels": base_displacement,
                "row_acceleration": acceleration,
                "exposure_extent_pixels": exposure_extent,
                "gain_over_average_db": gain,
                "delta_from_dense_db": delta,
                "global_circle_translation_xy": result.diagnostics[
                    "fourier_circle_translation_xy"],
                "local_chart_x_standard_deviation": float(np.std(chart_x)),
                "atlas_prior_center": result.diagnostics[
                    "flow_atlas_prior_center"],
                "atlas_authority_mean": result.diagnostics[
                    "correction_authority_mean"],
            }
    summary = {
        case: {method: _mean(rows, case, method) for method in methods}
        for case in cases
    }
    result = {
        "experiment": "accelerated_rolling_shutter_flow_atlas_v1",
        "scope": (
            "row-dependent relative motion and exposure integration inside "
            "one global/local positive flow atlas"
        ),
        "identifiability_boundary": (
            "shared constant-velocity row warp is a common pair gauge; this "
            "battery measures observable row-dependent relative acceleration"
        ),
        "size": int(size),
        "maximum_passes": int(passes),
        "cases": cases,
        "sources": list(sources(size)),
        "summary": summary,
        "mean_gain_over_average_db": float(np.mean(gains)),
        "minimum_gain_over_average_db": float(np.min(gains)),
        "mean_delta_from_dense_db": float(np.mean(deltas)),
        "minimum_delta_from_dense_db": float(np.min(deltas)),
        "positive_gain_trials": int(np.sum(np.asarray(gains) > 0.0)),
        "positive_delta_trials": int(np.sum(np.asarray(deltas) > 0.0)),
        "audits": audits,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("personal_deblurrer/rolling_shutter_atlas_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(
                f"  {method:29s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(
        "gain over average mean/min "
        f"{result['mean_gain_over_average_db']:.3f}/"
        f"{result['minimum_gain_over_average_db']:.3f} dB; positive "
        f"{result['positive_gain_trials']}/12"
    )
    print(
        "delta from dense mean/min "
        f"{result['mean_delta_from_dense_db']:.3f}/"
        f"{result['minimum_delta_from_dense_db']:.3f} dB; positive "
        f"{result['positive_delta_trials']}/12"
    )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
