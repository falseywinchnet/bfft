#!/usr/bin/env python3
"""Measure flow-atlas recovery under unequal exposure and sensor clipping."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus
from .radiometric_transport import transport_radiometric_pair
from .run_visibility_benchmark import _layered_pair


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
        "complementary_exposure": 0.70,
        "severe_complementary_exposure": 0.55,
    }
    methods = (
        "best_raw_capture",
        "raw_average",
        "uncensored_symmetric_average",
        "censored_symmetric_average",
        "radiometric_dense_common_gauge",
        "radiometric_unified_flow_atlas",
    )
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    atlas_gains = []
    ratio_errors = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        truth, observations = _layered_pair(
            background,
            3.0,
            noise_sigma=0.002,
            seed=41000 + 47 * source_index,
        )
        for case, low_gain in cases.items():
            gains = (low_gain, 1.0 / low_gain)
            clipped = tuple(
                np.clip(gain * observation, 0.0, 1.0)
                for gain, observation in zip(gains, observations)
            )
            radiometric = transport_radiometric_pair(*clipped)
            precision = radiometric.precision
            precision_for_image = (
                precision if truth.ndim == 2 else precision[..., None])
            gauge_stack = np.stack(radiometric.images, axis=0)
            censored_average = np.sum(
                precision_for_image * gauge_stack, axis=0
            ) / np.maximum(np.sum(precision_for_image, axis=0), 1e-8)
            result = deblur_flow_fiber_consensus(
                clipped[0], clipped[1], passes=passes)
            images = {
                "best_raw_capture": min(
                    clipped,
                    key=lambda image: float(np.mean((image - truth) ** 2)),
                ),
                "raw_average": np.mean(clipped, axis=0),
                "uncensored_symmetric_average": np.mean(gauge_stack, axis=0),
                "censored_symmetric_average": censored_average,
                "radiometric_dense_common_gauge": result.common_image,
                "radiometric_unified_flow_atlas": result.image,
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
                scored["radiometric_unified_flow_atlas"]["psnr"]
                - scored["raw_average"]["psnr"]
            )
            true_ratio = gains[1] / gains[0]
            ratio_error = abs(
                math.log(result.diagnostics[
                    "relative_gain_second_over_first"] / true_ratio)
            )
            atlas_gains.append(gain)
            ratio_errors.append(ratio_error)
            audits[f"{source}/{case}"] = {
                "exposure_gains": list(gains),
                "true_relative_gain": true_ratio,
                "estimated_relative_gain": result.diagnostics[
                    "relative_gain_second_over_first"],
                "absolute_log_gain_error": ratio_error,
                "radiometric_authority": result.diagnostics[
                    "radiometric_authority"],
                "sensor_precision_mean": result.diagnostics[
                    "sensor_precision_mean"],
                "hard_upper_clip_fraction": result.diagnostics[
                    "hard_upper_clip_fraction"],
                "gain_over_raw_average_db": gain,
                "flow_atlas_authority_mean": result.diagnostics[
                    "correction_authority_mean"],
            }
    summary = {
        case: {method: _mean(rows, case, method) for method in methods}
        for case in cases
    }
    result = {
        "experiment": "continuous_radiometric_flow_atlas_v1",
        "scope": (
            "symmetric quantile exposure gauge, continuous sensor-bound "
            "precision, global/local Fourier-circle atlas, no selected frame"
        ),
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma_before_exposure": 0.002,
        "cases": cases,
        "sources": list(sources(size)),
        "summary": summary,
        "mean_gain_over_raw_average_db": float(np.mean(atlas_gains)),
        "minimum_gain_over_raw_average_db": float(np.min(atlas_gains)),
        "positive_gain_trials": int(np.sum(np.asarray(atlas_gains) > 0.0)),
        "mean_absolute_log_gain_error": float(np.mean(ratio_errors)),
        "maximum_absolute_log_gain_error": float(np.max(ratio_errors)),
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
        default=Path("personal_deblurrer/radiometric_flow_atlas_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(
                f"  {method:32s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(
        "gain over raw average mean/min "
        f"{result['mean_gain_over_raw_average_db']:.3f}/"
        f"{result['minimum_gain_over_raw_average_db']:.3f} dB; positive "
        f"{result['positive_gain_trials']}/12"
    )
    print(
        "absolute log-gain error mean/max "
        f"{result['mean_absolute_log_gain_error']:.4f}/"
        f"{result['maximum_absolute_log_gain_error']:.4f}"
    )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
