#!/usr/bin/env python3
"""Measure rotational consensus estimation, oracle gap, and gauge abstention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .spatial_estimation import deblur_rotation_consensus
from .spatial_transport import (
    SpatialReflectedExposureOperator,
    refine_spatial_exposure,
    rotational_exposure,
)


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    true_angles = np.asarray((-4.0, 0.0, 4.0), dtype=np.float64)
    true_extent = 4.0
    fields = tuple(
        rotational_exposure(
            (size, size),
            mean_angle_degrees=float(angle),
            exposure_degrees=true_extent,
            atoms=9,
        )
        for angle in true_angles
    )
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    angle_errors: list[float] = []
    extent_errors: list[float] = []
    abstention_changes: list[float] = []
    for source_index, (source, truth) in enumerate(sources(size).items()):
        observations = []
        for capture, field in enumerate(fields):
            blurred = SpatialReflectedExposureOperator(field).forward(truth)
            rng = np.random.default_rng(9000 + 31 * source_index + capture)
            observations.append(np.clip(
                blurred + rng.normal(0.0, 0.002, blurred.shape), 0.0, 1.0))
        estimated = deblur_rotation_consensus(
            observations, duty_cycle=1.0, passes=passes)
        oracle = refine_spatial_exposure(
            observations[1], fields[1], passes=passes, ratio_limit=4.0)
        methods = {
            "raw_average": np.mean(observations, axis=0),
            "best_capture": min(
                observations,
                key=lambda item: float(np.mean((item - truth) ** 2)),
            ),
            "estimated_consensus": estimated.image,
            "known_center_oracle": oracle.image,
        }
        for method, image in methods.items():
            rows.append({
                "source": source,
                "method": method,
                **_score(image, truth),
            })
        angle_error = np.abs(
            estimated.estimate.relative_mean_angles_degrees - true_angles)
        extent_error = np.abs(
            estimated.estimate.exposure_extents_degrees - true_extent)
        angle_errors.extend(angle_error.tolist())
        extent_errors.extend(extent_error.tolist())

        ambiguous_field = rotational_exposure(
            (size, size),
            mean_angle_degrees=0.0,
            exposure_degrees=6.0,
            atoms=9,
        )
        ambiguous = SpatialReflectedExposureOperator(
            ambiguous_field).forward(truth)
        abstained = deblur_rotation_consensus(
            [ambiguous, ambiguous, ambiguous], passes=16)
        abstention_changes.append(float(np.max(np.abs(
            abstained.image - ambiguous))))
        audits[source] = {
            "estimated": estimated.diagnostics,
            "pair_evidence": [item.__dict__ for item in (
                estimated.estimate.pair_evidence)],
            "angle_absolute_error_degrees": angle_error.tolist(),
            "extent_absolute_error_degrees": extent_error.tolist(),
            "ambiguity_decision": abstained.diagnostics[
                "estimation_decision"],
            "ambiguity_maximum_change": abstention_changes[-1],
        }

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary = {}
    for method in (
        "raw_average", "best_capture", "estimated_consensus",
        "known_center_oracle",
    ):
        summary[method] = _mean([
            {key: float(row[key]) for key in score_keys}
            for row in rows if row["method"] == method
        ])
    result = {
        "experiment": "multi_observation_rotation_consensus_v1",
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "true_relative_mean_angles_degrees": true_angles.tolist(),
        "true_exposure_extent_degrees": true_extent,
        "sources": list(sources(size)),
        "summary": summary,
        "mean_angle_absolute_error_degrees": float(np.mean(angle_errors)),
        "maximum_angle_absolute_error_degrees": float(np.max(angle_errors)),
        "mean_extent_absolute_error_degrees": float(np.mean(extent_errors)),
        "maximum_extent_absolute_error_degrees": float(np.max(extent_errors)),
        "maximum_ambiguity_change": float(np.max(abstention_changes)),
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
        "--out", type=Path,
        default=Path("personal_deblurrer/spatial_estimation_results"))
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for method, score in result["summary"].items():
        print(
            f"{method:24s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        "angle MAE/max: "
        f"{result['mean_angle_absolute_error_degrees']:.5f}/"
        f"{result['maximum_angle_absolute_error_degrees']:.5f} deg")
    print(
        "extent MAE/max: "
        f"{result['mean_extent_absolute_error_degrees']:.5f}/"
        f"{result['maximum_extent_absolute_error_degrees']:.5f} deg")
    print(f"maximum ambiguity change: {result['maximum_ambiguity_change']:.3e}")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
