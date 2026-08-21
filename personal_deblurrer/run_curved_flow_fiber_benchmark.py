#!/usr/bin/env python3
"""Falsify and measure the Fourier-circle atlas on curved layered motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import rotate

from denoiser.run_2d_denoiser_battery import metrics, sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _curved_layer_pair(
    background: np.ndarray,
    angle_degrees: float,
    *,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rotate one occluding appearance about a remote center in two frames."""
    height, width = background.shape
    yy, xx = np.mgrid[:height, :width]
    mask = (
        ((xx - 0.62 * width) / (0.24 * width)) ** 2
        + ((yy - 0.47 * height) / (0.30 * height)) ** 2
        < 1.0
    ).astype(np.float64)
    foreground = np.clip(
        0.08 + 0.84 * (1.0 - background)
        + 0.08 * np.sin(0.31 * xx + 0.17 * yy),
        0.0,
        1.0,
    )
    truth = mask * foreground + (1.0 - mask) * background
    observations = []
    for capture, angle in enumerate((-angle_degrees, angle_degrees)):
        moved_mask = rotate(
            mask,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        moved_foreground = rotate(
            foreground,
            angle,
            reshape=False,
            order=1,
            mode="reflect",
            prefilter=False,
        )
        observation = (
            moved_mask * moved_foreground
            + (1.0 - moved_mask) * background
        )
        rng = np.random.default_rng(seed + capture)
        observations.append(np.clip(
            observation + rng.normal(0.0, noise_sigma, observation.shape),
            0.0,
            1.0,
        ))
    return truth, observations


def _mean(
    rows: list[dict[str, object]], case: str, method: str,
) -> dict[str, float]:
    chosen = [
        row for row in rows
        if row["case"] == case and row["method"] == method
    ]
    keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    return {
        key: float(np.mean([float(row[key]) for row in chosen]))
        for key in keys
    }


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {"moderate_curvature": 3.0, "larger_curvature": 5.0}
    methods = (
        "best_capture", "unregistered_average", "dense_common_gauge",
        "fourier_circle_atlas_measure", "unified_flow_atlas_output",
    )
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, angle in cases.items():
            truth, observations = _curved_layer_pair(
                background,
                angle,
                noise_sigma=0.002,
                seed=31000 + 43 * source_index,
            )
            result = deblur_flow_fiber_consensus(
                observations[0], observations[1], passes=passes)
            circle = (
                result.fiber_solution.image
                if result.fiber_solution is not None else result.common_image
            )
            images = {
                "best_capture": min(
                    observations,
                    key=lambda item: float(np.mean((item - truth) ** 2)),
                ),
                "unregistered_average": np.mean(observations, axis=0),
                "dense_common_gauge": result.common_image,
                "fourier_circle_atlas_measure": circle,
                "unified_flow_atlas_output": result.image,
            }
            scores = {}
            for method, image in images.items():
                score = _score(image, truth)
                scores[method] = score
                rows.append({
                    "source": source,
                    "case": case,
                    "method": method,
                    **score,
                })
            audits[f"{source}/{case}"] = {
                "angle_each_side_degrees": angle,
                "circle_delta_from_dense_db": (
                    scores["fourier_circle_atlas_measure"]["psnr"]
                    - scores["dense_common_gauge"]["psnr"]
                ),
                "unified_delta_from_dense_db": (
                    scores["unified_flow_atlas_output"]["psnr"]
                    - scores["dense_common_gauge"]["psnr"]
                ),
                "fourier_circle_translation_xy": result.diagnostics.get(
                    "fourier_circle_translation_xy"),
                "circle_dispersion_pixels": result.diagnostics.get(
                    "fourier_circle_transport", {}).get(
                        "translation_dispersion_pixels"),
                "correction_authority_mean": result.diagnostics.get(
                    "correction_authority_mean"),
            }
    summary = {
        case: {method: _mean(rows, case, method) for method in methods}
        for case in cases
    }
    deltas = [
        float(audit["unified_delta_from_dense_db"])
        for audit in audits.values()
    ]
    result = {
        "experiment": "curved_layer_fourier_circle_atlas_v2",
        "scope": (
            "one rotating occluding appearance over a static background; "
            "relative displacement direction varies across the raster"
        ),
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "cases": cases,
        "sources": list(sources(size)),
        "summary": summary,
        "mean_unified_delta_from_dense_db": float(np.mean(deltas)),
        "minimum_unified_delta_from_dense_db": float(np.min(deltas)),
        "positive_unified_trials": int(np.sum(np.asarray(deltas) > 0.0)),
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
        default=Path("personal_deblurrer/curved_flow_fiber_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(
                f"  {method:30s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(
        "unified delta from dense mean/min "
        f"{result['mean_unified_delta_from_dense_db']:.3f}/"
        f"{result['minimum_unified_delta_from_dense_db']:.3f} dB; positive "
        f"{result['positive_unified_trials']}/12"
    )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
