#!/usr/bin/env python3
"""Benchmark unified spatial exposure transport across warp/mixing limits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .spatial_transport import (
    SpatialReflectedExposureOperator,
    refine_spatial_exposure,
    rotational_exposure,
    shear_path_exposure,
)


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {
        "deterministic_shear": shear_path_exposure(
            (size, size), shear=0.07, residual_length=0.0, atoms=1),
        "shear_plus_centered_mix": shear_path_exposure(
            (size, size), shear=0.07, residual_length=7.0, atoms=9),
        "deterministic_rotation": rotational_exposure(
            (size, size), mean_angle_degrees=4.0,
            exposure_degrees=0.0, atoms=1),
        "rotation_plus_exposure_mix": rotational_exposure(
            (size, size), mean_angle_degrees=4.0,
            exposure_degrees=8.0, atoms=9),
        "centered_rotational_exposure": rotational_exposure(
            (size, size), mean_angle_degrees=0.0,
            exposure_degrees=8.0, atoms=9),
    }
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for case_index, (case, field) in enumerate(cases.items()):
        operator = SpatialReflectedExposureOperator(field)
        for source_index, (source, truth) in enumerate(sources(size).items()):
            rng = np.random.default_rng(7000 + 31 * case_index + source_index)
            observation = np.clip(
                operator.forward(truth)
                + rng.normal(0.0, 0.002, truth.shape),
                0.0,
                1.0,
            )
            result = refine_spatial_exposure(
                observation, field, passes=passes, ratio_limit=4.0)
            for method, image in (
                ("observation", observation),
                ("barycentric_pullback", result.barycentric_seed),
                ("continuous_spatial_exposure", result.image),
            ):
                rows.append({
                    "case": case,
                    "source": source,
                    "method": method,
                    **_score(image, truth),
                })
            audits[case] = result.diagnostics

    summary: dict[str, object] = {}
    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    for case in cases:
        summary[case] = {}
        for method in (
            "observation", "barycentric_pullback", "continuous_spatial_exposure"
        ):
            subset = [
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == case and row["method"] == method
            ]
            summary[case][method] = _mean(subset)
    result = {
        "experiment": "barycentric_first_spatial_positive_exposure_v1",
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "sources": list(sources(size)),
        "summary": summary,
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
        default=Path("personal_deblurrer/spatial_results"))
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(
                f"  {method:28s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
