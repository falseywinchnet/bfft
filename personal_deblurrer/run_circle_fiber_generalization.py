#!/usr/bin/env python3
"""Falsify the Fourier-circle flow atlas on smooth spatial deformation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus
from .run_dense_estimation_benchmark import _fields, _nominal_flow
from .spatial_transport import SpatialReflectedExposureOperator


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, object]], method: str) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method]
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
    duty_cycle = 0.5
    atoms = 7
    fields = _fields(
        _nominal_flow((size, size)), duty_cycle=duty_cycle, atoms=atoms)
    methods = (
        "unregistered_average", "dense_common_gauge",
        "raw_fourier_circle_atlas", "unified_flow_measure_output",
    )
    rows = []
    audits = {}
    for source_index, (source, truth) in enumerate(sources(size).items()):
        observations = []
        for capture, field in enumerate(fields):
            blurred = SpatialReflectedExposureOperator(field).forward(truth)
            rng = np.random.default_rng(15000 + 37 * source_index + capture)
            observations.append(np.clip(
                blurred + rng.normal(0.0, 0.002, blurred.shape), 0.0, 1.0))
        result = deblur_flow_fiber_consensus(
            observations[0], observations[1],
            duty_cycle=duty_cycle, atoms=atoms, passes=passes)
        images = {
            "unregistered_average": np.mean(observations, axis=0),
            "dense_common_gauge": result.common_image,
            "raw_fourier_circle_atlas": (
                result.fiber_solution.image
                if result.fiber_solution is not None else result.common_image),
            "unified_flow_measure_output": result.image,
        }
        scores = {}
        for method, image in images.items():
            score = _score(image, truth)
            scores[method] = score
            rows.append({"source": source, "method": method, **score})
        audits[source] = {
            "raw_delta_from_dense_db": (
                scores["raw_fourier_circle_atlas"]["psnr"]
                - scores["dense_common_gauge"]["psnr"]),
            "output_delta_from_dense_db": (
                scores["unified_flow_measure_output"]["psnr"]
                - scores["dense_common_gauge"]["psnr"]),
            "circle_translation_xy": result.diagnostics.get(
                "fourier_circle_translation_xy"),
            "circle_dispersion_pixels": result.diagnostics[
                "fourier_circle_transport"]["translation_dispersion_pixels"],
            "fold_fractions": result.diagnostics["fold_fractions"],
        }
    raw_deltas = [audit["raw_delta_from_dense_db"] for audit in audits.values()]
    result = {
        "experiment": "fourier_circle_atlas_smooth_deformation_generalization_v2",
        "size": int(size),
        "maximum_passes": int(passes),
        "summary": {method: _mean(rows, method) for method in methods},
        "mean_raw_delta_from_dense_db": float(np.mean(raw_deltas)),
        "minimum_raw_delta_from_dense_db": float(np.min(raw_deltas)),
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
        default=Path("personal_deblurrer/circle_fiber_generalization"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for method, score in result["summary"].items():
        print(f"{method:30s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        f"raw delta mean/min {result['mean_raw_delta_from_dense_db']:.3f}/"
        f"{result['minimum_raw_delta_from_dense_db']:.3f} dB")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
