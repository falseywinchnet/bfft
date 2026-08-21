#!/usr/bin/env python3
"""Measure the positive multi-sheet representation with known soft geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .multisheet_transport import solve_multisheet_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


def _field(
    name: str,
    shape: tuple[int, int],
    displacement_xy: tuple[float, float],
) -> SpatialExposureField:
    displacement = np.empty((1, *shape, 2), dtype=np.float64)
    displacement[0, ..., 0] = displacement_xy[0]
    displacement[0, ..., 1] = displacement_xy[1]
    return SpatialExposureField(
        name,
        displacement,
        np.ones((1, *shape), dtype=np.float64),
    )


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
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    displacement = 3.0
    for source_index, (source, background) in enumerate(sources(size).items()):
        yy, xx = np.mgrid[:size, :size]
        radius = (
            ((xx - 0.5 * size) / (0.22 * size)) ** 2
            + ((yy - 0.5 * size) / (0.28 * size)) ** 2
        )
        alpha = np.clip(4.0 * (1.0 - radius), 0.0, 1.0)
        foreground = np.clip(0.1 + 0.8 * (1.0 - background), 0.0, 1.0)
        truth = (1.0 - alpha) * background + alpha * foreground
        identity = _field("stationary_sheet", (size, size), (0.0, 0.0))
        moving = (
            _field("moving_sheet_a", (size, size), (-displacement, 0.0)),
            _field("moving_sheet_b", (size, size), (displacement, 0.0)),
        )
        observations = []
        sensor_ownership = []
        for frame, field in enumerate(moving):
            operator = SpatialReflectedExposureOperator(field)
            moved_alpha = operator.forward(alpha)
            observation = (
                (1.0 - moved_alpha) * background
                + moved_alpha * operator.forward(foreground))
            rng = np.random.default_rng(31000 + 37 * source_index + frame)
            observations.append(np.clip(
                observation + rng.normal(0.0, 0.002, observation.shape),
                0.0,
                1.0,
            ))
            sensor_ownership.append(np.stack(
                (1.0 - moved_alpha, moved_alpha), axis=0))
        result = solve_multisheet_consensus(
            observations,
            ((identity, moving[0]), (identity, moving[1])),
            sensor_ownership=np.stack(sensor_ownership, axis=0),
            reference_ownership=np.stack((1.0 - alpha, alpha), axis=0),
            passes=passes,
        )
        methods = {
            "best_capture": min(
                observations,
                key=lambda item: float(np.mean((item - truth) ** 2)),
            ),
            "unregistered_average": np.mean(observations, axis=0),
            "known_measure_multisheet_oracle": result.image,
        }
        scored = {}
        for method, image in methods.items():
            score = _score(image, truth)
            scored[method] = score
            rows.append({"source": source, "method": method, **score})
        audits[source] = {
            "gain_over_average_db": (
                scored["known_measure_multisheet_oracle"]["psnr"]
                - scored["unregistered_average"]["psnr"]),
            "terminal_residual_rms": result.diagnostics[
                "terminal_residual_rms"],
            "ownership_entropy_mean": result.diagnostics[
                "reference_ownership_entropy_mean"],
            "permutation_role": result.diagnostics["permutation_role"],
        }

    methods = (
        "best_capture", "unregistered_average",
        "known_measure_multisheet_oracle",
    )
    result = {
        "experiment": "positive_multisheet_representation_oracle_v1",
        "scope": (
            "known continuous ownership and known sheet geometry; "
            "representation control, not blind estimation"),
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "displacement_each_side_pixels": displacement,
        "sources": list(sources(size)),
        "summary": {method: _mean(rows, method) for method in methods},
        "minimum_gain_over_average_db": float(min(
            audit["gain_over_average_db"] for audit in audits.values())),
        "mean_gain_over_average_db": float(np.mean([
            audit["gain_over_average_db"] for audit in audits.values()])),
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
        default=Path("personal_deblurrer/multisheet_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for method, score in result["summary"].items():
        print(f"{method:35s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        f"gain mean/min {result['mean_gain_over_average_db']:.3f}/"
        f"{result['minimum_gain_over_average_db']:.3f} dB")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
