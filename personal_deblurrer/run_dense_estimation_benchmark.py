#!/usr/bin/env python3
"""Benchmark dense barycentric-flow estimation, exposure removal, and gauges."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import map_coordinates

from denoiser.run_2d_denoiser_battery import metrics, sources

from .dense_estimation import deblur_dense_pair_consensus
from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _nominal_flow(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    result = np.empty((height, width, 2), dtype=np.float64)
    result[..., 0] = (
        3.0
        + 0.02 * (yy - center_y)
        + 0.65 * np.sin(2.0 * np.pi * yy / height)
    )
    result[..., 1] = (
        -2.0
        + 0.015 * (xx - center_x)
        + 0.45 * np.sin(2.0 * np.pi * xx / width)
    )
    return result


def _true_pair_sampling_flow(nominal: np.ndarray) -> np.ndarray:
    """Solve s-.5u(s)=p+.5u(p) for the exact barycentric pair map."""
    height, width = nominal.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    sensor_x = xx + nominal[..., 0]
    sensor_y = yy + nominal[..., 1]
    for _ in range(40):
        sampled = np.stack([
            map_coordinates(
                nominal[..., channel],
                (sensor_y, sensor_x),
                order=1,
                mode="reflect",
                prefilter=False,
            )
            for channel in range(2)
        ], axis=2)
        next_x = xx + 0.5 * nominal[..., 0] + 0.5 * sampled[..., 0]
        next_y = yy + 0.5 * nominal[..., 1] + 0.5 * sampled[..., 1]
        update = max(
            float(np.max(np.abs(next_x - sensor_x))),
            float(np.max(np.abs(next_y - sensor_y))),
        )
        sensor_x, sensor_y = next_x, next_y
        if update <= 1e-10:
            break
    return np.stack((sensor_x - xx, sensor_y - yy), axis=2)


def _fields(
    flow: np.ndarray,
    *,
    duty_cycle: float,
    atoms: int,
) -> tuple[SpatialExposureField, SpatialExposureField]:
    height, width = flow.shape[:2]
    times = np.linspace(-0.5, 0.5, atoms)[:, None, None, None]
    residual = times * float(duty_cycle) * flow[None, ...]
    weight = np.ones((atoms, height, width), dtype=np.float64)
    return tuple(
        SpatialExposureField.from_barycentric_paths(
            name=f"dense_benchmark_capture_{index}",
            barycentric_flow_xy=sign * 0.5 * flow,
            residual_displacements_xy=residual,
            weights=weight,
        )
        for index, sign in enumerate((-1.0, 1.0))
    )  # type: ignore[return-value]


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    duty_cycle = 0.5
    atoms = 7
    nominal = _nominal_flow((size, size))
    true_sampling = _true_pair_sampling_flow(nominal)
    fields = _fields(nominal, duty_cycle=duty_cycle, atoms=atoms)
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    endpoint_errors = []
    ambiguity_changes = []
    for source_index, (source, truth) in enumerate(sources(size).items()):
        observations = []
        for capture, field in enumerate(fields):
            blurred = SpatialReflectedExposureOperator(field).forward(truth)
            rng = np.random.default_rng(15000 + 37 * source_index + capture)
            observations.append(np.clip(
                blurred + rng.normal(0.0, 0.002, blurred.shape), 0.0, 1.0))
        estimated = deblur_dense_pair_consensus(
            observations[0],
            observations[1],
            duty_cycle=duty_cycle,
            atoms=atoms,
            passes=passes,
        )
        oracle = solve_spatial_field_consensus(
            observations, fields, passes=passes, ratio_limit=4.0)
        methods = {
            "best_capture": min(
                observations,
                key=lambda item: float(np.mean((item - truth) ** 2)),
            ),
            "unregistered_average": np.mean(observations, axis=0),
            "estimated_dense_consensus": estimated.image,
            "known_field_consensus_oracle": oracle.image,
        }
        for method, image in methods.items():
            rows.append({
                "source": source,
                "method": method,
                **_score(image, truth),
            })
        endpoint = np.sqrt(np.sum(
            (estimated.estimate.forward_sampling_flow_xy - true_sampling) ** 2,
            axis=2,
        ))
        endpoint_errors.extend(endpoint.ravel().tolist())

        common = SpatialReflectedExposureOperator(fields[0]).forward(truth)
        abstained = deblur_dense_pair_consensus(
            common, common, duty_cycle=duty_cycle, passes=16)
        ambiguity_change = float(np.max(np.abs(abstained.image - common)))
        ambiguity_changes.append(ambiguity_change)
        audits[source] = {
            "estimated": estimated.diagnostics,
            "flow_endpoint_error_mean": float(np.mean(endpoint)),
            "flow_endpoint_error_q90": float(np.quantile(endpoint, 0.9)),
            "flow_endpoint_error_max": float(np.max(endpoint)),
            "ambiguity_decision": abstained.diagnostics["estimation_decision"],
            "ambiguity_maximum_change": ambiguity_change,
        }

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary = {}
    for method in (
        "best_capture", "unregistered_average", "estimated_dense_consensus",
        "known_field_consensus_oracle",
    ):
        summary[method] = _mean([
            {key: float(row[key]) for key in score_keys}
            for row in rows if row["method"] == method
        ])
    result = {
        "experiment": "continuous_dense_barycentric_flow_consensus_v1",
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "duty_cycle": duty_cycle,
        "exposure_atoms": atoms,
        "flow_components": (
            "translation_plus_affine_shear_plus_smooth_local_deformation"),
        "sources": list(sources(size)),
        "summary": summary,
        "mean_flow_endpoint_error_pixels": float(np.mean(endpoint_errors)),
        "q90_flow_endpoint_error_pixels": float(np.quantile(endpoint_errors, 0.9)),
        "maximum_flow_endpoint_error_pixels": float(np.max(endpoint_errors)),
        "maximum_ambiguity_change": float(np.max(ambiguity_changes)),
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
        default=Path("personal_deblurrer/dense_estimation_results"))
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for method, score in result["summary"].items():
        print(f"{method:30s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        "flow endpoint mean/q90/max: "
        f"{result['mean_flow_endpoint_error_pixels']:.4f}/"
        f"{result['q90_flow_endpoint_error_pixels']:.4f}/"
        f"{result['maximum_flow_endpoint_error_pixels']:.4f} px")
    print(f"maximum ambiguity change: {result['maximum_ambiguity_change']:.3e}")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
