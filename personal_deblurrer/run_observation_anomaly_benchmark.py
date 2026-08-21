#!/usr/bin/env python3
"""Benchmark compound affine blur and bounded sensor anomalies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .composed_transport import (
    compose_positive_transports,
    radial_scale_measure,
    refine_consolidated_transport,
)
from .observation_anomalies import (
    astigmatic_scale_measure,
    bounded_linear_sensor_observation,
    ghost_measure,
    rotation_exposure_measure,
    shear_exposure_measure,
)


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _compose(sequence, name: str):
    result = sequence[0]
    for transport in sequence[1:]:
        result = compose_positive_transports(result, transport, name=name)
    return result


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    shape = (int(size), int(size))
    center = np.asarray((0.5 * (size - 1), 0.5 * (size - 1)))
    radial_a = radial_scale_measure(
        shape,
        fractional_extent=0.04,
        center_xy=tuple(center + np.asarray((-4.0, 3.0))),
    ).to_transport(shape)
    radial_b = radial_scale_measure(
        shape,
        fractional_extent=0.055,
        center_xy=tuple(center + np.asarray((5.0, -2.0))),
    ).to_transport(shape)
    rotation = rotation_exposure_measure(
        shape,
        exposure_degrees=2.5,
        center_xy=tuple(center + np.asarray((2.0, -3.0))),
    ).to_transport(shape)
    ghost = ghost_measure((3.0, -2.0), ghost_mass=0.07).to_transport(shape)
    astigmatic = astigmatic_scale_measure(
        shape,
        fractional_extent=0.04,
        angle_degrees=31.0,
        center_xy=tuple(center + np.asarray((-2.0, 1.0))),
    ).to_transport(shape)
    shear = shear_exposure_measure(
        shape, fractional_extent=0.025).to_transport(shape)
    cases = {
        "decentered_unequal_double_radial": _compose(
            (radial_a, radial_b), "decentered_unequal_double_radial"),
        "radial_rotation_ghost": _compose(
            (radial_a, rotation, ghost), "radial_rotation_ghost"),
        "astigmatic_shear_ghost": _compose(
            (astigmatic, shear, ghost), "astigmatic_shear_ghost"),
    }
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for case, transport in cases.items():
        for source, truth in sources(size).items():
            observation = transport.forward(truth)
            result = refine_consolidated_transport(
                observation, transport, passes=passes)
            for method, image in (
                ("observation", observation),
                ("normalized_adjoint_seed", result.adjoint_seed),
                ("consolidated_positive_inverse", result.image),
            ):
                rows.append({
                    "case": case,
                    "source": source,
                    "method": method,
                    **_score(image, truth),
                })
            audits[case] = result.diagnostics

    sensor_transport = compose_positive_transports(radial_a, radial_b)
    sensor_case = "double_radial_saturated_quantized_missing"
    yy, xx = np.mgrid[:size, :size]
    invalid = ((7 * xx + 11 * yy) % 53) == 0
    sensor_records = []
    for source, truth in sources(size).items():
        clean = sensor_transport.forward(truth)
        sensor = bounded_linear_sensor_observation(
            clean,
            exposure_gain=1.8,
            quantization_levels=32,
            invalid_mask=invalid,
        )
        naive = refine_consolidated_transport(
            sensor.measured, sensor_transport, passes=passes)
        bounded = refine_consolidated_transport(
            sensor.transport_center,
            sensor_transport,
            passes=passes,
            observation_bounds=sensor.bounds,
        )
        for method, image in (
            ("sensor_codes", sensor.measured),
            ("naive_equality_inverse", naive.image),
            ("bounded_interval_inverse", bounded.image),
        ):
            rows.append({
                "case": sensor_case,
                "source": source,
                "method": method,
                **_score(image, truth),
            })
        sensor_records.append(sensor.diagnostics)
        audits[sensor_case] = bounded.diagnostics

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary: dict[str, object] = {}
    for case, transport in cases.items():
        methods = {}
        for method in (
            "observation", "normalized_adjoint_seed",
            "consolidated_positive_inverse",
        ):
            methods[method] = _mean([
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == case and row["method"] == method
            ])
        summary[case] = {
            "methods": methods,
            "contribution_count": transport.contribution_count,
            "operator_storage_bytes": transport.storage_bytes,
        }
    summary[sensor_case] = {
        "methods": {
            method: _mean([
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == sensor_case and row["method"] == method
            ])
            for method in (
                "sensor_codes", "naive_equality_inverse",
                "bounded_interval_inverse",
            )
        },
        "mean_maximum_code_fraction": float(np.mean([
            record["maximum_code_fraction"] for record in sensor_records])),
        "invalid_fraction": float(sensor_records[0]["invalid_fraction"]),
        "quantization_levels": 32,
        "exposure_gain": 1.8,
    }
    result = {
        "experiment": "unified_observation_transport_anomaly_battery_v1",
        "scope": "known_measure_and_known_sensor_bounds_not_blind_estimation",
        "operator_decomposition": False,
        "family_classification": False,
        "size": int(size),
        "passes": int(passes),
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
        "--out",
        type=Path,
        default=Path("personal_deblurrer/observation_anomaly_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, record in result["summary"].items():
        print(case)
        for method, score in record["methods"].items():
            print(
                f"  {method:32s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
