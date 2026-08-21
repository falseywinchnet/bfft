#!/usr/bin/env python3
"""Measure oblique and curved path charts against their positive-only basin."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .decomposition import two_stage_deblur_known
from .kernels import curved_path_kernel, line_kernel
from .synthetic import degrade


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases: list[tuple[str, float, float, object]] = []
    for angle in range(0, 180, 15):
        cases.append(("line", float(angle), 0.0, line_kernel(11.0, angle)))
    for bend in (-16.0, -8.0, 2.0, 4.0, 8.0, 12.0, 16.0):
        for angle in (0.0, 30.0, 45.0, 90.0, 120.0, 150.0):
            cases.append((
                "curve", angle, bend,
                curved_path_kernel(11.0, angle, bend),
            ))

    rows: list[dict[str, object]] = []
    audits: dict[str, dict[str, object]] = {}
    for case_index, (kind, angle, bend, kernel) in enumerate(cases):
        case = f"{kind}_angle_{angle:g}" + (
            f"_bend_{bend:g}" if kind == "curve" else "")
        for source_index, (source_name, truth) in enumerate(sources(size).items()):
            observation = degrade(
                truth,
                kernel,
                gaussian_sigma=0.002,
                seed=1000 + 31 * case_index + source_index,
                boundary="reflect",
            )
            positive = two_stage_deblur_known(
                observation,
                kernel,
                passes=passes,
                path_authority_scale=0.0,
            )
            unified = two_stage_deblur_known(
                observation,
                kernel,
                passes=passes,
            )
            scores = {
                "observation": _score(observation, truth),
                "positive_basin": _score(positive.image, truth),
                "path_chart": _score(unified.image, truth),
            }
            for method, score in scores.items():
                rows.append({
                    "case": case,
                    "kind": kind,
                    "angle_degrees": angle,
                    "bend": bend,
                    "source": source_name,
                    "method": method,
                    **score,
                })
            characteristic = unified.diagnostics["characteristic_transport"]
            line_constraint = characteristic.get("line_constraint", {})
            audits[case] = {
                "selected": bool(characteristic.get("selected", False)),
                "method": characteristic.get("method"),
                "line_constraint_chart": line_constraint.get("chart"),
                "line_constraint_authority": float(
                    characteristic.get("line_constraint_authority", 0.0)),
                "line_constraint_evaluated": bool(
                    line_constraint.get("evaluated", False)),
                "authority": float(characteristic.get("authority", 0.0)),
                "tangent_turn_degrees": float(
                    characteristic.get("fitted_tangent_turn_degrees", 0.0)),
                "minor_major_covariance_ratio": float(
                    characteristic.get("minor_major_covariance_ratio", 0.0)),
                "uncertainty_rms": float(
                    characteristic.get("uncertainty_rms", 0.0)),
                "path_length": float(characteristic.get("path_length", 0.0)),
                "jacobian_min": float(
                    characteristic.get("jacobian_min", 1.0)),
                "jacobian_max": float(
                    characteristic.get("jacobian_max", 1.0)),
                "endpoint_disagreement_rms": float(
                    characteristic.get("endpoint_disagreement_rms", 0.0)),
                "endpoint_seed_basin_rms": float(
                    characteristic.get("endpoint_seed_basin_rms", 0.0)),
                "correction_rms": float(
                    characteristic.get("correction_rms", 0.0)),
                "correction_spectral_energy_retained": float(
                    characteristic.get(
                        "correction_spectral_energy_retained", 0.0)),
                "operator_backend": characteristic.get("operator_backend"),
                "operator_plan_reused_across_branches": bool(
                    characteristic.get(
                        "operator_plan_reused_across_branches", False)),
                "observation_unchanged": bool(
                    unified.diagnostics["observation_unchanged"]),
            }

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for kind in ("line", "curve"):
        grouped[kind] = {}
        for method in ("observation", "positive_basin", "path_chart"):
            subset = [
                {
                    key: float(row[key])
                    for key in (
                        "mse", "ssim", "variance_ratio",
                        "central_range_ratio", "edge_retention",
                        "mean_bias", "psnr",
                    )
                }
                for row in rows
                if row["kind"] == kind and row["method"] == method
            ]
            grouped[kind][method] = _mean(subset)
    result: dict[str, object] = {
        "experiment": "continuous_positive_exposure_transport_v3",
        "size": int(size),
        "passes": int(passes),
        "noise_sigma": 0.002,
        "sources": list(sources(size)),
        "summary": grouped,
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
        default=Path("personal_deblurrer/path_chart_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for kind, methods in result["summary"].items():
        print(kind)
        for method, score in methods.items():
            print(
                f"  {method:18s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
