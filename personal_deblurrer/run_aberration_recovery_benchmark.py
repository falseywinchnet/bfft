#!/usr/bin/env python3
"""Benchmark blind relative affine-aberration recovery and its common gauge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .aberration_recovery import (
    covariance_field_matrices,
    recover_affine_aberration_multicapture,
)
from .composed_transport import compose_positive_transports, radial_scale_measure
from .observation_anomalies import astigmatic_scale_measure, ghost_measure


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _operators(shape: tuple[int, int]):
    parameters = (
        (0.015, 0.010, 0.0),
        (0.030, 0.018, 45.0),
        (0.045, 0.025, 90.0),
        (0.060, 0.015, 135.0),
    )
    result = []
    for radial_extent, astigmatic_extent, angle in parameters:
        radial = radial_scale_measure(
            shape, fractional_extent=radial_extent).to_transport(shape)
        astigmatic = astigmatic_scale_measure(
            shape,
            fractional_extent=astigmatic_extent,
            angle_degrees=angle,
        ).to_transport(shape)
        result.append(compose_positive_transports(radial, astigmatic))
    return tuple(result)


def _relative_correlation(actual: np.ndarray, estimated: np.ndarray) -> float:
    actual = actual - np.mean(actual, axis=0, keepdims=True)
    estimated = estimated - np.mean(estimated, axis=0, keepdims=True)
    margin = max(int(round(0.10 * actual.shape[1])), 4)
    first = actual[:, margin:-margin, margin:-margin].ravel()
    second = estimated[:, margin:-margin, margin:-margin].ravel()
    return float(np.dot(first, second) / np.sqrt(
        np.dot(first, first) * np.dot(second, second)))


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    shape = (int(size), int(size))
    base = _operators(shape)
    offsets = ((2.0, -1.0), (-1.5, 2.5), (3.0, 1.0), (-2.0, -2.5))
    ghosted = tuple(
        compose_positive_transports(
            operator,
            ghost_measure(offset, ghost_mass=0.025 + 0.012 * index)
            .to_transport(shape),
        )
        for index, (operator, offset) in enumerate(zip(base, offsets))
    )
    common = radial_scale_measure(
        shape, fractional_extent=0.03).to_transport(shape)
    common_gauge = tuple(
        compose_positive_transports(common, operator) for operator in base)
    cases = {
        "relative_radial_astigmatic": (base, None),
        "relative_radial_astigmatic_ghost": (ghosted, None),
        "relative_plus_common_radial_gauge": (common_gauge, common),
    }
    rows: list[dict[str, object]] = []
    operator_records: dict[str, list[dict[str, float]]] = {
        case: [] for case in cases}
    audits: dict[str, object] = {}
    for case, (operators, common_operator) in cases.items():
        for source, truth in sources(size).items():
            observations = tuple(operator.forward(truth) for operator in operators)
            result = recover_affine_aberration_multicapture(
                observations,
                passes=passes,
                patch_size=min(32, size),
                stride=min(16, max(size // 4, 8)),
            )
            best = max(observations, key=lambda item: _score(item, truth)["psnr"])
            average = np.mean(observations, axis=0)
            for method, image in (
                ("best_capture", best),
                ("observation_average", average),
                ("estimated_aberration_inverse", result.image),
            ):
                rows.append({
                    "case": case,
                    "source": source,
                    "method": method,
                    **_score(image, truth),
                })
            actual = np.stack([
                operator.local_moment_jet().covariance for operator in operators
            ])
            estimated = covariance_field_matrices(
                result.transport_result.fields)
            fitted = result.aberration_jet.fitted_covariance_fields
            record = {
                "raw_relative_tensor_correlation": _relative_correlation(
                    actual, estimated),
                "quadratic_jet_relative_tensor_correlation": (
                    _relative_correlation(actual, fitted)),
                "crossfit_predictive_authority": float(
                    result.aberration_jet.diagnostics[
                        "crossfit_predictive_authority"]),
            }
            if common_operator is not None:
                gauge_image = common_operator.forward(truth)
                record.update({
                    "result_psnr_to_common_gauge": _score(
                        result.image, gauge_image)["psnr"],
                    "average_psnr_to_common_gauge": _score(
                        average, gauge_image)["psnr"],
                    "common_gauge_psnr_to_sharp_truth": _score(
                        gauge_image, truth)["psnr"],
                })
            operator_records[case].append(record)
            audits[case] = result.diagnostics

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary = {}
    for case in cases:
        methods = {}
        for method in (
            "best_capture", "observation_average",
            "estimated_aberration_inverse",
        ):
            methods[method] = _mean([
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == case and row["method"] == method
            ])
        records = operator_records[case]
        summary[case] = {
            "methods": methods,
            "mean_raw_relative_tensor_correlation": float(np.mean([
                item["raw_relative_tensor_correlation"] for item in records])),
            "mean_quadratic_jet_relative_tensor_correlation": float(np.mean([
                item["quadratic_jet_relative_tensor_correlation"]
                for item in records])),
            "mean_crossfit_predictive_authority": float(np.mean([
                item["crossfit_predictive_authority"] for item in records])),
        }
        if case == "relative_plus_common_radial_gauge":
            summary[case].update({
                key: float(np.mean([item[key] for item in records]))
                for key in (
                    "result_psnr_to_common_gauge",
                    "average_psnr_to_common_gauge",
                    "common_gauge_psnr_to_sharp_truth",
                )
            })
    result = {
        "experiment": "blind_relative_affine_aberration_recovery_v1",
        "truth_used_for_estimation": False,
        "family_classification": False,
        "common_aberration_identifiable": False,
        "capture_count": 4,
        "size": int(size),
        "passes": int(passes),
        "sources": list(sources(size)),
        "summary": summary,
        "operator_records": operator_records,
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
        default=Path("personal_deblurrer/aberration_recovery_results"))
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, record in result["summary"].items():
        print(case)
        for method, score in record["methods"].items():
            print(
                f"  {method:32s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
        print(
            f"  tensor correlation raw/jet "
            f"{record['mean_raw_relative_tensor_correlation']:.4f}/"
            f"{record['mean_quadratic_jet_relative_tensor_correlation']:.4f}; "
            f"crossfit authority {record['mean_crossfit_predictive_authority']:.4f}"
        )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
