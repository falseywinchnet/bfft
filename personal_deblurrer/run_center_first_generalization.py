#!/usr/bin/env python3
"""Generalize one center-first atlas across sources and blur/noise regimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import ssim

from .kernels import (
    curved_path_kernel,
    disk_kernel,
    gaussian_kernel,
    line_kernel,
    translated_kernel,
)
from .multicapture_transport import deblur_multicapture_consensus
from .multicapture_posterior import solve_multicapture_transport_posterior
from .real_capture_evaluation import (
    _fourier_amplification,
    _gradient_energy,
)
from .source_portfolio import research_source_portfolio
from .spatial_transport import (
    CompactGlobalExposureField,
    CompactGlobalReflectedExposureOperator,
    CovarianceReflectedExposureOperator,
    SpatialReflectedExposureOperator,
    pullback_barycentric_values,
    pullback_compact_global_values,
    rotational_exposure,
)
from .synthetic import degrade
from .uncertainty import estimate_noise_discrepancy


_TRANSLATIONS = np.asarray((
    (-1.5, -0.5),
    (1.5, 0.5),
    (0.5, -1.5),
    (-0.5, 1.5),
), dtype=np.float64)


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(image, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    mse = float(np.mean((candidate - reference) ** 2))
    if reference.ndim == 2:
        structural = ssim(reference, candidate)
    else:
        structural = float(np.mean([
            ssim(reference[..., channel], candidate[..., channel])
            for channel in range(reference.shape[2])
        ]))
    return {
        "mse": mse,
        "psnr": float(-10.0 * math.log10(max(
            mse, np.finfo(float).tiny))),
        "ssim": structural,
    }


def _sensor_noise(
    image: np.ndarray,
    *,
    read_sigma: float,
    shot_peak: float,
    seed: int,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    value = np.asarray(image, dtype=np.float64)
    if shot_peak > 0.0:
        value = random.poisson(
            np.maximum(value, 0.0) * shot_peak) / shot_peak
    if read_sigma > 0.0:
        value = value + random.normal(0.0, read_sigma, value.shape)
    return np.clip(value, 0.0, 1.0)


def _translate(image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    field = CompactGlobalExposureField(
        "synthetic_deterministic_translation",
        image.shape[:2],
        np.zeros((1, 2), dtype=np.float64),
        np.ones(1, dtype=np.float64),
        displacement,
    )
    return CompactGlobalReflectedExposureOperator(field).forward(image)


def _global_observations(
    truth: np.ndarray,
    *,
    common: bool,
    read_sigma: float,
    shot_peak: float,
    source_seed: int,
) -> tuple[np.ndarray, ...]:
    kernels = (
        gaussian_kernel(1.6),
        disk_kernel(2.5),
        line_kernel(9.0, 35.0),
        curved_path_kernel(9.0, 115.0, 4.0),
    )
    if common:
        kernels = (gaussian_kernel(2.2),) * len(_TRANSLATIONS)
    return tuple(degrade(
        truth,
        translated_kernel(kernel, shift),
        gaussian_sigma=read_sigma,
        poisson_peak=shot_peak,
        seed=source_seed + capture,
        boundary="reflect",
    ) for capture, (kernel, shift) in enumerate(zip(kernels, _TRANSLATIONS)))


def _spatial_covariance_observations(
    truth: np.ndarray,
    *,
    source_seed: int,
) -> tuple[np.ndarray, ...]:
    height, width = truth.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    normalized_x = xx / max(width - 1, 1)
    normalized_y = yy / max(height - 1, 1)
    observations = []
    for capture, shift in enumerate(_TRANSLATIONS):
        angle = (
            np.deg2rad(25.0 * capture)
            + 0.75 * np.pi * normalized_x
            - 0.20 * np.pi * normalized_y)
        major = 2.0 + 2.5 * (
            0.25 + 0.75 * (normalized_x if capture % 2 == 0
                           else 1.0 - normalized_y))
        minor = 0.25 + 0.30 * normalized_y
        cosine = np.cos(angle)
        sine = np.sin(angle)
        covariance = np.empty((height, width, 2, 2), dtype=np.float64)
        covariance[..., 0, 0] = (
            major * cosine ** 2 + minor * sine ** 2)
        covariance[..., 0, 1] = (major - minor) * cosine * sine
        covariance[..., 1, 0] = covariance[..., 0, 1]
        covariance[..., 1, 1] = (
            major * sine ** 2 + minor * cosine ** 2)
        blurred = CovarianceReflectedExposureOperator(covariance).forward(truth)
        shifted = _translate(blurred, shift)
        observations.append(_sensor_noise(
            shifted,
            read_sigma=0.002,
            shot_peak=800.0,
            seed=source_seed + capture,
        ))
    return tuple(observations)


def _rotational_observations(
    truth: np.ndarray,
    *,
    source_seed: int,
) -> tuple[np.ndarray, ...]:
    observations = []
    for capture, mean_angle in enumerate((-3.0, -1.0, 1.0, 3.0)):
        field = rotational_exposure(
            truth.shape[:2],
            mean_angle_degrees=mean_angle,
            exposure_degrees=4.0 + 0.5 * capture,
            atoms=9,
        )
        blurred = SpatialReflectedExposureOperator(field).forward(truth)
        observations.append(_sensor_noise(
            blurred,
            read_sigma=0.002,
            shot_peak=800.0,
            seed=source_seed + capture,
        ))
    return tuple(observations)


def _centered_observations(result) -> tuple[np.ndarray, ...]:
    centered = []
    for image, field in zip(result.radiometric_images, result.fields):
        if isinstance(field, CompactGlobalExposureField):
            value, _ = pullback_compact_global_values(image, field)
        else:
            value, _ = pullback_barycentric_values(
                image, field.barycentric_field())
        centered.append(value)
    return tuple(centered)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    methods = tuple(dict.fromkeys(str(row["method"]) for row in rows))
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([float(row[key]) for row in subset]))
            for key in ("mse", "psnr", "ssim")
        }
    return summary


def _posterior_delta_summary(
    rows: list[dict[str, object]],
) -> dict[str, float]:
    center = {
        (row["source"], row["regime"]): float(row["psnr"])
        for row in rows
        if row["method"] == "center_transport_average"
    }
    deltas = np.asarray([
        float(row["psnr"]) - center[(row["source"], row["regime"])]
        for row in rows
        if row["method"] == "joint_blur_noise_posterior"
    ])
    return {
        "posterior_psnr_delta_over_center_mean": float(np.mean(deltas)),
        "posterior_psnr_delta_over_center_minimum": float(np.min(deltas)),
        "posterior_improvement_fraction": float(np.mean(deltas > 0.0)),
    }


def run(
    *,
    size: int = 64,
    passes: int = 24,
    patch_size: int = 32,
    stride: int = 24,
) -> dict[str, object]:
    started = time.perf_counter()
    sources = research_source_portfolio(size)
    regimes = (
        "complementary_global_shift_mix",
        "spatial_covariance_shift_mix",
        "rotational_warp_plus_exposure",
        "common_blur_shift_gauge",
        "photon_limited_complementary",
    )
    rows = []
    audits = {}
    for source_index, (source_name, truth) in enumerate(sources.items()):
        for regime_index, regime in enumerate(regimes):
            seed = 51000 + 101 * source_index + 17 * regime_index
            if regime == "complementary_global_shift_mix":
                observations = _global_observations(
                    truth, common=False, read_sigma=0.002,
                    shot_peak=800.0, source_seed=seed)
            elif regime == "spatial_covariance_shift_mix":
                observations = _spatial_covariance_observations(
                    truth, source_seed=seed)
            elif regime == "rotational_warp_plus_exposure":
                observations = _rotational_observations(
                    truth, source_seed=seed)
            elif regime == "common_blur_shift_gauge":
                observations = _global_observations(
                    truth, common=True, read_sigma=0.002,
                    shot_peak=800.0, source_seed=seed)
            else:
                observations = _global_observations(
                    truth, common=False, read_sigma=0.008,
                    shot_peak=80.0, source_seed=seed)

            result = deblur_multicapture_consensus(
                observations,
                passes=passes,
                descent_method="optimal_positive_line",
                mixing_patch_size=patch_size,
                mixing_stride=stride,
            )
            centered = _centered_observations(result)
            posterior = solve_multicapture_transport_posterior(result)
            raw_average = np.mean(observations, axis=0)
            center_average = np.mean(centered, axis=0)
            best_capture = min(
                observations,
                key=lambda image: float(np.mean((image - truth) ** 2)),
            )
            candidates = {
                "raw_average": raw_average,
                "center_transport_average": center_average,
                "center_first_atlas": result.image,
                "center_inverse_posterior": posterior.inverse_image,
                "joint_blur_noise_posterior": posterior.image,
                "best_capture_evaluation_only": best_capture,
            }
            for method, image in candidates.items():
                rows.append({
                    "source": source_name,
                    "regime": regime,
                    "method": method,
                    **_score(image, truth),
                })
            centered_stack = np.stack(centered)
            closure = float(np.sqrt(np.mean((
                result.predicted_transport_gauge_observations
                - centered_stack) ** 2)))
            disagreement = float(np.sqrt(np.mean((
                centered_stack - center_average[None, ...]) ** 2)))
            atlas_record = result.diagnostics["spatial_mixing_atlas"]
            noise_records = [estimate_noise_discrepancy(
                observation, prediction)
                for observation, prediction in zip(
                    centered,
                    result.predicted_transport_gauge_observations,
                )]
            audits[f"{source_name}/{regime}"] = {
                "center_average_disagreement_rms": disagreement,
                "forward_closure_rms": closure,
                "forward_closure_over_center_average": (
                    closure / max(disagreement, 1e-12)),
                "edge_concentration_over_center_average": (
                    _gradient_energy(result.image)
                    / max(_gradient_energy(center_average), 1e-12)),
                "fourier_circle_amplification": _fourier_amplification(
                    result.image, centered),
                "uncertainty_rms": float(np.sqrt(np.mean(
                    result.uncertainty ** 2))),
                "estimated_read_sigma_median": float(np.median([
                    item.read_sigma for item in noise_records])),
                "estimated_structured_residual_rms_median": float(np.median([
                    item.structured_rms for item in noise_records])),
                "estimated_outlier_fraction_median": float(np.median([
                    item.outlier_fraction for item in noise_records])),
                "passes_used": result.diagnostics["passes_used"],
                "stopped_by": result.diagnostics["stopped_by"],
                "operator_batch_backend": result.diagnostics[
                    "operator_batch_backend"],
                "local_deviation_authority_range": atlas_record[
                    "local_deviation_authority_range"],
                "posterior": posterior.diagnostics,
            }
    by_regime = {}
    for regime in regimes:
        regime_rows = [row for row in rows if row["regime"] == regime]
        atlas = [row for row in regime_rows if row["method"] == "center_first_atlas"]
        posterior = [row for row in regime_rows
                     if row["method"] == "joint_blur_noise_posterior"]
        center = [row for row in regime_rows
                  if row["method"] == "center_transport_average"]
        improvements = np.asarray([
            float(a["psnr"]) - float(c["psnr"])
            for a, c in zip(atlas, center)
        ])
        posterior_improvements = np.asarray([
            float(a["psnr"]) - float(c["psnr"])
            for a, c in zip(posterior, center)
        ])
        by_regime[regime] = {
            "summary": _summarize(regime_rows),
            "atlas_psnr_delta_over_center_mean": float(np.mean(improvements)),
            "atlas_psnr_delta_over_center_minimum": float(np.min(improvements)),
            "atlas_improvement_fraction": float(np.mean(improvements > 0.0)),
            "posterior_psnr_delta_over_center_mean": float(np.mean(
                posterior_improvements)),
            "posterior_psnr_delta_over_center_minimum": float(np.min(
                posterior_improvements)),
            "posterior_improvement_fraction": float(np.mean(
                posterior_improvements > 0.0)),
        }
    all_atlas = [row for row in rows if row["method"] == "center_first_atlas"]
    all_center = [row for row in rows
                  if row["method"] == "center_transport_average"]
    all_posterior = [row for row in rows
                     if row["method"] == "joint_blur_noise_posterior"]
    all_improvements = np.asarray([
        float(a["psnr"]) - float(c["psnr"])
        for a, c in zip(all_atlas, all_center)
    ])
    all_posterior_improvements = np.asarray([
        float(a["psnr"]) - float(c["psnr"])
        for a, c in zip(all_posterior, all_center)
    ])
    development_rows = [
        row for row in rows
        if not str(row["source"]).startswith("v3_skimage/")
    ]
    chronological_holdout_rows = [
        row for row in rows
        if str(row["source"]).startswith("v3_skimage/")
    ]
    return {
        "experiment": "center_first_no_selection_generalization_v2",
        "size": int(size),
        "passes": int(passes),
        "patch_size": int(patch_size),
        "stride": int(stride),
        "source_count": len(sources),
        "sources": list(sources),
        "v3_skimage_source_count": sum(
            name.startswith("v3_skimage/") for name in sources),
        "regimes": list(regimes),
        "selection_policy": (
            "one_continuous_center_inverse_noise_posterior_for_every_source_"
            "and_regime_no_blur_family_source_or_capture_selection"),
        "by_source_portfolio": {
            "denoiser_development": {
                "source_count": sum(
                    not name.startswith("v3_skimage/") for name in sources),
                **_posterior_delta_summary(development_rows),
            },
            "v3_chronological_holdout": {
                "source_count": sum(
                    name.startswith("v3_skimage/") for name in sources),
                "method_inheritance": "none",
                **_posterior_delta_summary(chronological_holdout_rows),
            },
        },
        "by_regime": by_regime,
        "overall_summary": _summarize(rows),
        "overall_atlas_psnr_delta_over_center_mean": float(np.mean(
            all_improvements)),
        "overall_atlas_psnr_delta_over_center_minimum": float(np.min(
            all_improvements)),
        "overall_atlas_improvement_fraction": float(np.mean(
            all_improvements > 0.0)),
        "overall_posterior_psnr_delta_over_center_mean": float(np.mean(
            all_posterior_improvements)),
        "overall_posterior_psnr_delta_over_center_minimum": float(np.min(
            all_posterior_improvements)),
        "overall_posterior_improvement_fraction": float(np.mean(
            all_posterior_improvements > 0.0)),
        "audits": audits,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        size=args.size,
        passes=args.passes,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "source_count": report["source_count"],
        "v3_skimage_source_count": report["v3_skimage_source_count"],
        "by_regime": {
            name: {
                "psnr_delta": record[
                    "atlas_psnr_delta_over_center_mean"],
                "minimum_delta": record[
                    "atlas_psnr_delta_over_center_minimum"],
                "improvement_fraction": record[
                    "atlas_improvement_fraction"],
                "posterior_psnr_delta": record[
                    "posterior_psnr_delta_over_center_mean"],
                "posterior_minimum_delta": record[
                    "posterior_psnr_delta_over_center_minimum"],
                "posterior_improvement_fraction": record[
                    "posterior_improvement_fraction"],
            }
            for name, record in report["by_regime"].items()
        },
        "overall_psnr_delta": report[
            "overall_atlas_psnr_delta_over_center_mean"],
        "overall_improvement_fraction": report[
            "overall_atlas_improvement_fraction"],
        "overall_posterior_psnr_delta": report[
            "overall_posterior_psnr_delta_over_center_mean"],
        "overall_posterior_improvement_fraction": report[
            "overall_posterior_improvement_fraction"],
        "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
