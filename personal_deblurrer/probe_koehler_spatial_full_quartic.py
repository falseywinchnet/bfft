#!/usr/bin/env python3
"""Measure full-K4 evidence inside the accepted center-first atlas charts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .dense_estimation import _luminance
from .full_quartic_transport import estimate_full_quartic_transport
from .multicapture_transport import (
    deblur_multicapture_consensus,
    estimate_spatial_mixing_covariance_atlas,
)
from .spatial_transport import (
    CompactGlobalExposureField,
    pullback_compact_global_values,
)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    data: Path,
    *,
    patch_size: int = 192,
    stride: int = 128,
) -> dict[str, object]:
    started = time.perf_counter()
    paths = tuple(data / f"blurry_1_{index}.jpg" for index in range(1, 13))
    before = {path.name: _sha256(path) for path in paths}
    raw = tuple(_load(path) for path in paths)
    coordinates = deblur_multicapture_consensus(
        raw, passes=0, descent_method="optimal_positive_line")
    if not all(
        isinstance(field, CompactGlobalExposureField)
        for field in coordinates.fields
    ):
        raise RuntimeError("spatial K4 probe requires compact center fields")
    centered = tuple(
        pullback_compact_global_values(image, field)[0]
        for image, field in zip(
            coordinates.radiometric_images, coordinates.fields)
    )
    covariance_fields, _, atlas = estimate_spatial_mixing_covariance_atlas(
        centered,
        patch_size=patch_size,
        stride=stride,
    )
    extent = int(atlas["patch_size"])
    half = extent // 2
    pad_after = extent - half
    padded = tuple(np.pad(
        _luminance(image),
        ((half, pad_after), (half, pad_after)),
        mode="reflect",
    ) for image in centered)
    chart_records = []
    for covariance_record in atlas["chart_records"]:
        center_x, center_y = covariance_record["center_xy"]
        patches = tuple(
            image[
                center_y:center_y + extent,
                center_x:center_x + extent,
            ]
            for image in padded
        )
        chart_covariances = covariance_fields[:, center_y, center_x]
        adaptive_range = covariance_record[
            "adaptive_maximum_frequency_range"]
        maximum_frequency = max(
            0.12, min(0.30, 2.0 * max(adaptive_range)))
        estimate = estimate_full_quartic_transport(
            patches,
            chart_covariances,
            maximum_frequency=maximum_frequency,
        )
        chart_records.append({
            "center_xy": [center_x, center_y],
            "positive_chart_mass": covariance_record[
                "positive_chart_mass"],
            "local_covariance_deviation_authority": covariance_record[
                "local_deviation_authority"],
            "maximum_frequency_cycles_per_pixel": maximum_frequency,
            "shape_authority": estimate.authority,
            "crossfit_predictive_authority": estimate.diagnostics[
                "crossfit_predictive_authority"],
            "tensor_signal_authority": estimate.diagnostics[
                "tensor_signal_authority"],
            "baseline_relative_log_magnitude_rms": estimate.diagnostics[
                "baseline_relative_log_magnitude_rms"],
            "fitted_relative_log_magnitude_rms": estimate.diagnostics[
                "fitted_relative_log_magnitude_rms"],
            "maximum_transported_tensor_magnitude": float(np.max(np.abs(
                estimate.standardized_cumulants))),
            "standardized_cumulants": (
                estimate.standardized_cumulants.tolist()),
            "dictionary_weights": estimate.dictionary_weights.tolist(),
            "active_dictionary_components": estimate.diagnostics[
                "active_dictionary_components"],
            "chart_covariances": chart_covariances.tolist(),
        })
    authority = np.asarray([
        item["shape_authority"] for item in chart_records])
    crossfit = np.asarray([
        item["crossfit_predictive_authority"] for item in chart_records])
    chart_mass = np.asarray([
        item["positive_chart_mass"] for item in chart_records])
    local_covariance_authority = np.asarray([
        item["local_covariance_deviation_authority"]
        for item in chart_records])
    combined_authority = authority * local_covariance_authority
    after = {path.name: _sha256(path) for path in paths}
    return {
        "experiment": "koehler_scene1_spatial_full_quartic_probe_v1",
        "capture_count": len(centered),
        "patch_size": extent,
        "stride": int(atlas["stride"]),
        "chart_count": len(chart_records),
        "estimation_order": (
            "radiometric_then_deterministic_center_then_covariance_then_k4"),
        "selection_policy": (
            "all_overlapping_charts_and_captures_retain_positive_mass"),
        "shape_authority_range": [
            float(np.min(authority)), float(np.max(authority))],
        "shape_authority_weighted_mean": float(np.sum(
            chart_mass * authority) / np.sum(chart_mass)),
        "crossfit_predictive_authority_range": [
            float(np.min(crossfit)), float(np.max(crossfit))],
        "combined_covariance_k4_authority_range": [
            float(np.min(combined_authority)),
            float(np.max(combined_authority)),
        ],
        "nonzero_shape_chart_fraction": float(np.mean(authority > 1e-8)),
        "chart_records": chart_records,
        "covariance_atlas_summary": atlas,
        "source_sha256": after,
        "all_sources_unchanged": before == after,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg"),
    )
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.data, patch_size=args.patch_size, stride=args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "chart_count": report["chart_count"],
        "shape_authority_range": report["shape_authority_range"],
        "shape_authority_weighted_mean": report[
            "shape_authority_weighted_mean"],
        "crossfit_predictive_authority_range": report[
            "crossfit_predictive_authority_range"],
        "combined_covariance_k4_authority_range": report[
            "combined_covariance_k4_authority_range"],
        "nonzero_shape_chart_fraction": report[
            "nonzero_shape_chart_fraction"],
        "wall_seconds": report["wall_seconds"],
        "all_sources_unchanged": report["all_sources_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
