#!/usr/bin/env python3
"""Estimate the Köhler full-K4 gauge without constructing exposure fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .full_quartic_transport import estimate_full_quartic_transport
from .multicapture_transport import (
    _graph_coordinates,
    _minimum_trace_positive_covariances,
)
from .radiometric_transport import _quantile_log_gain
from .relative_mixing_transport import (
    estimate_relative_mixing_from_spectra,
    prepare_mixing_magnitude_spectrum,
)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def run(data: Path) -> dict[str, object]:
    started = time.perf_counter()
    paths = [data / f"blurry_1_{index}.jpg" for index in range(1, 13)]
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    raw = tuple(_load(path) for path in paths)
    count = len(raw)
    edges = tuple(combinations(range(count), 2))
    gain_values = []
    gain_weights = []
    for first, second in edges:
        log_gain, support = _quantile_log_gain(raw[first], raw[second])
        gain_values.append(float(np.clip(log_gain, -np.log(8.0), np.log(8.0))))
        gain_weights.append(max(float(support) / 38.0, 1e-3))
    log_exposures, _ = _graph_coordinates(
        count, edges, np.asarray(gain_values), np.asarray(gain_weights))
    log_exposures = log_exposures[:, 0]
    radiometric_authority = float(1.0 - np.exp(-(
        np.std(log_exposures) / 0.12) ** 4))
    normalized = tuple(
        (1.0 - radiometric_authority) * image
        + radiometric_authority * image / np.exp(log_exposures[index])
        for index, image in enumerate(raw)
    )
    spectra = tuple(
        prepare_mixing_magnitude_spectrum(item) for item in normalized)
    covariance_values = []
    covariance_weights = []
    for first, second in edges:
        estimate = estimate_relative_mixing_from_spectra(
            spectra[first], spectra[second])
        covariance = estimate.covariance_difference_second_minus_first
        covariance_values.append((
            covariance[0, 0], covariance[0, 1], covariance[1, 1]))
        covariance_weights.append(max(estimate.authority ** 2, 1e-4))
    components, graph_residual = _graph_coordinates(
        count,
        edges,
        np.asarray(covariance_values),
        np.asarray(covariance_weights),
    )
    relative = np.empty((count, 2, 2), dtype=np.float64)
    relative[:, 0, 0] = components[:, 0]
    relative[:, 0, 1] = components[:, 1]
    relative[:, 1, 0] = components[:, 1]
    relative[:, 1, 1] = components[:, 2]
    covariances, common_covariance_gauge, _ = (
        _minimum_trace_positive_covariances(relative))
    shape = estimate_full_quartic_transport(normalized, covariances)
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return {
        "experiment": "koehler_scene1_full_quartic_estimation_only_v1",
        "capture_count": count,
        "shape_authority": shape.authority,
        "maximum_transported_tensor_magnitude": float(np.max(np.abs(
            shape.standardized_cumulants))),
        "common_covariance_gauge": common_covariance_gauge.tolist(),
        "covariance_graph_residual_rms": float(np.sqrt(np.mean(
            graph_residual * graph_residual))),
        "full_quartic_diagnostics": shape.diagnostics,
        "all_sources_unchanged": before == after,
        "source_sha256": after,
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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.data)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(json.dumps({
        "shape_authority": report["shape_authority"],
        "maximum_transported_tensor_magnitude": report[
            "maximum_transported_tensor_magnitude"],
        "crossfit_predictive_authority": report[
            "full_quartic_diagnostics"]["crossfit_predictive_authority"],
        "baseline_rms": report["full_quartic_diagnostics"][
            "baseline_relative_log_magnitude_rms"],
        "fitted_rms": report["full_quartic_diagnostics"][
            "fitted_relative_log_magnitude_rms"],
        "wall_seconds": report["wall_seconds"],
        "all_sources_unchanged": report["all_sources_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
