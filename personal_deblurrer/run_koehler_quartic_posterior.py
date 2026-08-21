#!/usr/bin/env python3
"""Run the real twelve-capture covariance/K4 image posterior gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .multicapture_transport import deblur_multicapture_consensus
from .quartic_gauge_posterior import solve_quartic_gauge_posterior
from .radiometric_transport import _soft_sensor_precision
from .real_capture_evaluation import (
    _fourier_amplification,
    _gradient_energy,
    _local_envelope_excursion,
    _save_image,
)
from .run_koehler_multicapture_benchmark import _reference_score
from .spatial_transport import (
    CompactGlobalExposureField,
    pullback_compact_global_values,
)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weighted_rms(
    residual: np.ndarray,
    weights: np.ndarray,
) -> float:
    weight = weights if residual.ndim == 3 else weights[..., None]
    return float(np.sqrt(
        np.sum(weight * residual * residual)
        / max(float(np.sum(np.broadcast_to(weight, residual.shape))), 1e-12)))


def run(
    data: Path,
    output: Path,
    *,
    passes: int = 32,
    accepted_adaptive_result: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    capture_paths = tuple(
        data / f"blurry_1_{index}.jpg" for index in range(1, 13))
    reference_path = data / "ground_truth_1.jpg"
    source_paths = (*capture_paths, reference_path)
    before = {path.name: _sha256(path) for path in source_paths}
    raw = tuple(_load(path) for path in capture_paths)
    reference = _load(reference_path)

    # Reuse the exchange-symmetric radiometric, center, and covariance graph.
    # Zero passes asks only for its measured coordinates; the reconstruction
    # below is performed by the two-measure posterior.
    coordinates = deblur_multicapture_consensus(
        raw,
        passes=0,
        descent_method="optimal_positive_line",
    )
    if not all(
        isinstance(field, CompactGlobalExposureField)
        for field in coordinates.fields
    ):
        raise RuntimeError("real quartic gate requires compact global fields")
    centered_images = []
    centered_precision = []
    radiometric_authority = float(
        coordinates.diagnostics["radiometric_authority"])
    for raw_image, normalized, field in zip(
        raw, coordinates.radiometric_images, coordinates.fields
    ):
        image, _ = pullback_compact_global_values(normalized, field)
        sensor_precision = (
            (1.0 - radiometric_authority)
            + radiometric_authority * _soft_sensor_precision(raw_image))
        precision, _ = pullback_compact_global_values(
            sensor_precision, field)
        centered_images.append(image)
        centered_precision.append(np.maximum(precision, 0.0))
    centered = tuple(centered_images)
    precision = np.stack(centered_precision, axis=0)
    covariances = np.asarray(
        coordinates.diagnostics["frame_covariances"], dtype=np.float64)

    posterior = solve_quartic_gauge_posterior(
        centered,
        covariances,
        frame_weights=precision / len(centered),
        passes=passes,
        descent_method="optimal_positive_line",
    )
    center_average = np.sum(
        precision[..., None] * np.stack(centered, axis=0), axis=0
    ) / np.maximum(np.sum(precision, axis=0)[..., None], 1e-12)
    candidates = {
        "center_transport_average": center_average,
        "global_covariance_transport": posterior.covariance_solution.image,
        "relative_k4_transport": posterior.quartic_solution.image,
        "covariance_k4_posterior": posterior.image,
    }
    if accepted_adaptive_result is not None and accepted_adaptive_result.exists():
        candidates["accepted_adaptive_covariance_atlas"] = _load(
            accepted_adaptive_result)
    reference_scores = {
        name: _reference_score(image, reference)
        for name, image in candidates.items()
    }
    centered_stack = np.stack(centered, axis=0)
    center_disagreement = _weighted_rms(
        centered_stack - center_average[None, ...], precision)
    branch_records = {}
    for name, image in candidates.items():
        branch_records[name] = {
            "web_reference": reference_scores[name],
            "edge_concentration_over_center_average": (
                _gradient_energy(image)
                / max(_gradient_energy(center_average), 1e-12)),
            "local_centered_observation_envelope_excursion": (
                _local_envelope_excursion(image, centered)),
            "fourier_circle_amplification": _fourier_amplification(
                image, centered),
        }
    covariance_closure = float(posterior.diagnostics[
        "covariance_forward_closure_rms"])
    quartic_closure = float(posterior.diagnostics[
        "quartic_forward_closure_rms"])
    posterior_predictions = (
        posterior.predicted_transport_gauge_observations)
    posterior_closure = _weighted_rms(
        posterior_predictions - centered_stack, precision)

    output.mkdir(parents=True, exist_ok=True)
    for name, image in candidates.items():
        _save_image(output / f"{name}.png", image)
    uncertainty = np.mean(posterior.uncertainty, axis=2)
    uncertainty /= max(float(np.quantile(uncertainty, 0.99)), 1e-12)
    _save_image(output / "posterior_uncertainty.png", uncertainty)
    after = {path.name: _sha256(path) for path in source_paths}
    report = {
        "experiment": "koehler_scene1_quartic_image_posterior_v1",
        "status": "measured_not_predeclared",
        "capture_count": len(centered),
        "passes": int(passes),
        "reference_scope": (
            "one compressed scene-1 web JPEG; not the official roughly-200-"
            "sample Koehler trajectory evaluation"),
        "posterior_policy": (
            "covariance_and_relative_k4_retained_as_positive_measures_no_"
            "winner_selection"),
        "reference_scores": reference_scores,
        "branch_records": branch_records,
        "center_average_disagreement_rms": center_disagreement,
        "covariance_forward_closure_rms": covariance_closure,
        "quartic_forward_closure_rms": quartic_closure,
        "posterior_forward_closure_rms": posterior_closure,
        "covariance_closure_over_center_average": (
            covariance_closure / max(center_disagreement, 1e-12)),
        "quartic_closure_over_center_average": (
            quartic_closure / max(center_disagreement, 1e-12)),
        "posterior_closure_over_center_average": (
            posterior_closure / max(center_disagreement, 1e-12)),
        "covariance_posterior_mass": posterior.covariance_posterior_mass,
        "quartic_posterior_mass": posterior.quartic_posterior_mass,
        "quartic_estimation": posterior.quartic_estimate.diagnostics,
        "posterior_diagnostics": posterior.diagnostics,
        "coordinate_diagnostics": coordinates.diagnostics,
        "all_sources_unchanged": before == after,
        "source_sha256": after,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument(
        "--accepted-adaptive-result",
        type=Path,
        default=Path(
            "personal_deblurrer/real_capture_results/"
            "personal_deblurrer_koehler_multicapture_center_first_atlas_v6/"
            "deblurred.png"),
    )
    args = parser.parse_args()
    report = run(
        args.data,
        args.out,
        passes=args.passes,
        accepted_adaptive_result=args.accepted_adaptive_result,
    )
    print(json.dumps({
        "reference_scores": report["reference_scores"],
        "covariance_posterior_mass": report["covariance_posterior_mass"],
        "quartic_posterior_mass": report["quartic_posterior_mass"],
        "closure_ratios": {
            "covariance": report[
                "covariance_closure_over_center_average"],
            "quartic": report["quartic_closure_over_center_average"],
            "posterior": report["posterior_closure_over_center_average"],
        },
        "wall_seconds": report["wall_seconds"],
        "all_sources_unchanged": report["all_sources_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
