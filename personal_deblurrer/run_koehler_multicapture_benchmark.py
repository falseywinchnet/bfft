#!/usr/bin/env python3
"""Run the symmetric twelve-capture transport on Köhler scene 1 web JPEGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from denoiser.run_2d_denoiser_battery import metrics

from .multicapture_transport import deblur_multicapture_consensus
from .multicapture_posterior import solve_multicapture_transport_posterior
from .real_capture_evaluation import (
    _fourier_amplification,
    _gradient_energy,
    _local_envelope_excursion,
    _save_image,
)
from .spatial_transport import pullback_barycentric_coordinates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.asarray((0.2126, 0.7152, 0.0722))


def _reference_score(
    image: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    luminance = _luminance(image)
    record = metrics(
        luminance[32:-32, 32:-32],
        _luminance(reference)[32:-32, 32:-32],
    )
    return {
        "web_reference_psnr": float(-10.0 * math.log10(max(
            float(record["mse"]), np.finfo(float).tiny))),
        "web_reference_ssim": float(record["ssim"]),
    }


def run(
    data: Path,
    output: Path,
    passes: int,
    descent_method: str = "optimal_positive_line",
    mixing_patch_size: int | None = None,
    mixing_stride: int | None = None,
    quartic_shape: bool = False,
    full_quartic_shape: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    capture_paths = [data / f"blurry_1_{index}.jpg" for index in range(1, 13)]
    reference_path = data / "ground_truth_1.jpg"
    before = {path.name: _sha256(path) for path in capture_paths + [reference_path]}
    observations = tuple(_load(path) for path in capture_paths)
    reference = _load(reference_path)
    result = deblur_multicapture_consensus(
        observations,
        passes=passes,
        descent_method=descent_method,
        mixing_patch_size=mixing_patch_size,
        mixing_stride=mixing_stride,
        quartic_shape=quartic_shape,
        full_quartic_shape=full_quartic_shape,
    )
    posterior = solve_multicapture_transport_posterior(result)

    centered_observations = []
    for image, field in zip(result.radiometric_images, result.fields):
        centered, _, _ = pullback_barycentric_coordinates(
            image, field.barycentric_field())
        centered_observations.append(centered)
    centered_average = np.mean(centered_observations, axis=0)
    unregistered_average = np.mean(observations, axis=0)
    candidate_images = {
        "unregistered_average": unregistered_average,
        "center_transport_average": centered_average,
        "multicapture_positive_transport": result.image,
        "center_inverse_noise_posterior": posterior.image,
    }
    individual_scores = [
        _reference_score(image, reference) for image in observations]
    best_capture_psnr = max(
        item["web_reference_psnr"] for item in individual_scores)
    best_capture_ssim = max(
        item["web_reference_ssim"] for item in individual_scores)
    reference_scores = {
        name: _reference_score(image, reference)
        for name, image in candidate_images.items()
    }
    candidate_audits = {
        name: {
            "edge_concentration_over_center_average": (
                _gradient_energy(image)
                / max(_gradient_energy(centered_average), 1e-12)),
            "local_centered_observation_envelope_excursion": (
                _local_envelope_excursion(image, centered_observations)),
            "fourier_circle_amplification": _fourier_amplification(
                image, centered_observations),
        }
        for name, image in candidate_images.items()
    }
    reference_scores["best_individual_capture_oracle"] = {
        "web_reference_psnr": best_capture_psnr,
        "web_reference_ssim": best_capture_ssim,
        "role": "evaluation_only_not_capture_selection_in_reconstruction",
    }
    predictions = result.predicted_transport_gauge_observations
    centered_stack = np.stack(centered_observations, axis=0)
    forward_closure_rms = float(np.sqrt(np.mean(
        (predictions - centered_stack) ** 2)))
    average_closure_rms = float(np.sqrt(np.mean(
        (centered_stack - centered_average[None, ...]) ** 2)))
    output.mkdir(parents=True, exist_ok=True)
    _save_image(output / "deblurred.png", result.image)
    _save_image(output / "posterior_deblurred.png", posterior.image)
    _save_image(output / "center_transport_average.png", centered_average)
    uncertainty = np.mean(result.uncertainty, axis=2)
    uncertainty /= max(float(np.quantile(uncertainty, 0.99)), 1e-12)
    _save_image(output / "uncertainty.png", uncertainty)
    posterior_uncertainty = np.mean(posterior.uncertainty, axis=2)
    posterior_uncertainty /= max(
        float(np.quantile(posterior_uncertainty, 0.99)), 1e-12)
    _save_image(output / "posterior_uncertainty.png", posterior_uncertainty)
    after = {path.name: _sha256(path) for path in capture_paths + [reference_path]}
    report = {
        "experiment": "koehler_scene1_twelve_capture_positive_transport_v1",
        "status": "measured_not_predeclared",
        "capture_count": len(observations),
        "passes": int(passes),
        "descent_method": descent_method,
        "mixing_patch_size": mixing_patch_size,
        "mixing_stride": mixing_stride,
        "quartic_shape": bool(quartic_shape),
        "full_quartic_shape": bool(full_quartic_shape),
        "reference_scope": (
            "one compressed scene-1 web JPEG; not the official roughly-200-"
            "sample Koehler trajectory evaluation"),
        "reference_scores": reference_scores,
        "candidate_audits": candidate_audits,
        "individual_capture_scores": individual_scores,
        "forward_closure_rms": forward_closure_rms,
        "center_average_closure_rms": average_closure_rms,
        "forward_closure_over_center_average": (
            forward_closure_rms / max(average_closure_rms, 1e-12)),
        "edge_concentration_over_center_average": (
            _gradient_energy(result.image)
            / max(_gradient_energy(centered_average), 1e-12)),
        "local_centered_observation_envelope_excursion": (
            _local_envelope_excursion(result.image, centered_observations)),
        "fourier_circle_amplification_over_centered_observations": (
            _fourier_amplification(result.image, centered_observations)),
        "uncertainty_rms": float(np.sqrt(np.mean(result.uncertainty ** 2))),
        "posterior_uncertainty_rms": float(np.sqrt(np.mean(
            posterior.uncertainty ** 2))),
        "posterior_diagnostics": posterior.diagnostics,
        "algorithm_diagnostics": result.diagnostics,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "all_sources_unchanged": before == after,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/real_capture_results/koehler_multicapture"))
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--descent-method",
        choices=("optimal_positive_line", "multiplicative"),
        default="optimal_positive_line",
    )
    parser.add_argument("--mixing-patch-size", type=int)
    parser.add_argument("--mixing-stride", type=int)
    parser.add_argument("--quartic-shape", action="store_true")
    parser.add_argument("--full-quartic-shape", action="store_true")
    args = parser.parse_args()
    report = run(
        args.data,
        args.out,
        args.passes,
        args.descent_method,
        args.mixing_patch_size,
        args.mixing_stride,
        args.quartic_shape,
        args.full_quartic_shape,
    )
    print(json.dumps({
        "reference_scores": report["reference_scores"],
        "forward_closure_over_center_average": report[
            "forward_closure_over_center_average"],
        "fourier_outer_ratio": report[
            "fourier_circle_amplification_over_centered_observations"
        ]["outer_three_mean_ratio"],
        "all_sources_unchanged": report["all_sources_unchanged"],
        "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
