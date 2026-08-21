"""Immutable real-pair evaluation for the unified positive flow atlas.

No-reference image quality is not truth.  This module therefore records the
source bytes, reconstructs both transported observations through the fitted
forward model, and reports several independent diagnostics rather than
collapsing them into a preferred-frame or preferred-method score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from .dense_estimation import _luminance
from .flow_fiber_estimation import (
    FlowFiberConsensusResult,
    deblur_flow_fiber_consensus,
)
from .radiometric_transport import transport_radiometric_pair


@dataclass(frozen=True)
class CapturePairEvaluation:
    result: FlowFiberConsensusResult
    predicted_observations: np.ndarray
    diagnostics: dict[str, object]


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _gradient_energy(image: np.ndarray) -> float:
    value = _luminance(image)
    dx = np.diff(value, axis=1)
    dy = np.diff(value, axis=0)
    return float(0.5 * (np.mean(np.abs(dx)) + np.mean(np.abs(dy))))


def _local_envelope_excursion(
    image: np.ndarray,
    observations: Sequence[np.ndarray],
    *,
    radius: int = 2,
) -> dict[str, float]:
    output = _luminance(image)
    stack = np.stack([_luminance(item) for item in observations], axis=0)
    lower = minimum_filter(np.min(stack, axis=0), size=2 * radius + 1,
                           mode="reflect")
    upper = maximum_filter(np.max(stack, axis=0), size=2 * radius + 1,
                           mode="reflect")
    excursion = np.maximum(lower - output, 0.0) + np.maximum(output - upper, 0.0)
    return {
        "mean": float(np.mean(excursion)),
        "q95": float(np.quantile(excursion, 0.95)),
        "maximum": float(np.max(excursion)),
        "nonzero_fraction": float(np.mean(excursion > 1e-6)),
        "radius_pixels": int(radius),
    }


def _radial_fourier_power(image: np.ndarray, annuli: int = 10) -> np.ndarray:
    value = _luminance(image)
    value = value - np.mean(value)
    height, width = value.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fftshift(np.fft.fft2(value * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0.0, np.sqrt(0.5), annuli + 1)
    profile = np.empty(annuli, dtype=np.float64)
    for index in range(annuli):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        profile[index] = float(np.mean(power[mask])) if np.any(mask) else 0.0
    return profile


def _fourier_amplification(
    image: np.ndarray,
    observations: Sequence[np.ndarray],
) -> dict[str, object]:
    output = _radial_fourier_power(image)
    reference = np.mean(np.stack([
        _radial_fourier_power(item) for item in observations
    ], axis=0), axis=0)
    # Normalize away overall contrast so the record isolates redistribution
    # across Fourier circles rather than exposure gain.
    output /= max(float(np.sum(output)), np.finfo(float).tiny)
    reference /= max(float(np.sum(reference)), np.finfo(float).tiny)
    ratio = output / np.maximum(reference, np.finfo(float).tiny)
    return {
        "annular_power_ratio": ratio.tolist(),
        "maximum_ratio": float(np.max(ratio)),
        "outer_three_mean_ratio": float(np.mean(ratio[-3:])),
        "role": (
            "ringing_warning_not_quality_objective_or_method_selection"),
    }


def evaluate_capture_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    duty_cycle: float = 0.0,
    passes: int = 64,
    automatic_relative_mixing: bool = True,
) -> CapturePairEvaluation:
    """Deblur a pair and audit its ability to explain both observations."""
    raw = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    if raw[0].shape != raw[1].shape:
        raise ValueError("real capture pair must share one raster")
    if any(np.any(~np.isfinite(item)) for item in raw):
        raise ValueError("real capture pair must be finite")
    radiometric = transport_radiometric_pair(*raw)
    result = deblur_flow_fiber_consensus(
        *raw, duty_cycle=duty_cycle, passes=passes,
        automatic_relative_mixing=automatic_relative_mixing)
    if result.fiber_solution is None:
        predictions = np.stack((result.image, result.image), axis=0)
        prediction_source = "common_gauge_abstention"
    else:
        predictions = result.fiber_solution.predicted_observations
        prediction_source = "positive_flow_atlas_forward_model"
    transported = np.stack(radiometric.images, axis=0)
    residual = predictions - transported
    precision = radiometric.precision
    precision_for_image = (
        precision if residual.ndim == 3 else precision[..., None])
    forward_rms = float(np.sqrt(
        np.sum(precision_for_image * residual * residual)
        / max(float(np.sum(np.broadcast_to(
            precision_for_image, residual.shape))), 1e-12)
    ))
    common = np.sum(
        precision_for_image * transported, axis=0
    ) / np.maximum(np.sum(precision_for_image, axis=0), 1e-12)
    baseline_residual = transported - common[None, ...]
    baseline_rms = float(np.sqrt(
        np.sum(precision_for_image * baseline_residual * baseline_residual)
        / max(float(np.sum(np.broadcast_to(
            precision_for_image, baseline_residual.shape))), 1e-12)
    ))
    observation_edges = float(np.mean([
        _gradient_energy(item) for item in radiometric.images
    ]))
    output_edges = _gradient_energy(result.image)
    uncertainty = np.asarray(result.uncertainty, dtype=np.float64)
    diagnostics = {
        "evaluation_method": "immutable_pair_forward_closure_audit_v1",
        "truth_status": "no_reference_not_ground_truth",
        "prediction_source": prediction_source,
        "forward_closure_rms": forward_rms,
        "shared_radiometric_average_residual_rms": baseline_rms,
        "forward_closure_over_pair_disagreement": (
            forward_rms / max(baseline_rms, 1e-12)),
        "edge_energy_observation_mean": observation_edges,
        "edge_energy_output": output_edges,
        "edge_concentration_ratio": (
            output_edges / max(observation_edges, 1e-12)),
        "local_observation_envelope_excursion": _local_envelope_excursion(
            result.image, radiometric.images),
        "fourier_circle_amplification": _fourier_amplification(
            result.image, radiometric.images),
        "uncertainty_rms": float(np.sqrt(np.mean(uncertainty ** 2))),
        "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
        "algorithm_diagnostics": _json_ready(result.diagnostics),
        "interpretation": (
            "forward closure tests observation consistency; edge and Fourier "
            "records expose possible sharpening or ringing but cannot prove "
            "fidelity without a sharp reference"),
    }
    return CapturePairEvaluation(result, predictions, diagnostics)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as source:
        source.load()
        mode = "L" if source.mode in ("1", "L", "I", "F") else "RGB"
        converted = source.convert(mode)
        array = np.asarray(converted, dtype=np.float64) / 255.0
    return array, mode


def _save_image(path: Path, image: np.ndarray) -> None:
    value = np.clip(np.asarray(image), 0.0, 1.0)
    Image.fromarray(np.round(255.0 * value).astype(np.uint8)).save(path)


def evaluate_capture_files(
    first_path: Path,
    second_path: Path,
    output: Path,
    *,
    duty_cycle: float = 0.0,
    passes: int = 64,
    automatic_relative_mixing: bool = True,
) -> dict[str, object]:
    """Evaluate two files while proving their source bytes were untouched."""
    paths = (first_path.expanduser().resolve(), second_path.expanduser().resolve())
    if paths[0] == paths[1]:
        raise ValueError("capture paths must name two distinct source files")
    before = [_sha256(path) for path in paths]
    arrays_and_modes = [_load_image(path) for path in paths]
    arrays = [item[0] for item in arrays_and_modes]
    evaluation = evaluate_capture_pair(
        arrays[0], arrays[1], duty_cycle=duty_cycle, passes=passes,
        automatic_relative_mixing=automatic_relative_mixing)
    output.mkdir(parents=True, exist_ok=True)
    _save_image(output / "deblurred.png", evaluation.result.image)
    uncertainty = np.asarray(evaluation.result.uncertainty)
    if uncertainty.ndim == 3:
        uncertainty = np.mean(uncertainty, axis=2)
    uncertainty /= max(float(np.quantile(uncertainty, 0.99)), 1e-12)
    _save_image(output / "uncertainty.png", uncertainty)
    for index, prediction in enumerate(evaluation.predicted_observations):
        _save_image(output / f"predicted_observation_{index}.png", prediction)
    after = [_sha256(path) for path in paths]
    provenance = [{
        "path": str(path),
        "sha256_before": before[index],
        "sha256_after": after[index],
        "source_unchanged": before[index] == after[index],
        "byte_count": path.stat().st_size,
        "decoded_shape": list(arrays[index].shape),
        "decoded_mode": arrays_and_modes[index][1],
    } for index, path in enumerate(paths)]
    report = {
        **evaluation.diagnostics,
        "source_provenance": provenance,
        "all_sources_unchanged": all(
            item["source_unchanged"] for item in provenance),
        "outputs": {
            "deblurred": str((output / "deblurred.png").resolve()),
            "uncertainty": str((output / "uncertainty.png").resolve()),
            "predicted_observations": [
                str((output / f"predicted_observation_{index}.png").resolve())
                for index in range(2)
            ],
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duty-cycle", type=float, default=0.0)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--disable-relative-mixing", action="store_true",
        help="diagnostic ablation: remove identifiable differential mixing",
    )
    args = parser.parse_args()
    report = evaluate_capture_files(
        args.first, args.second, args.out,
        duty_cycle=args.duty_cycle, passes=args.passes,
        automatic_relative_mixing=not args.disable_relative_mixing)
    print(json.dumps({
        "all_sources_unchanged": report["all_sources_unchanged"],
        "forward_closure_rms": report["forward_closure_rms"],
        "forward_closure_over_pair_disagreement": report[
            "forward_closure_over_pair_disagreement"],
        "edge_concentration_ratio": report["edge_concentration_ratio"],
        "fourier_outer_ratio": report[
            "fourier_circle_amplification"]["outer_three_mean_ratio"],
        "output": report["outputs"]["deblurred"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
