#!/usr/bin/env python3
"""Compare N=2048 phase fusion with 1024/2048/4096 densification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.io import wavfile

from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    DEFAULT_OFFSETS,
    fourier_unrolled_densified,
)
from experiments.ostensibly_frontend.uncertainty_fusion import (
    strongest_ridge_audit,
)
from experiments.ostensibly_frontend.supported_hodge_fusion import (
    supported_hodge_fusion,
)


def _audit(value: np.ndarray) -> dict[str, object]:
    return {
        key: item for key, item in strongest_ridge_audit(value).items()
        if not isinstance(item, np.ndarray)
    }


def periodic_row_modulation(
    field: np.ndarray,
    *,
    period: int = 16,
) -> dict[str, object]:
    """Measure a repeated row-phase defect after removing the slow envelope."""
    value = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    row_energy = np.mean(np.log1p(
        value / max(float(np.quantile(value, 0.90)), 1e-30)), axis=1)
    residual = row_energy - ndi.gaussian_filter1d(
        row_energy, sigma=float(period), mode="reflect")
    phase_means = np.asarray([
        np.mean(residual[phase::period]) for phase in range(period)
    ])
    residual_scale = max(float(np.std(residual)), 1e-30)
    return {
        "period_rows": period,
        "phase_means": phase_means.tolist(),
        "phase_peak_to_peak_over_residual_std": float(
            np.ptp(phase_means) / residual_scale),
        "phase_std_over_residual_std": float(
            np.std(phase_means) / residual_scale),
    }


def _display(field: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    positive = value[value > 0.0]
    scale = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    return np.log1p(value / max(scale * 0.015, 1e-30))


def _render(path: Path, result, supported) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _render_with_pillow(path, result, supported)

    baseline = result.aperture_fields[result.apertures.index(2048)]
    addition = np.maximum(result.fused - baseline, 0.0)
    fields = list(result.aperture_fields) + [
        result.fused,
        addition,
        result.support_union,
        supported.field,
        supported.refined_confidence,
        np.hypot(
            supported.transverse_time_derivative,
            supported.transverse_row_derivative,
        ),
    ]
    titles = [f"phase-fused N={value}, mapped to N=2048" for value in result.apertures] + [
        "fourier_unrolled_densified: registered per-pixel maximum",
        "positive contribution beyond N=2048 alone",
        "three-FFT reassigned power: alignment landmarks only",
        "supported Hodge fusion: integrable short-time + long-frequency field",
        "accepted independent support after one residual remeasurement",
        "rejected transverse contradiction (cannot alter scalar readout)",
    ]
    figure, axes = plt.subplots(len(fields), 1, figsize=(14, 20), sharex=True)
    for axis, field, title in zip(axes, fields, titles):
        axis.imshow(np.flipud(_display(field)), aspect="auto", cmap="magma")
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel("row")
    axes[-1].set_xlabel("frame (hop 512)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return "matplotlib"


def _render_with_pillow(path: Path, result, supported) -> str:
    """Dependency-light static renderer for the project measurement venv."""
    from PIL import Image, ImageDraw

    baseline = result.aperture_fields[result.apertures.index(2048)]
    fields = list(result.aperture_fields) + [
        result.fused,
        np.maximum(result.fused - baseline, 0.0),
        result.support_union,
        supported.field,
        supported.refined_confidence,
        np.hypot(
            supported.transverse_time_derivative,
            supported.transverse_row_derivative,
        ),
    ]
    titles = [f"N={value} mapped to N=2048" for value in result.apertures] + [
        "fourier_unrolled_densified",
        "positive contribution beyond N=2048",
        "reassigned support (alignment only)",
        "supported Hodge fusion",
        "accepted support after residualization",
        "rejected transverse contradiction",
    ]
    stops = np.asarray((
        (2, 7, 30), (43, 15, 87), (105, 28, 109),
        (181, 54, 86), (238, 116, 51), (252, 238, 95),
    ), dtype=np.float64)
    positions = np.linspace(0.0, 1.0, stops.shape[0])
    width, panel_height, label_height = 1400, 220, 30
    canvas = Image.new("RGB", (width, len(fields) * (panel_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (field, title) in enumerate(zip(fields, titles)):
        display = _display(field)
        display /= max(float(np.quantile(display, 0.999)), 1e-30)
        display = np.clip(np.flipud(display), 0.0, 1.0)
        flat = display.ravel()
        channel = np.stack([
            np.interp(flat, positions, stops[:, component])
            for component in range(3)
        ], axis=1).reshape((*display.shape, 3)).astype(np.uint8)
        image = Image.fromarray(channel).resize(
            (width, panel_height), resample=Image.Resampling.BILINEAR)
        top = index * (panel_height + label_height)
        draw.text((8, top + 7), title, fill="black")
        canvas.paste(image, (0, top + label_height))
    canvas.save(path)
    return "pillow"


def run(wav: Path, out: Path, *, duration: float) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    sample_rate, samples = wavfile.read(wav)
    samples = np.asarray(samples[: int(round(duration * sample_rate))])
    started = time.perf_counter()
    result = fourier_unrolled_densified(
        samples,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        target_crop_rows=256,
        hop_length=512,
        offsets=DEFAULT_OFFSETS,
    )
    supported = supported_hodge_fusion(result)
    elapsed = time.perf_counter() - started
    base_index = result.apertures.index(2048)
    baseline = result.aperture_fields[base_index]
    aperture_audits = {
        str(n_fft): {
            "ridge": _audit(field),
            "periodic_16_row_modulation": periodic_row_modulation(field),
        }
        for n_fft, field in zip(result.apertures, result.aperture_fields)
    }
    baseline_audit = _audit(baseline)
    fused_audit = _audit(result.fused)
    supported_audit = _audit(supported.field)
    report = {
        "wav": str(wav),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(samples.shape[0] / sample_rate),
        "apertures": list(result.apertures),
        "target_n_fft": result.target_n_fft,
        "target_crop_rows": result.target_crop_rows,
        "source_crop_rows": list(result.source_crop_rows),
        "offsets_samples": list(DEFAULT_OFFSETS),
        "shape": list(result.fused.shape),
        "elapsed_seconds": elapsed,
        "terminal_semantics": (
            "literal_per_pixel_maximum_of_registered_linear_double_irfft_fields"
        ),
        "reassigned_power_semantics": (
            "coordinate_witness_for_cross_aperture_transport_only"
        ),
        "aperture_audits": aperture_audits,
        "baseline_2048": baseline_audit,
        "densified": {
            "ridge": fused_audit,
            "periodic_16_row_modulation": periodic_row_modulation(result.fused),
        },
        "supported_hodge": {
            "ridge": supported_audit,
            "periodic_16_row_modulation": periodic_row_modulation(
                supported.field),
            "diagnostics": supported.diagnostics,
        },
        "change": {
            "void_fraction_delta": float(
                fused_audit["void_fraction"] - baseline_audit["void_fraction"]),
            "path_q10_ratio": float(
                fused_audit["path_q10"]
                / max(float(baseline_audit["path_q10"]), 1e-30)),
            "path_median_ratio": float(
                fused_audit["path_median"]
                / max(float(baseline_audit["path_median"]), 1e-30)),
            "supported_void_fraction_delta": float(
                supported_audit["void_fraction"]
                - baseline_audit["void_fraction"]),
            "supported_path_q10_ratio": float(
                supported_audit["path_q10"]
                / max(float(baseline_audit["path_q10"]), 1e-30)),
            "supported_path_median_ratio": float(
                supported_audit["path_median"]
                / max(float(baseline_audit["path_median"]), 1e-30)),
        },
        "aperture_contribution": {},
        "cross_aperture_registration": list(result.diagnostics),
    }
    winner = np.argmax(result.aligned_fields, axis=0)
    active_threshold = 0.03 * max(float(np.quantile(result.fused, 0.995)), 1e-30)
    active = result.fused > active_threshold
    addition = np.maximum(result.fused - baseline, 0.0)
    report["aperture_contribution"] = {
        "winner_fraction_all_pixels": {
            str(n_fft): float(np.mean(winner == index))
            for index, n_fft in enumerate(result.apertures)
        },
        "winner_fraction_active_pixels": {
            str(n_fft): float(np.mean(winner[active] == index))
            for index, n_fft in enumerate(result.apertures)
        },
        "added_l1_fraction_of_fused": float(
            np.sum(addition) / max(float(np.sum(result.fused)), 1e-30)),
        "material_addition_fraction_of_active_pixels": float(np.mean(
            addition[active] > active_threshold)),
    }
    np.savez_compressed(
        out / "fourier_unrolled_densified.npz",
        apertures=np.asarray(result.apertures),
        centers=result.centers,
        source_crop_rows=np.asarray(result.source_crop_rows),
        aperture_fields=result.aperture_fields,
        reassigned_support=result.reassigned_support,
        aligned_fields=result.aligned_fields,
        aligned_support=result.aligned_support,
        flows=result.flows,
        fused=result.fused,
        support_union=result.support_union,
        supported_hodge=supported.field,
        supported_hodge_initial=supported.initial_field,
        supported_confidence=supported.support_confidence,
        supported_refined_confidence=supported.refined_confidence,
        supported_transverse_time=supported.transverse_time_derivative,
        supported_transverse_row=supported.transverse_row_derivative,
    )
    report["static_renderer"] = _render(
        out / "fourier_unrolled_densified.png", result, supported)
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args.wav, args.out, duration=args.duration), indent=2))


if __name__ == "__main__":
    main()
