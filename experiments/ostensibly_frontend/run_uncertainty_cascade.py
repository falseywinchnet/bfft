#!/usr/bin/env python3
"""Run registered phase-lattice maximum fusion and texture-cartoon cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from experiments.ostensibly_frontend.uncertainty_fusion import (
    phase_lattice_observations,
    registered_perceptual_maximum,
    strongest_ridge_audit,
    texture_cartoon_cascade,
)


def _jsonable_audit(audit: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in audit.items()
        if not isinstance(value, np.ndarray)
    }


def run(
    wav_path: Path,
    out_dir: Path,
    *,
    duration: float = 1.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    crop_rows: int = 256,
    offsets: tuple[int, ...] = (-379, -241, -64, 0, 83, 214, 427),
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate, samples = wavfile.read(wav_path)
    samples = np.asarray(samples[: int(round(duration * sample_rate))])
    started = time.perf_counter()
    observations = phase_lattice_observations(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        crop_rows=crop_rows,
        offsets=offsets,
    )
    lattice_ms = 1000.0 * (time.perf_counter() - started)
    started_fusion = time.perf_counter()
    registration = registered_perceptual_maximum(observations, offsets)
    fusion_ms = 1000.0 * (time.perf_counter() - started_fusion)
    started_cascade = time.perf_counter()
    cascade = texture_cartoon_cascade(registration.fused)
    cascade_ms = 1000.0 * (time.perf_counter() - started_cascade)

    reference = registration.observations[offsets.index(0)]
    raw_audit = strongest_ridge_audit(reference)
    fused_audit = strongest_ridge_audit(registration.fused)
    final_audit = strongest_ridge_audit(np.abs(cascade.texture_cartoon))
    magnitude_final_audit = strongest_ridge_audit(
        cascade.magnitude_texture_cartoon)
    first_complement_error = float(np.max(np.abs(
        cascade.input_scaled
        - cascade.first_cartoon
        - cascade.first_texture
    )))
    second_complement_error = float(np.max(np.abs(
        cascade.first_texture
        - cascade.texture_cartoon
        - cascade.second_texture
    )))
    metrics: dict[str, object] = {
        "wav": str(wav_path),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(samples.shape[0] / sample_rate),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "crop_rows": crop_rows,
        "offsets_samples": list(offsets),
        "observation_shape": list(observations.shape),
        "lattice_ms": lattice_ms,
        "registration_and_max_ms": fusion_ms,
        "two_meyer_splits_ms": cascade_ms,
        "total_ms": 1000.0 * (time.perf_counter() - started),
        "raw_reference_ridge": _jsonable_audit(raw_audit),
        "registered_maximum_ridge": _jsonable_audit(fused_audit),
        "texture_cartoon_ridge": _jsonable_audit(final_audit),
        "magnitude_texture_cartoon_ridge": _jsonable_audit(
            magnitude_final_audit),
        "void_fraction_ratio_fused_to_raw": float(
            fused_audit["void_fraction"]
            / max(float(raw_audit["void_fraction"]), 1e-30)
        ),
        "first_complement_max_error": first_complement_error,
        "second_complement_max_error": second_complement_error,
        "registration": list(registration.diagnostics),
        "cascade_semantics": (
            "registered_per_pixel_max_then_keep_texture_then_keep_its_cartoon"
        ),
    }
    np.savez_compressed(
        out_dir / "uncertainty_cascade_arrays.npz",
        offsets=np.asarray(offsets),
        observations=registration.observations,
        aligned=registration.aligned,
        fused=registration.fused,
        flows=registration.flows,
        observability=registration.observability,
        input_scaled=cascade.input_scaled,
        first_cartoon=cascade.first_cartoon,
        first_texture=cascade.first_texture,
        texture_cartoon=cascade.texture_cartoon,
        second_texture=cascade.second_texture,
        magnitude_texture_cartoon=cascade.magnitude_texture_cartoon,
        magnitude_second_texture=cascade.magnitude_second_texture,
        promoted_trace=cascade.promoted_trace,
        raw_ridge_path=raw_audit["path_rows"],
        fused_ridge_path=fused_audit["path_rows"],
        final_ridge_path=final_audit["path_rows"],
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args.wav, args.out, duration=args.duration), indent=2))


if __name__ == "__main__":
    main()
