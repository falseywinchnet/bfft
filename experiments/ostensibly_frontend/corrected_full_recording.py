"""Four-second-window full-recording build from the corrected trace field."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.io import wavfile

from experiments.ostensibly_frontend.full_recording import (
    phone_boundary_proposals,
    supported_activity_intervals,
)
from experiments.ostensibly_frontend.uncertainty_fusion import (
    _as_float_audio,
)
from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified_at_centers,
    reassigned_texture_baseline,
)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path)
    return int(sample_rate), _as_float_audio(samples)


def build_corrected_full_recording(
    wav_path: Path,
    *,
    apertures: tuple[int, ...] = DEFAULT_APERTURES,
    target_n_fft: int = 2048,
    hop_length: int = 512,
    crop_rows: int = 256,
    window_seconds: float = 4.0,
    halo_seconds: float = 0.50,
    offsets: tuple[int, ...] = (-379, -241, -64, 0, 83, 214, 427),
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Build the promoted positive trace plane in bounded analysis windows."""
    sample_rate, samples = read_wav_mono(wav_path)
    frame_count = 1 + samples.size // hop_length
    core_frames = max(8, int(round(window_seconds * sample_rate / hop_length)))
    halo_frames = max(4, int(round(halo_seconds * sample_rate / hop_length)))
    n2048_registered_maximum = np.empty(
        (crop_rows, frame_count), dtype=np.float64)
    multiscale_registered_maximum = np.empty_like(n2048_registered_maximum)
    reassigned_support_power = np.empty_like(n2048_registered_maximum)
    chunks: list[dict[str, object]] = []
    started = time.perf_counter()
    for core_start in range(0, frame_count, core_frames):
        core_stop = min(core_start + core_frames, frame_count)
        window_start = max(0, core_start - halo_frames)
        window_stop = min(frame_count, core_stop + halo_frames)
        frame_indices = np.arange(window_start, window_stop, dtype=np.int64)
        centers = frame_indices * hop_length
        chunk_started = time.perf_counter()
        densified = fourier_unrolled_densified_at_centers(
            samples,
            centers,
            apertures=apertures,
            target_n_fft=target_n_fft,
            target_crop_rows=crop_rows,
            hop_length=hop_length,
            offsets=offsets,
        )
        local_start = core_start - window_start
        local_stop = local_start + (core_stop - core_start)
        destination = slice(core_start, core_stop)
        source = slice(local_start, local_stop)
        base_index = densified.apertures.index(target_n_fft)
        n2048_registered_maximum[:, destination] = (
            densified.aperture_fields[base_index, :, source])
        multiscale_registered_maximum[:, destination] = densified.fused[:, source]
        reassigned_support_power[:, destination] = densified.support_union[:, source]
        flow_rms = [
            float(record.get("flow_rms_pixels", 0.0))
            for phase in densified.phase_fusions
            for record in phase.diagnostics
        ]
        cross_aperture_flow_rms = [
            float(record.get("flow_rms_pixels", 0.0))
            for record in densified.diagnostics
        ]
        chunk_record = {
            "core_frames": [core_start, core_stop],
            "window_frames": [window_start, window_stop],
            "window_seconds": [
                window_start * hop_length / sample_rate,
                window_stop * hop_length / sample_rate,
            ],
            "elapsed_ms": 1000.0 * (time.perf_counter() - chunk_started),
            "flow_rms_mean_pixels": float(np.mean(flow_rms)),
            "flow_rms_max_pixels": float(np.max(flow_rms)),
            "cross_aperture_flow_rms_mean_pixels": float(
                np.mean(cross_aperture_flow_rms)),
            "cross_aperture_flow_rms_max_pixels": float(
                np.max(cross_aperture_flow_rms)),
        }
        chunks.append(chunk_record)
        if progress is not None:
            progress(chunk_record)
    promoted = reassigned_texture_baseline(
        multiscale_registered_maximum, reassigned_support_power)
    # Harmonic existence locates a candidate phone; comparison consumes the
    # complete promoted field, including its nuisance/noise geometry.
    final = promoted.merged
    activity_score, active, speech, activity_diagnostics = supported_activity_intervals(
        final,
        samples,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    phones = phone_boundary_proposals(
        final,
        speech,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    return {
        "samples": samples,
        "sample_rate": sample_rate,
        "reference": n2048_registered_maximum,
        "n2048_registered_maximum": n2048_registered_maximum,
        "registered_maximum": multiscale_registered_maximum,
        "reassigned_support_power": reassigned_support_power,
        "baseline_cartoon": promoted.cartoon,
        "baseline_texture": promoted.texture,
        "trace_field": final,
        "activity_score": activity_score,
        "active": active,
        "speech": speech,
        "phones": phones,
        "chunks": chunks,
        "diagnostics": {
            "method": (
                "multiscale_phase_lattices_to_registered_maximum_then_"
                "3x3_average_plus_amplitude_gauged_reassigned_texture"
            ),
            "apertures": list(apertures),
            "target_n_fft": target_n_fft,
            "hop_length": hop_length,
            "crop_rows": crop_rows,
            "duration_seconds": samples.size / sample_rate,
            "frames": frame_count,
            "offsets_samples": list(offsets),
            "reassigned_texture_amplitude_scale": (
                promoted.texture_amplitude_scale),
            "core_window_seconds": window_seconds,
            "halo_seconds": halo_seconds,
            "chunk_count": len(chunks),
            "speech_intervals": len(speech),
            "phone_intervals": len(phones),
            "total_ms": 1000.0 * (time.perf_counter() - started),
            **activity_diagnostics,
        },
    }
