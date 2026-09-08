#!/usr/bin/env python3
"""Measure Cleanup/SHARK speech anchors and save transport-ready crops."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from experiments.ostensibly_frontend.cleanup_shark_vad import (
    CleanupSharkConfig,
    SpeechState,
    analyze_cleanup_shark,
)


def _read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, source = wavfile.read(path)
    values = np.asarray(source)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        values = values.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        values = values.astype(np.float64, copy=False)
    if values.ndim == 2:
        values = np.mean(values, axis=1)
    if values.ndim != 1:
        raise ValueError("audio must be mono or samples-by-channels")
    return int(sample_rate), np.ascontiguousarray(values)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    difference = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(
        zip(
            np.flatnonzero(difference == 1).tolist(),
            np.flatnonzero(difference == -1).tolist(),
            strict=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sample_rate, samples = _read_wav_mono(args.wav)
    with np.load(args.recording) as source:
        frame_count = int(source["trace_field"].shape[1])
        old_active = np.asarray(source["active"], dtype=bool)
    config = CleanupSharkConfig()
    analysis = analyze_cleanup_shark(
        samples, sample_rate, frame_count=frame_count, config=config
    )
    new_active = analysis.speech_mask
    old_runs = _runs(old_active)
    new_runs = _runs(new_active)
    record = {
        "wav": str(args.wav),
        "recording": str(args.recording),
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "hop_length": config.hop_length,
        "config": asdict(config),
        "statistics": {
            "noise_frames": analysis.noise_frame_count,
            "old_active_fraction": float(np.mean(old_active)),
            "fused_active_fraction": float(np.mean(new_active)),
            "old_interval_count": len(old_runs),
            "fused_interval_count": len(new_runs),
            "mask_agreement": float(np.mean(old_active == new_active)),
            "old_support_retained": float(np.mean(new_active[old_active])),
            "new_activity_outside_old_support": float(
                np.mean(new_active[~old_active])
            ),
            "voiced_fraction": float(np.mean(analysis.voiced_mask)),
            "unvoiced_fraction": float(np.mean(analysis.unvoiced_mask)),
            "vocalization_fraction": float(np.mean(analysis.vocalization_mask)),
            "vocalization_count": len(analysis.vocalizations),
        },
        "intervals": [asdict(interval) for interval in analysis.intervals],
        "vocalizations": [
            asdict(interval) for interval in analysis.vocalizations
        ],
        "state_names": {str(int(state)): state.name for state in SpeechState},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        cleanup_similarity=analysis.cleanup_similarity.astype(np.float32),
        cleanup_center=analysis.cleanup_center.astype(np.float32),
        cleanup_atd=analysis.cleanup_atd.astype(np.float32),
        cleanup_probability=analysis.cleanup_probability.astype(np.float32),
        pitch_periodicity=analysis.pitch_periodicity.astype(np.float32),
        pitch_hz=analysis.pitch_hz.astype(np.float32),
        spectral_prominence_db=analysis.spectral_prominence_db.astype(np.float32),
        energy_snr_db=analysis.energy_snr_db.astype(np.float32),
        shark_score=analysis.shark_score.astype(np.float32),
        fused_score=analysis.fused_score.astype(np.float32),
        unvoiced_score=analysis.unvoiced_score.astype(np.float32),
        spectral_gain=analysis.spectral_gain.astype(np.float32),
        denoised_magnitude=analysis.denoised_magnitude.astype(np.float32),
        state=analysis.state,
        speech_mask=analysis.speech_mask,
        voiced_mask=analysis.voiced_mask,
        unvoiced_mask=analysis.unvoiced_mask,
        vocalization_mask=analysis.vocalization_mask,
        old_active=old_active,
    )
    print(json.dumps(record["statistics"], indent=2))


if __name__ == "__main__":
    main()
