#!/usr/bin/env python3
"""Resegment a saved promoted trace field using waveform-supported activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from experiments.ostensibly_frontend.full_recording import (
    phone_boundary_proposals,
    serializable_intervals,
    supported_activity_intervals,
)
from experiments.ostensibly_frontend.uncertainty_fusion import _as_float_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_npz", type=Path)
    parser.add_argument("recording_json", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--minimum-seconds", type=float, default=0.08)
    parser.add_argument("--close-gap-seconds", type=float, default=0.10)
    parser.add_argument("--padding-seconds", type=float, default=0.0)
    parser.add_argument("--maximum-phone-seconds", type=float, default=0.24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recording = np.load(args.recording_npz)
    field_key = (
        "trace_field" if "trace_field" in recording.files
        else "registered_maximum"
    )
    field = np.asarray(recording[field_key], dtype=np.float64)
    metadata = json.loads(args.recording_json.read_text())
    sample_rate, samples = wavfile.read(args.wav)
    samples = _as_float_audio(samples)
    if int(sample_rate) != int(metadata["sample_rate"]):
        raise ValueError("waveform and recording sample rates disagree")
    hop_length = int(metadata["diagnostics"]["hop_length"])
    score, active, speech, diagnostics = supported_activity_intervals(
        field,
        samples,
        sample_rate=int(sample_rate),
        hop_length=hop_length,
        minimum_seconds=args.minimum_seconds,
        close_gap_seconds=args.close_gap_seconds,
        padding_seconds=args.padding_seconds,
    )
    phones = phone_boundary_proposals(
        field,
        speech,
        sample_rate=int(sample_rate),
        hop_length=hop_length,
        maximum_seconds=args.maximum_phone_seconds,
    )
    result = {
        **metadata,
        "source_recording_json": str(args.recording_json),
        "segmentation_diagnostics": {
            **diagnostics,
            "speech_intervals": len(speech),
            "phone_intervals": len(phones),
            "active_seconds": float(np.sum(active) * hop_length / sample_rate),
            "maximum_phone_seconds": args.maximum_phone_seconds,
        },
        "speech": serializable_intervals(speech),
        "phones": serializable_intervals(phones),
    }
    output_json = args.out / "resegmented_recording.json"
    output_npz = args.out / "resegmented_activity.npz"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_npz, activity_score=score, active=active)
    print(json.dumps({
        "diagnostics": result["segmentation_diagnostics"],
        "recording_json": str(output_json),
        "activity_npz": str(output_npz),
    }, indent=2))


if __name__ == "__main__":
    main()
