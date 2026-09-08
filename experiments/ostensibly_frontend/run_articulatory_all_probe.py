#!/usr/bin/env python3
"""Calibrate a vowel seed and render one continuous articulatory “all”."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .articulatory_acoustics import (
    estimate_acoustic_trajectory,
    stable_vowel_span,
    waveguide_transfer_resonances,
)
from .articulatory_tract import (
    initial_all_trajectory,
    synthesize_articulatory_trajectory,
    tract_diameter_from_controls,
)
from .uncertainty_fusion import _as_float_audio


def _vowel_seed_search(target_formants: np.ndarray) -> dict[str, object]:
    parameters = []
    profiles = []
    for tongue_index in np.linspace(12.0, 23.0, 12):
        for tongue_diameter in np.linspace(2.0, 3.2, 7):
            for lip_diameter in np.linspace(0.6, 1.4, 5):
                parameters.append((tongue_index, tongue_diameter, lip_diameter))
                profiles.append(
                    tract_diameter_from_controls(
                        tongue_index,
                        tongue_diameter,
                        lip_diameter,
                        38.0,
                        1.45,
                    )
                )
    resonances = waveguide_transfer_resonances(np.stack(profiles))
    weights = np.asarray((1.0, 1.0, 0.25))
    errors = np.log(np.maximum(resonances, 1.0) / target_formants[None, :])
    loss = np.sum(weights[None, :] * errors * errors, axis=1)
    loss += 100.0 * np.sum(~np.isfinite(resonances), axis=1)
    ranking = np.argsort(loss)
    top = []
    for index in ranking[:8]:
        top.append({
            "tongue_index": float(parameters[index][0]),
            "tongue_diameter": float(parameters[index][1]),
            "lip_diameter": float(parameters[index][2]),
            "resonances_hz": resonances[index].tolist(),
            "log_formant_loss": float(loss[index]),
        })
    return {"selected": top[0], "top": top}


def _calibrated_all_trajectory(
    duration: float,
    f0_hz: float,
    seed: dict[str, object],
):
    frames = initial_all_trajectory(duration)
    index_shift = float(seed["tongue_index"]) - 17.7
    diameter_shift = float(seed["tongue_diameter"]) - 2.05
    lip_shift = float(seed["lip_diameter"]) - 0.82
    f0_scale = f0_hz / 138.0
    return tuple(
        replace(
            frame,
            f0_hz=frame.f0_hz * f0_scale,
            tongue_index=frame.tongue_index + index_shift,
            tongue_diameter=frame.tongue_diameter + diameter_shift,
            lip_diameter=frame.lip_diameter + lip_shift,
        )
        for frame in frames
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("vocalizations", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.vocalizations.read_text(encoding="utf-8"))
    first = document["vocalizations"][0]
    frame0, frame1 = int(first["frame0"]), int(first["frame1"])
    sample_rate, source = wavfile.read(args.wav)
    waveform = _as_float_audio(source)
    target = waveform[frame0 * 512 : frame1 * 512]
    duration = target.size / sample_rate
    target_acoustics = estimate_acoustic_trajectory(target, int(sample_rate))
    target_span = stable_vowel_span(target_acoustics)
    target_slice = slice(*target_span)
    target_f0 = float(np.nanmedian(target_acoustics.f0_hz[target_slice]))
    target_formants = np.nanmedian(
        target_acoustics.formants_hz[target_slice], axis=0
    )
    if not np.all(np.isfinite(target_formants)):
        raise RuntimeError("the target vowel anchor does not expose three formants")

    seed_search = _vowel_seed_search(target_formants)
    frames = _calibrated_all_trajectory(
        duration, target_f0, seed_search["selected"]
    )
    synthetic = synthesize_articulatory_trajectory(frames, sample_rate=int(sample_rate))
    target_rms = float(np.sqrt(np.mean(target * target)))
    synthetic_rms = float(np.sqrt(np.mean(synthetic.samples * synthetic.samples)))
    level_gain = target_rms / max(synthetic_rms, 1e-12)
    level_matched_samples = synthetic.samples * level_gain
    synthetic_acoustics = estimate_acoustic_trajectory(
        level_matched_samples, int(sample_rate)
    )
    synthetic_span = stable_vowel_span(synthetic_acoustics)
    synthetic_slice = slice(*synthetic_span)
    synthetic_formants = np.nanmedian(
        synthetic_acoustics.formants_hz[synthetic_slice], axis=0
    )
    target_register = np.nanmedian(
        target_acoustics.register_semitones[target_slice], axis=0
    )
    synthetic_register = np.nanmedian(
        synthetic_acoustics.register_semitones[synthetic_slice], axis=0
    )

    args.out.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        args.out / "target-all-context.wav",
        int(sample_rate),
        np.asarray(np.clip(target, -1.0, 1.0) * 32767.0, dtype=np.int16),
    )
    wavfile.write(
        args.out / "articulatory-all-initial.wav",
        int(sample_rate),
        np.asarray(np.clip(level_matched_samples, -1.0, 1.0) * 32767.0, dtype=np.int16),
    )
    np.savez_compressed(
        args.out / "articulatory-all-initial.npz",
        target_samples=target.astype(np.float32),
        synthetic_samples=level_matched_samples.astype(np.float32),
        synthetic_physical_samples=synthetic.samples.astype(np.float32),
        target_f0_hz=target_acoustics.f0_hz.astype(np.float32),
        target_formants_hz=target_acoustics.formants_hz.astype(np.float32),
        target_register_semitones=target_acoustics.register_semitones.astype(np.float32),
        synthetic_f0_hz=synthetic_acoustics.f0_hz.astype(np.float32),
        synthetic_formants_hz=synthetic_acoustics.formants_hz.astype(np.float32),
        synthetic_register_semitones=synthetic_acoustics.register_semitones.astype(np.float32),
        tract_diameter=synthetic.tract_diameter.astype(np.float32),
        **{
            f"control_{name}": values.astype(np.float32)
            for name, values in synthetic.controls.items()
        },
    )
    result = {
        "method": "vowel_seeded_continuous_articulatory_analysis_by_synthesis",
        "target": {
            "label": "all",
            "frame_span": [frame0, frame1],
            "duration_seconds": duration,
            "stable_vowel_frames": list(target_span),
            "stable_f0_hz": target_f0,
            "stable_formants_hz": target_formants.tolist(),
            "stable_register_semitones": target_register.tolist(),
        },
        "vowel_seed_search": seed_search,
        "synthetic": {
            "target_rms": target_rms,
            "physical_rms": synthetic_rms,
            "level_gain": level_gain,
            "stable_vowel_frames": list(synthetic_span),
            "stable_formants_hz": synthetic_formants.tolist(),
            "stable_register_semitones": synthetic_register.tolist(),
            "register_error_semitones": (
                synthetic_register - target_register
            ).tolist(),
            "trajectory": [asdict(frame) for frame in frames],
        },
    }
    (args.out / "articulatory-all-initial.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
