#!/usr/bin/env python3
"""Audit Cleanup/SHARK temporal anchors against held-out ARCTIC timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .cleanup_shark_vad import (
    CleanupSharkConfig,
    analyze_cleanup_shark,
    fuse_cleanup_shark_evidence,
    speech_state_machine,
)
from .run_calibrate_geometry_fusion import DEFAULT_UTTERANCES
from .run_arctic_calibration import HOP_LENGTH, TARGET_SAMPLE_RATE
from .run_occupation_cloud_battery import _load_channel_audio
from .speech_anchor_audit import (
    binary_counts,
    load_lab_segments,
    match_boundaries,
    predicted_state_geometry,
    summarize_counts,
    target_state_geometry,
)


def _strengths(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("certifier strengths must be decimals") from error
    if not values or any(not 0.0 <= value <= 0.25 for value in values):
        raise argparse.ArgumentTypeError("certifier strengths must lie in [0, 0.25]")
    return tuple(sorted(set(values)))


def _add_counts(total: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0) + int(value)


def _boundary_summary(rows: list[dict[str, object]], kind: str) -> dict[str, object]:
    predicted = sum(int(row[f"{kind}_predicted"]) for row in rows)
    target = sum(int(row[f"{kind}_target"]) for row in rows)
    matched = sum(int(row[f"{kind}_matched"]) for row in rows)
    errors = [
        int(error)
        for row in rows
        for error in row[f"{kind}_errors_frames"]
    ]
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    return {
        "predicted": predicted,
        "target": target,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
        "median_absolute_error_frames": float(np.median(errors)) if errors else None,
        "mean_absolute_error_frames": float(np.mean(errors)) if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--tolerance-ms", type=float, default=40.0)
    parser.add_argument(
        "--voice-certifier-strengths",
        type=_strengths,
        default=(0.0, 0.125, 0.25),
    )
    parser.add_argument(
        "--structure-certifier-strengths",
        type=_strengths,
        default=(0.0, 0.125, 0.25),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if (
        not utterances
        or any(value not in DEFAULT_UTTERANCES for value in utterances)
        or not np.isfinite(args.noise_db)
        or args.tolerance_ms < 0.0
    ):
        raise ValueError("speech-anchor audit configuration is invalid")

    config = CleanupSharkConfig()
    tolerance_frames = int(round(
        args.tolerance_ms * TARGET_SAMPLE_RATE / (1000.0 * HOP_LENGTH)
    ))
    variants = {
        **{
            f"certifier_v{voice:g}_u{structure:g}": []
            for voice in args.voice_certifier_strengths
            for structure in args.structure_certifier_strengths
        },
        "pitch_only": [],
    }
    for speaker in ("bdl", "slt"):
        for utterance_index, utterance in enumerate(utterances):
            samples = _load_channel_audio(
                args.arctic_root, speaker, utterance, utterance_index, args.noise_db
            )
            analysis = analyze_cleanup_shark(samples, TARGET_SAMPLE_RATE, config=config)
            segments = load_lab_segments(
                args.arctic_root / speaker / "lab" / f"{utterance}.lab"
            )
            target_mask, target_crop_edges, target_manner_edges = target_state_geometry(
                segments,
                analysis.state.size,
                sample_rate=TARGET_SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )
            spectral_structure = expit(
                (analysis.spectral_prominence_db - config.prominence_threshold_db) / 2.0
            )
            energy_structure = expit((analysis.energy_snr_db - 3.0) / 2.0)
            structure = np.maximum(spectral_structure, energy_structure)
            variant_inputs = {}
            for voice_strength in args.voice_certifier_strengths:
                for structure_strength in args.structure_certifier_strengths:
                    name = f"certifier_v{voice_strength:g}_u{structure_strength:g}"
                    if (
                        voice_strength == config.cleanup_voice_certifier_strength
                        and structure_strength
                        == config.cleanup_structure_certifier_strength
                    ):
                        inferred = analysis.state, analysis.speech_mask
                    else:
                        fused, unvoiced = fuse_cleanup_shark_evidence(
                            analysis.shark_score,
                            analysis.cleanup_probability,
                            structure,
                            voice_certifier_strength=voice_strength,
                            structure_certifier_strength=structure_strength,
                        )
                        inferred = speech_state_machine(
                            fused,
                            unvoiced,
                            sample_rate=TARGET_SAMPLE_RATE,
                            config=config,
                        )[:2]
                    variant_inputs[name] = inferred
            variant_inputs["pitch_only"] = speech_state_machine(
                analysis.shark_score,
                np.zeros_like(analysis.unvoiced_score),
                sample_rate=TARGET_SAMPLE_RATE,
                config=config,
            )[:2]
            for name, (state, speech_mask) in variant_inputs.items():
                crop_edges, manner_edges = predicted_state_geometry(
                    state, speech_mask
                )
                crop_matched, crop_errors = match_boundaries(
                    crop_edges, target_crop_edges, tolerance_frames=tolerance_frames
                )
                manner_matched, manner_errors = match_boundaries(
                    manner_edges, target_manner_edges, tolerance_frames=tolerance_frames
                )
                variants[name].append({
                    "speaker": speaker,
                    "utterance": utterance,
                    "frame_counts": binary_counts(speech_mask, target_mask),
                    "crop_predicted": len(crop_edges),
                    "crop_target": len(target_crop_edges),
                    "crop_matched": crop_matched,
                    "crop_errors_frames": list(crop_errors),
                    "manner_predicted": len(manner_edges),
                    "manner_target": len(target_manner_edges),
                    "manner_matched": manner_matched,
                    "manner_errors_frames": list(manner_errors),
                })
            print(json.dumps({"speaker": speaker, "utterance": utterance}), flush=True)

    summaries = {}
    for name, rows in variants.items():
        counts: dict[str, int] = {}
        for row in rows:
            _add_counts(counts, row["frame_counts"])
        summaries[name] = {
            "frames": summarize_counts(counts),
            "crop_edges": _boundary_summary(rows, "crop"),
            "manner_edges": _boundary_summary(rows, "manner"),
        }
    result = {
        "method": "label_held_out_cleanup_shark_temporal_anchor_audit",
        "selection_status": "timing labels used only after state inference",
        "speakers": ["bdl", "slt"],
        "utterances": list(utterances),
        "noise_db": args.noise_db,
        "voice_certifier_strengths": list(args.voice_certifier_strengths),
        "structure_certifier_strengths": list(args.structure_certifier_strengths),
        "certifier_law": {
            "voiced": "shark * (1 - voice_strength * (1 - cleanup_probability))",
            "unvoiced": "sqrt(structure) * cleanup_probability ** (2 * structure_strength)",
            "selected_default_voice_strength": config.cleanup_voice_certifier_strength,
            "selected_default_structure_strength": config.cleanup_structure_certifier_strength,
        },
        "tolerance_ms": args.tolerance_ms,
        "tolerance_frames": tolerance_frames,
        "summaries": summaries,
        "rows": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
