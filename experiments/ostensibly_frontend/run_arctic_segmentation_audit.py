#!/usr/bin/env python3
"""Measure promoted phone-boundary proposals against timed ARCTIC labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from experiments.ostensibly_frontend.full_recording import (
    phone_boundary_proposals,
    supported_activity_intervals,
)
from experiments.ostensibly_frontend.run_arctic_calibration import (
    HOP_LENGTH,
    OFFSETS,
    TARGET_SAMPLE_RATE,
    _timed_phones,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    apply_ssb_probe_channel,
)
from experiments.ostensibly_frontend.uncertainty_fusion import _as_float_audio


def match_boundaries(reference, proposed, tolerance_seconds):
    """Return an order-free one-to-one nearest-boundary audit."""

    reference = np.asarray(reference, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    pairs = sorted(
        (abs(float(observed - target)), i, j, float(observed - target))
        for i, target in enumerate(reference)
        for j, observed in enumerate(proposed)
        if abs(float(observed - target)) <= tolerance_seconds
    )
    used_reference = set()
    used_proposed = set()
    signed_errors = []
    for _, i, j, signed in pairs:
        if i in used_reference or j in used_proposed:
            continue
        used_reference.add(i)
        used_proposed.add(j)
        signed_errors.append(signed)
    matched = len(signed_errors)
    precision = matched / max(len(proposed), 1)
    recall = matched / max(len(reference), 1)
    return {
        "matched": matched,
        "reference_count": int(len(reference)),
        "proposed_count": int(len(proposed)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
        "median_absolute_error_seconds": (
            float(np.median(np.abs(signed_errors))) if signed_errors else None
        ),
        "median_signed_error_seconds": (
            float(np.median(signed_errors)) if signed_errors else None
        ),
    }


ACTIVITY_VARIANTS = {
    "baseline": (0.08, 0.10, 0.00, 0.24),
    "bridge_250ms": (0.08, 0.25, 0.00, 0.24),
    "bridge_250ms_pad_40ms": (0.08, 0.25, 0.04, 0.24),
    "bridge_350ms_pad_60ms": (0.08, 0.35, 0.06, 0.24),
    "short_40ms_bridge_250ms_pad_40ms": (0.04, 0.25, 0.04, 0.24),
    "timed_q95_160ms": (0.04, 0.25, 0.04, 0.16),
}


def _trace(wav_path, speaker, utterance_index, noise_db):
    sample_rate, samples = wavfile.read(wav_path)
    samples = _as_float_audio(samples)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, TARGET_SAMPLE_RATE, int(sample_rate))
    samples = apply_ssb_probe_channel(
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        noise_db=noise_db,
        seed=10_000 * (1 + utterance_index) + sum(map(ord, speaker)),
    )
    densified = fourier_unrolled_densified(
        samples,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=HOP_LENGTH,
        target_crop_rows=256,
        offsets=OFFSETS,
    )
    baseline = reassigned_texture_baseline(
        densified.fused, densified.support_union
    )
    return samples, baseline.merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", default="clb")
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    utterances = tuple(args.utterances.split(","))
    details = []
    aggregate_reference = {name: [] for name in ACTIVITY_VARIANTS}
    aggregate_proposed = {name: [] for name in ACTIVITY_VARIANTS}
    time_offset = 0.0
    for utterance_index, utterance in enumerate(utterances):
        wav_path = args.arctic_root / args.speaker / "wav" / f"{utterance}.wav"
        lab_path = args.arctic_root / args.speaker / "lab" / f"{utterance}.lab"
        timed = _timed_phones(lab_path)
        samples, trace = _trace(
            wav_path, args.speaker, utterance_index, args.noise_db
        )
        reference_boundaries = sorted(set(
            [row[1] for row in timed] + [row[2] for row in timed]
        ))
        phone_centers = np.asarray([
            0.5 * (row[1] + row[2]) for row in timed
        ])
        center_frames = np.clip(
            np.rint(phone_centers * TARGET_SAMPLE_RATE / HOP_LENGTH).astype(int),
            0, max(trace.shape[1] - 1, 0),
        )
        variants = {}
        maximum_duration = max(reference_boundaries + [0.0])
        for name, (
            minimum, close_gap, padding, maximum_phone
        ) in ACTIVITY_VARIANTS.items():
            _, active, speech, activity_diagnostics = supported_activity_intervals(
                trace,
                samples,
                sample_rate=TARGET_SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                minimum_seconds=minimum,
                close_gap_seconds=close_gap,
                padding_seconds=padding,
            )
            phones = phone_boundary_proposals(
                trace,
                speech,
                sample_rate=TARGET_SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                maximum_seconds=maximum_phone,
            )
            proposed_boundaries = sorted(set(
                [row.seconds0 for row in phones]
                + [row.seconds1 for row in phones]
            ))
            maximum_duration = max(
                maximum_duration, max(proposed_boundaries + [0.0])
            )
            aggregate_reference[name].extend(
                time_offset + np.asarray(reference_boundaries)
            )
            aggregate_proposed[name].extend(
                time_offset + np.asarray(proposed_boundaries)
            )
            variants[name] = {
                "proposed_phone_count": len(phones),
                "region_count_ratio": len(phones) / max(len(timed), 1),
                "phone_center_activity_recall": float(
                    np.mean(active[center_frames])
                ),
                "activity_diagnostics": activity_diagnostics,
                "boundary_metrics": {
                    str(milliseconds): match_boundaries(
                        reference_boundaries,
                        proposed_boundaries,
                        milliseconds / 1000.0,
                    )
                    for milliseconds in (20, 40, 80)
                },
            }
        time_offset += maximum_duration + 1.0
        detail = {
            "utterance": utterance,
            "reference_phone_count": len(timed),
            "variants": variants,
        }
        details.append(detail)
        print(
            utterance,
            {name: round(row["region_count_ratio"], 3)
             for name, row in variants.items()},
            flush=True,
        )
    result = {
        "method": "promoted_trace_activity_and_geometry_novelty_boundaries",
        "speaker": args.speaker,
        "utterances": list(utterances),
        "noise_db": args.noise_db,
        "details": details,
        "activity_variants": {
            name: {
                "minimum_seconds": values[0],
                "close_gap_seconds": values[1],
                "padding_seconds": values[2],
                "maximum_phone_seconds": values[3],
            }
            for name, values in ACTIVITY_VARIANTS.items()
        },
        "aggregate": {
            name: {
                "proposed_phone_count": sum(
                    row["variants"][name]["proposed_phone_count"]
                    for row in details
                ),
                "phone_center_activity_recall": float(np.mean([
                    row["variants"][name]["phone_center_activity_recall"]
                    for row in details
                ])),
                "boundary_metrics": {
                    str(milliseconds): match_boundaries(
                        aggregate_reference[name],
                        aggregate_proposed[name],
                        milliseconds / 1000.0,
                    )
                    for milliseconds in (20, 40, 80)
                },
            }
            for name in ACTIVITY_VARIANTS
        },
        "aggregate_reference_phone_count": sum(
            row["reference_phone_count"] for row in details
        ),
    }
    output = args.out / "arctic_segmentation_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
