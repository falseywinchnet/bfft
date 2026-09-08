#!/usr/bin/env python3
"""Audit compact center-cropped phones synthesized inside triphone context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from experiments.ostensibly_frontend.phone_match import (
    fingerprint_whole_phone_patch,
)
from experiments.ostensibly_frontend.run_whole_patch_inventory import (
    LABELS,
    OFFSETS,
    ROUTE_SIZE,
    VOICE_PROBES,
    audit,
    class_centroids,
    routing_statistics,
)
from experiments.ostensibly_frontend.synthetic_consonants import (
    CONSONANTS,
    synthesize_consonant,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    DIPHTHONGS,
    FORMANTS,
    apply_ssb_probe_channel,
    synthesize_english_vowel,
)


VOWELS = tuple(FORMANTS) + tuple(DIPHTHONGS)
LEFT_VOWELS = ("AH", "IY", "UW", "AE", "AO")
RIGHT_VOWELS = ("IY", "AO", "AE", "UW", "AH")
LEFT_CONSONANTS = ("B", "D", "G", "M", "L")
RIGHT_CONSONANTS = ("T", "K", "N", "S", "D")
VOWEL_DURATIONS = (0.090, 0.110, 0.130, 0.150, 0.170)
CONSONANT_DURATIONS = (0.060, 0.075, 0.090, 0.105, 0.120)
CONTEXT_DURATION = 0.070
SAMPLE_RATE = 48_000
HOP_LENGTH = 512


def _phone_samples(label, voice, duration, seed):
    if label in VOWELS:
        return synthesize_english_vowel(
            label,
            voice,
            sample_rate=SAMPLE_RATE,
            duration=duration,
            seed=seed,
        )
    return synthesize_consonant(
        label,
        voice,
        sample_rate=SAMPLE_RATE,
        duration=duration,
        seed=seed,
    )


def contextual_fingerprint(label, voice_index, seed_offset):
    voice = VOICE_PROBES[voice_index]
    label_index = LABELS.index(label)
    context_shift = 2 if seed_offset else 0
    context_index = (label_index + voice_index + context_shift) % 5
    if label in VOWELS:
        left_label = LEFT_CONSONANTS[context_index]
        right_label = RIGHT_CONSONANTS[context_index]
        durations = VOWEL_DURATIONS
    else:
        left_label = LEFT_VOWELS[context_index]
        right_label = RIGHT_VOWELS[context_index]
        durations = CONSONANT_DURATIONS
    duration_index = (voice_index + (1 if seed_offset else 0)) % 5
    target_duration = durations[duration_index]
    seed = seed_offset + 1000 * voice_index + label_index
    left = _phone_samples(
        left_label, voice, CONTEXT_DURATION, seed + 101
    )
    target = _phone_samples(label, voice, target_duration, seed)
    right = _phone_samples(
        right_label, voice, CONTEXT_DURATION, seed + 211
    )
    carrier = apply_ssb_probe_channel(
        np.concatenate((left, target, right)),
        sample_rate=SAMPLE_RATE,
        seed=seed + 307,
    )
    densified = fourier_unrolled_densified(
        carrier,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=HOP_LENGTH,
        target_crop_rows=256,
        offsets=OFFSETS,
    )
    baseline = reassigned_texture_baseline(
        densified.fused, densified.support_union
    )
    centers = HOP_LENGTH * np.arange(baseline.merged.shape[1])
    target_start = left.size
    target_stop = left.size + target.size
    selected = np.flatnonzero(
        (centers >= target_start) & (centers < target_stop)
    )
    if selected.size < 2:
        raise RuntimeError("contextual phone crop contains fewer than two frames")
    return fingerprint_whole_phone_patch(baseline.merged[:, selected])


def build_records(seed_offset):
    return [
        {
            "label": label,
            "voice": voice.name,
            "fingerprint": contextual_fingerprint(
                label, voice_index, seed_offset
            ),
        }
        for voice_index, voice in enumerate(VOICE_PROBES)
        for label in LABELS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("building contextual reference battery", flush=True)
    references = build_records(0)
    print("building contextual held-out battery", flush=True)
    queries = build_records(100_000)
    centroids = class_centroids(references)
    statistics = routing_statistics(references)
    validation, details = audit(queries, centroids, statistics)
    np.savez_compressed(
        args.out / "whole_patch_templates.npz",
        labels=np.asarray(LABELS),
        fields=np.stack([centroids[label].field for label in LABELS]),
        anchor_rows=np.asarray([centroids[label].anchor_row for label in LABELS]),
        source_centroid_rows=np.asarray([
            centroids[label].source_centroid_row for label in LABELS]),
        source_centroid_frames=np.asarray([
            centroids[label].source_centroid_frame for label in LABELS]),
        centroid_rows=np.asarray([
            centroids[label].centroid_row for label in LABELS]),
        centroid_frames=np.asarray([
            centroids[label].centroid_frame for label in LABELS]),
        duration_frames=np.asarray([
            centroids[label].duration_frames for label in LABELS]),
        harmonicities=np.asarray([
            centroids[label].harmonicity for label in LABELS]),
        routing_means=np.stack([statistics[label][0] for label in LABELS]),
        routing_scales=np.stack([statistics[label][1] for label in LABELS]),
    )
    result = {
        "method": (
            "compact_contextual_center_crop_then_selected_reassigned_texture_baseline"
        ),
        "route_size": ROUTE_SIZE,
        "vowel_durations": list(VOWEL_DURATIONS),
        "consonant_durations": list(CONSONANT_DURATIONS),
        "context_duration": CONTEXT_DURATION,
        "validation": validation,
        "details": details,
    }
    output = args.out / "whole_patch_inventory_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
