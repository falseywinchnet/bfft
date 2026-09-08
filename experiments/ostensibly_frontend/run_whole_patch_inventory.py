#!/usr/bin/env python3
"""Audit centroid-registered whole-field geometry on a synthetic phone battery."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.phone_match import (
    WholePatchFingerprint,
    fingerprint_whole_phone_patch,
    whole_patch_distance,
)
from experiments.ostensibly_frontend.synthetic_consonants import (
    CONSONANTS,
    synthesize_consonant,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    DIPHTHONGS,
    FORMANTS,
    VOICE_PROBES,
    apply_ssb_probe_channel,
    synthesize_english_vowel,
)
from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)


VOWELS = tuple(FORMANTS) + tuple(DIPHTHONGS)
LABELS = VOWELS + CONSONANTS
OFFSETS = (-379, -241, -64, 0, 83, 214, 427)
ROUTE_SIZE = 24


def _samples(
    label: str,
    voice_index: int,
    seed_offset: int,
    noise_db: float = 28.0,
) -> np.ndarray:
    voice = VOICE_PROBES[voice_index]
    label_index = LABELS.index(label)
    seed = seed_offset + 1000 * voice_index + label_index
    if label in FORMANTS or label in DIPHTHONGS:
        value = synthesize_english_vowel(label, voice, seed=seed)
    else:
        value = synthesize_consonant(label, voice, seed=seed)
    return apply_ssb_probe_channel(
        value,
        noise_db=noise_db,
        seed=seed_offset + 2000 * voice_index + label_index,
    )


def _fingerprint(
    label: str,
    voice_index: int,
    seed_offset: int,
    noise_db: float = 28.0,
) -> WholePatchFingerprint:
    return _component_fingerprints(
        label, voice_index, seed_offset, noise_db=noise_db
    )["merged"]


def _component_fingerprints(
    label: str,
    voice_index: int,
    seed_offset: int,
    noise_db: float = 28.0,
) -> dict[str, WholePatchFingerprint]:
    densified = fourier_unrolled_densified(
        _samples(label, voice_index, seed_offset, noise_db=noise_db),
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=512,
        target_crop_rows=256,
        offsets=OFFSETS,
    )
    baseline = reassigned_texture_baseline(
        densified.fused, densified.support_union)
    return {
        "merged": fingerprint_whole_phone_patch(baseline.merged),
        "cartoon": fingerprint_whole_phone_patch(baseline.cartoon),
        "texture": fingerprint_whole_phone_patch(baseline.texture),
        "registered_maximum": fingerprint_whole_phone_patch(densified.fused),
    }


def build_records(
    seed_offset: int,
    noise_db: float = 28.0,
) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "voice": voice.name,
            "fingerprint": _fingerprint(
                label, voice_index, seed_offset, noise_db=noise_db
            ),
        }
        for voice_index, voice in enumerate(VOICE_PROBES)
        for label in LABELS
    ]


def class_centroids(
    records: list[dict[str, object]],
) -> dict[str, WholePatchFingerprint]:
    result = {}
    for label in LABELS:
        values = [
            item["fingerprint"] for item in records if item["label"] == label
        ]
        field = np.mean(np.stack([item.field for item in values]), axis=0)
        field /= max(float(np.linalg.norm(field)), 1e-30)
        result[label] = replace(
            values[0],
            anchor_row=int(round(np.mean([item.anchor_row for item in values]))),
            source_centroid_row=float(np.mean([
                item.source_centroid_row for item in values])),
            source_centroid_frame=float(np.mean([
                item.source_centroid_frame for item in values])),
            duration_frames=int(round(np.mean([
                item.duration_frames for item in values]))),
            harmonicity=float(np.mean([item.harmonicity for item in values])),
            field=field,
        )
    return result


def routing_statistics(
    records: list[dict[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    floors = np.asarray((5.0, 5.0, 0.10, 0.25), dtype=np.float64)
    for label in LABELS:
        values = [
            item["fingerprint"] for item in records if item["label"] == label
        ]
        features = np.asarray([
            (
                item.anchor_row,
                item.source_centroid_row,
                item.source_centroid_frame
                / max(item.duration_frames - 1, 1),
                np.log(max(item.harmonicity, 1e-12)),
            )
            for item in values
        ])
        result[label] = (np.mean(features, axis=0), np.maximum(
            np.std(features, axis=0), floors))
    return result


def route_labels(
    query: WholePatchFingerprint,
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    count: int = ROUTE_SIZE,
) -> list[str]:
    feature = np.asarray((
        query.anchor_row,
        query.source_centroid_row,
        query.source_centroid_frame / max(query.duration_frames - 1, 1),
        np.log(max(query.harmonicity, 1e-12)),
    ))
    scored = [
        (label, float(np.linalg.norm((feature - mean) / scale)))
        for label, (mean, scale) in statistics.items()
    ]
    return [label for label, _ in sorted(scored, key=lambda x: (x[1], x[0]))[:count]]


def audit(
    queries: list[dict[str, object]],
    centroids: dict[str, WholePatchFingerprint],
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    hits = {1: 0, 3: 0, 5: 0}
    route_hits = 0
    details = []
    for index, item in enumerate(queries):
        query = item["fingerprint"]
        candidates = route_labels(query, statistics)
        route_hits += int(item["label"] in candidates)
        ranking = sorted(
            (
                (label, whole_patch_distance(query, centroids[label]))
                for label in candidates
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        for k in hits:
            hits[k] += int(any(
                label == item["label"] for label, _ in ranking[:k]))
        details.append({
            "label": item["label"],
            "voice": item["voice"],
            "route": candidates,
            "top5": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:5]
            ],
        })
        if (index + 1) % 20 == 0:
            print(f"audited {index + 1}/{len(queries)}", flush=True)
    count = len(queries)
    return {
        "cases": count,
        "route_size": ROUTE_SIZE,
        "route_recall": route_hits / count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
    }, details


def audit_minimum_witness(
    queries: list[dict[str, object]],
    witnesses: dict[str, tuple[WholePatchFingerprint, ...]],
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    hits = {1: 0, 3: 0, 5: 0}
    route_hits = 0
    for item in queries:
        query = item["fingerprint"]
        candidates = route_labels(query, statistics)
        route_hits += int(item["label"] in candidates)
        ranking = sorted(
            (
                (
                    label,
                    min(whole_patch_distance(query, reference)
                        for reference in witnesses[label]),
                )
                for label in candidates
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        for count in hits:
            hits[count] += int(any(
                label == item["label"] for label, _ in ranking[:count]
            ))
    count = len(queries)
    return {
        "cases": count,
        "route_size": ROUTE_SIZE,
        "route_recall": route_hits / count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--noise-db", type=float, default=28.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("building reference battery", flush=True)
    references = build_records(0, noise_db=args.noise_db)
    print("building held-out battery", flush=True)
    queries = build_records(100_000, noise_db=args.noise_db)
    centroids = class_centroids(references)
    statistics = routing_statistics(references)
    witnesses = {
        label: tuple(
            item["fingerprint"]
            for item in references if item["label"] == label
        )
        for label in LABELS
    }
    report, details = audit(queries, centroids, statistics)
    witness_report = audit_minimum_witness(queries, witnesses, statistics)
    np.savez_compressed(
        args.out / "whole_patch_templates.npz",
        labels=np.asarray(LABELS),
        fields=np.stack([centroids[label].field for label in LABELS]),
        anchor_rows=np.asarray([centroids[label].anchor_row for label in LABELS]),
        centroid_rows=np.asarray([
            centroids[label].centroid_row for label in LABELS]),
        centroid_frames=np.asarray([
            centroids[label].centroid_frame for label in LABELS]),
        source_centroid_rows=np.asarray([
            centroids[label].source_centroid_row for label in LABELS]),
        source_centroid_frames=np.asarray([
            centroids[label].source_centroid_frame for label in LABELS]),
        duration_frames=np.asarray([
            centroids[label].duration_frames for label in LABELS]),
        harmonicities=np.asarray([
            centroids[label].harmonicity for label in LABELS]),
        routing_means=np.stack([statistics[label][0] for label in LABELS]),
        routing_scales=np.stack([statistics[label][1] for label in LABELS]),
        witness_fields=np.stack([
            np.stack([item.field for item in witnesses[label]])
            for label in LABELS
        ]),
        witness_anchor_rows=np.stack([
            np.asarray([item.anchor_row for item in witnesses[label]])
            for label in LABELS
        ]),
        witness_source_centroid_rows=np.stack([
            np.asarray([
                item.source_centroid_row for item in witnesses[label]
            ])
            for label in LABELS
        ]),
        witness_source_centroid_frames=np.stack([
            np.asarray([
                item.source_centroid_frame for item in witnesses[label]
            ])
            for label in LABELS
        ]),
        witness_centroid_rows=np.stack([
            np.asarray([item.centroid_row for item in witnesses[label]])
            for label in LABELS
        ]),
        witness_centroid_frames=np.stack([
            np.asarray([item.centroid_frame for item in witnesses[label]])
            for label in LABELS
        ]),
        witness_duration_frames=np.stack([
            np.asarray([item.duration_frames for item in witnesses[label]])
            for label in LABELS
        ]),
        witness_harmonicities=np.stack([
            np.asarray([item.harmonicity for item in witnesses[label]])
            for label in LABELS
        ]),
    )
    output = {
        "method": (
            "harmonic_metadata_routes_then_one_centroid_registered_"
            "whole_reassigned_texture_baseline_raster_distance"
        ),
        "frontend": (
            "3x3_average_of_multiscale_registered_maximum_plus_"
            "q995_amplitude_gauged_sqrt_reassigned_support_power"
        ),
        "apertures": list(DEFAULT_APERTURES),
        "offsets_samples": list(OFFSETS),
        "channel_noise_db": args.noise_db,
        "labels": list(LABELS),
        "voices": [voice.__dict__ for voice in VOICE_PROBES],
        "validation": report,
        "minimum_witness_validation": witness_report,
        "details": details,
    }
    path = args.out / "whole_patch_inventory_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation": report,
        "minimum_witness_validation": witness_report,
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
