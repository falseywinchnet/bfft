#!/usr/bin/env python3
"""Cross-speaker phone calibration on CMU ARCTIC timed labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from experiments.ostensibly_frontend.phone_match import (
    WholePatchFingerprint,
    fingerprint_whole_phone_patch,
    whole_patch_distance,
)
from experiments.ostensibly_frontend.run_radio_whole_patch_match import (
    _load_templates,
    _route,
)
from experiments.ostensibly_frontend.run_whole_patch_inventory import (
    LABELS,
    OFFSETS,
    ROUTE_SIZE,
    route_labels,
    routing_statistics,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    apply_ssb_probe_channel,
)
from experiments.ostensibly_frontend.uncertainty_fusion import _as_float_audio


PHONE_MAP = {
    **{label.lower(): label for label in LABELS},
    "ax": "AH",
}
TARGET_SAMPLE_RATE = 48_000
HOP_LENGTH = 512
DEFAULT_UTTERANCES = (
    "arctic_a0108",
    "arctic_a0019",
    "arctic_a0117",
    "arctic_a0041",
    "arctic_a0095",
)


def _timed_phones(path: Path):
    start = 0.0
    output = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0].startswith("#"):
            continue
        stop = float(fields[0])
        raw_label = fields[2].lower()
        label = PHONE_MAP.get(raw_label)
        if label is not None and stop > start:
            output.append((label, start, stop, raw_label))
        start = stop
    return output


def _extract_speaker(
    root: Path,
    speaker: str,
    utterances,
    noise_db: float,
):
    records = []
    for utterance_index, utterance in enumerate(utterances):
        wav_path = root / speaker / "wav" / f"{utterance}.wav"
        lab_path = root / speaker / "lab" / f"{utterance}.lab"
        sample_rate, samples = wavfile.read(wav_path)
        samples = _as_float_audio(samples)
        if int(sample_rate) != TARGET_SAMPLE_RATE:
            samples = resample_poly(
                samples, TARGET_SAMPLE_RATE, int(sample_rate)
            )
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
        centers_seconds = (
            HOP_LENGTH * np.arange(baseline.merged.shape[1])
            / TARGET_SAMPLE_RATE
        )
        for label, seconds0, seconds1, raw_label in _timed_phones(lab_path):
            indices = np.flatnonzero(
                (centers_seconds >= seconds0) & (centers_seconds < seconds1)
            )
            if not indices.size:
                midpoint = 0.5 * (seconds0 + seconds1)
                indices = np.asarray((int(np.argmin(
                    np.abs(centers_seconds - midpoint)
                )),))
            fingerprint = fingerprint_whole_phone_patch(
                baseline.merged[:, indices]
            )
            records.append({
                "label": label,
                "speaker": speaker,
                "utterance": utterance,
                "seconds0": seconds0,
                "seconds1": seconds1,
                "raw_label": raw_label,
                "fingerprint": fingerprint,
            })
        print(
            f"{speaker}: extracted {utterance_index + 1}/{len(utterances)} utterances",
            flush=True,
        )
    return records


def _centroids(records):
    result = {}
    for label in LABELS:
        values = [
            item["fingerprint"] for item in records if item["label"] == label
        ]
        if not values:
            raise ValueError(f"calibration speaker has no {label} phone")
        field = np.mean(np.stack([item.field for item in values]), axis=0)
        field /= max(float(np.linalg.norm(field)), 1e-30)
        result[label] = replace(
            values[0],
            anchor_row=int(round(np.mean([item.anchor_row for item in values]))),
            source_centroid_row=float(np.mean([
                item.source_centroid_row for item in values
            ])),
            source_centroid_frame=float(np.mean([
                item.source_centroid_frame for item in values
            ])),
            duration_frames=int(round(np.mean([
                item.duration_frames for item in values
            ]))),
            harmonicity=float(np.mean([item.harmonicity for item in values])),
            field=field,
        )
    return {label: (value,) for label, value in result.items()}


def _witnesses(records, maximum=5):
    return {
        label: tuple(
            item["fingerprint"]
            for item in records if item["label"] == label
        )[:maximum]
        for label in LABELS
    }


def _evaluate(queries, templates, route):
    hits = {1: 0, 3: 0, 5: 0}
    route_hits = 0
    details = []
    for item in queries:
        query = item["fingerprint"]
        candidates = route(query)
        route_hits += int(item["label"] in candidates)
        ranking = sorted(
            (
                (
                    label,
                    min(whole_patch_distance(query, reference)
                        for reference in templates[label]),
                )
                for label in candidates
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        for count in hits:
            hits[count] += int(any(
                label == item["label"] for label, _ in ranking[:count]
            ))
        details.append({
            "target": item["label"],
            "utterance": item["utterance"],
            "seconds0": item["seconds0"],
            "seconds1": item["seconds1"],
            "route": candidates,
            "top5": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:5]
            ],
        })
    count = len(queries)
    return {
        "cases": count,
        "route_recall": route_hits / max(count, 1),
        **{f"top{k}_recall": value / max(count, 1) for k, value in hits.items()},
    }, details


def _save_records(path: Path, records):
    np.savez_compressed(
        path,
        labels=np.asarray([item["label"] for item in records]),
        speakers=np.asarray([item["speaker"] for item in records]),
        utterances=np.asarray([item["utterance"] for item in records]),
        seconds0=np.asarray([item["seconds0"] for item in records]),
        seconds1=np.asarray([item["seconds1"] for item in records]),
        fields=np.stack([item["fingerprint"].field for item in records]),
        anchor_rows=np.asarray([
            item["fingerprint"].anchor_row for item in records
        ]),
        source_centroid_rows=np.asarray([
            item["fingerprint"].source_centroid_row for item in records
        ]),
        source_centroid_frames=np.asarray([
            item["fingerprint"].source_centroid_frame for item in records
        ]),
        duration_frames=np.asarray([
            item["fingerprint"].duration_frames for item in records
        ]),
        harmonicities=np.asarray([
            item["fingerprint"].harmonicity for item in records
        ]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("analytic_templates", type=Path)
    parser.add_argument("--speakers", default="bdl,slt")
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    speakers = tuple(args.speakers.split(","))
    if len(speakers) != 2:
        raise ValueError("calibration currently requires exactly two speakers")
    utterances = tuple(args.utterances.split(","))
    records = {
        speaker: _extract_speaker(
            args.arctic_root, speaker, utterances, args.noise_db
        )
        for speaker in speakers
    }
    for speaker in speakers:
        _save_records(args.out / f"{speaker}_phone_fingerprints.npz", records[speaker])

    analytic_templates, analytic_routing = _load_templates(
        args.analytic_templates, "centroid"
    )
    analytic_route = lambda query: _route(
        query, analytic_routing, count=ROUTE_SIZE
    )
    results = {}
    details = {}
    for target_speaker in speakers:
        metric, rows = _evaluate(
            records[target_speaker], analytic_templates, analytic_route
        )
        results[f"analytic_to_{target_speaker}"] = metric
        details[f"analytic_to_{target_speaker}"] = rows

    for source_speaker, target_speaker in (
        (speakers[0], speakers[1]), (speakers[1], speakers[0])
    ):
        routing_records = [
            {**item, "fingerprint": item["fingerprint"]}
            for item in records[source_speaker]
        ]
        statistics = routing_statistics(routing_records)
        route = lambda query, statistics=statistics: route_labels(
            query, statistics
        )
        for mode, templates in (
            ("centroid", _centroids(records[source_speaker])),
            ("minimum_witness", _witnesses(records[source_speaker])),
        ):
            key = f"{source_speaker}_to_{target_speaker}_{mode}"
            metric, rows = _evaluate(records[target_speaker], templates, route)
            results[key] = metric
            details[key] = rows
    result = {
        "method": (
            "timed_cmu_arctic_phone_crops_through_selected_baseline_and_"
            "target_matched_channel"
        ),
        "label_provenance": (
            "CMU ARCTIC automatic CMU-Sphinx/FestVox labels; not hand-corrected"
        ),
        "speakers": list(speakers),
        "utterances": list(utterances),
        "noise_db": args.noise_db,
        "results": results,
        "details": details,
    }
    output = args.out / "arctic_calibration_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
