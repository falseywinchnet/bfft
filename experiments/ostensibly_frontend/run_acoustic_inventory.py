"""Audit a deterministic formant inventory through the Ostensibly front end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import bfft
from experiments.ostensibly_frontend.phone_match import (
    PhoneFingerprint,
    fingerprint_phone_patch,
    phone_distance,
)
from experiments.ostensibly_frontend.run_frontend_probe import (
    exact_double_irfft_texture,
    scale_for_meyer,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    FORMANTS,
    VOICE_PROBES,
    apply_ssb_probe_channel,
    synthesize_vowel,
)


def build_fingerprint(label: str, voice_index: int, seed_offset: int) -> PhoneFingerprint:
    voice = VOICE_PROBES[voice_index]
    vowel_index = list(FORMANTS).index(label)
    samples = synthesize_vowel(
        label,
        voice,
        seed=seed_offset + 1000 * voice_index + vowel_index,
    )
    samples = apply_ssb_probe_channel(
        samples,
        seed=seed_offset + 2000 * voice_index + vowel_index,
    )
    _, _, texture = exact_double_irfft_texture(
        samples,
        n_fft=2048,
        hop_length=512,
        crop_rows=256,
    )
    meyer_input, _ = scale_for_meyer(texture)
    cartoon, _ = bfft.meyer_split(meyer_input, threads=4)
    return fingerprint_phone_patch(cartoon)


def build_records(seed_offset: int) -> list[dict]:
    records = []
    for voice_index, voice in enumerate(VOICE_PROBES):
        for label in FORMANTS:
            records.append({
                "label": label,
                "voice": voice.name,
                "f0": voice.f0,
                "tract_scale": voice.tract_scale,
                "fingerprint": build_fingerprint(label, voice_index, seed_offset),
            })
    return records


def audit_records(queries: list[dict], references: list[dict]) -> tuple[dict, list[dict]]:
    correct = 0
    top3_correct = 0
    top5_correct = 0
    margins = []
    cases = []
    for query in queries:
        scores: dict[str, float] = {}
        for reference in references:
            if reference["voice"] == query["voice"]:
                continue
            distance = phone_distance(query["fingerprint"], reference["fingerprint"])
            scores[reference["label"]] = min(
                scores.get(reference["label"], float("inf")), distance
            )
        ranking = sorted(scores.items(), key=lambda item: (item[1], item[0]))
        predicted = ranking[0][0]
        correct += int(predicted == query["label"])
        top3_correct += int(any(
            label == query["label"] for label, _ in ranking[:3]
        ))
        top5_correct += int(any(
            label == query["label"] for label, _ in ranking[:5]
        ))
        true_distance = scores[query["label"]]
        impostor = min(
            value for label, value in scores.items() if label != query["label"]
        )
        margins.append(impostor - true_distance)
        cases.append({
            "label": query["label"],
            "voice": query["voice"],
            "predicted": predicted,
            "top5": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:5]
            ],
        })
    audit = {
        "cases": len(queries),
        "correct": correct,
        "top1_accuracy": correct / len(queries),
        "top3_recall": top3_correct / len(queries),
        "top5_recall": top5_correct / len(queries),
        "median_margin": float(np.median(margins)),
        "positive_margins": int(np.count_nonzero(np.asarray(margins) > 0.0)),
        "passes_minimum_gate": bool(
            correct / len(queries) >= 0.60 and np.median(margins) > 0.0
        ),
        "passes_topk_gate": bool(top3_correct / len(queries) >= 0.90),
    }
    return audit, cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = build_records(seed_offset=0)
    validation_records = build_records(seed_offset=100_000)
    calibration_audit, calibration_cases = audit_records(records, records)
    validation_audit, validation_cases = audit_records(validation_records, records)
    output = {
        "method": "deterministic glottal/formant probes through declared SSB channel",
        "formants_hz": FORMANTS,
        "voices": [voice.__dict__ for voice in VOICE_PROBES],
        "calibration_audit": calibration_audit,
        "validation_audit": validation_audit,
        "calibration_cases": calibration_cases,
        "validation_cases": validation_cases,
    }
    path = args.out / "acoustic_inventory_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "calibration_audit": calibration_audit,
        "validation_audit": validation_audit,
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
