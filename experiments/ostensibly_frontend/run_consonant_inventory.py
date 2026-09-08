"""Held-out top-k audit for deterministic English consonant probes."""

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
from experiments.ostensibly_frontend.synthetic_consonants import (
    CONSONANTS,
    synthesize_consonant,
)
from experiments.ostensibly_frontend.synthetic_vowels import (
    VOICE_PROBES,
    apply_ssb_probe_channel,
)


def fingerprint(label: str, voice_index: int, seed_offset: int) -> PhoneFingerprint:
    voice = VOICE_PROBES[voice_index]
    phone_index = CONSONANTS.index(label)
    samples = synthesize_consonant(
        label,
        voice,
        seed=seed_offset + 1000 * voice_index + phone_index,
    )
    samples = apply_ssb_probe_channel(
        samples,
        seed=seed_offset + 2000 * voice_index + phone_index,
    )
    _, _, texture = exact_double_irfft_texture(
        samples, n_fft=2048, hop_length=512, crop_rows=256
    )
    meyer_input, _ = scale_for_meyer(texture)
    cartoon, _ = bfft.meyer_split(meyer_input, threads=4)
    return fingerprint_phone_patch(cartoon)


def records(seed_offset: int) -> list[dict]:
    return [
        {
            "label": label,
            "voice": voice.name,
            "fingerprint": fingerprint(label, voice_index, seed_offset),
        }
        for voice_index, voice in enumerate(VOICE_PROBES)
        for label in CONSONANTS
    ]


def audit(queries: list[dict], references: list[dict]) -> tuple[dict, list[dict]]:
    hits = {1: 0, 3: 0, 5: 0, 8: 0}
    details = []
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
        for top_k in hits:
            hits[top_k] += int(any(
                label == query["label"] for label, _ in ranking[:top_k]
            ))
        details.append({
            "label": query["label"],
            "voice": query["voice"],
            "top8": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:8]
            ],
        })
    count = len(queries)
    report = {
        "cases": count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
    }
    report["passes_topk_gate"] = bool(report["top5_recall"] >= 0.90)
    return report, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = records(0) + records(200_000)
    validation = records(100_000)
    report, details = audit(validation, reference)
    output = {
        "method": "declared consonant mechanics through SSB and IRODFT/Meyer",
        "phones": list(CONSONANTS),
        "validation": report,
        "details": details,
    }
    path = args.out / "consonant_inventory_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": report, "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
