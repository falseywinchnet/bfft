"""Explicit-phoneme synthetic vowel inventory with a cross-voice audit gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import librosa
import numpy as np

import bfft
from experiments.ostensibly_frontend.full_recording import TimeInterval
from experiments.ostensibly_frontend.phone_match import (
    PhoneFingerprint,
    fingerprint_phone_patch,
    phone_distance,
    rank_phone_templates,
)
from experiments.ostensibly_frontend.run_frontend_probe import (
    exact_double_irfft_texture,
    scale_for_meyer,
)


# Apple Speech Manager Table 4-3 symbols, mapped to familiar ARPABET labels.
# AH uses Apple's stressed "bud" vowel UX; AX is retained separately.
VOWELS = {
    "IY": "IY",
    "IH": "IH",
    "EY": "EY",
    "EH": "EH",
    "AE": "AE",
    "AA": "AA",
    "AO": "AO",
    "UW": "UW",
    "UH": "UH",
    "OW": "OW",
    "AW": "AW",
    "OY": "OY",
    "AY": "AY",
    "AH": "UX",
    "AX": "AX",
}


def synthesize_phone(symbol: str, voice: str, path: Path) -> None:
    phonemic_text = f"[[inpt PHON]]{symbol}[[inpt TEXT]]"
    subprocess.run(
        ["say", "-v", voice, "-r", "150", "-o", str(path), phonemic_text],
        check=True,
    )


def template_from_audio(path: Path, sample_rate: int) -> PhoneFingerprint:
    samples, _ = librosa.load(path, sr=sample_rate, mono=True)
    samples, _ = librosa.effects.trim(samples, top_db=40)
    _, _, texture = exact_double_irfft_texture(
        samples,
        n_fft=2048,
        hop_length=512,
        crop_rows=256,
    )
    meyer_input, _ = scale_for_meyer(texture)
    cartoon, _ = bfft.meyer_split(meyer_input, threads=4)
    return fingerprint_phone_patch(cartoon)


def leave_voice_out_audit(records: list[dict]) -> dict:
    correct = 0
    margins: list[float] = []
    cases = []
    for query in records:
        scores: dict[str, float] = {}
        for reference in records:
            if reference["voice"] == query["voice"]:
                continue
            distance = phone_distance(query["fingerprint"], reference["fingerprint"])
            label = reference["label"]
            scores[label] = min(scores.get(label, float("inf")), distance)
        ranking = sorted(scores.items(), key=lambda item: (item[1], item[0]))
        predicted = ranking[0][0]
        correct += int(predicted == query["label"])
        true_distance = scores[query["label"]]
        impostor_distance = min(
            distance for label, distance in scores.items()
            if label != query["label"]
        )
        margin = impostor_distance - true_distance
        margins.append(margin)
        cases.append({
            "label": query["label"],
            "voice": query["voice"],
            "predicted": predicted,
            "true_distance": true_distance,
            "impostor_distance": impostor_distance,
            "margin": margin,
            "top3": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:3]
            ],
        })
    accuracy = correct / max(len(records), 1)
    return {
        "cases": len(records),
        "correct": correct,
        "top1_accuracy": accuracy,
        "median_margin": float(np.median(margins)) if margins else None,
        "positive_margins": int(np.count_nonzero(np.asarray(margins) > 0.0)),
        "passes_minimum_gate": bool(accuracy >= 0.60 and np.median(margins) > 0.0),
        "details": cases,
    }


def serialize_fingerprint(value: PhoneFingerprint) -> dict:
    return {
        "anchor_row": value.anchor_row,
        "duration_frames": value.duration_frames,
        "radial": value.radial.tolist(),
        "spatial": value.spatial.tolist(),
        "intensity": value.intensity.tolist(),
        "row_profile": value.row_profile.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frontend_npz", type=Path)
    parser.add_argument("frontend_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voices", default="Samantha,Daniel,Fred")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "synthetic_audio"
    audio_dir.mkdir(exist_ok=True)

    frontend = np.load(args.frontend_npz)
    cartoon = np.asarray(frontend["cartoon"], dtype=np.float64)
    frontend_report = json.loads(args.frontend_report.read_text(encoding="utf-8"))
    sample_rate = int(frontend_report["sample_rate"])
    phones = [TimeInterval(**item) for item in frontend_report["phones"]]
    voices = tuple(item.strip() for item in args.voices.split(",") if item.strip())

    records: list[dict] = []
    templates: dict[str, list[PhoneFingerprint]] = {
        label: [] for label in VOWELS
    }
    for voice in voices:
        for label, symbol in VOWELS.items():
            path = audio_dir / f"{voice}_{label}.aiff"
            if not path.exists():
                synthesize_phone(symbol, voice, path)
            fingerprint = template_from_audio(path, sample_rate)
            records.append({
                "label": label,
                "symbol": symbol,
                "voice": voice,
                "audio": str(path),
                "fingerprint": fingerprint,
            })
            templates[label].append(fingerprint)

    audit = leave_voice_out_audit(records)
    frozen_templates = {label: tuple(values) for label, values in templates.items()}
    hypotheses = []
    if audit["passes_minimum_gate"]:
        for index, interval in enumerate(phones):
            query = fingerprint_phone_patch(
                cartoon[:, interval.frame0 : interval.frame1]
            )
            hypotheses.append({
                "index": index,
                "seconds": [interval.seconds0, interval.seconds1],
                "anchor_row": query.anchor_row,
                "top_k_vowels": [
                    {"phone": label, "distance": distance}
                    for label, distance in rank_phone_templates(
                        query, frozen_templates, top_k=args.top_k
                    )
                ],
            })

    output = {
        "method": "explicit Apple phoneme synthesis plus anchored Eikonal geometry",
        "voices": list(voices),
        "symbols": VOWELS,
        "audit": audit,
        "radio_hypotheses_emitted": bool(audit["passes_minimum_gate"]),
        "hypotheses": hypotheses,
        "templates": [
            {
                **{key: value for key, value in record.items() if key != "fingerprint"},
                "fingerprint": serialize_fingerprint(record["fingerprint"]),
            }
            for record in records
        ],
    }
    path = args.out / "direct_vowel_inventory.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "templates": len(records),
        "audit": {key: value for key, value in audit.items() if key != "details"},
        "radio_hypotheses": len(hypotheses),
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
