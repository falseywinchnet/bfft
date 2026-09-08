"""Build a multi-voice synthetic vowel inventory and score the radio speech."""

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
    rank_phone_templates,
)
from experiments.ostensibly_frontend.run_frontend_probe import (
    exact_double_irfft_texture,
    scale_for_meyer,
)


# hVd/hVt carriers keep the vowel away from synthesis onset and release.
VOWEL_CARRIERS = {
    "IY": "heed",
    "IH": "hid",
    "EH": "head",
    "AE": "had",
    "AA": "hod",
    "AO": "hawed",
    "UH": "hood",
    "UW": "who'd",
    "AH": "hud",
    "ER": "heard",
    "EY": "hayed",
    "AY": "hide",
    "OW": "hoed",
    "AW": "how'd",
    "OY": "hoy",
}


def synthesize(word: str, voice: str, path: Path) -> None:
    subprocess.run(
        ["say", "-v", voice, "-r", "150", "-o", str(path), word],
        check=True,
    )


def vowel_template(path: Path, sample_rate: int) -> PhoneFingerprint:
    samples, _ = librosa.load(path, sr=sample_rate, mono=True)
    samples, _ = librosa.effects.trim(samples, top_db=35)
    _, _, texture = exact_double_irfft_texture(
        samples,
        n_fft=2048,
        hop_length=512,
        crop_rows=256,
    )
    meyer_input, _ = scale_for_meyer(texture)
    cartoon, _ = bfft.meyer_split(meyer_input, threads=4)
    # Discard carrier onset and release.  Multiple voices provide the missing
    # duration/context variation without fitting a user-specific model.
    frame0 = int(round(cartoon.shape[1] * 0.30))
    frame1 = max(frame0 + 2, int(round(cartoon.shape[1] * 0.70)))
    return fingerprint_phone_patch(cartoon[:, frame0:frame1])


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

    archive = np.load(args.frontend_npz)
    cartoon = np.asarray(archive["cartoon"], dtype=np.float64)
    report = json.loads(args.frontend_report.read_text(encoding="utf-8"))
    sample_rate = int(report["sample_rate"])
    phones = [TimeInterval(**item) for item in report["phones"]]
    voices = tuple(item.strip() for item in args.voices.split(",") if item.strip())

    templates: dict[str, list[PhoneFingerprint]] = {
        label: [] for label in VOWEL_CARRIERS
    }
    inventory_records = []
    for voice in voices:
        for label, word in VOWEL_CARRIERS.items():
            path = audio_dir / f"{voice}_{label}_{word.replace(chr(39), '')}.aiff"
            if not path.exists():
                synthesize(word, voice, path)
            fingerprint = vowel_template(path, sample_rate)
            templates[label].append(fingerprint)
            inventory_records.append({
                "label": label,
                "word": word,
                "voice": voice,
                "audio": str(path),
                "fingerprint": serialize_fingerprint(fingerprint),
            })
    frozen_templates = {
        label: tuple(values) for label, values in templates.items()
    }

    hypotheses = []
    for index, interval in enumerate(phones):
        patch = cartoon[:, interval.frame0 : interval.frame1]
        query = fingerprint_phone_patch(patch)
        ranking = rank_phone_templates(
            query,
            frozen_templates,
            top_k=args.top_k,
        )
        hypotheses.append({
            "index": index,
            "seconds": [interval.seconds0, interval.seconds1],
            "frames": [interval.frame0, interval.frame1],
            "anchor_row": query.anchor_row,
            "top_k_vowels": [
                {"phone": phone, "distance": distance}
                for phone, distance in ranking
            ],
        })

    output = {
        "method": "frequency-anchored multi-ridge Eikonal nearest witness",
        "voices": list(voices),
        "vowels": list(VOWEL_CARRIERS),
        "templates": inventory_records,
        "hypotheses": hypotheses,
        "limitations": [
            "vowel-only inventory",
            "carrier nucleus is selected by a fixed central crop",
            "no SSB channel simulation has yet been applied to synthetic references",
            "phone boundaries are proposals, not forced alignments",
        ],
    }
    output_path = args.out / "vowel_top_k.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "voices": list(voices),
        "templates": len(inventory_records),
        "queries": len(hypotheses),
        "output": str(output_path),
        "first_hypotheses": hypotheses[:8],
    }, indent=2))


if __name__ == "__main__":
    main()
