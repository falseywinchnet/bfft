#!/usr/bin/env python3
"""Build a reusable timed-phone fingerprint bank from ARCTIC speakers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.ostensibly_frontend.run_arctic_calibration import (
    _extract_speaker,
    _save_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speakers", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    utterances = tuple(args.utterances.split(","))
    summaries = {}
    for speaker in args.speakers.split(","):
        records = _extract_speaker(
            args.arctic_root, speaker, utterances, args.noise_db
        )
        _save_records(
            args.out / f"{speaker}_phone_fingerprints.npz", records
        )
        counts = Counter(item["label"] for item in records)
        summaries[speaker] = {
            "utterance_count": len(utterances),
            "phone_count": len(records),
            "phone_type_count": len(counts),
            "minimum_witnesses_per_phone": min(counts.values()),
            "maximum_witnesses_per_phone": max(counts.values()),
        }
    result = {
        "method": "timed_arctic_phone_fingerprint_bank",
        "speakers": args.speakers.split(","),
        "utterances": list(utterances),
        "noise_db": args.noise_db,
        "summaries": summaries,
    }
    output = args.out / "fingerprint_bank.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
