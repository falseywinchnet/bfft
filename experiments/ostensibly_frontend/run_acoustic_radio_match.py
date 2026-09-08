"""Emit calibrated top-k vowel evidence for every proposed radio phone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.full_recording import TimeInterval
from experiments.ostensibly_frontend.phone_match import (
    fingerprint_phone_patch,
    phone_distance,
)
from experiments.ostensibly_frontend.run_acoustic_inventory import (
    audit_records,
    build_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frontend_npz", type=Path)
    parser.add_argument("frontend_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    references = build_records(seed_offset=0) + build_records(seed_offset=200_000)
    validation = build_records(seed_offset=100_000)
    validation_audit, _ = audit_records(validation, references)
    if not validation_audit["passes_topk_gate"]:
        raise RuntimeError(
            "synthetic inventory failed its held-out top-k gate; refusing radio labels"
        )

    archive = np.load(args.frontend_npz)
    cartoon = np.asarray(archive["cartoon"], dtype=np.float64)
    report = json.loads(args.frontend_report.read_text(encoding="utf-8"))
    intervals = [TimeInterval(**item) for item in report["phones"]]
    hypotheses = []
    for index, interval in enumerate(intervals):
        query = fingerprint_phone_patch(
            cartoon[:, interval.frame0 : interval.frame1]
        )
        class_scores: dict[str, float] = {}
        for reference in references:
            distance = phone_distance(query, reference["fingerprint"])
            label = reference["label"]
            class_scores[label] = min(
                class_scores.get(label, float("inf")), distance
            )
        ranking = sorted(class_scores.items(), key=lambda item: (item[1], item[0]))
        best = ranking[0][1]
        hypotheses.append({
            "index": index,
            "seconds": [interval.seconds0, interval.seconds1],
            "frames": [interval.frame0, interval.frame1],
            "duration_seconds": interval.seconds1 - interval.seconds0,
            "anchor_row": query.anchor_row,
            "top_k": [
                {
                    "phone": label,
                    "distance": distance,
                    "distance_from_best": distance - best,
                }
                for label, distance in ranking[: args.top_k]
            ],
        })

    output = {
        "method": "ranked nearest witnesses in registered IRODFT/Meyer geometry",
        "validation_audit": validation_audit,
        "reference_witnesses": len(references),
        "inventory_kind": "deterministic monophthong probes",
        "hypotheses": hypotheses,
        "limitations": [
            "vowels only; consonant likelihood is not yet represented",
            "synthetic held-out top-k recall does not establish radio accuracy",
            "phone boundaries remain overcomplete proposals",
            "word-chain decoding must not consume these as hard labels",
        ],
    }
    path = args.out / "radio_vowel_top_k.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation_audit": validation_audit,
        "reference_witnesses": len(references),
        "radio_intervals": len(hypotheses),
        "output": str(path),
        "first_hypotheses": hypotheses[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
