#!/usr/bin/env python3
"""Match radio phones to a frozen generic real-speech multiscale bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.multiscale_phone_geometry import (
    duration_coupled_similarity,
    multiscale_phone_descriptor,
)
from experiments.ostensibly_frontend.phone_match import WholePatchFingerprint
from experiments.ostensibly_frontend.run_whole_patch_inventory import (
    ROUTE_SIZE,
    route_labels,
    routing_statistics,
)


def _fingerprints(saved):
    output = []
    for index, field in enumerate(saved["fields"]):
        output.append(WholePatchFingerprint(
            anchor_row=int(saved["anchor_rows"][index]),
            source_centroid_row=float(saved["source_centroid_rows"][index]),
            source_centroid_frame=float(saved["source_centroid_frames"][index]),
            centroid_row=0.5 * (field.shape[0] - 1),
            centroid_frame=0.5 * (field.shape[1] - 1),
            duration_frames=int(saved["duration_frames"][index]),
            harmonicity=float(saved["harmonicities"][index]),
            field=np.asarray(field, dtype=np.float64),
        ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_fingerprints_npz", type=Path)
    parser.add_argument("recording_json", type=Path)
    parser.add_argument("training_directory", type=Path)
    parser.add_argument("--training-speakers", default="bdl,slt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    training = []
    for speaker in args.training_speakers.split(","):
        saved = np.load(
            args.training_directory / f"{speaker}_phone_fingerprints.npz"
        )
        for label, fingerprint in zip(
            saved["labels"].astype(str), _fingerprints(saved)
        ):
            training.append({
                "label": str(label),
                "speaker": speaker,
                "fingerprint": fingerprint,
            })
    statistics = routing_statistics(training)
    banks = {}
    for item in training:
        banks.setdefault(item["label"], []).append((
            multiscale_phone_descriptor(item["fingerprint"].field),
            item["fingerprint"].duration_frames,
        ))

    query_saved = np.load(args.query_fingerprints_npz)
    queries = _fingerprints(query_saved)
    metadata = json.loads(args.recording_json.read_text())
    if len(queries) != len(metadata["phones"]):
        raise ValueError("query fingerprints and phone intervals disagree")
    rows = []
    for index, (query, interval) in enumerate(zip(
        queries, metadata["phones"]
    )):
        route = route_labels(query, statistics, count=ROUTE_SIZE)
        descriptor = multiscale_phone_descriptor(query.field)
        ranking = sorted(
            (
                max(
                    duration_coupled_similarity(
                        descriptor,
                        query.duration_frames,
                        witness_descriptor,
                        witness_duration,
                    )
                    for witness_descriptor, witness_duration in banks[label]
                ),
                label,
            )
            for label in route
        )
        ranking.reverse()
        rows.append({
            **interval,
            "anchor_row": query.anchor_row,
            "source_centroid_row": query.source_centroid_row,
            "source_centroid_frame": query.source_centroid_frame,
            "duration_frames": query.duration_frames,
            "harmonicity": query.harmonicity,
            "route": route,
            "top5": [
                {"phone": label, "distance": 1.0 - similarity}
                for similarity, label in ranking[:5]
            ],
        })
        if (index + 1) % 50 == 0:
            print(f"matched {index + 1}/{len(queries)}", flush=True)
    result = {
        "method": (
            "generic_real_speech_witness_bank_with_raw_retaining_"
            "multiscale_gabor_and_duration_coupled_geometry"
        ),
        "selection_status": (
            "candidate terminal geometry validated on unseen CLB speaker"
        ),
        "training_speakers": args.training_speakers.split(","),
        "route_size": ROUTE_SIZE,
        "phone_count": len(rows),
        "phones": rows,
    }
    output = args.out / "radio_multiscale_top_k.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phone_count": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
