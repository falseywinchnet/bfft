#!/usr/bin/env python3
"""Validate the candidate terminal geometry on an unseen ARCTIC speaker."""

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
from experiments.ostensibly_frontend.run_arctic_calibration import (
    DEFAULT_UTTERANCES,
    _extract_speaker,
    _save_records,
)
from experiments.ostensibly_frontend.run_whole_patch_inventory import (
    ROUTE_SIZE,
    route_labels,
    routing_statistics,
)


def _loaded_records(path: Path):
    saved = np.load(path)
    records = []
    for index, field in enumerate(saved["fields"]):
        fingerprint = WholePatchFingerprint(
            anchor_row=int(saved["anchor_rows"][index]),
            source_centroid_row=float(saved["source_centroid_rows"][index]),
            source_centroid_frame=float(saved["source_centroid_frames"][index]),
            centroid_row=0.5 * (field.shape[0] - 1),
            centroid_frame=0.5 * (field.shape[1] - 1),
            duration_frames=int(saved["duration_frames"][index]),
            harmonicity=float(saved["harmonicities"][index]),
            field=np.asarray(field, dtype=np.float64),
        )
        records.append({
            "label": str(saved["labels"][index]),
            "speaker": str(saved["speakers"][index]),
            "utterance": str(saved["utterances"][index]),
            "fingerprint": fingerprint,
        })
    return records


def _unit(field):
    vector = np.asarray(field, dtype=np.float64).ravel()
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def _evaluate(training, queries, mode):
    statistics = routing_statistics(training)
    descriptors = {}
    for item in training:
        descriptor = (
            _unit(item["fingerprint"].field)
            if mode == "raw"
            else multiscale_phone_descriptor(item["fingerprint"].field)
        )
        descriptors.setdefault(item["label"], []).append(descriptor)
    descriptors = {
        label: np.stack(values) for label, values in descriptors.items()
    }
    hits = {1: 0, 3: 0, 5: 0}
    route_hits = 0
    details = []
    for item in queries:
        query = item["fingerprint"]
        candidates = route_labels(query, statistics, count=ROUTE_SIZE)
        route_hits += int(item["label"] in candidates)
        descriptor = (
            _unit(query.field)
            if mode == "raw"
            else multiscale_phone_descriptor(query.field)
        )
        if mode == "multiscale_gabor_duration":
            ranking = sorted(
                (
                    max(
                        duration_coupled_similarity(
                            descriptor,
                            query.duration_frames,
                            reference_descriptor,
                            reference["fingerprint"].duration_frames,
                        )
                        for reference_descriptor, reference in zip(
                            descriptors[label],
                            (row for row in training if row["label"] == label),
                        )
                    ),
                    label,
                )
                for label in candidates
            )
        else:
            ranking = sorted(
                (
                    float(np.max(descriptors[label] @ descriptor)),
                    label,
                )
                for label in candidates
            )
        ranked = [label for _, label in reversed(ranking)]
        for count in hits:
            hits[count] += int(item["label"] in ranked[:count])
        details.append({
            "target": item["label"],
            "utterance": item["utterance"],
            "route": candidates,
            "top5": ranked[:5],
        })
    count = len(queries)
    return {
        "cases": count,
        "route_recall": route_hits / count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
    }, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("training_directory", type=Path)
    parser.add_argument("--training-speakers", default="bdl,slt")
    parser.add_argument("--holdout-speaker", default="clb")
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--reuse-holdout-fingerprints", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    training_speakers = tuple(args.training_speakers.split(","))
    training = []
    for speaker in training_speakers:
        training.extend(_loaded_records(
            args.training_directory / f"{speaker}_phone_fingerprints.npz"
        ))
    holdout_path = args.out / f"{args.holdout_speaker}_phone_fingerprints.npz"
    if args.reuse_holdout_fingerprints and holdout_path.exists():
        holdout = _loaded_records(holdout_path)
    else:
        holdout = _extract_speaker(
            args.arctic_root,
            args.holdout_speaker,
            tuple(args.utterances.split(",")),
            args.noise_db,
        )
        _save_records(holdout_path, holdout)
    results = {}
    details = {}
    for mode in ("raw", "multiscale_gabor", "multiscale_gabor_duration"):
        results[mode], details[mode] = _evaluate(training, holdout, mode)
        print(mode, results[mode], flush=True)
    result = {
        "method": "combined_real_speaker_bank_to_unseen_speaker",
        "training_speakers": list(training_speakers),
        "holdout_speaker": args.holdout_speaker,
        "utterances": args.utterances.split(","),
        "noise_db": args.noise_db,
        "results": results,
        "details": details,
    }
    output = args.out / "arctic_holdout_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
