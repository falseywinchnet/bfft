#!/usr/bin/env python3
"""Freeze whole-patch fingerprints for a saved phone segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.phone_match import (
    fingerprint_whole_phone_patch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_npz", type=Path)
    parser.add_argument("recording_json", type=Path)
    parser.add_argument("--field-key", default="trace_field")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recording = np.load(args.recording_npz)
    field_key = args.field_key
    if field_key not in recording.files:
        raise ValueError(f"recording does not contain field {field_key!r}")
    field = np.asarray(recording[field_key], dtype=np.float64)
    intervals = json.loads(args.recording_json.read_text())["phones"]
    fingerprints = [
        fingerprint_whole_phone_patch(
            field[:, int(interval["frame0"]):int(interval["frame1"])]
        )
        for interval in intervals
    ]
    output = args.out / "radio_query_fingerprints.npz"
    np.savez_compressed(
        output,
        fields=np.stack([item.field for item in fingerprints]),
        anchor_rows=np.asarray([item.anchor_row for item in fingerprints]),
        source_centroid_rows=np.asarray([
            item.source_centroid_row for item in fingerprints]),
        source_centroid_frames=np.asarray([
            item.source_centroid_frame for item in fingerprints]),
        centroid_rows=np.asarray([item.centroid_row for item in fingerprints]),
        centroid_frames=np.asarray([
            item.centroid_frame for item in fingerprints]),
        duration_frames=np.asarray([
            item.duration_frames for item in fingerprints]),
        harmonicities=np.asarray([item.harmonicity for item in fingerprints]),
        frame0=np.asarray([int(item["frame0"]) for item in intervals]),
        frame1=np.asarray([int(item["frame1"]) for item in intervals]),
    )
    print(json.dumps({"count": len(fingerprints), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
