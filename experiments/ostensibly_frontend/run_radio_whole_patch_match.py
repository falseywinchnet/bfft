#!/usr/bin/env python3
"""Apply whole-fused-field synthetic centroids to saved radio phone proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.phone_match import (
    WholePatchFingerprint,
    fingerprint_whole_phone_patch,
    whole_patch_distance,
)
from experiments.ostensibly_frontend.run_whole_patch_inventory import ROUTE_SIZE


def _load_templates(path: Path, template_mode: str) -> tuple[
    dict[str, tuple[WholePatchFingerprint, ...]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    saved = np.load(path)
    labels = [str(value) for value in saved["labels"]]
    templates = {}
    routing = {}
    for index, label in enumerate(labels):
        if template_mode == "minimum-witness":
            if "witness_fields" not in saved.files:
                raise ValueError("template file does not contain witness fields")
            templates[label] = tuple(
                WholePatchFingerprint(
                    anchor_row=int(saved["witness_anchor_rows"][index, witness]),
                    source_centroid_row=float(
                        saved["witness_source_centroid_rows"][index, witness]),
                    source_centroid_frame=float(
                        saved["witness_source_centroid_frames"][index, witness]),
                    centroid_row=float(
                        saved["witness_centroid_rows"][index, witness]),
                    centroid_frame=float(
                        saved["witness_centroid_frames"][index, witness]),
                    duration_frames=int(
                        saved["witness_duration_frames"][index, witness]),
                    harmonicity=float(
                        saved["witness_harmonicities"][index, witness]),
                    field=np.asarray(
                        saved["witness_fields"][index, witness],
                        dtype=np.float64,
                    ),
                )
                for witness in range(saved["witness_fields"].shape[1])
            )
        else:
            templates[label] = (WholePatchFingerprint(
                anchor_row=int(saved["anchor_rows"][index]),
                source_centroid_row=float(saved["source_centroid_rows"][index]),
                source_centroid_frame=float(saved["source_centroid_frames"][index]),
                centroid_row=float(saved["centroid_rows"][index]),
                centroid_frame=float(saved["centroid_frames"][index]),
                duration_frames=int(saved["duration_frames"][index]),
                harmonicity=float(saved["harmonicities"][index]),
                field=np.asarray(saved["fields"][index], dtype=np.float64),
            ),)
        routing[label] = (
            np.asarray(saved["routing_means"][index], dtype=np.float64),
            np.asarray(saved["routing_scales"][index], dtype=np.float64),
        )
    return templates, routing


def _route(
    query: WholePatchFingerprint,
    routing: dict[str, tuple[np.ndarray, np.ndarray]],
    count: int = ROUTE_SIZE,
) -> list[str]:
    feature = np.asarray((
        query.anchor_row,
        query.source_centroid_row,
        query.source_centroid_frame / max(query.duration_frames - 1, 1),
        np.log(max(query.harmonicity, 1e-12)),
    ))
    scored = [
        (label, float(np.linalg.norm((feature - mean) / scale)))
        for label, (mean, scale) in routing.items()
    ]
    return [label for label, _ in sorted(scored, key=lambda x: (x[1], x[0]))[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_npz", type=Path)
    parser.add_argument("recording_json", type=Path)
    parser.add_argument("templates_npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route-size", type=int, default=ROUTE_SIZE)
    parser.add_argument(
        "--template-mode",
        choices=("centroid", "minimum-witness"),
        default="centroid",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recording = np.load(args.recording_npz)
    field_key = (
        "trace_field" if "trace_field" in recording.files
        else "registered_maximum"
    )
    field = np.asarray(recording[field_key], dtype=np.float64)
    metadata = json.loads(args.recording_json.read_text())
    templates, routing = _load_templates(args.templates_npz, args.template_mode)
    rows = []
    query_fingerprints: list[WholePatchFingerprint] = []
    for index, interval in enumerate(metadata["phones"]):
        frame0 = int(interval["frame0"])
        frame1 = int(interval["frame1"])
        query = fingerprint_whole_phone_patch(field[:, frame0:frame1])
        query_fingerprints.append(query)
        candidates = _route(query, routing, count=args.route_size)
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
        rows.append({
            **interval,
            "anchor_row": query.anchor_row,
            "source_centroid_row": query.source_centroid_row,
            "source_centroid_frame": query.source_centroid_frame,
            "duration_frames": query.duration_frames,
            "harmonicity": query.harmonicity,
            "route": candidates,
            "top5": [
                {"phone": label, "distance": distance}
                for label, distance in ranking[:5]
            ],
        })
        if (index + 1) % 25 == 0:
            print(f"matched {index + 1}/{len(metadata['phones'])}", flush=True)
    result = {
        "method": (
            "harmonic_metadata_routes_then_one_centroid_registered_"
            "whole_reassigned_texture_baseline_raster_distance"
        ),
        "recording_field": field_key,
        "route_size": args.route_size,
        "template_mode": args.template_mode,
        "recording": str(args.recording_npz),
        "phone_count": len(rows),
        "phones": rows,
    }
    path = args.out / "radio_whole_patch_top_k.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.out / "radio_query_fingerprints.npz",
        fields=np.stack([item.field for item in query_fingerprints]),
        anchor_rows=np.asarray([item.anchor_row for item in query_fingerprints]),
        source_centroid_rows=np.asarray([
            item.source_centroid_row for item in query_fingerprints]),
        source_centroid_frames=np.asarray([
            item.source_centroid_frame for item in query_fingerprints]),
        centroid_rows=np.asarray([
            item.centroid_row for item in query_fingerprints]),
        centroid_frames=np.asarray([
            item.centroid_frame for item in query_fingerprints]),
        duration_frames=np.asarray([
            item.duration_frames for item in query_fingerprints]),
        harmonicities=np.asarray([
            item.harmonicity for item in query_fingerprints]),
    )
    print(json.dumps({"phone_count": len(rows), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
