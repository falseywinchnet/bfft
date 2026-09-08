#!/usr/bin/env python3
"""Audit which selected-baseline component transfers phone geometry."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.phone_match import whole_patch_distance
from experiments.ostensibly_frontend.run_whole_patch_inventory import (
    LABELS,
    ROUTE_SIZE,
    VOICE_PROBES,
    _component_fingerprints,
    route_labels,
    routing_statistics,
)


COMPONENTS = ("merged", "cartoon", "texture", "registered_maximum")


def build_records(seed_offset: int):
    records = []
    for voice_index, voice in enumerate(VOICE_PROBES):
        for label in LABELS:
            records.append({
                "label": label,
                "voice": voice.name,
                "fingerprints": _component_fingerprints(
                    label, voice_index, seed_offset
                ),
            })
    return records


def centroids(records, component):
    result = {}
    for label in LABELS:
        values = [
            item["fingerprints"][component]
            for item in records if item["label"] == label
        ]
        field = np.mean(np.stack([item.field for item in values]), axis=0)
        field /= max(float(np.linalg.norm(field)), 1e-30)
        result[label] = replace(
            values[0],
            anchor_row=int(round(np.mean([item.anchor_row for item in values]))),
            source_centroid_row=float(np.mean([
                item.source_centroid_row for item in values])),
            source_centroid_frame=float(np.mean([
                item.source_centroid_frame for item in values])),
            duration_frames=int(round(np.mean([
                item.duration_frames for item in values]))),
            harmonicity=float(np.mean([item.harmonicity for item in values])),
            field=field,
        )
    return result


def audit(records, component, component_centroids, route_statistics):
    hits = {1: 0, 3: 0, 5: 0}
    route_hits = 0
    for item in records:
        route_query = item["fingerprints"]["merged"]
        query = item["fingerprints"][component]
        candidates = route_labels(route_query, route_statistics)
        route_hits += int(item["label"] in candidates)
        ranking = sorted(
            (
                (label, whole_patch_distance(query, component_centroids[label]))
                for label in candidates
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        for count in hits:
            hits[count] += int(any(
                label == item["label"] for label, _ in ranking[:count]
            ))
    count = len(records)
    return {
        "cases": count,
        "route_size": ROUTE_SIZE,
        "route_recall": route_hits / count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("building component reference battery", flush=True)
    references = build_records(0)
    print("building component held-out battery", flush=True)
    queries = build_records(100_000)
    merged_route_records = [
        {**item, "fingerprint": item["fingerprints"]["merged"]}
        for item in references
    ]
    statistics = routing_statistics(merged_route_records)
    component_centroids = {
        component: centroids(references, component) for component in COMPONENTS
    }
    validation = {}
    for component in COMPONENTS:
        print(f"auditing {component}", flush=True)
        validation[component] = audit(
            queries, component, component_centroids[component], statistics
        )
    np.savez_compressed(
        args.out / "component_templates.npz",
        labels=np.asarray(LABELS),
        component_names=np.asarray(COMPONENTS),
        fields=np.stack([
            np.stack([
                component_centroids[component][label].field for label in LABELS
            ])
            for component in COMPONENTS
        ]),
        routing_means=np.stack([statistics[label][0] for label in LABELS]),
        routing_scales=np.stack([statistics[label][1] for label in LABELS]),
    )
    result = {
        "method": "symmetric_selected_baseline_component_ablation",
        "components": list(COMPONENTS),
        "validation": validation,
    }
    output = args.out / "component_inventory_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
