#!/usr/bin/env python3
"""Audit terminal geometries on frozen timed CMU ARCTIC phone crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.multiscale_phone_geometry import (
    multiscale_phone_descriptor,
)


def _unit_rows(values):
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape((array.shape[0], -1))
    return flat / np.maximum(
        np.linalg.norm(flat, axis=1, keepdims=True), 1e-30
    )


def _descriptors(fields, mode):
    if mode == "raw":
        return _unit_rows(fields)
    if mode == "multiscale_gabor":
        return np.stack([
            multiscale_phone_descriptor(field) for field in fields
        ])
    raise KeyError(mode)


def _evaluate(source, target, routes, mode):
    source_labels = np.asarray(source["labels"]).astype(str)
    target_labels = np.asarray(target["labels"]).astype(str)
    labels = tuple(sorted(set(source_labels)))
    source_descriptors = _descriptors(source["fields"], mode)
    target_descriptors = _descriptors(target["fields"], mode)
    hits = {1: 0, 3: 0, 5: 0}
    attractors = {}
    for index, (descriptor, target_label) in enumerate(zip(
        target_descriptors, target_labels, strict=True
    )):
        candidates = routes[index]
        ranking = sorted(
            (
                float(np.max(
                    source_descriptors[source_labels == label] @ descriptor
                )),
                label,
            )
            for label in candidates
        )
        ranked = [label for _, label in reversed(ranking)]
        attractors[ranked[0]] = attractors.get(ranked[0], 0) + 1
        for count in hits:
            hits[count] += int(target_label in ranked[:count])
    count = len(target_labels)
    return {
        "cases": count,
        **{f"top{k}_recall": value / count for k, value in hits.items()},
        "top1_attractors": sorted(
            attractors.items(), key=lambda item: (-item[1], item[0])
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_directory", type=Path)
    parser.add_argument("calibration_audit_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    saved = {
        speaker: np.load(
            args.calibration_directory / f"{speaker}_phone_fingerprints.npz"
        )
        for speaker in ("bdl", "slt")
    }
    prior = json.loads(args.calibration_audit_json.read_text())
    results = {}
    for source, target in (("bdl", "slt"), ("slt", "bdl")):
        detail_key = f"{source}_to_{target}_minimum_witness"
        routes = [row["route"] for row in prior["details"][detail_key]]
        for mode in ("raw", "multiscale_gabor"):
            key = f"{source}_to_{target}_{mode}"
            results[key] = _evaluate(
                saved[source], saved[target], routes, mode
            )
            print(key, results[key], flush=True)
    result = {
        "method": (
            "fixed_route_cross_speaker_witness_ablation_on_timed_phone_crops"
        ),
        "multiscale_weights": {
            "raw": 0.1,
            "gaussian_sigma_2x1.2": 4.0,
            "gaussian_sigma_4x2.4": 8.0,
            "pooled_gabor": 2.0,
        },
        "results": results,
    }
    output = args.out / "arctic_metric_ablation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
