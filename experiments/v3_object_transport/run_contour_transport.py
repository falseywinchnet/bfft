#!/usr/bin/env python3
"""Audit exact one-sided contour participation on the frozen V3 controls."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.contour_transport import (
    build_contour_transport,
    summarize_contour_transport,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_bundle,
    _load_complex,
)


def _audit(kernel: np.ndarray, stages, landmarks: dict) -> dict:
    labels = stages["compound_labels"]
    height, width = labels.shape
    points = {}
    for name, specification in landmarks.items():
        xn, yn = specification["xy"]
        x = int(np.clip(round(float(xn) * (width - 1)), 0, width - 1))
        y = int(np.clip(round(float(yn) * (height - 1)), 0, height - 1))
        points[name] = {
            "region": int(labels[y, x]),
            "instance": specification["instance"],
            "xy": [x, y],
        }
    pairs = []
    for first_name, second_name in combinations(points, 2):
        first, second = points[first_name], points[second_name]
        similarity = float(kernel[first["region"], second["region"]])
        pairs.append({
            "first": first_name,
            "second": second_name,
            "same_instance": first["instance"] == second["instance"],
            "contour_participation": similarity,
            "distance": 1.0 - similarity,
        })
    same = np.asarray([
        pair["distance"] for pair in pairs if pair["same_instance"]
    ], dtype=np.float64)
    different = np.asarray([
        pair["distance"] for pair in pairs if not pair["same_instance"]
    ], dtype=np.float64)
    auc = None
    if len(same) and len(different):
        comparison = same[:, None] - different[None, :]
        auc = float(
            np.mean(comparison < 0.0) + 0.5 * np.mean(comparison == 0.0))
    return {
        "points": points,
        "pairs": pairs,
        "same_instance_median_distance": (
            float(np.median(same)) if len(same) else None),
        "different_instance_median_distance": (
            float(np.median(different)) if len(different) else None),
        "closer_pair_auc": auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "contour_transport")
    output.mkdir(parents=True, exist_ok=True)
    landmark_specification = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "exact connected one-sided contour participation; landmarks are "
            "evaluation-only"
        ),
        "images": {},
    }
    for name in CONTROLS:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        bundle = _load_bundle(image_dir / "compound_incidence_bundle.npz")
        stages = np.load(image_dir / "v3_stages.npz")
        transport = build_contour_transport(complex_, bundle)
        audit = _audit(
            transport["region_kernel"],
            stages,
            landmark_specification[name],
        )
        opposite = transport["opposite_participation"]
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "contour_transport.npz",
            incidence_component=transport["incidence_component"],
            component_owner=transport["component_owner"],
            component_length=transport["component_length"],
            component_arcs=transport["component_arcs"],
            component_closed_arcs=transport["component_closed_arcs"],
            pair_component=transport["pair_component"],
            pair_region=transport["pair_region"],
            pair_length=transport["pair_length"],
            pair_fraction=transport["pair_fraction"],
            opposite_data=opposite.data,
            opposite_indices=opposite.indices,
            opposite_indptr=opposite.indptr,
            opposite_shape=np.asarray(opposite.shape, dtype=np.int32),
            region_kernel=np.asarray(
                transport["region_kernel"], dtype=np.float32),
        )
        report["images"][name] = {
            "summary": summarize_contour_transport(transport),
            "audit": audit,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
