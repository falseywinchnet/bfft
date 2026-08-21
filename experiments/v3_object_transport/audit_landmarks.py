#!/usr/bin/env python3
"""Evaluate frozen sparse landmarks without feeding them to the model."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDMARKS = (
    ROOT / "experiments" / "v3_object_transport" / "assets"
    / "landmarks.json"
)


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).ravel()]


def audit_image(image_dir: Path, landmarks: dict) -> dict:
    stages = np.load(image_dir / "v3_stages.npz")
    leaf_complex = np.load(image_dir / "leaf_region_complex.npz")
    leaf_labels = stages["compound_leaf_labels"]
    height, width = leaf_labels.shape
    points = {}
    for name, specification in landmarks.items():
        xn, yn = specification["xy"]
        x = int(np.clip(round(float(xn) * (width - 1)), 0, width - 1))
        y = int(np.clip(round(float(yn) * (height - 1)), 0, height - 1))
        leaf = int(leaf_labels[y, x])
        points[name] = {
            "xy": [x, y],
            "instance": specification["instance"],
            "leaf": leaf,
            "compound": int(stages["compound_labels"][y, x]),
            "historical_family": int(
                stages["historical_family_labels"][y, x]),
            "structural": int(stages["structural_labels"][y, x]),
            "target_mean": _vector(leaf_complex["node__target_mean"][leaf]),
            "cartoon_mean": _vector(leaf_complex["node__cartoon_mean"][leaf]),
            "texture_rms": _vector(
                leaf_complex["node__texture_target_rms"][leaf]),
            "structural_purity": float(
                leaf_complex["node__structural_purity"][leaf]),
        }
        for field in ("cartoon", "texture", "residual"):
            mean_key = f"node__fused_{field}_mean"
            rms_key = f"node__fused_{field}_rms"
            if mean_key in leaf_complex and rms_key in leaf_complex:
                points[name][f"fused_{field}_mean"] = float(
                    leaf_complex[mean_key][leaf])
                points[name][f"fused_{field}_rms"] = float(
                    leaf_complex[rms_key][leaf])
    pairs = []
    for first_name, second_name in combinations(points, 2):
        first = points[first_name]
        second = points[second_name]
        target_distance = float(np.linalg.norm(
            np.asarray(first["target_mean"])
            - np.asarray(second["target_mean"])))
        cartoon_distance = float(np.linalg.norm(
            np.asarray(first["cartoon_mean"])
            - np.asarray(second["cartoon_mean"])))
        texture_distance = float(np.linalg.norm(
            np.asarray(first["texture_rms"])
            - np.asarray(second["texture_rms"])))
        record = {
            "first": first_name,
            "second": second_name,
            "same_instance": first["instance"] == second["instance"],
            "same_leaf": first["leaf"] == second["leaf"],
            "same_compound": first["compound"] == second["compound"],
            "same_historical_family": (
                first["historical_family"] == second["historical_family"]),
            "same_structural": first["structural"] == second["structural"],
            "target_distance": target_distance,
            "cartoon_distance": cartoon_distance,
            "texture_rms_distance": texture_distance,
        }
        for field in ("cartoon", "texture", "residual"):
            mean_name = f"fused_{field}_mean"
            rms_name = f"fused_{field}_rms"
            if mean_name in first and mean_name in second:
                record[f"fused_{field}_mean_distance"] = abs(
                    first[mean_name] - second[mean_name])
                record[f"fused_{field}_rms_distance"] = abs(
                    first[rms_name] - second[rms_name])
        pairs.append(record)

    decisions = (
        "same_leaf", "same_compound", "same_historical_family",
        "same_structural",
    )
    positive = [pair for pair in pairs if pair["same_instance"]]
    negative = [pair for pair in pairs if not pair["same_instance"]]
    quotient_audit = {}
    for name in decisions:
        quotient_audit[name] = {
            "same_instance_recall": (
                float(np.mean([pair[name] for pair in positive]))
                if positive else None
            ),
            "different_instance_false_join": (
                float(np.mean([pair[name] for pair in negative]))
                if negative else None
            ),
        }
    content_audit = {}
    distance_names = [
        "target_distance", "cartoon_distance", "texture_rms_distance",
    ]
    if pairs and "fused_cartoon_mean_distance" in pairs[0]:
        distance_names.extend(
            f"fused_{field}_{stat}_distance"
            for field in ("cartoon", "texture", "residual")
            for stat in ("mean", "rms")
        )
    for name in distance_names:
        if not positive or not negative:
            content_audit[name] = None
            continue
        same = np.asarray([pair[name] for pair in positive], dtype=np.float64)
        different = np.asarray(
            [pair[name] for pair in negative], dtype=np.float64)
        comparison = same[:, None] - different[None, :]
        content_audit[name] = {
            "same_instance_median": float(np.median(same)),
            "different_instance_median": float(np.median(different)),
            "closer_pair_auc": float(
                np.mean(comparison < 0.0)
                + 0.5 * np.mean(comparison == 0.0)
            ),
        }
    return {
        "shape": [height, width],
        "landmarks": points,
        "pairs": pairs,
        "quotient_audit": quotient_audit,
        "content_distance_audit": content_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    specification = json.loads(args.landmarks.read_text())
    images = {
        name: audit_image(args.results / name, landmarks)
        for name, landmarks in specification["images"].items()
    }
    report = {
        "purpose": (
            "held-out sparse landmark audit; landmark coordinates and instance "
            "labels never enter V3 or any quotient"
        ),
        "images": images,
    }
    output = args.out or (args.results / "landmark_audit.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
