#!/usr/bin/env python3
"""Run the seed-free connection bloom and matched nulls on frozen controls."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.connection_bloom import (
    aggregate_region_embedding,
    bloom_region_embedding,
    fit_joint_whitener,
    relation_features,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT / "experiments" / "v3_object_transport" / "results"
    / "v3_object_transport_audit_256_bundle"
)
DEFAULT_LANDMARKS = (
    ROOT / "experiments" / "v3_object_transport" / "assets"
    / "landmarks.json"
)
CONTROLS = ("pikachu_hard", "coffee", "astronaut", "checker", "coins")


def _load_complex(path: Path) -> dict:
    archive = np.load(path)
    section: dict[str, dict[str, np.ndarray]] = {
        "node": {}, "edge": {}, "arc": {}, "ancestry": {},
    }
    topology: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        "edgel": {}, "arc": {}, "junction": {},
    }
    for key in archive.files:
        if "__" not in key:
            continue
        prefix, name = key.split("__", 1)
        if prefix in section:
            section[prefix][name] = archive[key]
        elif prefix == "topology":
            if "__" in name:
                topology_section, field = name.split("__", 1)
                topology[topology_section][field] = archive[key]
            else:
                topology[name] = archive[key]
    labels = archive["labels"]
    return {
        "labels": labels,
        "region_count": int(labels.max(initial=-1)) + 1,
        "topology": topology,
        **section,
    }


def _load_bundle(path: Path) -> dict:
    archive = np.load(path)
    result: dict[str, dict[str, np.ndarray]] = {
        "incidence": {}, "continuation": {},
    }
    for key in archive.files:
        prefix, name = key.split("__", 1)
        result[prefix][name] = archive[key]
    return result


def _pair_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    euclidean = float(np.linalg.norm(first - second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine = (
        1.0 if denominator <= 1e-30
        else float(1.0 - np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    )
    return euclidean, cosine


def _auc(pairs: list[dict], name: str) -> dict | None:
    same = np.asarray([
        pair[name] for pair in pairs if pair["same_instance"]
    ], dtype=np.float64)
    different = np.asarray([
        pair[name] for pair in pairs if not pair["same_instance"]
    ], dtype=np.float64)
    if not len(same) or not len(different):
        return None
    comparison = same[:, None] - different[None, :]
    return {
        "same_instance_median": float(np.median(same)),
        "different_instance_median": float(np.median(different)),
        "closer_pair_auc": float(
            np.mean(comparison < 0.0) + 0.5 * np.mean(comparison == 0.0)),
    }


def _audit_embedding(
    embedding: np.ndarray,
    stages: np.lib.npyio.NpzFile,
    landmarks: dict,
) -> dict:
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
        first = points[first_name]
        second = points[second_name]
        euclidean, cosine = _pair_distance(
            embedding[first["region"]], embedding[second["region"]])
        pairs.append({
            "first": first_name,
            "second": second_name,
            "same_instance": first["instance"] == second["instance"],
            "euclidean_distance": euclidean,
            "cosine_distance": cosine,
        })
    return {
        "points": points,
        "pairs": pairs,
        "euclidean": _auc(pairs, "euclidean_distance"),
        "cosine": _auc(pairs, "cosine_distance"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "connection_bloom")
    output.mkdir(parents=True, exist_ok=True)
    landmark_specification = json.loads(args.landmarks.read_text())["images"]

    controls = {}
    for name in CONTROLS:
        image_dir = args.results / name
        controls[name] = {
            "complex": _load_complex(image_dir / "compound_region_complex.npz"),
            "bundle": _load_bundle(image_dir / "compound_incidence_bundle.npz"),
            "stages": np.load(image_dir / "v3_stages.npz"),
        }

    arms = {
        "full": {
            "include_fused": True,
            "shuffled_outside": False,
            "arc_crossing": True,
            "junctions": True,
        },
        "no_fused": {
            "include_fused": False,
            "shuffled_outside": False,
            "arc_crossing": True,
            "junctions": True,
        },
        "shuffled_outside": {
            "include_fused": True,
            "shuffled_outside": True,
            "arc_crossing": True,
            "junctions": True,
        },
        "no_crossing_or_junction": {
            "include_fused": True,
            "shuffled_outside": False,
            "arc_crossing": False,
            "junctions": False,
        },
    }
    report = {
        "purpose": (
            "seed-free normalized heat bloom on directed V3 incidences; "
            "landmarks are evaluation-only"
        ),
        "controls": list(CONTROLS),
        "arms": {},
    }

    for arm_name, configuration in arms.items():
        raw = {}
        schema = None
        for name, control in controls.items():
            values, names = relation_features(
                control["complex"],
                control["bundle"],
                include_fused=configuration["include_fused"],
                shuffled_outside=configuration["shuffled_outside"],
                shuffle_key=name,
            )
            if schema is None:
                schema = names
            elif names != schema:
                raise RuntimeError("control relation schemas disagree")
            raw[name] = values
        assert schema is not None
        whitener = fit_joint_whitener(raw.values(), schema)
        arm_report = {
            "raw_relation_channels": len(schema),
            "retained_covariance_modes": int(whitener.basis.shape[1]),
            "images": {},
        }
        for name, control in controls.items():
            whitened = whitener.transform(raw[name])
            if arm_name == "full":
                raw_embedding = aggregate_region_embedding(
                    whitened,
                    control["bundle"]["incidence"]["region"],
                    int(control["complex"]["region_count"]),
                )
            embedding, topology_summary = bloom_region_embedding(
                control["complex"],
                control["bundle"],
                whitened,
                include_arc_crossing=configuration["arc_crossing"],
                include_junctions=configuration["junctions"],
            )
            audit = _audit_embedding(
                embedding,
                control["stages"],
                landmark_specification[name],
            )
            image_output = output / name
            image_output.mkdir(parents=True, exist_ok=True)
            arrays = {"region_embedding": embedding}
            if arm_name == "full":
                arrays["raw_region_embedding"] = raw_embedding
                audit["raw_relations"] = _audit_embedding(
                    raw_embedding,
                    control["stages"],
                    landmark_specification[name],
                )
            np.savez_compressed(
                image_output / f"{arm_name}.npz", **arrays)
            arm_report["images"][name] = {
                "topology": topology_summary,
                "audit": audit,
            }
        report["arms"][arm_name] = arm_report

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
