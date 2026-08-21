#!/usr/bin/env python3
"""Evaluate the localized signed connection heat kernel and null controls."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.connection_bloom import (
    connection_green_gram,
    connection_heat_gram,
    connection_heat_response,
    fit_joint_whitener,
    region_source_matrix,
    relation_features,
    signed_incidence_connection,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_bundle,
    _load_complex,
)


ANCHORS = {
    "pikachu_hard": ("body", "left_ear_tip", "black_surround"),
    "coffee": ("cup_wall", "plate", "spoon_handle"),
    "astronaut": ("flag_blue", "suit"),
    "checker": ("black_a",),
    "coins": ("coin_00", "ground"),
}


def _landmark_regions(stages, landmarks: dict) -> tuple[list[str], np.ndarray]:
    labels = stages["compound_labels"]
    height, width = labels.shape
    names = list(landmarks)
    regions = []
    for name in names:
        xn, yn = landmarks[name]["xy"]
        x = int(np.clip(round(float(xn) * (width - 1)), 0, width - 1))
        y = int(np.clip(round(float(yn) * (height - 1)), 0, height - 1))
        regions.append(int(labels[y, x]))
    return names, np.asarray(regions, dtype=np.int32)


def _audit_similarity(
    names: list[str],
    similarity: np.ndarray,
    landmarks: dict,
    distance: np.ndarray | None = None,
) -> dict:
    pairs = []
    for first, second in combinations(range(len(names)), 2):
        measured_distance = (
            float(1.0 - similarity[first, second])
            if distance is None
            else float(distance[first, second])
        )
        pairs.append({
            "first": names[first],
            "second": names[second],
            "same_instance": (
                landmarks[names[first]]["instance"]
                == landmarks[names[second]]["instance"]
            ),
            "participation_similarity": float(similarity[first, second]),
            "participation_distance": measured_distance,
        })
    same = np.asarray([
        pair["participation_distance"]
        for pair in pairs if pair["same_instance"]
    ], dtype=np.float64)
    different = np.asarray([
        pair["participation_distance"]
        for pair in pairs if not pair["same_instance"]
    ], dtype=np.float64)
    if len(same) and len(different):
        comparison = same[:, None] - different[None, :]
        auc = float(
            np.mean(comparison < 0.0) + 0.5 * np.mean(comparison == 0.0))
    else:
        auc = None
    return {
        "pairs": pairs,
        "same_instance_median_distance": (
            float(np.median(same)) if len(same) else None),
        "different_instance_median_distance": (
            float(np.median(different)) if len(different) else None),
        "closer_pair_auc": auc,
    }


def _region_amplitude(
    response: np.ndarray,
    incidence_region: np.ndarray,
    region_count: int,
) -> np.ndarray:
    labels = np.asarray(incidence_region, dtype=np.int32)
    count = np.bincount(labels, minlength=region_count).astype(np.float64)
    square = np.bincount(
        labels, weights=response * response, minlength=region_count)
    return np.sqrt(square / np.maximum(count, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "signed_connection")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]

    controls = {}
    full_raw = {}
    no_fused_raw = {}
    full_schema = None
    no_fused_schema = None
    for name in CONTROLS:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        bundle = _load_bundle(image_dir / "compound_incidence_bundle.npz")
        stages = np.load(image_dir / "v3_stages.npz")
        controls[name] = {
            "complex": complex_, "bundle": bundle, "stages": stages,
        }
        full_raw[name], schema = relation_features(complex_, bundle)
        no_fused_raw[name], schema_without = relation_features(
            complex_, bundle, include_fused=False)
        full_schema = schema if full_schema is None else full_schema
        no_fused_schema = (
            schema_without if no_fused_schema is None else no_fused_schema)
        if schema != full_schema or schema_without != no_fused_schema:
            raise RuntimeError("control relation schemas disagree")
    assert full_schema is not None and no_fused_schema is not None
    full_whitener = fit_joint_whitener(full_raw.values(), full_schema)
    no_fused_whitener = fit_joint_whitener(
        no_fused_raw.values(), no_fused_schema)

    arms = {
        "signed_full": {
            "raw": full_raw, "whitener": full_whitener, "mode": "signed",
            "junctions": True,
        },
        "signed_no_fused": {
            "raw": no_fused_raw, "whitener": no_fused_whitener,
            "mode": "signed", "junctions": True,
        },
        "signed_shuffled_outside": {
            "raw": None, "whitener": full_whitener, "mode": "signed",
            "junctions": True,
        },
        "unsigned_full": {
            "raw": full_raw, "whitener": full_whitener, "mode": "unsigned",
            "junctions": True,
        },
        "topology_only": {
            "raw": full_raw, "whitener": full_whitener, "mode": "topology",
            "junctions": True,
        },
        "signed_no_junction": {
            "raw": full_raw, "whitener": full_whitener, "mode": "signed",
            "junctions": False,
        },
    }
    report = {
        "purpose": (
            "localized unit-time heat kernel of the signed empirical "
            "incidence connection; all landmarks are evaluation-only"
        ),
        "fibre": {
            "full_channels": len(full_schema),
            "full_covariance_modes": int(full_whitener.basis.shape[1]),
            "no_fused_channels": len(no_fused_schema),
            "no_fused_covariance_modes": int(
                no_fused_whitener.basis.shape[1]),
        },
        "arms": {},
    }

    for arm_name, configuration in arms.items():
        arm_report = {"images": {}}
        for name, control in controls.items():
            if arm_name == "signed_shuffled_outside":
                raw, schema = relation_features(
                    control["complex"], control["bundle"],
                    shuffled_outside=True, shuffle_key=name)
                if schema != full_schema:
                    raise RuntimeError("shuffled relation schema changed")
            else:
                raw = configuration["raw"][name]
            whitened = configuration["whitener"].transform(raw)
            connection, _degree, topology_summary = signed_incidence_connection(
                control["complex"],
                control["bundle"],
                whitened,
                mode=configuration["mode"],
                include_junctions=configuration["junctions"],
            )
            landmark_names, regions = _landmark_regions(
                control["stages"], landmarks[name])
            sources = region_source_matrix(
                control["bundle"]["incidence"]["region"], regions)
            gram, similarity = connection_heat_gram(connection, sources)
            heat_audit = _audit_similarity(
                landmark_names, similarity, landmarks[name])
            (
                green_gram,
                green_similarity,
                green_resistance,
                green_solver,
            ) = connection_green_gram(connection, sources)
            green_audit = _audit_similarity(
                landmark_names,
                green_similarity,
                landmarks[name],
                distance=green_resistance,
            )

            image_output = output / name
            image_output.mkdir(parents=True, exist_ok=True)
            arrays = {
                "landmark_names": np.asarray(landmark_names, dtype="U64"),
                "landmark_regions": regions,
                "landmark_heat_gram": gram,
                "landmark_similarity": similarity,
                "landmark_green_gram": green_gram,
                "landmark_green_similarity": green_similarity,
                "landmark_green_resistance": green_resistance,
            }
            if arm_name == "signed_full":
                anchor_names = ANCHORS[name]
                anchor_regions = np.asarray([
                    regions[landmark_names.index(anchor)]
                    for anchor in anchor_names
                ], dtype=np.int32)
                amplitudes = []
                for region in anchor_regions:
                    source = region_source_matrix(
                        control["bundle"]["incidence"]["region"],
                        np.asarray([region], dtype=np.int32),
                    )
                    response = connection_heat_response(connection, source)
                    amplitudes.append(_region_amplitude(
                        response,
                        control["bundle"]["incidence"]["region"],
                        int(control["complex"]["region_count"]),
                    ))
                arrays.update({
                    "anchor_names": np.asarray(anchor_names, dtype="U64"),
                    "anchor_regions": anchor_regions,
                    "anchor_region_amplitude": np.asarray(amplitudes),
                })
            np.savez_compressed(
                image_output / f"{arm_name}.npz", **arrays)
            arm_report["images"][name] = {
                "topology": topology_summary,
                "heat_audit": heat_audit,
                "green_audit": green_audit,
                "green_solver": green_solver,
            }
        report["arms"][arm_name] = arm_report

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
