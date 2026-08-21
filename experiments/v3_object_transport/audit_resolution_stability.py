#!/usr/bin/env python3
"""Evaluation-only aperture scale-space across 128/256/384 V3 audits."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_128 = ROOT / "experiments/v3_object_transport/results/v3_object_transport_128"
DEFAULT_384 = ROOT / "experiments/v3_object_transport/results/v3_object_transport_384"
APERTURES = (0.0, 0.005, 0.01, 0.02, 0.04)


def _probe_distribution(
    labels: np.ndarray,
    xy: list[float],
    aperture: float,
    region_count: int,
) -> np.ndarray:
    height, width = labels.shape
    x = float(xy[0]) * (width - 1)
    y = float(xy[1]) * (height - 1)
    if aperture == 0.0:
        result = np.zeros(region_count, dtype=np.float64)
        result[int(labels[round(y), round(x)])] = 1.0
        return result
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    sigma = float(aperture) * max(height, width)
    weight = np.exp(-0.5 * (
        (xx - x) ** 2 + (yy - y) ** 2) / (sigma * sigma))
    result = np.bincount(
        labels.ravel(), weights=weight.ravel(), minlength=region_count)
    return result / max(float(np.sum(result)), 1e-30)


def _audit(
    kernel: np.ndarray,
    labels: np.ndarray,
    landmarks: dict,
    aperture: float,
) -> dict:
    names = list(landmarks)
    probe = np.asarray([
        _probe_distribution(
            labels, landmarks[name]["xy"], aperture, len(kernel))
        for name in names
    ])
    gram = probe @ np.asarray(kernel, dtype=np.float64) @ probe.T
    norm = np.sqrt(np.maximum(np.diag(gram), 0.0))
    similarity = np.divide(
        gram, norm[:, None] * norm[None, :], out=np.zeros_like(gram),
        where=norm[:, None] * norm[None, :] > 1e-30)
    pairs = []
    for first, second in combinations(range(len(names)), 2):
        pairs.append({
            "first": names[first],
            "second": names[second],
            "same_instance": (
                landmarks[names[first]]["instance"]
                == landmarks[names[second]]["instance"]),
            "similarity": float(similarity[first, second]),
            "distance": float(1.0 - similarity[first, second]),
        })
    same = np.asarray([
        pair["distance"] for pair in pairs if pair["same_instance"]])
    different = np.asarray([
        pair["distance"] for pair in pairs if not pair["same_instance"]])
    auc = None
    if len(same) and len(different):
        comparison = same[:, None] - different[None, :]
        auc = float(
            np.mean(comparison < 0.0)
            + 0.5 * np.mean(comparison == 0.0))
    return {"closer_pair_auc": auc, "pairs": pairs}


def _kernels(path: Path, name: str) -> dict[str, np.ndarray]:
    support = np.load(path / "support_manifold_transport" / name / "full.npz")
    return {
        "canonical": np.load(
            path / "participation_algebra" / name
            / "participation_algebra.npz")["complete"],
        "assembly": support["region_kernel"],
        "hierarchical": support["complete_kernel"],
        "flat": support["flat_complete_kernel"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-128", type=Path, default=DEFAULT_128)
    parser.add_argument("--results-256", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--results-384", type=Path, default=DEFAULT_384)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = {128: args.results_128, 256: args.results_256, 384: args.results_384}
    output = args.out or (args.results_256 / "resolution_stability")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "evaluation-only exact-point and Gaussian-aperture scale-space; "
            "no aperture enters inference or selects a reported arm"
        ),
        "apertures": list(APERTURES),
        "resolutions": {},
    }
    exact_pair_similarity = {}
    for side, path in paths.items():
        side_report = {}
        for name in CONTROLS:
            labels = np.load(path / name / "v3_stages.npz")["compound_labels"]
            image_report = {}
            for kernel_name, kernel in _kernels(path, name).items():
                image_report[kernel_name] = {
                    str(aperture): _audit(
                        kernel, labels, landmarks[name], aperture)
                    for aperture in APERTURES
                }
                exact_pair_similarity[(side, name, kernel_name)] = {
                    (pair["first"], pair["second"]): pair["similarity"]
                    for pair in image_report[kernel_name]["0.0"]["pairs"]
                }
            side_report[name] = image_report
        report["resolutions"][str(side)] = side_report

    log_scale = np.log(np.asarray((128.0, 256.0, 384.0)))
    right_fraction = float(
        (log_scale[1] - log_scale[0]) / (log_scale[2] - log_scale[0]))
    ridge = {}
    for name in CONTROLS:
        ridge[name] = {}
        for kernel_name in ("canonical", "assembly", "hierarchical", "flat"):
            middle = exact_pair_similarity[(256, name, kernel_name)]
            low = exact_pair_similarity[(128, name, kernel_name)]
            high = exact_pair_similarity[(384, name, kernel_name)]
            ridge[name][kernel_name] = [
                {
                    "first": pair[0], "second": pair[1],
                    "similarity_128": float(low[pair]),
                    "similarity_256": float(middle[pair]),
                    "similarity_384": float(high[pair]),
                    "middle_log_scale_excess": float(
                        middle[pair]
                        - ((1.0 - right_fraction) * low[pair]
                           + right_fraction * high[pair]))
                }
                for pair in middle
            ]
    report["exact_point_scale_curvature"] = ridge
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
