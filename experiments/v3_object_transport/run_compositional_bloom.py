#!/usr/bin/env python3
"""Evaluate generic path composition before introducing amodal state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from experiments.v3_object_transport.compositional_bloom import (
    spectral_exponential_bloom,
    typed_order_two_bloom,
)
from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
    normalized_linear_kernel,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
)
from experiments.v3_object_transport.run_contour_transport import _audit


def _sparse_kernel(path: Path, prefix: str) -> np.ndarray:
    archive = np.load(path)
    shape = tuple(int(value) for value in archive[f"{prefix}_shape"])
    return sparse.csr_matrix((
        archive[f"{prefix}_data"],
        archive[f"{prefix}_indices"],
        archive[f"{prefix}_indptr"],
    ), shape=shape).toarray()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "compositional_bloom")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads((
        Path(__file__).resolve().parent / "assets" / "landmarks.json"
    ).read_text())["images"]
    report = {
        "purpose": (
            "falsify generic transitive completion using a spectral "
            "exponential and the complete typed order-two path kernel"
        ),
        "images": {},
    }
    for name in CONTROLS:
        embedding = np.load(
            args.results / "connection_bloom" / name / "full.npz"
        )["region_embedding"]
        role = normalized_linear_kernel(embedding)
        contour = np.load(
            args.results / "contour_transport" / name
            / "contour_transport.npz")["region_kernel"].astype(np.float64)
        enclosure = np.load(
            args.results / "relative_enclosure" / name
            / "relative_enclosure.npz")["region_kernel"].astype(np.float64)
        winding = _sparse_kernel(
            args.results / "contour_cycle_nesting" / name
            / "contour_cycle_nesting.npz", "centered_kernel")
        base = complete_kernel_algebra({
            "role": role, "contour": contour, "enclosure": enclosure,
        })["complete"]
        extended = complete_kernel_algebra({
            "role": role, "contour": contour, "enclosure": enclosure,
            "winding": winding,
        })["complete"]
        arms = {
            "base_spectral_exponential": spectral_exponential_bloom(base),
            "winding_spectral_exponential": spectral_exponential_bloom(
                extended),
            "base_typed_order_two": typed_order_two_bloom(
                (role, contour, enclosure)),
            "winding_typed_order_two": typed_order_two_bloom(
                (role, contour, enclosure, winding)),
        }
        stages = np.load(args.results / name / "v3_stages.npz")
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "compositional_bloom.npz",
            **{key: value.astype(np.float32) for key, value in arms.items()},
        )
        report["images"][name] = {
            key: _audit(value, stages, landmarks[name])
            for key, value in arms.items()
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
