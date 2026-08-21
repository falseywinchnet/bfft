#!/usr/bin/env python3
"""Audit the complete role/contour/enclosure participation algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
    normalized_linear_kernel,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
)
from experiments.v3_object_transport.run_contour_transport import _audit


def _load_sparse_kernel(path: Path, prefix: str) -> np.ndarray:
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
    parser.add_argument(
        "--include-winding", action="store_true",
        help="evaluate the centered contour-winding coordinate candidate")
    args = parser.parse_args()
    default_name = (
        "participation_algebra_winding"
        if args.include_winding else "participation_algebra"
    )
    output = args.out or (args.results / default_name)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": (
            "complete non-empty tensor algebra of role, one-sided contour, "
            "and bounded-enclosure participation"
            + (
                ", with centered contour winding as a candidate fourth "
                "coordinate" if args.include_winding else ""
            )
            + "; landmarks are evaluation-only"
        ),
        "images": {},
    }
    for name in CONTROLS:
        embedding = np.load(
            args.results / "connection_bloom" / name / "full.npz"
        )["region_embedding"]
        role = normalized_linear_kernel(embedding)
        contour = np.load(
            args.results / "contour_transport" / name / "contour_transport.npz"
        )["region_kernel"]
        enclosure = np.load(
            args.results / "relative_enclosure" / name
            / "relative_enclosure.npz"
        )["region_kernel"]
        kernels = {
            "role": role,
            "contour": contour,
            "enclosure": enclosure,
        }
        if args.include_winding:
            kernels["winding"] = _load_sparse_kernel(
                args.results / "contour_cycle_nesting" / name
                / "contour_cycle_nesting.npz",
                "centered_kernel",
            )
        algebra = complete_kernel_algebra(kernels)
        stages = np.load(args.results / name / "v3_stages.npz")
        frozen = json.loads((
            Path(__file__).resolve().parent / "assets" / "landmarks.json"
        ).read_text())["images"][name]
        audit = _audit(algebra["complete"], stages, frozen)
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "participation_algebra.npz",
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in algebra.items()
            },
        )
        report["images"][name] = {"audit": audit}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
