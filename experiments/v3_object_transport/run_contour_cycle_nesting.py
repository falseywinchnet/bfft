#!/usr/bin/env python3
"""Audit exact contour-cycle winding and centered participation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.contour_cycle_nesting import (
    build_contour_cycle_nesting,
    summarize_contour_cycle_nesting,
)
from experiments.v3_object_transport.contour_transport import (
    build_contour_transport,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_bundle,
    _load_complex,
)
from experiments.v3_object_transport.run_contour_transport import _audit


def _save_participation(prefix: str, value, output: dict) -> None:
    output[f"{prefix}_data"] = value.data
    output[f"{prefix}_indices"] = value.indices
    output[f"{prefix}_indptr"] = value.indptr
    output[f"{prefix}_shape"] = np.asarray(value.shape, dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--level", choices=("compound", "leaf"), default="compound")
    parser.add_argument(
        "--controls", default=",".join(CONTROLS),
        help="comma-separated frozen controls")
    args = parser.parse_args()
    controls = tuple(
        name.strip() for name in args.controls.split(",") if name.strip())
    unknown = sorted(set(controls) - set(CONTROLS))
    if unknown:
        parser.error(f"unknown controls: {', '.join(unknown)}")
    default_name = (
        "contour_cycle_nesting" if args.level == "compound"
        else "contour_cycle_nesting_leaf"
    )
    output = args.out or (args.results / default_name)
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "exact mod-2 winding supports and covariance-normalized centered "
            "participation from every closed contour component"
        ),
        "level": args.level,
        "images": {},
    }
    for name in controls:
        image_dir = args.results / name
        prefix = "compound" if args.level == "compound" else "leaf"
        complex_ = _load_complex(image_dir / f"{prefix}_region_complex.npz")
        bundle = _load_bundle(image_dir / f"{prefix}_incidence_bundle.npz")
        contour = build_contour_transport(complex_, bundle)
        nesting = build_contour_cycle_nesting(complex_, bundle, contour)
        stages = {"compound_labels": complex_["labels"]}
        archive = {
            key: value for key, value in nesting.items()
            if not key.endswith("_participation")
            and not key.endswith("_kernel")
        }
        _save_participation(
            "overlap", nesting["overlap_participation"], archive)
        _save_participation(
            "centered", nesting["centered_participation"], archive)
        _save_participation("overlap_kernel", nesting["overlap_kernel"], archive)
        _save_participation(
            "centered_kernel", nesting["centered_kernel"], archive)
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "contour_cycle_nesting.npz", **archive)
        report["images"][name] = {
            "summary": summarize_contour_cycle_nesting(nesting),
            "overlap_audit": _audit(
                nesting["overlap_kernel"], stages, landmarks[name]),
            "centered_audit": _audit(
                nesting["centered_kernel"], stages, landmarks[name]),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
