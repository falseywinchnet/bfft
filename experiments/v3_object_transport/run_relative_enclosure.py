#!/usr/bin/env python3
"""Audit all bounded relative-complement manifolds on the V3 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.relative_enclosure import (
    build_relative_enclosures,
    summarize_relative_enclosures,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_LANDMARKS,
    DEFAULT_RESULTS,
    _load_complex,
)
from experiments.v3_object_transport.run_contour_transport import _audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "relative_enclosure")
    output.mkdir(parents=True, exist_ok=True)
    landmarks = json.loads(args.landmarks.read_text())["images"]
    report = {
        "purpose": (
            "all exact frame-bounded components of every vertex-deleted V3 "
            "region complex; landmarks are evaluation-only"
        ),
        "images": {},
    }
    for name in CONTROLS:
        image_dir = args.results / name
        complex_ = _load_complex(image_dir / "compound_region_complex.npz")
        stages = np.load(image_dir / "v3_stages.npz")
        enclosure = build_relative_enclosures(complex_)
        audit = _audit(enclosure["region_kernel"], stages, landmarks[name])
        participation = enclosure["participation"]
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "relative_enclosure.npz",
            manifold_owner=enclosure["manifold_owner"],
            manifold_offset=enclosure["manifold_offset"],
            manifold_member=enclosure["manifold_member"],
            manifold_member_area_fraction=enclosure[
                "manifold_member_area_fraction"],
            manifold_area=enclosure["manifold_area"],
            participation_data=participation.data,
            participation_indices=participation.indices,
            participation_indptr=participation.indptr,
            participation_shape=np.asarray(participation.shape, dtype=np.int32),
            region_kernel=np.asarray(enclosure["region_kernel"], dtype=np.float32),
        )
        report["images"][name] = {
            "summary": summarize_relative_enclosures(enclosure),
            "audit": audit,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
