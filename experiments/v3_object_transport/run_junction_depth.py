#!/usr/bin/env python3
"""Measure oriented junction caps on all frozen V3 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3_object_transport.junction_depth import (
    build_junction_depth,
    summarize_junction_depth,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
    _load_complex,
)


def _directed_mass(depth: dict, region_count: int) -> np.ndarray:
    mass = np.zeros((region_count, region_count), dtype=np.float64)
    offset = depth["other_region_offset"]
    for identifier, cap in enumerate(depth["cap_region"]):
        start, stop = int(offset[identifier]), int(offset[identifier + 1])
        other = depth["other_region"][start:stop]
        # This product is a display sufficient statistic only.  Both raw
        # factors remain in the saved evidence and no depth decision uses it.
        value = (
            float(depth["cap_adjacent_pairs"][identifier])
            * float(depth["tangent_anisotropy"][identifier])
        )
        mass[int(cap), other] += value
    return mass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "junction_depth")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": (
            "raw oriented cap/stem evidence from exact V3 junction sectors; "
            "no object or depth partition is inferred"
        ),
        "images": {},
    }
    for name in CONTROLS:
        complex_ = _load_complex(
            args.results / name / "compound_region_complex.npz")
        depth = build_junction_depth(complex_)
        directed = _directed_mass(depth, int(complex_["region_count"]))
        image_output = output / name
        image_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            image_output / "junction_depth.npz",
            **depth,
            directed_display_mass=np.asarray(directed, dtype=np.float32),
            antisymmetric_display_mass=np.asarray(
                directed - directed.T, dtype=np.float32),
        )
        report["images"][name] = {
            "summary": summarize_junction_depth(depth),
            "directed_region_pairs": int(np.count_nonzero(directed)),
            "reciprocal_region_pairs": int(np.count_nonzero(
                (directed > 0.0) & (directed.T > 0.0)) // 2),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
