#!/usr/bin/env python3
"""Run continuous step-3 morphology and lifted geometry on an initial window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .depthmap_geometry import DepthmapGeometryConfig, depthmap_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step3", type=Path)
    parser.add_argument("--frames", type=int, default=94)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("frames must be positive")

    with np.load(args.step3) as saved:
        trace = np.asarray(saved["trace_field"][:, : args.frames], dtype=np.float64)
    config = DepthmapGeometryConfig()
    residual, scale_stack, geometry = depthmap_geometry(trace, config)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out / "depthmap_geometry.npz"
    np.savez_compressed(
        arrays_path,
        trace=trace,
        residual=residual,
        scale_stack=scale_stack,
        extrema=geometry.extrema,
        hit_counts=geometry.hit_counts,
        segments=geometry.segments,
        points=geometry.points,
    )
    metadata = {
        "source": str(args.step3),
        "trace_shape": list(trace.shape),
        "config": asdict(config),
        "residual_nonzero_fraction": float(np.mean(residual > 0.0)),
        "extrema_count": int(geometry.extrema.shape[0]),
        "segment_count": int(geometry.segments.shape[0]),
        "point_count": int(geometry.points.shape[0]),
    }
    metadata_path = args.out / "depthmap_geometry.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"arrays": str(arrays_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
