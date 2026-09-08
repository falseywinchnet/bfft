#!/usr/bin/env python3
"""Reclassify saved radio regions with the occupation-phone atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_phone_atlas import load_phone_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import (
    affine_marginal_cloud,
    PointCloudConfig,
    marginal_copula_cloud,
)
from .run_occupation_cloud_battery import _trace_to_cloud
from .run_occupation_context_battery import _subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_npz", type=Path)
    parser.add_argument("region_lattice", type=Path)
    parser.add_argument("phone_atlas", type=Path)
    parser.add_argument("--field", default="trace_field")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--point-count", type=int, default=8192)
    parser.add_argument("--projection-points", type=int, default=4096)
    parser.add_argument("--gauge", choices=("copula", "affine"), default="copula")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--clouds-in",
        type=Path,
        help="reuse previously extracted raw region clouds and skip morphology",
    )
    parser.add_argument("--clouds-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.rows < 1
        or args.point_count < 1
        or args.projection_points < 1
        or args.top_k < 1
    ):
        raise ValueError("radio phone atlas resolutions must be positive")
    regions_document = json.loads(args.region_lattice.read_text())
    cached_clouds = None
    field = None
    if args.clouds_in is not None:
        with np.load(args.clouds_in) as document:
            cached_clouds = np.asarray(document["clouds"], dtype=np.float64)
        if cached_clouds.shape[0] != len(regions_document["phones"]):
            raise ValueError("region cloud and lattice counts disagree")
    else:
        with np.load(args.recording_npz) as document:
            field = np.asarray(document[args.field], dtype=np.float64)
    atlas = load_phone_atlas(args.phone_atlas)
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climbers = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.point_count)
    rows = []
    raw_clouds = []
    started = perf_counter()
    for index, region in enumerate(regions_document["phones"]):
        frame0 = int(region["frame0"])
        frame1 = int(region["frame1"])
        if cached_clouds is not None:
            raw_cloud = cached_clouds[index]
        else:
            trace = field[: args.rows, frame0:frame1]
            if trace.shape[1] < 1:
                raise ValueError(f"empty radio region {index}")
            raw_cloud = _trace_to_cloud(
                trace, morphology, ridge, climbers, cloud_config
            )
        raw_clouds.append(np.asarray(raw_cloud, dtype=np.float32))
        cloud = _subset(
            (
                marginal_copula_cloud(raw_cloud)
                if args.gauge == "copula"
                else affine_marginal_cloud(raw_cloud)
            ),
            args.projection_points,
        )
        ranking = atlas.rank(cloud, chunk_size=args.chunk_size)
        rows.append(
            {
                **{key: value for key, value in region.items() if key != "top5"},
                "top5": ranking[: args.top_k],
            }
        )
        if (index + 1) % 25 == 0:
            print(f"classified {index + 1}/{len(regions_document['phones'])}", flush=True)
    elapsed = perf_counter() - started
    result = {
        "method": f"{args.gauge}_marginal_occupation_phone_atlas_swd",
        "selection_status": "same radio regions; new occupation geometry",
        "source_regions": str(args.region_lattice),
        "recording": str(args.recording_npz),
        "phone_atlas": str(args.phone_atlas),
        "region_clouds": str(args.clouds_in) if args.clouds_in is not None else None,
        "phone_count": len(rows),
        "elapsed_seconds": elapsed,
        "phones": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.clouds_out is not None:
        args.clouds_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.clouds_out,
            clouds=np.stack(raw_clouds),
            frame0=np.asarray([int(row["frame0"]) for row in rows]),
            frame1=np.asarray([int(row["frame1"]) for row in rows]),
        )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "phone_count": len(rows),
                "elapsed_seconds": elapsed,
                "clouds_output": (
                    str(args.clouds_out) if args.clouds_out is not None else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
