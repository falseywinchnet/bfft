#!/usr/bin/env python3
"""Classify cached radio regions by cross-speaker distance-profile centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_phone_atlas import load_phone_atlas
from .distance_profile_geometry import load_distance_profile_atlas, ranking_profile
from .occupation_point_cloud import marginal_copula_cloud
from .run_occupation_context_battery import _subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("region_lattice", type=Path)
    parser.add_argument("profile_atlas", type=Path)
    parser.add_argument("--bdl-atlas", type=Path, required=True)
    parser.add_argument("--slt-atlas", type=Path, required=True)
    parser.add_argument("--projection-points", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=39)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    regions = json.loads(args.region_lattice.read_text())
    with np.load(args.region_clouds) as document:
        clouds = np.asarray(document["clouds"], dtype=np.float64)
    if clouds.shape[0] != len(regions["phones"]):
        raise ValueError("region cloud and lattice counts disagree")
    profile_atlas = load_distance_profile_atlas(args.profile_atlas)
    phone_atlases = {"bdl": load_phone_atlas(args.bdl_atlas), "slt": load_phone_atlas(args.slt_atlas)}
    rows = []
    started = perf_counter()
    for index, (cloud, region) in enumerate(zip(clouds, regions["phones"], strict=True)):
        query = _subset(marginal_copula_cloud(cloud), args.projection_points)
        profiles = {
            channel: ranking_profile(atlas.rank(query), profile_atlas.labels)
            for channel, atlas in phone_atlases.items()
        }
        ranking = profile_atlas.rank(profiles)
        rows.append(
            {
                **{key: value for key, value in region.items() if key != "top5"},
                "top5": ranking[: args.top_k],
            }
        )
        if (index + 1) % 50 == 0:
            print(f"classified {index + 1}/{clouds.shape[0]} profiles", flush=True)
    result = {
        "method": "cross_speaker_complete_distance_profile_centroids",
        "selection_status": "generic BDL/SLT calibration; no radio labels",
        "phone_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "phones": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "phone_count": len(rows), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
