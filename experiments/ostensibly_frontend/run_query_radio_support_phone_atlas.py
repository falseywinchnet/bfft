#!/usr/bin/env python3
"""Classify cached radio regions by density-free local support geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_support_phone_atlas import load_support_phone_atlas
from .occupation_point_cloud import marginal_copula_cloud


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("region_lattice", type=Path)
    parser.add_argument("support_atlas", type=Path)
    parser.add_argument("--top-k", type=int, default=39)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lattice = json.loads(args.region_lattice.read_text())
    with np.load(args.region_clouds) as document:
        clouds = np.asarray(document["clouds"], dtype=np.float64)
    if clouds.shape[0] != len(lattice["phones"]):
        raise ValueError("region cloud and lattice counts disagree")
    atlas = load_support_phone_atlas(args.support_atlas)
    rows = []
    started = perf_counter()
    for index, (cloud, region) in enumerate(zip(clouds, lattice["phones"], strict=True)):
        ranking = atlas.rank(marginal_copula_cloud(cloud))
        rows.append(
            {
                **{key: value for key, value in region.items() if key != "top5"},
                "top5": ranking[: args.top_k],
            }
        )
        if (index + 1) % 50 == 0:
            print(f"classified {index + 1}/{clouds.shape[0]} support regions", flush=True)
    result = {
        "method": "density_free_local_support_phone_atlas",
        "selection_status": "generic support channel; no radio labels",
        "phone_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "phones": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "phone_count": len(rows), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
