#!/usr/bin/env python3
"""Audit a compiled support atlas on labeled cached ARCTIC phones."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_support_phone_atlas import load_support_phone_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig, marginal_copula_cloud
from .run_occupation_context_battery import _cache_name, _window_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("support_atlas", type=Path)
    parser.add_argument("--query-speaker", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
    payload = {
        "rows": args.rows,
        "noise_db": args.noise_db,
        "morphology": asdict(DepthmapGeometryConfig()),
        "ridge": asdict(HessianRidgeConfig()),
        "climbers": asdict(CrazyClimberConfig()),
        "point_cloud": asdict(PointCloudConfig(point_count=32768)),
    }
    atlas = load_support_phone_atlas(args.support_atlas)
    rows = []
    started = perf_counter()
    for utterance in utterances:
        for window in _window_list(args.arctic_root, args.query_speaker, utterance):
            path = args.raw_cache / _cache_name(
                window,
                {**payload, "key": window.key, "representation": "raw_occupation_point_cloud_v1"},
            )
            cloud = marginal_copula_cloud(np.asarray(np.load(path), dtype=np.float64))
            ranking = atlas.rank(cloud)
            rank = next(index + 1 for index, item in enumerate(ranking) if item["phone"] == window.label)
            rows.append({"label": window.label, "rank": rank})
    ranks = [row["rank"] for row in rows]
    result = {
        "query_speaker": args.query_speaker,
        "support_atlas": str(args.support_atlas),
        "elapsed_seconds": perf_counter() - started,
        "summary": {
            "count": len(ranks),
            "top1": sum(rank == 1 for rank in ranks),
            "top5": sum(rank <= 5 for rank in ranks),
            "top10": sum(rank <= 10 for rank in ranks),
            "median_rank": float(np.median(ranks)),
        },
        "phones": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), **result["summary"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
