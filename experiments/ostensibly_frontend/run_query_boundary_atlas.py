#!/usr/bin/env python3
"""Query a compiled diphone atlas with one cached phone span."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_boundary_atlas import load_boundary_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_context_battery import _cache_name, _window_list
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("compiled_atlas", type=Path)
    parser.add_argument("--query-speaker", default="slt")
    parser.add_argument("--query-utterance", default="arctic_a0019")
    parser.add_argument("--query-start-index", type=int, required=True)
    parser.add_argument("--phone-count", type=int, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--points-per-phone", type=int)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.query_start_index < 0 or args.phone_count < 2:
        raise ValueError("atlas query requires a valid multi-phone span")

    atlas = load_boundary_atlas(args.compiled_atlas)
    provenance = dict(atlas.provenance or {})
    compiled_points = provenance.get("points_per_phone")
    points_per_phone = (
        args.points_per_phone
        if args.points_per_phone is not None
        else int(compiled_points or 2048)
    )
    if compiled_points is not None and points_per_phone != int(compiled_points):
        raise ValueError("query points-per-phone must match the compiled atlas")

    point_config = PointCloudConfig(point_count=32768)
    cache_payload = {
        "rows": args.rows,
        "noise_db": args.noise_db,
        "morphology": asdict(DepthmapGeometryConfig()),
        "ridge": asdict(HessianRidgeConfig()),
        "climbers": asdict(CrazyClimberConfig()),
        "point_cloud": asdict(point_config),
    }

    def load_raw(window) -> np.ndarray:
        path = args.raw_cache / _cache_name(
            window,
            {
                **cache_payload,
                "key": window.key,
                "representation": "raw_occupation_point_cloud_v1",
            },
        )
        if not path.exists():
            raise FileNotFoundError(
                f"raw cloud missing for {window.key}; populate --raw-cache first"
            )
        return np.asarray(np.load(path), dtype=np.float64)

    windows = _window_list(
        args.arctic_root, args.query_speaker, args.query_utterance
    )
    stop = args.query_start_index + args.phone_count
    if stop > len(windows):
        raise ValueError("atlas query span extends past the utterance")
    span = tuple(windows[args.query_start_index:stop])
    raw_clouds = tuple(load_raw(window) for window in span)

    started = perf_counter()
    rankings = []
    target_ranks = []
    for boundary_index, pair in enumerate(zip(raw_clouds, raw_clouds[1:])):
        query_pair = lane_chunks(joint_gauge_chunks(pair, points_per_phone))
        ranking = atlas.rank(query_pair, chunk_size=args.chunk_size)
        target_pair = [span[boundary_index].label, span[boundary_index + 1].label]
        target_rank = next(
            (
                item["rank"]
                for item in ranking
                if item["phones"] == target_pair
            ),
            None,
        )
        target_ranks.append(target_rank)
        rankings.append(
            {
                "boundary_index": boundary_index,
                "query_phones": target_pair,
                "target_pair_rank": target_rank,
                "pair_type_count": len(ranking),
                "ranking": ranking,
            }
        )
    query_seconds = perf_counter() - started
    result = {
        "method": "compiled_fixed_quantile_boundary_swd_atlas_query",
        "query": {
            "speaker": args.query_speaker,
            "utterance": args.query_utterance,
            "start_index": args.query_start_index,
            "phones": [window.label for window in span],
            "seconds": [span[0].seconds0, span[-1].seconds1],
        },
        "reference": {
            "speaker": provenance.get("speaker"),
            "utterances": provenance.get("utterances", []),
            "compiled_atlas": str(args.compiled_atlas),
        },
        "pipeline": {
            "quantile_count": atlas.quantile_count,
            "projection_count": atlas.projection_count,
            "direction_count": atlas.projections.shape[2],
            "occurrence_count": atlas.projections.shape[0],
            "points_per_phone": points_per_phone,
            "chunk_size": args.chunk_size,
        },
        "summary": {
            "target_boundary_pair_ranks": target_ranks,
            "query_seconds": query_seconds,
        },
        "joint_boundary_pair_swd_rankings": rankings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "query_phones": result["query"]["phones"],
                "target_boundary_pair_ranks": target_ranks,
                "query_seconds": query_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
