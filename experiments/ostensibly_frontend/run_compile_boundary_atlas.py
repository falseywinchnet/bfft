#!/usr/bin/env python3
"""Compile a reusable fixed-quantile diphone SWD atlas from cached raw clouds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_boundary_atlas import compile_boundary_atlas, save_boundary_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import CloudFitConfig, PointCloudConfig
from .run_arctic_calibration import DEFAULT_UTTERANCES
from .run_occupation_context_battery import _cache_name, _window_list
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", default="bdl")
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--points-per-phone", type=int, default=2048)
    parser.add_argument("--quantiles", type=int, default=256)
    parser.add_argument("--projection-count", type=int, default=64)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
    if not utterances:
        raise ValueError("compiled atlas requires reference utterances")
    if args.points_per_phone < 1 or args.quantiles < 1:
        raise ValueError("compiled atlas resolutions must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    occurrences = []
    started = perf_counter()
    for utterance in utterances:
        windows = _window_list(args.arctic_root, args.speaker, utterance)
        for start, (left, right) in enumerate(zip(windows, windows[1:])):
            cloud = lane_chunks(
                joint_gauge_chunks(
                    (load_raw(left), load_raw(right)), args.points_per_phone
                )
            )
            occurrences.append(
                (
                    (left.label, right.label),
                    f"{args.speaker}:{utterance}:{start}",
                    cloud,
                )
            )
        print(
            f"loaded {utterance}: {len(occurrences)} diphone occurrences",
            flush=True,
        )
    load_seconds = perf_counter() - started
    compile_started = perf_counter()
    config = CloudFitConfig(
        fit_point_count=2 * args.points_per_phone,
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=1e-6,
        distance_mode="sliced_wasserstein",
        sliced_projection_count=args.projection_count,
        global_iterations=1,
        population_size=4,
    )
    atlas = compile_boundary_atlas(occurrences, config, args.quantiles)
    atlas = replace(
        atlas,
        provenance={
            "speaker": args.speaker,
            "utterances": list(utterances),
            "points_per_phone": args.points_per_phone,
            "rows": args.rows,
            "noise_db": args.noise_db,
            "raw_cache": str(args.raw_cache),
        },
    )
    compile_seconds = perf_counter() - compile_started
    save_boundary_atlas(args.out, atlas)
    size_bytes = args.out.stat().st_size
    pair_type_count = len({tuple(row) for row in atlas.phones.tolist()})
    print(
        json.dumps(
            {
                "output": str(args.out),
                "speaker": args.speaker,
                "utterance_count": len(utterances),
                "occurrence_count": atlas.projections.shape[0],
                "pair_type_count": pair_type_count,
                "quantile_count": atlas.quantile_count,
                "direction_count": atlas.projections.shape[2],
                "size_bytes": size_bytes,
                "load_seconds": load_seconds,
                "compile_seconds": compile_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
