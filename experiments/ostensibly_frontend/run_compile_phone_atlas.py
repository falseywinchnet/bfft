#!/usr/bin/env python3
"""Compile marginal-copula occupation projections for reference phones."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_phone_atlas import compile_phone_atlas, save_phone_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import (
    affine_marginal_cloud,
    CloudFitConfig,
    PointCloudConfig,
    marginal_copula_cloud,
)
from .run_arctic_calibration import DEFAULT_UTTERANCES
from .run_occupation_context_battery import _cache_name, _window_list
from .run_occupation_context_battery import _subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", default="bdl")
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--quantiles", type=int, default=256)
    parser.add_argument("--projection-count", type=int, default=64)
    parser.add_argument("--projection-points", type=int, default=4096)
    parser.add_argument("--gauge", choices=("copula", "affine"), default="copula")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
    if not utterances:
        raise ValueError("phone atlas requires reference utterances")
    if args.projection_points < 1:
        raise ValueError("phone atlas projection point count must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    point_config = PointCloudConfig(point_count=32768)
    payload = {
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
                **payload,
                "key": window.key,
                "representation": "raw_occupation_point_cloud_v1",
            },
        )
        if not path.exists():
            raise FileNotFoundError(f"raw cloud missing for {window.key}")
        return np.asarray(np.load(path), dtype=np.float64)

    occurrences = []
    for utterance in utterances:
        for ordinal, window in enumerate(
            _window_list(args.arctic_root, args.speaker, utterance)
        ):
            occurrences.append(
                (
                    window.label,
                    f"{args.speaker}:{utterance}:{ordinal}",
                    _subset(
                        (
                            marginal_copula_cloud(load_raw(window))
                            if args.gauge == "copula"
                            else affine_marginal_cloud(load_raw(window))
                        ),
                        args.projection_points,
                    ),
                )
            )
    started = perf_counter()
    config = CloudFitConfig(
        distance_mode="sliced_wasserstein",
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=1e-6,
        sliced_projection_count=args.projection_count,
        global_iterations=1,
        population_size=4,
    )
    atlas = compile_phone_atlas(
        occurrences,
        config,
        quantile_count=args.quantiles,
        provenance={
            "speaker": args.speaker,
            "utterances": list(utterances),
            "rows": args.rows,
            "noise_db": args.noise_db,
            "representation": f"{args.gauge}_marginal_occupation_cloud_v1",
            "projection_points": args.projection_points,
        },
    )
    compile_seconds = perf_counter() - started
    save_phone_atlas(args.out, atlas)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "occurrence_count": atlas.projections.shape[0],
                "label_count": len(set(atlas.labels.tolist())),
                "size_bytes": args.out.stat().st_size,
                "compile_seconds": compile_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
