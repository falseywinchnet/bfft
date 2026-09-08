#!/usr/bin/env python3
"""Populate reusable raw occupation clouds for complete ARCTIC speakers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .run_arctic_calibration import DEFAULT_UTTERANCES
from .run_occupation_cloud_battery import (
    _load_channel_baseline,
    _trace_to_cloud,
    _window_trace,
)
from .run_occupation_context_battery import _cache_name, _window_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
    args.cache.mkdir(parents=True, exist_ok=True)
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climbers = CrazyClimberConfig()
    point_config = PointCloudConfig(point_count=32768)
    payload = {
        "rows": args.rows,
        "noise_db": args.noise_db,
        "morphology": asdict(morphology),
        "ridge": asdict(ridge),
        "climbers": asdict(climbers),
        "point_cloud": asdict(point_config),
    }
    built = reused = 0
    started = perf_counter()
    for utterance_index, utterance in enumerate(utterances):
        windows = _window_list(args.arctic_root, args.speaker, utterance)
        pending = []
        for window in windows:
            path = args.cache / _cache_name(
                window,
                {
                    **payload,
                    "key": window.key,
                    "representation": "raw_occupation_point_cloud_v1",
                },
            )
            if path.exists():
                reused += 1
            else:
                pending.append((window, path))
        if pending:
            baseline = _load_channel_baseline(
                args.arctic_root,
                args.speaker,
                utterance,
                utterance_index,
                args.noise_db,
            )
            for window, path in pending:
                trace, _ = _window_trace(
                    baseline, window.seconds0, window.seconds1, args.rows
                )
                cloud = _trace_to_cloud(
                    trace, morphology, ridge, climbers, point_config
                )
                temporary = path.with_name(
                    f".{path.stem}.{os.getpid()}.temporary.npy"
                )
                np.save(temporary, np.asarray(cloud, dtype=np.float32))
                os.replace(temporary, path)
                built += 1
        print(
            f"cached {args.speaker}/{utterance}: built={built} reused={reused}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "speaker": args.speaker,
                "utterance_count": len(utterances),
                "built": built,
                "reused": reused,
                "elapsed_seconds": perf_counter() - started,
                "cache": str(args.cache),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
