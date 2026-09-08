#!/usr/bin/env python3
"""Compile raw phone occurrences for runtime generic word realization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .generic_word_template_bank import (
    EDGE_CONTEXT,
    compile_generic_word_template_bank,
    save_generic_word_template_bank,
)
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_context_battery import _cache_name, _window_list
from .run_occupation_cloud_battery import (
    _load_channel_baseline,
    _trace_to_cloud,
    _window_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="Build uncached utterances transiently and retain only bank points",
    )
    parser.add_argument("--points-per-occurrence", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climbers = CrazyClimberConfig()
    point_cloud = PointCloudConfig(point_count=32768)
    payload = {
        "rows": 128,
        "noise_db": 8.0,
        "morphology": asdict(morphology),
        "ridge": asdict(ridge),
        "climbers": asdict(climbers),
        "point_cloud": asdict(point_cloud),
    }
    occurrences = []
    for utterance in (value for value in args.utterances.split(",") if value):
        windows = _window_list(args.arctic_root, args.speaker, utterance)
        baseline = None
        for ordinal, window in enumerate(windows):
            path = args.raw_cache / _cache_name(
                window,
                {
                    **payload,
                    "key": window.key,
                    "representation": "raw_occupation_point_cloud_v1",
                },
            )
            if path.exists():
                cloud = np.asarray(np.load(path), dtype=np.float64)
            elif args.build_missing:
                if baseline is None:
                    utterance_number = int(utterance.rsplit("a", 1)[-1])
                    baseline = _load_channel_baseline(
                        args.arctic_root,
                        args.speaker,
                        utterance,
                        utterance_number - 1,
                        8.0,
                    )
                trace, _ = _window_trace(
                    baseline, window.seconds0, window.seconds1, 128
                )
                cloud = _trace_to_cloud(
                    trace,
                    morphology,
                    ridge,
                    climbers,
                    point_cloud,
                )
            else:
                raise FileNotFoundError(
                    f"raw cloud missing for {window.key}; pass --build-missing"
                )
            occurrences.append(
                (
                    window.label,
                    f"{args.speaker}:{utterance}:{ordinal}",
                    windows[ordinal - 1].label if ordinal else EDGE_CONTEXT,
                    (
                        windows[ordinal + 1].label
                        if ordinal + 1 < len(windows)
                        else EDGE_CONTEXT
                    ),
                    cloud,
                )
            )
    bank = compile_generic_word_template_bank(
        occurrences, args.points_per_occurrence
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_generic_word_template_bank(args.out, bank)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "occurrences": int(bank.labels.size),
                "labels": len(set(bank.labels.tolist())),
                "points_per_occurrence": bank.points_per_occurrence,
                "size_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
