#!/usr/bin/env python3
"""Compile density-free phone support from cached generic speech clouds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .compiled_support_phone_atlas import (
    compile_support_phone_atlas,
    save_support_phone_atlas,
)
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig, marginal_copula_cloud
from .run_occupation_context_battery import _cache_name, _window_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--minimum-cell-count", type=int, default=2)
    parser.add_argument("--sigma-cells", type=float, default=1.5)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
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
            {**payload, "key": window.key, "representation": "raw_occupation_point_cloud_v1"},
        )
        if not path.exists():
            raise FileNotFoundError(f"raw cloud missing for {window.key}")
        return np.asarray(np.load(path), dtype=np.float64)

    occurrences = []
    for utterance in utterances:
        for ordinal, window in enumerate(_window_list(args.arctic_root, args.speaker, utterance)):
            occurrences.append(
                (
                    window.label,
                    f"{args.speaker}:{utterance}:{ordinal}",
                    marginal_copula_cloud(load_raw(window)),
                )
            )
    atlas = compile_support_phone_atlas(
        occurrences,
        bins=args.bins,
        minimum_cell_count=args.minimum_cell_count,
        sigma_cells=args.sigma_cells,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_support_phone_atlas(args.out, atlas)
    print(json.dumps({"output": str(args.out), "occurrences": len(occurrences), "labels": len(set(atlas.labels.tolist())), "size_bytes": args.out.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
