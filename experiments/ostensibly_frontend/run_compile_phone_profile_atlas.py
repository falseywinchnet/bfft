#!/usr/bin/env python3
"""Compile cross-speaker centroids of complete phone-distance profiles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .compiled_phone_atlas import load_phone_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .distance_profile_geometry import fit_distance_profile_atlas, ranking_profile
from .occupation_point_cloud import PointCloudConfig, marginal_copula_cloud
from .run_occupation_context_battery import _cache_name, _subset, _window_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--bdl-atlas", type=Path, required=True)
    parser.add_argument("--slt-atlas", type=Path, required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--projection-points", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(item for item in args.utterances.split(",") if item)
    atlases = {"bdl": load_phone_atlas(args.bdl_atlas), "slt": load_phone_atlas(args.slt_atlas)}
    labels = tuple(sorted(set(atlases["bdl"].labels.astype(str).tolist())))
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

    profiles = []
    targets = []
    channels = []
    witnesses = []
    for query_speaker, channel in (("slt", "bdl"), ("bdl", "slt")):
        atlas = atlases[channel]
        for utterance in utterances:
            for ordinal, window in enumerate(_window_list(args.arctic_root, query_speaker, utterance)):
                cloud = _subset(marginal_copula_cloud(load_raw(window)), args.projection_points)
                profiles.append(ranking_profile(atlas.rank(cloud), labels))
                targets.append(window.label)
                channels.append(channel)
                witnesses.append(f"{query_speaker}:{utterance}:{ordinal}")
        print(f"profiled {query_speaker} against {channel}: {len(profiles)} total", flush=True)
    fitted = fit_distance_profile_atlas(
        np.stack(profiles), np.asarray(targets), np.asarray(channels), ("bdl", "slt"), labels
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "ostensibly_distance_profile_atlas_v1", "channels": list(fitted.channels), "labels": list(fitted.labels)}
    np.savez(
        args.out,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        feature_centers=fitted.feature_centers,
        feature_scales=fitted.feature_scales,
        class_centroids=fitted.class_centroids,
        class_counts=fitted.class_counts,
        calibration_profiles=np.stack(profiles).astype(np.float32),
        calibration_targets=np.asarray(targets),
        calibration_channels=np.asarray(channels),
        witnesses=np.asarray(witnesses),
    )
    print(json.dumps({"output": str(args.out), "profile_count": len(profiles), "labels": len(labels)}, indent=2))


if __name__ == "__main__":
    main()
