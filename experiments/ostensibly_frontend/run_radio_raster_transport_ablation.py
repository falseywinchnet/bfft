#!/usr/bin/env python3
"""Measure pre-lift radio-floor transport on a whole-word control battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .conditional_ridge_atlas import (
    conditional_ridge_distance,
    conditional_ridge_signature,
)
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .radio_raster_transport import (
    estimate_radio_raster_profile,
    transport_reference_raster,
)
from .run_occupation_cloud_battery import _trace_to_cloud
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError("ablation grid must contain finite values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positive_control_json", type=Path)
    parser.add_argument("positive_control_npz", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--noise-scales", default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--phases", default="0,0.3333333333,0.6666666667")
    parser.add_argument("--row-blur-sigma", type=float, default=0.0)
    parser.add_argument("--time-blur-sigma", type=float, default=0.0)
    parser.add_argument("--amplitude-gamma", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    noise_scales = _csv_floats(args.noise_scales)
    phases = _csv_floats(args.phases)
    metadata = json.loads(args.positive_control_json.read_text(encoding="utf-8"))
    with np.load(args.positive_control_npz, allow_pickle=False) as document:
        query = (
            np.asarray(document["query_surface"], dtype=np.float64),
            np.asarray(document["query_mass"], dtype=np.float64),
        )
        packed_traces = np.asarray(document["trace_fields"], dtype=np.float64)
        offsets = np.asarray(document["trace_offsets"], dtype=np.int64)
    rows = metadata["ranking"]
    if offsets.shape != (len(rows) + 1,) or offsets[0] != 0:
        raise ValueError("positive-control trace cache disagrees with ranking")
    traces = tuple(
        packed_traces[:, int(start) : int(stop)]
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )
    with np.load(args.recording, allow_pickle=False) as document:
        profile = estimate_radio_raster_profile(
            np.asarray(document["trace_field"], dtype=np.float64),
            np.asarray(document["active"], dtype=bool),
            rows=packed_traces.shape[0],
        )
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climber = CrazyClimberConfig()
    cloud = PointCloudConfig(point_count=args.points)
    settings = []
    for noise_scale in noise_scales:
        started = perf_counter()
        ranked = []
        for ordinal, (row, trace) in enumerate(zip(rows, traces, strict=True)):
            distances = []
            for phase in phases:
                transported = transport_reference_raster(
                    trace,
                    profile,
                    noise_scale=noise_scale,
                    phase=phase,
                    row_blur_sigma=args.row_blur_sigma,
                    time_blur_sigma=args.time_blur_sigma,
                    amplitude_gamma=args.amplitude_gamma,
                )
                points = _trace_to_cloud(
                    transported, morphology, ridge, climber, cloud
                )
                signature = conditional_ridge_signature(
                    lane_chunks(joint_gauge_chunks((points,), 8192)), 64, 24
                )
                distances.append(
                    conditional_ridge_distance(
                        query[0], query[1], signature[0], signature[1], 4
                    )
                )
            ranked.append(
                {
                    "phones": list(row["phones"]),
                    "text": str(row["text"]),
                    "distance_mean": float(np.mean(distances)),
                    "distance_maximum": float(np.max(distances)),
                    "phase_distances": distances,
                    "ordinal": ordinal,
                }
            )
        ranked.sort(
            key=lambda row: (row["distance_mean"], row["phones"], row["text"])
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        target_phones = tuple(metadata["target"]["phones"])
        target = next(
            row for row in ranked if tuple(row["phones"]) == target_phones
        )
        setting = {
            "noise_scale": noise_scale,
            "target": target,
            "ranking": ranked,
            "elapsed_seconds": perf_counter() - started,
        }
        settings.append(setting)
        print(
            json.dumps(
                {
                    "noise_scale": noise_scale,
                    "target_rank": target["rank"],
                    "target_distance": target["distance_mean"],
                    "elapsed_seconds": setting["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    output = {
        "method": "measured_correlated_radio_floor_before_ridge_lift",
        "profile": {
            "noise_to_active_ratio": profile.noise_to_active_ratio,
            "noise_shape": list(profile.noise_field.shape),
            "quantile": profile.quantile,
        },
        "phases": list(phases),
        "row_blur_sigma": args.row_blur_sigma,
        "time_blur_sigma": args.time_blur_sigma,
        "amplitude_gamma": args.amplitude_gamma,
        "points": args.points,
        "settings": settings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
