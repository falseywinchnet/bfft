#!/usr/bin/env python3
"""Retrieve repeated listener words using identically formed radio clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import (
    CloudFitConfig,
    PointCloudConfig,
    sliced_wasserstein_projection,
    sliced_wasserstein_projection_distance,
)
from .run_occupation_cloud_battery import _trace_to_cloud
from .run_occupation_context_battery import _subset
from .run_occupation_word_span_battery import lane_chunks


def gauge_cloud(points: np.ndarray, mode: str) -> np.ndarray:
    """Apply a pitch/time gauge with explicit marginal-retention semantics."""

    cloud = np.asarray(points, dtype=np.float64).copy()
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
        raise ValueError("affine word gauge requires a nonempty N x 3 cloud")
    if mode not in {
        "copula",
        "affine",
        "affine_row_rank_time",
        "rank_row_affine_time",
    }:
        raise ValueError("unsupported word-cloud gauge")
    count = cloud.shape[0]
    if mode in {"affine", "affine_row_rank_time"}:
        row_center = float(np.median(cloud[:, 0]))
        row_scale = float(
            np.quantile(cloud[:, 0], 0.9) - np.quantile(cloud[:, 0], 0.1)
        )
        cloud[:, 0] = (cloud[:, 0] - row_center) / max(row_scale, 1e-12)
    else:
        cloud[:, 0] = (rankdata(cloud[:, 0], method="average") - 0.5) / count
    if mode in {"affine", "rank_row_affine_time"}:
        frame0 = float(np.min(cloud[:, 1]))
        frame_scale = float(np.max(cloud[:, 1]) - frame0)
        cloud[:, 1] = (cloud[:, 1] - frame0) / max(frame_scale, 1e-12)
    else:
        cloud[:, 1] = (rankdata(cloud[:, 1], method="average") - 0.5) / count
    return cloud


def affine_gauge_cloud(points: np.ndarray) -> np.ndarray:
    """Compatibility name for the marginal-preserving affine gauge."""

    return gauge_cloud(points, "affine")


def support_grid(
    points: np.ndarray, bins: int, minimum_cell_count: int
) -> np.ndarray:
    """Rasterize density-free support from a gauged occupation cloud."""

    if bins < 16 or minimum_cell_count < 1:
        raise ValueError("support grid needs at least 16 bins and positive depth")
    cloud = np.asarray(points, dtype=np.float64)
    row = cloud[:, 0]
    if float(np.min(row)) < 0.0 or float(np.max(row)) > 1.0:
        row = 0.5 + 0.5 * row
    coordinates = np.column_stack((row, cloud[:, 1]))
    coordinates = np.clip(coordinates, 0.0, np.nextafter(1.0, 0.0))
    cells = np.floor(coordinates * bins).astype(np.int64)
    counts = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(counts, (cells[:, 0], cells[:, 1]), 1)
    return counts >= minimum_cell_count


def support_miss_distance(
    left: np.ndarray, right: np.ndarray, sigma_cells: float
) -> float:
    """Symmetric saturating nearest-support miss cost."""

    if sigma_cells <= 0.0 or not np.any(left) or not np.any(right):
        raise ValueError("support distance requires nonempty support and sigma")
    left_distance = ndi.distance_transform_edt(~left)
    right_distance = ndi.distance_transform_edt(~right)

    def miss(distance: np.ndarray) -> float:
        return float(
            np.mean(1.0 - np.exp(-0.5 * (distance / sigma_cells) ** 2))
        )

    return 0.5 * (miss(right_distance[left]) + miss(left_distance[right]))


def gauge_region_lanes(
    clouds: tuple[np.ndarray, ...], points_per_region: int, mode: str
) -> np.ndarray:
    """Gauge frequency jointly and time locally over ordered phone regions."""

    if not clouds or points_per_region < 1:
        raise ValueError("region-lane gauge requires clouds and point depth")
    chunks = [
        _subset(np.asarray(cloud, dtype=np.float64), points_per_region).copy()
        for cloud in clouds
    ]
    total = sum(chunk.shape[0] for chunk in chunks)
    combined_rows = np.concatenate([chunk[:, 0] for chunk in chunks])
    if mode in {"affine", "affine_row_rank_time"}:
        center = float(np.median(combined_rows))
        scale = float(
            np.quantile(combined_rows, 0.9)
            - np.quantile(combined_rows, 0.1)
        )
        gauged_rows = (combined_rows - center) / max(scale, 1e-12)
    else:
        gauged_rows = (rankdata(combined_rows, method="average") - 0.5) / total
    offset = 0
    for chunk in chunks:
        count = chunk.shape[0]
        chunk[:, 0] = gauged_rows[offset : offset + count]
        offset += count
        if mode in {"affine", "rank_row_affine_time"}:
            frame0 = float(np.min(chunk[:, 1]))
            scale = float(np.max(chunk[:, 1]) - frame0)
            chunk[:, 1] = (chunk[:, 1] - frame0) / max(scale, 1e-12)
        else:
            chunk[:, 1] = (
                rankdata(chunk[:, 1], method="average") - 0.5
            ) / count
    return lane_chunks(tuple(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fusion_lattice", type=Path)
    parser.add_argument("transcript_alignment", type=Path)
    parser.add_argument("recording_npz", type=Path)
    parser.add_argument("--section", default="Gemini listener")
    parser.add_argument("--recording-field", default="trace_field")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--point-count", type=int, default=8192)
    parser.add_argument("--projection-points", type=int, default=4096)
    parser.add_argument("--projection-count", type=int, default=32)
    parser.add_argument("--height-metric-scale", type=float, default=1e-6)
    parser.add_argument(
        "--distance-mode", choices=("swd", "support"), default="swd"
    )
    parser.add_argument("--support-bins", type=int, default=64)
    parser.add_argument("--support-minimum-cell-count", type=int, default=2)
    parser.add_argument("--support-sigma-cells", type=float, default=1.5)
    parser.add_argument(
        "--gauge",
        choices=(
            "copula",
            "affine",
            "affine_row_rank_time",
            "rank_row_affine_time",
        ),
        default="copula",
    )
    parser.add_argument("--word-clouds-in", type=Path)
    parser.add_argument("--word-clouds-out", type=Path)
    parser.add_argument(
        "--query-mode", choices=("whole_trace", "region_lanes"), default="whole_trace"
    )
    parser.add_argument("--region-clouds", type=Path)
    parser.add_argument("--points-per-region", type=int, default=1024)
    parser.add_argument("--word-limit", type=int)
    parser.add_argument("--exact-span-only", action="store_true")
    parser.add_argument(
        "--candidate-mode",
        choices=("all", "same_region_count"),
        default="all",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fusion = json.loads(args.fusion_lattice.read_text())
    alignment = json.loads(args.transcript_alignment.read_text())
    spans = alignment["audits"][args.section]["word_region_spans"]
    if args.word_limit is not None:
        spans = spans[: args.word_limit]
    phone_rows = fusion["phones"]
    with np.load(args.recording_npz) as document:
        field = np.asarray(document[args.recording_field], dtype=np.float64)
    cached_clouds = None
    if args.word_clouds_in is not None:
        with np.load(args.word_clouds_in) as document:
            cached_clouds = np.asarray(document["clouds"], dtype=np.float64)
    region_clouds = None
    if args.query_mode == "region_lanes":
        if args.region_clouds is None:
            raise ValueError("region-lane retrieval requires --region-clouds")
        with np.load(args.region_clouds) as document:
            region_clouds = np.asarray(document["clouds"], dtype=np.float64)
        if region_clouds.shape[0] != len(phone_rows):
            raise ValueError("region cloud and fusion lattice counts disagree")

    distance_config = CloudFitConfig(
        fit_point_count=args.projection_points,
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=args.height_metric_scale,
        distance_mode="sliced_wasserstein",
        sliced_projection_count=args.projection_count,
        global_iterations=1,
        population_size=4,
    )
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climbers = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.point_count)
    tokens = []
    projections = []
    supports = []
    raw_clouds = []
    started = perf_counter()
    for span in spans:
        if span["region0"] is None or span["region1"] is None:
            continue
        region0 = int(span["region0"])
        region1 = int(span["region1"])
        if not 0 <= region0 < region1 <= len(phone_rows):
            continue
        frame0 = int(phone_rows[region0]["frame0"])
        frame1 = int(phone_rows[region1 - 1]["frame1"])
        if args.query_mode == "region_lanes":
            cloud = None
            ordered = gauge_region_lanes(
                tuple(region_clouds[region0:region1]),
                args.points_per_region,
                args.gauge,
            )
        elif cached_clouds is None:
            cloud = _trace_to_cloud(
                field[: args.rows, frame0:frame1],
                morphology,
                ridge,
                climbers,
                cloud_config,
            )
        else:
            if len(tokens) >= cached_clouds.shape[0]:
                raise ValueError("word cloud cache has too few entries")
            cloud = cached_clouds[len(tokens)]
        if args.query_mode == "whole_trace":
            raw_clouds.append(np.asarray(cloud, dtype=np.float32))
            ordered = gauge_cloud(cloud, args.gauge)
        if args.distance_mode == "swd":
            projections.append(
                sliced_wasserstein_projection(ordered, distance_config)
            )
        else:
            supports.append(
                support_grid(
                    ordered,
                    args.support_bins,
                    args.support_minimum_cell_count,
                )
            )
        tokens.append(
            {
                "word_index": int(span["word_index"]),
                "word": str(span["word"]),
                "phones": list(span["phones"]),
                "region0": region0,
                "region1": region1,
                "frame0": frame0,
                "frame1": frame1,
            }
        )
        if len(tokens) % 25 == 0:
            print(f"projected {len(tokens)} listener words", flush=True)
    if (
        args.query_mode == "whole_trace"
        and cached_clouds is not None
        and cached_clouds.shape[0] != len(tokens)
    ):
        raise ValueError("word cloud cache and aligned token counts disagree")
    if args.word_clouds_out is not None:
        if args.query_mode != "whole_trace":
            raise ValueError("word cloud output is only valid for whole-trace mode")
        args.word_clouds_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.word_clouds_out,
            clouds=np.stack(raw_clouds),
        )

    repeated = []
    for index, token in enumerate(tokens):
        if args.exact_span_only and (
            token["region1"] - token["region0"] != len(token["phones"])
        ):
            continue
        eligible = {
            other
            for other, candidate in enumerate(tokens)
            if other != index
            and (
                not args.exact_span_only
                or candidate["region1"] - candidate["region0"]
                == len(candidate["phones"])
            )
            and (
                args.candidate_mode == "all"
                or candidate["region1"] - candidate["region0"]
                == token["region1"] - token["region0"]
            )
        }
        positives = {
            other
            for other, candidate in enumerate(tokens)
            if other in eligible and candidate["word"] == token["word"]
        }
        if not positives:
            continue
        if args.distance_mode == "swd":
            distance = lambda other: sliced_wasserstein_projection_distance(
                projections[index], projections[other]
            )
        else:
            distance = lambda other: support_miss_distance(
                supports[index], supports[other], args.support_sigma_cells
            )
        ranking = sorted(
            (distance(other), tokens[other]["word"], other)
            for other in eligible
        )
        rank = next(
            position
            for position, (_, _, other) in enumerate(ranking, start=1)
            if other in positives
        )
        repeated.append(
            {
                **token,
                "same_word_occurrences": len(positives),
                "candidate_count": len(eligible),
                "nearest_same_word_rank": rank,
                "top_neighbors": [
                    {
                        "rank": position,
                        "distance": float(distance),
                        "word": word,
                        "word_index": tokens[other]["word_index"],
                    }
                    for position, (distance, word, other) in enumerate(
                        ranking[:10], start=1
                    )
                ],
            }
        )
    ranks = [row["nearest_same_word_rank"] for row in repeated]
    result = {
        "method": "listener_aligned_identically_formed_radio_word_cloud_retrieval",
        "selection_status": "diagnostic transcript labels; no reference speech",
        "section": args.section,
        "parameters": {
            "rows": args.rows,
            "point_count": args.point_count,
            "projection_points": args.projection_points,
            "projection_count": args.projection_count,
            "height_metric_scale": args.height_metric_scale,
            "gauge": args.gauge,
            "exact_span_only": args.exact_span_only,
            "candidate_mode": args.candidate_mode,
            "distance_mode": args.distance_mode,
            "support_bins": args.support_bins,
            "support_minimum_cell_count": args.support_minimum_cell_count,
            "support_sigma_cells": args.support_sigma_cells,
            "query_mode": args.query_mode,
            "points_per_region": args.points_per_region,
        },
        "elapsed_seconds": perf_counter() - started,
        "token_count": len(tokens),
        "summary": {
            "repeated_token_count": len(ranks),
            "top1": sum(rank == 1 for rank in ranks),
            "top5": sum(rank <= 5 for rank in ranks),
            "top20": sum(rank <= 20 for rank in ranks),
            "mean_reciprocal_rank": (
                float(np.mean([1.0 / rank for rank in ranks])) if ranks else None
            ),
            "median_rank": float(np.median(ranks)) if ranks else None,
        },
        "tokens": repeated,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
