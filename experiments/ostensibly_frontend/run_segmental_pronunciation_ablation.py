#!/usr/bin/env python3
"""Rank pronunciations by contiguous, bottleneck-aware phone obligations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import fuse_rank_channels

from .conditional_ridge_atlas import (
    conditional_ridge_signature,
    load_conditional_ridge_atlas,
)
from .crazy_climber_geometry import (
    CrazyClimberConfig,
    HessianRidgeConfig,
    crazy_climber_occupation,
    multiscale_hessian_ridge_surface,
)
from .depthmap_geometry import (
    DepthmapGeometryConfig,
    multiscale_morphological_residual,
)
from .generic_pronunciation_catalog import GenericPronunciationCatalog
from .occupation_point_cloud import PointCloudConfig, occupation_to_point_cloud
from .physical_interval_geometry import (
    combine_physical_interval_atlases,
    load_physical_interval_atlas,
)
from .phone_rank_coverage import load_phone_rank_coverage
from .segmental_pronunciation_geometry import (
    PhoneSpanCost,
    best_contiguous_partition,
    debounced_state_landmarks,
    enclosing_speech_crop,
    score_to_dict,
    support_landmarks_within_crop,
)


def _phones(value: str) -> tuple[str, ...]:
    result = tuple(item for item in value.split(",") if item)
    if not result:
        raise ValueError("target phones must not be empty")
    return result


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows.sort(
        key=lambda row: (
            int(row["worst_rank"]),
            float(row["mean_rank"]),
            float(row["worst_distance"]),
            float(row["mean_distance"]),
            tuple(row["phones"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _trace_to_cloud(
    trace: np.ndarray,
    morphology: DepthmapGeometryConfig,
    ridge: HessianRidgeConfig,
    climber: CrazyClimberConfig,
    cloud: PointCloudConfig,
) -> np.ndarray:
    """Lift an already computed real step-3 raster without audio dependencies."""

    residual, _ = multiscale_morphological_residual(trace, morphology)
    ridge_surface = multiscale_hessian_ridge_surface(residual, ridge)
    occupation = crazy_climber_occupation(ridge_surface.saliency, climber)
    return occupation_to_point_cloud(occupation.weighted, cloud).points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("pronunciation_catalog", type=Path)
    parser.add_argument("--phone0", type=int, required=True)
    parser.add_argument("--phone1", type=int, required=True)
    parser.add_argument("--target-phones", required=True)
    parser.add_argument(
        "--boundary-source",
        choices=("region-gaps", "state-landmarks", "state-cropped-support"),
        default="region-gaps",
        help="Use the old active-region gaps or Cleanup/SHARK state transitions",
    )
    parser.add_argument(
        "--speech-state",
        type=Path,
        help="Cleanup/SHARK NPZ containing state and speech_mask",
    )
    parser.add_argument("--minimum-state-run-frames", type=int, default=3)
    parser.add_argument("--minimum-landmark-interval-frames", type=int, default=2)
    parser.add_argument(
        "--physical-atlas",
        type=Path,
        nargs="+",
        help="One or more cross-speaker physical interval atlases",
    )
    parser.add_argument(
        "--physical-weight",
        type=float,
        default=0.0,
        help="Cross-speaker-calibrated weight on physical rank percentiles",
    )
    parser.add_argument(
        "--fusion-policy",
        choices=("arithmetic", "geometric", "minimum", "maximum"),
        default="arithmetic",
    )
    parser.add_argument(
        "--physical-query-row-multiplier",
        type=float,
        default=1.0,
        help="Frozen unlabeled radio-to-atlas row-unit conversion",
    )
    parser.add_argument("--physical-interval-weight", type=float, default=1.0)
    parser.add_argument("--physical-trajectory-weight", type=float, default=0.5)
    parser.add_argument("--physical-mass-weight", type=float, default=0.25)
    parser.add_argument(
        "--rank-coverage",
        type=Path,
        help="Cross-speaker target-rank coverage law; annotates but never reorders",
    )
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument(
        "--gap-assignment",
        choices=("right", "midpoint", "left"),
        default="right",
        help="Place inactive inter-region frames in the right phone, split at "
        "the midpoint, or place them in the left phone",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    if args.phone0 < 0 or args.phone1 <= args.phone0:
        raise ValueError("invalid queried region span")

    lattice = json.loads(args.lattice.read_text(encoding="utf-8"))
    edge = next(
        row
        for row in lattice["edges"]
        if (int(row["phone0"]), int(row["phone1"]))
        == (args.phone0, args.phone1)
    )
    target = _phones(args.target_phones)
    phone_length = len(target)
    source_region_count = args.phone1 - args.phone0
    admitted = sorted(
        {
            tuple(str(phone) for phone in row["phones"])
            for row in edge["ambiguity_classes"]
            if len(row["phones"]) == phone_length
        }
    )
    catalog = GenericPronunciationCatalog.load(args.pronunciation_catalog)
    exhaustive = sorted(
        {
            tuple(value.split())
            for value, length in zip(
                catalog.phone_keys.astype(str), catalog.lengths, strict=True
            )
            if int(length) == phone_length
        }
    )
    if target not in admitted or target not in exhaustive:
        raise ValueError("target missing from a requested comparison inventory")

    with np.load(args.region_clouds, allow_pickle=False) as document:
        frame0 = np.asarray(document["frame0"], dtype=np.int64)
        frame1 = np.asarray(document["frame1"], dtype=np.int64)
    with np.load(args.recording, allow_pickle=False) as document:
        trace = np.asarray(document["trace_field"], dtype=np.float64)
    atlas = load_conditional_ridge_atlas(args.atlas)
    rank_coverage = (
        load_phone_rank_coverage(args.rank_coverage)
        if args.rank_coverage is not None
        else None
    )
    physical_atlas = None
    if args.physical_weight < 0.0:
        raise ValueError("physical weight must be nonnegative")
    if args.physical_query_row_multiplier <= 0.0:
        raise ValueError("physical query row multiplier must be positive")
    physical_component_weights = (
        args.physical_interval_weight,
        args.physical_trajectory_weight,
        args.physical_mass_weight,
    )
    if (
        any(not np.isfinite(value) or value < 0.0 for value in physical_component_weights)
        or sum(physical_component_weights) <= 0.0
    ):
        raise ValueError("physical component weights must be finite and nonzero")
    if args.physical_weight > 0.0:
        if not args.physical_atlas:
            raise ValueError("positive physical weight requires --physical-atlas")
        physical_atlas = combine_physical_interval_atlases(
            load_physical_interval_atlas(path) for path in args.physical_atlas
        )
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climber = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    gap_audit: list[dict[str, object]] = []
    state_audit: list[dict[str, object]] = []
    if args.boundary_source in ("state-landmarks", "state-cropped-support"):
        if args.speech_state is None:
            raise ValueError("state-based boundaries require --speech-state")
        with np.load(args.speech_state, allow_pickle=False) as document:
            speech_mask = np.asarray(document["speech_mask"], dtype=bool)
            state_boundary_values, state_rows = debounced_state_landmarks(
                document["state"],
                document["speech_mask"],
                int(frame0[args.phone0]),
                int(frame1[args.phone1 - 1]),
                minimum_run_frames=args.minimum_state_run_frames,
            )
        state_audit = [dict(row) for row in state_rows]
        if args.boundary_source == "state-landmarks":
            boundary_frames = list(state_boundary_values)
        else:
            crop0, crop1 = enclosing_speech_crop(
                speech_mask,
                int(frame0[args.phone0]),
                int(frame1[args.phone1 - 1]),
            )
            boundary_frames = list(
                support_landmarks_within_crop(
                    frame0[args.phone0 : args.phone1],
                    frame1[args.phone0 : args.phone1],
                    crop0,
                    crop1,
                    minimum_interval_frames=args.minimum_landmark_interval_frames,
                )
            )
    else:
        boundary_frames = [int(frame0[args.phone0])]
        for local_boundary in range(1, source_region_count):
            absolute_boundary = args.phone0 + local_boundary
            previous_stop = int(frame1[absolute_boundary - 1])
            next_start = int(frame0[absolute_boundary])
            if args.gap_assignment == "right":
                cut = previous_stop
            elif args.gap_assignment == "left":
                cut = next_start
            else:
                cut = (previous_stop + next_start) // 2
            boundary_frames.append(cut)
            gap_audit.append(
                {
                    "local_boundary": local_boundary,
                    "previous_active_stop": previous_stop,
                    "next_active_start": next_start,
                    "gap_frames": next_start - previous_stop,
                    "cut_frame": cut,
                }
            )
        boundary_frames.append(int(frame1[args.phone1 - 1]))
    partition_count = len(boundary_frames) - 1
    if partition_count < phone_length:
        raise ValueError("fewer measured state compartments than target phones")

    span_costs: dict[tuple[int, int, str], PhoneSpanCost] = {}
    span_audit = []
    packed_surfaces = []
    packed_masses = []
    packed_distances = []
    packed_ranks = []
    packed_witnesses = []
    packed_topology_ranks = []
    packed_physical_ranks = []
    packed_topology_distances = []
    packed_physical_distances = []
    atlas_labels = tuple(sorted(set(atlas.labels.astype(str))))
    if rank_coverage is not None and rank_coverage.label_count != len(atlas_labels):
        raise ValueError("rank coverage and phone atlas inventories differ")
    for local_start in range(partition_count):
        for local_stop in range(local_start + 1, partition_count + 1):
            absolute_start = args.phone0 + local_start
            absolute_stop = args.phone0 + local_stop
            trace_field = trace[
                :128,
                boundary_frames[local_start] : boundary_frames[local_stop],
            ]
            cloud = _trace_to_cloud(
                trace_field, morphology, ridge, climber, cloud_config
            )
            surface, mass = conditional_ridge_signature(
                cloud, atlas.time_bins, atlas.row_quantiles
            )
            topology_ranking = atlas.rank(cloud)
            if physical_atlas is None:
                ranking = topology_ranking
                physical_ranking = None
            else:
                physical_cloud = cloud.copy()
                physical_cloud[:, 0] *= args.physical_query_row_multiplier
                physical_ranking = physical_atlas.rank_with_weights(
                    physical_cloud,
                    interval_weight=args.physical_interval_weight,
                    trajectory_weight=args.physical_trajectory_weight,
                    mass_weight=args.physical_mass_weight,
                )
                ranking = fuse_rank_channels(
                    topology_ranking,
                    physical_ranking,
                    args.physical_weight,
                    policy=args.fusion_policy,
                )
                topology_by_phone = {
                    str(row["phone"]): row for row in topology_ranking
                }
                physical_by_phone = {
                    str(row["phone"]): row for row in physical_ranking
                }
                for row in ranking:
                    phone = str(row["phone"])
                    row["distance"] = float(row["score"])
                    row["witness"] = (
                        f"topology={topology_by_phone[phone]['witness']}|"
                        f"physical={physical_by_phone[phone]['witness']}"
                    )
            if rank_coverage is not None:
                ranking = rank_coverage.annotate(ranking)
            ranking_by_phone = {str(row["phone"]): row for row in ranking}
            for row in ranking:
                span_costs[(local_start, local_stop, str(row["phone"]))] = (
                    PhoneSpanCost(
                        distance=float(row["distance"]),
                        rank=int(row["rank"]),
                        witness=str(row["witness"]),
                        empirical_target_frequency_at_rank=(
                            float(row["empirical_target_frequency_at_rank"])
                            if "empirical_target_frequency_at_rank" in row
                            else None
                        ),
                        cumulative_target_coverage=(
                            float(row["cumulative_target_coverage"])
                            if "cumulative_target_coverage" in row
                            else None
                        ),
                        cumulative_target_coverage_lower_95=(
                            float(row["cumulative_target_coverage_lower_95"])
                            if "cumulative_target_coverage_lower_95" in row
                            else None
                        ),
                    )
                )
            span_audit.append(
                {
                    "region_span": [local_start, local_stop],
                    "absolute_region_span": [absolute_start, absolute_stop],
                    "frame_span": [
                        boundary_frames[local_start],
                        boundary_frames[local_stop],
                    ],
                    "top5": ranking[:5],
                }
            )
            packed_surfaces.append(surface.astype(np.float32))
            packed_masses.append(mass.astype(np.float32))
            packed_distances.append(
                [float(ranking_by_phone[phone]["distance"]) for phone in atlas_labels]
            )
            packed_ranks.append(
                [int(ranking_by_phone[phone]["rank"]) for phone in atlas_labels]
            )
            packed_witnesses.append(
                [str(ranking_by_phone[phone]["witness"]) for phone in atlas_labels]
            )
            topology_by_phone = {
                str(row["phone"]): row for row in topology_ranking
            }
            packed_topology_ranks.append(
                [int(topology_by_phone[phone]["rank"]) for phone in atlas_labels]
            )
            packed_topology_distances.append(
                [
                    float(topology_by_phone[phone]["distance"])
                    for phone in atlas_labels
                ]
            )
            if physical_ranking is None:
                packed_physical_ranks.append([-1 for _ in atlas_labels])
                packed_physical_distances.append(
                    [float("nan") for _ in atlas_labels]
                )
            else:
                physical_by_phone = {
                    str(row["phone"]): row for row in physical_ranking
                }
                packed_physical_ranks.append(
                    [int(physical_by_phone[phone]["rank"]) for phone in atlas_labels]
                )
                packed_physical_distances.append(
                    [
                        float(physical_by_phone[phone]["distance"])
                        for phone in atlas_labels
                    ]
                )
            print(
                json.dumps(
                    {
                        "completed_spans": len(span_audit),
                        "total_spans": partition_count * (partition_count + 1) // 2,
                        "region_span": [local_start, local_stop],
                    }
                ),
                flush=True,
            )

    def score_inventory(values: list[tuple[str, ...]]) -> list[dict[str, object]]:
        return _rank_rows(
            [
                score_to_dict(
                    best_contiguous_partition(phones, partition_count, span_costs)
                )
                for phones in values
            ]
        )

    admitted_ranking = score_inventory(admitted)
    exhaustive_ranking = score_inventory(exhaustive)
    admitted_target = next(
        row for row in admitted_ranking if tuple(row["phones"]) == target
    )
    exhaustive_target = next(
        row for row in exhaustive_ranking if tuple(row["phones"]) == target
    )
    output = {
        "method": (
            "contiguous_phone_obligations_calibrated_topology_physical_rank_fusion"
            if physical_atlas is not None
            else "contiguous_phone_obligations_rank_bottleneck_then_mean"
        ),
        "selection_status": (
            "diagnostic identity verifier; no transcript labels used in geometry"
        ),
        "phone_span": [args.phone0, args.phone1],
        "source_frame_span": [
            int(frame0[args.phone0]), int(frame1[args.phone1 - 1])
        ],
        "frame_span": [boundary_frames[0], boundary_frames[-1]],
        "boundary_source": args.boundary_source,
        "gap_assignment": (
            args.gap_assignment if args.boundary_source == "region-gaps" else None
        ),
        "speech_state": (
            str(args.speech_state) if args.speech_state is not None else None
        ),
        "minimum_state_run_frames": args.minimum_state_run_frames,
        "minimum_landmark_interval_frames": args.minimum_landmark_interval_frames,
        "boundary_frames": boundary_frames,
        "gap_audit": gap_audit,
        "state_audit": state_audit,
        "target_phones": list(target),
        "points_per_span": args.points,
        "atlas": str(args.atlas),
        "physical_atlases": [str(path) for path in (args.physical_atlas or ())],
        "physical_weight": args.physical_weight,
        "fusion_policy": args.fusion_policy,
        "physical_query_row_multiplier": args.physical_query_row_multiplier,
        "physical_component_weights": {
            "interval": args.physical_interval_weight,
            "trajectory": args.physical_trajectory_weight,
            "mass": args.physical_mass_weight,
        },
        "rank_coverage": (
            str(args.rank_coverage) if args.rank_coverage is not None else None
        ),
        "span_audit": span_audit,
        "admitted": {
            "candidate_count": len(admitted_ranking),
            "target": admitted_target,
            "ranking": admitted_ranking,
        },
        "exhaustive_same_length": {
            "candidate_count": len(exhaustive_ranking),
            "target": exhaustive_target,
            "ranking": exhaustive_ranking,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        surfaces=np.stack(packed_surfaces),
        masses=np.stack(packed_masses),
        span_bounds=np.asarray(
            [row["region_span"] for row in span_audit], dtype=np.int16
        ),
        phone_labels=np.asarray(atlas_labels),
        phone_distances=np.asarray(packed_distances, dtype=np.float32),
        phone_ranks=np.asarray(packed_ranks, dtype=np.int16),
        phone_witnesses=np.asarray(packed_witnesses),
        topology_phone_ranks=np.asarray(packed_topology_ranks, dtype=np.int16),
        physical_phone_ranks=np.asarray(packed_physical_ranks, dtype=np.int16),
        topology_phone_distances=np.asarray(
            packed_topology_distances, dtype=np.float32
        ),
        physical_phone_distances=np.asarray(
            packed_physical_distances, dtype=np.float32
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "admitted_target_rank": admitted_target["rank"],
                "admitted_count": len(admitted_ranking),
                "exhaustive_target_rank": exhaustive_target["rank"],
                "exhaustive_count": len(exhaustive_ranking),
                "target_cuts": admitted_target["cuts"],
                "elapsed_seconds": output["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
