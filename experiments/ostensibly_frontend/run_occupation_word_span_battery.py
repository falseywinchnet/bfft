#!/usr/bin/env python3
"""Compare one ordered phone span against every contiguous reference span."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import rankdata

from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import (
    CloudFitConfig,
    PointCloudConfig,
    SupportGeometryConfig,
    jittered_support_distance,
    marginal_copula_cloud,
    sliced_wasserstein_distance,
)
from .run_arctic_calibration import DEFAULT_UTTERANCES
from .run_occupation_cloud_battery import (
    ReferenceWindow,
    _load_channel_baseline,
    _trace_to_cloud,
    _window_trace,
)
from .run_occupation_context_battery import (
    _cache_name,
    _subset,
    _window_list,
    boundary_transition_distance,
    boundary_transition_law,
)


def ordered_span_cloud(
    clouds: tuple[np.ndarray, ...], points_per_phone: int
) -> np.ndarray:
    """Place an arbitrary nonempty phone sequence in equal ordered time lanes."""

    if not clouds or points_per_phone < 1:
        raise ValueError("ordered span requires clouds and points")
    count = len(clouds)
    chunks = []
    for slot, cloud in enumerate(clouds):
        selected = _subset(np.asarray(cloud, dtype=np.float64), points_per_phone).copy()
        if selected.ndim != 2 or selected.shape[1] != 3 or not selected.size:
            raise ValueError("ordered span contains an invalid phone cloud")
        selected[:, 1] = (slot + selected[:, 1]) / count
        chunks.append(selected)
    return np.concatenate(chunks, axis=0)


def span_transition_laws(clouds: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(
        boundary_transition_law(left, right)
        for left, right in zip(clouds, clouds[1:])
    )


def span_transition_distance(
    left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]
) -> float:
    if len(left) != len(right):
        raise ValueError("span transition counts must agree")
    if not left:
        return 0.0
    return float(
        np.mean(
            [
                boundary_transition_distance(a, b)
                for a, b in zip(left, right, strict=True)
            ]
        )
    )


def joint_gauge_chunks(
    clouds: tuple[np.ndarray, ...], points_per_phone: int
) -> tuple[np.ndarray, ...]:
    """Canonicalize frequency jointly while keeping each phone's local time."""

    if not clouds or points_per_phone < 1:
        raise ValueError("joint gauge requires clouds and points")
    chunks = []
    for cloud in clouds:
        selected = _subset(np.asarray(cloud, dtype=np.float64), points_per_phone).copy()
        if selected.ndim != 2 or selected.shape[1] != 3 or not selected.size:
            raise ValueError("joint gauge contains an invalid phone cloud")
        count = selected.shape[0]
        selected[:, 1] = (rankdata(selected[:, 1], method="average") - 0.5) / count
        chunks.append(selected)
    combined_rows = np.concatenate([chunk[:, 0] for chunk in chunks])
    canonical_rows = (
        rankdata(combined_rows, method="average") - 0.5
    ) / combined_rows.size
    offset = 0
    for chunk in chunks:
        chunk[:, 0] = canonical_rows[offset : offset + chunk.shape[0]]
        offset += chunk.shape[0]
    return tuple(chunks)


def lane_chunks(chunks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Place already time-canonical chunks into one ordered span."""

    if not chunks:
        raise ValueError("lane placement requires chunks")
    output = []
    count = len(chunks)
    for slot, chunk in enumerate(chunks):
        placed = np.asarray(chunk, dtype=np.float64).copy()
        placed[:, 1] = (slot + placed[:, 1]) / count
        output.append(placed)
    return np.concatenate(output, axis=0)


def timestamp_lane_chunks(
    chunks: tuple[np.ndarray, ...],
    intervals: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Place canonical chunks in their measured physical-time intervals."""

    if not chunks or len(intervals) != len(chunks):
        raise ValueError("timestamp lanes require one interval per chunk")
    starts = np.asarray([value[0] for value in intervals], dtype=np.float64)
    stops = np.asarray([value[1] for value in intervals], dtype=np.float64)
    if (
        not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(stops))
        or np.any(stops <= starts)
        or np.any(starts[1:] < stops[:-1])
    ):
        raise ValueError("timestamp lane intervals are invalid or overlap")
    origin = float(starts[0])
    scale = float(stops[-1] - origin)
    output = []
    for chunk, start, stop in zip(chunks, starts, stops, strict=True):
        placed = np.asarray(chunk, dtype=np.float64).copy()
        placed[:, 1] = (
            start - origin + placed[:, 1] * (stop - start)
        ) / scale
        output.append(placed)
    return np.concatenate(output, axis=0)


def _rank(rows: list[dict], field: str, marker: str) -> int:
    ordered = sorted(rows, key=lambda row: (row[field], row["key"]))
    return next(index + 1 for index, row in enumerate(ordered) if row[marker])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--query-speaker", default="slt")
    parser.add_argument("--reference-speaker", default="bdl")
    parser.add_argument("--query-utterance", default="arctic_a0019")
    parser.add_argument("--query-start-index", type=int, required=True)
    parser.add_argument("--phone-count", type=int, required=True)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--points-per-phone", type=int, default=2048)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.query_start_index < 0 or args.phone_count < 2:
        raise ValueError("word span requires a valid start and at least two phones")
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.raw_cache is not None:
        args.raw_cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    utterances = tuple(args.utterances.split(","))

    morphology_config = DepthmapGeometryConfig()
    ridge_config = HessianRidgeConfig()
    climber_config = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    support_config = SupportGeometryConfig(minimum_cell_count=1)
    distance_config = CloudFitConfig(
        fit_point_count=max(4096, args.points_per_phone * args.phone_count),
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=1e-6,
        distance_mode="sliced_wasserstein",
        sliced_projection_count=64,
        global_iterations=1,
        population_size=4,
    )
    cache_payload = {
        "rows": args.rows,
        "noise_db": args.noise_db,
        "morphology": asdict(morphology_config),
        "ridge": asdict(ridge_config),
        "climbers": asdict(climber_config),
        "point_cloud": asdict(cloud_config),
    }
    baselines: dict[tuple[str, str], np.ndarray] = {}
    clouds: dict[tuple[str, str, str, float, float], np.ndarray] = {}

    def get_cloud(window: ReferenceWindow) -> np.ndarray:
        if window.key in clouds:
            return clouds[window.key]
        path = args.cache / _cache_name(
            window, {**cache_payload, "key": window.key}
        )
        if path.exists():
            cloud = np.asarray(np.load(path), dtype=np.float64)
        else:
            baseline_key = (window.speaker, window.utterance)
            if baseline_key not in baselines:
                print(f"building baseline {window.speaker}/{window.utterance}", flush=True)
                baselines[baseline_key] = _load_channel_baseline(
                    args.arctic_root,
                    window.speaker,
                    window.utterance,
                    (
                        utterances.index(window.utterance)
                        if window.utterance in utterances
                        else DEFAULT_UTTERANCES.index(window.utterance)
                    ),
                    args.noise_db,
                )
            trace, _ = _window_trace(
                baselines[baseline_key],
                window.seconds0,
                window.seconds1,
                args.rows,
            )
            cloud = marginal_copula_cloud(
                _trace_to_cloud(
                    trace,
                    morphology_config,
                    ridge_config,
                    climber_config,
                    cloud_config,
                )
            )
            temporary = path.with_name(
                f".{path.stem}.{os.getpid()}.temporary.npy"
            )
            np.save(temporary, np.asarray(cloud, dtype=np.float32))
            os.replace(temporary, path)
        clouds[window.key] = cloud
        return cloud

    raw_clouds: dict[tuple[str, str, str, float, float], np.ndarray] = {}

    def get_raw_cloud(window: ReferenceWindow) -> np.ndarray:
        if args.raw_cache is None:
            raise ValueError("raw cloud cache was not configured")
        if window.key in raw_clouds:
            return raw_clouds[window.key]
        path = args.raw_cache / _cache_name(
            window,
            {
                **cache_payload,
                "key": window.key,
                "representation": "raw_occupation_point_cloud_v1",
            },
        )
        if path.exists():
            cloud = np.asarray(np.load(path), dtype=np.float64)
        else:
            baseline_key = (window.speaker, window.utterance)
            if baseline_key not in baselines:
                print(f"building baseline {window.speaker}/{window.utterance}", flush=True)
                baselines[baseline_key] = _load_channel_baseline(
                    args.arctic_root,
                    window.speaker,
                    window.utterance,
                    (
                        utterances.index(window.utterance)
                        if window.utterance in utterances
                        else DEFAULT_UTTERANCES.index(window.utterance)
                    ),
                    args.noise_db,
                )
            trace, _ = _window_trace(
                baselines[baseline_key],
                window.seconds0,
                window.seconds1,
                args.rows,
            )
            cloud = _trace_to_cloud(
                trace,
                morphology_config,
                ridge_config,
                climber_config,
                cloud_config,
            )
            temporary = path.with_name(
                f".{path.stem}.{os.getpid()}.temporary.npy"
            )
            np.save(temporary, np.asarray(cloud, dtype=np.float32))
            os.replace(temporary, path)
        raw_clouds[window.key] = cloud
        return cloud

    query_windows = _window_list(
        args.arctic_root, args.query_speaker, args.query_utterance
    )
    query_stop = args.query_start_index + args.phone_count
    if query_stop > len(query_windows):
        raise ValueError("query span extends past the utterance")
    query_span = tuple(query_windows[args.query_start_index:query_stop])
    query_clouds = tuple(get_cloud(window) for window in query_span)
    query_ordered = ordered_span_cloud(query_clouds, args.points_per_phone)
    query_transitions = span_transition_laws(query_clouds)
    query_joint_ordered = None
    query_joint_transitions = None
    if args.raw_cache is not None:
        query_raw_clouds = tuple(get_raw_cloud(window) for window in query_span)
        query_joint_ordered = lane_chunks(
            joint_gauge_chunks(query_raw_clouds, args.points_per_phone)
        )
        query_joint_transitions = tuple(
            boundary_transition_law(*joint_gauge_chunks(pair, args.points_per_phone))
            for pair in zip(query_raw_clouds, query_raw_clouds[1:])
        )

    rows = []
    started = perf_counter()
    for utterance in utterances:
        windows = _window_list(
            args.arctic_root, args.reference_speaker, utterance
        )
        for start in range(len(windows) - args.phone_count + 1):
            span = tuple(windows[start : start + args.phone_count])
            reference_clouds = tuple(get_cloud(window) for window in span)
            reference_ordered = ordered_span_cloud(
                reference_clouds, args.points_per_phone
            )
            key = (
                f"{args.reference_speaker}:{utterance}:{start}:"
                + "-".join(window.label for window in span)
            )
            row = {
                    "key": key,
                    "utterance": utterance,
                    "start_index": start,
                    "phones": [window.label for window in span],
                    "seconds": [span[0].seconds0, span[-1].seconds1],
                    "ordered_swd": sliced_wasserstein_distance(
                        query_ordered, reference_ordered, distance_config
                    ),
                    "support_distance": jittered_support_distance(
                        query_ordered, reference_ordered, support_config
                    ),
                    "transition_distance": span_transition_distance(
                        query_transitions, span_transition_laws(reference_clouds)
                    ),
                    "is_corresponding_span": bool(
                        utterance == args.query_utterance
                        and start == args.query_start_index
                    ),
                    "is_same_phone_sequence": bool(
                        tuple(window.label for window in span)
                        == tuple(window.label for window in query_span)
                    ),
                }
            if args.raw_cache is not None:
                reference_raw_clouds = tuple(get_raw_cloud(window) for window in span)
                reference_joint_ordered = lane_chunks(
                    joint_gauge_chunks(reference_raw_clouds, args.points_per_phone)
                )
                reference_joint_transitions = tuple(
                    boundary_transition_law(
                        *joint_gauge_chunks(pair, args.points_per_phone)
                    )
                    for pair in zip(reference_raw_clouds, reference_raw_clouds[1:])
                )
                row.update(
                    {
                        "joint_ordered_swd": sliced_wasserstein_distance(
                            query_joint_ordered,
                            reference_joint_ordered,
                            distance_config,
                        ),
                        "joint_support_distance": jittered_support_distance(
                            query_joint_ordered,
                            reference_joint_ordered,
                            support_config,
                        ),
                        "joint_transition_distance": span_transition_distance(
                            query_joint_transitions,
                            reference_joint_transitions,
                        ),
                    }
                )
            rows.append(row)
        print(f"scored {utterance}", flush=True)
    elapsed = perf_counter() - started
    corresponding = next(
        (row for row in rows if row["is_corresponding_span"]), None
    )
    fields = ["ordered_swd", "support_distance", "transition_distance"]
    if args.raw_cache is not None:
        fields.extend(
            [
                "joint_ordered_swd",
                "joint_support_distance",
                "joint_transition_distance",
            ]
        )
    summary = {
        "candidate_span_count": len(rows),
        "query_phones": [window.label for window in query_span],
        "elapsed_seconds": elapsed,
    }
    rankings = {}
    for field in fields:
        ranking = sorted(rows, key=lambda row: (row[field], row["key"]))
        rankings[field] = [
            {
                "rank": index + 1,
                "key": row["key"],
                "phones": row["phones"],
                "distance": row[field],
                "is_corresponding_span": row["is_corresponding_span"],
            }
            for index, row in enumerate(ranking[:20])
        ]
        summary[f"corresponding_{field}_rank"] = (
            _rank(rows, field, "is_corresponding_span")
            if corresponding is not None
            else None
        )
        summary[f"corresponding_{field}"] = (
            corresponding[field] if corresponding is not None else None
        )

    def build_boundary_pair_rankings(
        query_laws: tuple[np.ndarray, ...], joint: bool
    ) -> tuple[list[dict], list[int]]:
        boundary_pair_rankings = []
        target_pair_ranks = []
        for boundary_index, query_law in enumerate(query_laws):
            best_by_pair: dict[tuple[str, str], dict] = {}
            for utterance in utterances:
                windows = _window_list(
                    args.arctic_root, args.reference_speaker, utterance
                )
                for start, (left, right) in enumerate(zip(windows, windows[1:])):
                    if joint:
                        chunks = joint_gauge_chunks(
                            (get_raw_cloud(left), get_raw_cloud(right)),
                            args.points_per_phone,
                        )
                        reference_law = boundary_transition_law(*chunks)
                    else:
                        reference_law = boundary_transition_law(
                            get_cloud(left), get_cloud(right)
                        )
                    distance = boundary_transition_distance(query_law, reference_law)
                    pair = (left.label, right.label)
                    witness = {
                        "phones": list(pair),
                        "distance": distance,
                        "witness": f"{args.reference_speaker}:{utterance}:{start}",
                    }
                    current = best_by_pair.get(pair)
                    if current is None or (distance, witness["witness"]) < (
                        current["distance"],
                        current["witness"],
                    ):
                        best_by_pair[pair] = witness
            ranking = sorted(
                best_by_pair.values(),
                key=lambda item: (item["distance"], item["phones"]),
            )
            for rank, item in enumerate(ranking, start=1):
                item["rank"] = rank
            target_pair = [
                query_span[boundary_index].label,
                query_span[boundary_index + 1].label,
            ]
            target_rank = next(
                (
                    item["rank"]
                    for item in ranking
                    if item["phones"] == target_pair
                ),
                None,
            )
            target_pair_ranks.append(target_rank)
            boundary_pair_rankings.append(
                {
                    "boundary_index": boundary_index,
                    "query_phones": target_pair,
                    "target_pair_rank": target_rank,
                    "pair_type_count": len(ranking),
                    "ranking": ranking,
                }
            )
        return boundary_pair_rankings, target_pair_ranks

    boundary_pair_rankings, target_pair_ranks = build_boundary_pair_rankings(
        query_transitions, False
    )
    joint_boundary_pair_rankings = None
    joint_target_pair_ranks = None
    joint_boundary_pair_swd_rankings = None
    joint_target_pair_swd_ranks = None
    if query_joint_transitions is not None:
        joint_boundary_pair_rankings, joint_target_pair_ranks = (
            build_boundary_pair_rankings(query_joint_transitions, True)
        )
        reference_pair_clouds = []
        for utterance in utterances:
            windows = _window_list(
                args.arctic_root, args.reference_speaker, utterance
            )
            for start, (left, right) in enumerate(zip(windows, windows[1:])):
                ordered = lane_chunks(
                    joint_gauge_chunks(
                        (get_raw_cloud(left), get_raw_cloud(right)),
                        args.points_per_phone,
                    )
                )
                reference_pair_clouds.append(
                    (
                        (left.label, right.label),
                        f"{args.reference_speaker}:{utterance}:{start}",
                        ordered,
                    )
                )
        joint_boundary_pair_swd_rankings = []
        joint_target_pair_swd_ranks = []
        for boundary_index, pair in enumerate(
            zip(query_raw_clouds, query_raw_clouds[1:])
        ):
            query_pair = lane_chunks(
                joint_gauge_chunks(pair, args.points_per_phone)
            )
            best_by_pair: dict[tuple[str, str], dict] = {}
            for labels, witness_key, reference_pair in reference_pair_clouds:
                distance = sliced_wasserstein_distance(
                    query_pair, reference_pair, distance_config
                )
                witness = {
                    "phones": list(labels),
                    "distance": distance,
                    "witness": witness_key,
                }
                current = best_by_pair.get(labels)
                if current is None or (distance, witness_key) < (
                    current["distance"],
                    current["witness"],
                ):
                    best_by_pair[labels] = witness
            ranking = sorted(
                best_by_pair.values(),
                key=lambda item: (item["distance"], item["phones"]),
            )
            for rank, item in enumerate(ranking, start=1):
                item["rank"] = rank
            target_pair = [
                query_span[boundary_index].label,
                query_span[boundary_index + 1].label,
            ]
            target_rank = next(
                (
                    item["rank"]
                    for item in ranking
                    if item["phones"] == target_pair
                ),
                None,
            )
            joint_target_pair_swd_ranks.append(target_rank)
            joint_boundary_pair_swd_rankings.append(
                {
                    "boundary_index": boundary_index,
                    "query_phones": target_pair,
                    "target_pair_rank": target_rank,
                    "pair_type_count": len(ranking),
                    "ranking": ranking,
                }
            )
    summary["target_boundary_pair_ranks"] = target_pair_ranks
    if joint_target_pair_ranks is not None:
        summary["joint_target_boundary_pair_ranks"] = joint_target_pair_ranks
    if joint_target_pair_swd_ranks is not None:
        summary["joint_target_boundary_pair_swd_ranks"] = (
            joint_target_pair_swd_ranks
        )

    result = {
        "method": "ordered_whole_span_occupation_geometry_battery",
        "query": {
            "speaker": args.query_speaker,
            "utterance": args.query_utterance,
            "start_index": args.query_start_index,
            "phones": [window.label for window in query_span],
            "seconds": [query_span[0].seconds0, query_span[-1].seconds1],
        },
        "reference": {
            "speaker": args.reference_speaker,
            "utterances": list(utterances),
            "contiguous_only": True,
        },
        "pipeline": {
            "point_cloud": asdict(cloud_config),
            "support_geometry": asdict(support_config),
            "distance": asdict(distance_config),
            "points_per_phone": args.points_per_phone,
            "ordered_lanes": args.phone_count,
            "joint_frequency_gauge": bool(args.raw_cache is not None),
        },
        "summary": summary,
        "top_rankings": rankings,
        "boundary_pair_rankings": boundary_pair_rankings,
        "joint_boundary_pair_rankings": joint_boundary_pair_rankings,
        "joint_boundary_pair_swd_rankings": joint_boundary_pair_swd_rankings,
        "spans": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
