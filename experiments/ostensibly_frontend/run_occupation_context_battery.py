#!/usr/bin/env python3
"""Compare isolated and ordered three-phone occupation evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

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
from .run_arctic_calibration import DEFAULT_UTTERANCES, _timed_phones
from .run_occupation_cloud_battery import (
    ReferenceWindow,
    _load_channel_baseline,
    _trace_to_cloud,
    _window_trace,
    select_duration_matched_windows,
)
from .run_whole_patch_inventory import LABELS


SILENCE_LABEL = "<sil>"


def silence_occupation_cloud(point_count: int) -> np.ndarray:
    """Return a deterministic zero-mass sentinel in copula coordinates."""

    if point_count < 1:
        raise ValueError("silence cloud requires positive point count")
    frame = (np.arange(point_count, dtype=np.float64) + 0.5) / point_count
    return np.column_stack(
        (
            np.full(point_count, 0.5, dtype=np.float64),
            frame,
            np.zeros(point_count, dtype=np.float64),
        )
    )


def edge_context_windows(
    windows: list[ReferenceWindow],
    center_index: int,
    use_left: bool,
    use_right: bool,
) -> tuple[ReferenceWindow | None, ReferenceWindow, ReferenceWindow | None]:
    """Apply query-relative context topology to one reference center."""

    if not 0 <= center_index < len(windows):
        raise ValueError("context center index is outside the utterance")
    if use_left and center_index == 0:
        raise ValueError("context requires a missing left phone")
    if use_right and center_index + 1 >= len(windows):
        raise ValueError("context requires a missing right phone")
    return (
        windows[center_index - 1] if use_left else None,
        windows[center_index],
        windows[center_index + 1] if use_right else None,
    )


def _window_list(root: Path, speaker: str, utterance: str) -> list[ReferenceWindow]:
    path = root / speaker / "lab" / f"{utterance}.lab"
    return [
        ReferenceWindow(label, speaker, utterance, start, stop, raw)
        for label, start, stop, raw in _timed_phones(path)
        if label in LABELS
    ]


def _find_window_index(
    windows: list[ReferenceWindow], target: ReferenceWindow
) -> int:
    return next(index for index, item in enumerate(windows) if item.key == target.key)


def _subset(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
    return points[indices]


def ordered_context_cloud(
    clouds: tuple[np.ndarray, np.ndarray, np.ndarray],
    points_per_phone: int,
) -> np.ndarray:
    """Place three copula clouds in fixed left/center/right time lanes."""

    if points_per_phone < 1 or len(clouds) != 3:
        raise ValueError("ordered context requires three nonempty phone clouds")
    chunks = []
    for slot, cloud in enumerate(clouds):
        selected = _subset(np.asarray(cloud, dtype=np.float64), points_per_phone).copy()
        if selected.ndim != 2 or selected.shape[1] != 3 or not selected.size:
            raise ValueError("ordered context requires three nonempty phone clouds")
        selected[:, 1] = (slot + selected[:, 1]) / 3.0
        chunks.append(selected)
    return np.concatenate(chunks, axis=0)


def boundary_transition_law(
    left: np.ndarray,
    right: np.ndarray,
    quantile_count: int = 128,
    boundary_fraction: float = 0.25,
) -> np.ndarray:
    """Return translation-quotiented row transport across one phone boundary."""

    if quantile_count < 8 or not 0.0 < boundary_fraction <= 0.5:
        raise ValueError("invalid boundary transport resolution")
    departing = np.asarray(left, dtype=np.float64)
    arriving = np.asarray(right, dtype=np.float64)
    departing_rows = departing[
        departing[:, 1] >= 1.0 - boundary_fraction, 0
    ]
    arriving_rows = arriving[arriving[:, 1] <= boundary_fraction, 0]
    if not departing_rows.size or not arriving_rows.size:
        raise ValueError("phone cloud has no boundary occupation")
    quantiles = (np.arange(quantile_count, dtype=np.float64) + 0.5) / quantile_count
    displacement = np.quantile(arriving_rows, quantiles) - np.quantile(
        departing_rows, quantiles
    )
    return displacement - np.median(displacement)


def boundary_transition_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(difference * difference)))


def _label_ranking(rows: list[dict], field: str) -> list[dict]:
    ranking = []
    for label in sorted({row["label"] for row in rows}):
        best = min(
            (row for row in rows if row["label"] == label),
            key=lambda row: (row[field], row["key"]),
        )
        ranking.append(
            {
                "phone": label,
                "distance": best[field],
                "witness_key": best["key"],
                "context_labels": best["context_labels"],
            }
        )
    return sorted(ranking, key=lambda row: (row["distance"], row["phone"]))


def _rank(ranking: list[dict], label: str) -> int:
    return next(index + 1 for index, row in enumerate(ranking) if row["phone"] == label)


def proposal_shortlist(
    rankings: dict[str, list[dict]], top_k_per_channel: int
) -> list[dict]:
    """Return a provenance-preserving union of independent channel proposals."""

    if not rankings or top_k_per_channel < 1:
        raise ValueError("proposal shortlist needs rankings and positive top-k")
    proposals: dict[str, dict] = {}
    for channel, ranking in rankings.items():
        for rank, item in enumerate(ranking[:top_k_per_channel], start=1):
            proposal = proposals.setdefault(
                item["phone"], {"phone": item["phone"], "channel_ranks": {}}
            )
            proposal["channel_ranks"][channel] = rank
    for proposal in proposals.values():
        ranks = tuple(proposal["channel_ranks"].values())
        proposal["best_channel_rank"] = min(ranks)
        proposal["supporting_channel_count"] = len(ranks)
    return sorted(
        proposals.values(),
        key=lambda item: (
            item["best_channel_rank"],
            -item["supporting_channel_count"],
            item["phone"],
        ),
    )


def _robust_scale(rows: list[dict], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return max(float(np.median(values)), 1e-12)


def add_empirical_rank_fusion(
    rows: list[dict],
    fields: tuple[str, ...],
    output_field: str = "conservative_rank_fusion",
) -> None:
    """Fuse channels by the worst empirical percentile, without phone labels.

    Each channel is converted to its deterministic mid-rank percentile over the
    candidate inventory.  Taking the maximum is an intersection rule: a
    candidate only scores well when none of its evidence channels rejects it.
    """

    if not rows or not fields:
        raise ValueError("rank fusion requires rows and evidence fields")
    count = len(rows)
    percentiles = np.empty((count, len(fields)), dtype=np.float64)
    for column, field in enumerate(fields):
        order = sorted(
            range(count), key=lambda index: (rows[index][field], rows[index]["key"])
        )
        for rank, index in enumerate(order, start=1):
            percentiles[index, column] = (rank - 0.5) / count
    for index, row in enumerate(rows):
        row[f"{output_field}_channels"] = {
            field: float(percentiles[index, column])
            for column, field in enumerate(fields)
        }
        row[output_field] = float(np.max(percentiles[index]))


def _instance_rank(rows: list[dict], field: str, marker: str) -> int:
    ordered = sorted(rows, key=lambda row: (row[field], row["key"]))
    return next(index + 1 for index, row in enumerate(ordered) if row[marker])


def _cache_name(window: ReferenceWindow, payload: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{window.speaker}_{window.utterance}_{window.label}_{digest}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--query-speaker", default="slt")
    parser.add_argument("--reference-speaker", default="bdl")
    parser.add_argument("--query-utterance", default="arctic_a0019")
    parser.add_argument("--query-phone", default="R")
    parser.add_argument("--query-phone-ordinal", type=int, default=2)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--exclude-query-reference", action="store_true")
    parser.add_argument("--witnesses-per-label", type=int, default=2)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--distance-points", type=int, default=2048)
    parser.add_argument("--output-name", default="occupation_context_battery.json")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.query_phone_ordinal < 1 or args.witnesses_per_label < 1:
        raise ValueError("phone ordinal and witness count must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache if args.cache is not None else args.out / "cloud_cache"
    cache.mkdir(parents=True, exist_ok=True)
    utterances = tuple(args.utterances.split(","))
    if args.exclude_query_reference:
        utterances = tuple(
            utterance
            for utterance in utterances
            if utterance != args.query_utterance
        )
    if not utterances:
        raise ValueError("context battery requires reference utterances")

    morphology_config = DepthmapGeometryConfig()
    ridge_config = HessianRidgeConfig()
    climber_config = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    support_config = SupportGeometryConfig()
    distance_config = CloudFitConfig(
        fit_point_count=args.distance_points,
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=1e-6,
        distance_mode="sliced_wasserstein",
        sliced_projection_count=64,
        global_iterations=1,
        population_size=4,
    )

    phone_lists = {
        (args.reference_speaker, utterance): _window_list(
            args.arctic_root, args.reference_speaker, utterance
        )
        for utterance in utterances
    }
    query_windows = _window_list(
        args.arctic_root, args.query_speaker, args.query_utterance
    )
    query_occurrences = [
        (index, item)
        for index, item in enumerate(query_windows)
        if item.label == args.query_phone
    ]
    if args.query_phone_ordinal > len(query_occurrences):
        raise ValueError("query phone ordinal exceeds available occurrences")
    query_index, query_center = query_occurrences[args.query_phone_ordinal - 1]
    use_left_context = query_index > 0
    use_right_context = query_index + 1 < len(query_windows)
    query_triplet = edge_context_windows(
        query_windows,
        query_index,
        use_left_context,
        use_right_context,
    )
    corresponding_reference = None
    if (args.reference_speaker, args.query_utterance) in phone_lists:
        corresponding_reference = phone_lists[
            (args.reference_speaker, args.query_utterance)
        ][query_index]
        if corresponding_reference.label != query_center.label:
            raise ValueError(
                "query and reference label sequences do not share the selected phone"
            )

    available = [item for values in phone_lists.values() for item in values]
    forced = (
        (
            corresponding_reference.label,
            corresponding_reference.utterance,
            corresponding_reference.seconds0,
            corresponding_reference.seconds1,
        )
        if corresponding_reference is not None
        else None
    )
    selected = select_duration_matched_windows(
        available,
        query_center.duration,
        args.witnesses_per_label,
        forced=forced,
    )
    selected = [
        item
        for item in selected
        if (
            (
                not use_left_context
                or _find_window_index(
                    phone_lists[(item.speaker, item.utterance)], item
                )
                > 0
            )
            and (
                not use_right_context
                or _find_window_index(
                    phone_lists[(item.speaker, item.utterance)], item
                )
                < len(phone_lists[(item.speaker, item.utterance)]) - 1
            )
        )
    ]

    baselines: dict[tuple[str, str], np.ndarray] = {}
    clouds: dict[tuple[str, str, str, float, float], np.ndarray] = {}
    cache_payload = {
        "rows": args.rows,
        "noise_db": args.noise_db,
        "morphology": asdict(morphology_config),
        "ridge": asdict(ridge_config),
        "climbers": asdict(climber_config),
        "point_cloud": asdict(cloud_config),
    }

    def get_cloud(window: ReferenceWindow) -> np.ndarray:
        if window.key in clouds:
            return clouds[window.key]
        path = cache / _cache_name(window, {**cache_payload, "key": window.key})
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

    silence_cloud = silence_occupation_cloud(args.points)
    query_clouds = tuple(
        silence_cloud.copy() if window is None else get_cloud(window)
        for window in query_triplet
    )
    query_center_cloud = _subset(query_clouds[1], args.distance_points)
    query_context_cloud = ordered_context_cloud(
        query_clouds, args.distance_points
    )
    query_left_transition = boundary_transition_law(
        query_clouds[0], query_clouds[1]
    )
    query_right_transition = boundary_transition_law(
        query_clouds[1], query_clouds[2]
    )

    rows = []
    for index, center in enumerate(selected):
        windows = phone_lists[(center.speaker, center.utterance)]
        center_index = _find_window_index(windows, center)
        context = edge_context_windows(
            windows,
            center_index,
            use_left_context,
            use_right_context,
        )
        reference_clouds = tuple(
            silence_cloud.copy() if window is None else get_cloud(window)
            for window in context
        )
        reference_center_cloud = _subset(
            reference_clouds[1], args.distance_points
        )
        reference_context_cloud = ordered_context_cloud(
            reference_clouds, args.distance_points
        )
        center_distance = sliced_wasserstein_distance(
            query_center_cloud, reference_center_cloud, distance_config
        )
        support_distance = jittered_support_distance(
            query_clouds[1], reference_clouds[1], support_config
        )
        context_distance = sliced_wasserstein_distance(
            query_context_cloud, reference_context_cloud, distance_config
        )
        left_transition_distance = boundary_transition_distance(
            query_left_transition,
            boundary_transition_law(reference_clouds[0], reference_clouds[1]),
        )
        right_transition_distance = boundary_transition_distance(
            query_right_transition,
            boundary_transition_law(reference_clouds[1], reference_clouds[2]),
        )
        key = (
            f"{center.speaker}:{center.utterance}:{center.label}:"
            f"{center.seconds0:.5f}-{center.seconds1:.5f}"
        )
        rows.append(
            {
                "key": key,
                "label": center.label,
                "context_labels": [
                    item.label if item is not None else SILENCE_LABEL
                    for item in context
                ],
                "center_distance": center_distance,
                "support_distance": support_distance,
                "context_distance": context_distance,
                "left_transition_distance": left_transition_distance,
                "right_transition_distance": right_transition_distance,
                "is_corresponding_context": bool(
                    corresponding_reference is not None
                    and center.key == corresponding_reference.key
                ),
            }
        )
        print(
            f"score {index + 1}/{len(selected)} {center.label} "
            f"center={center_distance:.5f} context={context_distance:.5f} "
            f"support={support_distance:.5f} "
            f"transition={0.5 * (left_transition_distance + right_transition_distance):.5f}",
            flush=True,
        )

    scales = {
        field: _robust_scale(rows, field)
        for field in (
            "center_distance",
            "context_distance",
            "left_transition_distance",
            "right_transition_distance",
        )
    }
    for row in rows:
        row["relational_score"] = float(
            np.mean(
                [
                    row["center_distance"] / scales["center_distance"],
                    row["left_transition_distance"]
                    / scales["left_transition_distance"],
                    row["right_transition_distance"]
                    / scales["right_transition_distance"],
                ]
            )
        )
    fused_fields = (
        "center_distance",
        "left_transition_distance",
        "right_transition_distance",
    )
    add_empirical_rank_fusion(rows, fused_fields)
    center_ranking = _label_ranking(rows, "center_distance")
    support_ranking = _label_ranking(rows, "support_distance")
    context_ranking = _label_ranking(rows, "context_distance")
    relational_ranking = _label_ranking(rows, "relational_score")
    fused_ranking = _label_ranking(rows, "conservative_rank_fusion")
    proposal_rankings = {
        "center": center_ranking,
        "ordered_context": context_ranking,
        "boundary_transport": relational_ranking,
        "support_geometry": support_ranking,
    }
    proposal_shortlists = {
        f"top_{top_k}": proposal_shortlist(proposal_rankings, top_k)
        for top_k in (3, 5, 6, 8)
    }
    corresponding = next(
        (row for row in rows if row["is_corresponding_context"]), None
    )
    corresponding_rank = (
        _instance_rank(rows, "context_distance", "is_corresponding_context")
        if corresponding is not None
        else None
    )
    target_label = query_center.label
    best_target = next(
        row for row in context_ranking if row["phone"] == target_label
    )
    best_non_target = next(
        row for row in context_ranking if row["phone"] != target_label
    )
    best_relational_target = next(
        row for row in relational_ranking if row["phone"] == target_label
    )
    best_relational_non_target = next(
        row for row in relational_ranking if row["phone"] != target_label
    )
    best_fused_target = next(
        row for row in fused_ranking if row["phone"] == target_label
    )
    best_fused_non_target = next(
        row for row in fused_ranking if row["phone"] != target_label
    )
    summary = {
        "target_phone": target_label,
        "center_only_target_rank": _rank(center_ranking, target_label),
        "support_target_rank": _rank(support_ranking, target_label),
        "context_target_rank": _rank(context_ranking, target_label),
        "relational_target_rank": _rank(relational_ranking, target_label),
        "conservative_fusion_target_rank": _rank(fused_ranking, target_label),
        "label_count": len(context_ranking),
        "corresponding_context_instance_rank": corresponding_rank,
        "corresponding_support_instance_rank": (
            _instance_rank(rows, "support_distance", "is_corresponding_context")
            if corresponding is not None
            else None
        ),
        "corresponding_relational_instance_rank": (
            _instance_rank(rows, "relational_score", "is_corresponding_context")
            if corresponding is not None
            else None
        ),
        "corresponding_conservative_fusion_instance_rank": (
            _instance_rank(
                rows, "conservative_rank_fusion", "is_corresponding_context"
            )
            if corresponding is not None
            else None
        ),
        "instance_count": len(rows),
        "corresponding_center_distance": (
            corresponding["center_distance"] if corresponding is not None else None
        ),
        "corresponding_context_distance": (
            corresponding["context_distance"] if corresponding is not None else None
        ),
        "best_target_context_distance": best_target["distance"],
        "best_non_target_phone": best_non_target["phone"],
        "best_non_target_context_distance": best_non_target["distance"],
        "target_margin_vs_best_non_target": (
            best_non_target["distance"] - best_target["distance"]
        ),
        "best_target_relational_score": best_relational_target["distance"],
        "best_relational_non_target_phone": best_relational_non_target["phone"],
        "relational_margin_vs_best_non_target": (
            best_relational_non_target["distance"]
            - best_relational_target["distance"]
        ),
        "best_target_conservative_fusion": best_fused_target["distance"],
        "best_conservative_fusion_non_target_phone": best_fused_non_target["phone"],
        "conservative_fusion_margin_vs_best_non_target": (
            best_fused_non_target["distance"] - best_fused_target["distance"]
        ),
    }
    result = {
        "method": "ordered_three_phone_marginal_copula_transport",
        "query": {
            "speaker": args.query_speaker,
            "utterance": args.query_utterance,
            "center_phone": query_center.label,
            "context_labels": [
                item.label if item is not None else SILENCE_LABEL
                for item in query_triplet
            ],
            "seconds": [
                [item.seconds0, item.seconds1]
                if item is not None
                else [None, None]
                for item in query_triplet
            ],
        },
        "selection": {
            "reference_speaker": args.reference_speaker,
            "utterances": list(utterances),
            "excludes_query_utterance": args.query_utterance not in utterances,
            "uses_geometry": False,
            "witnesses_per_label": args.witnesses_per_label,
        },
        "pipeline": {
            "morphology": asdict(morphology_config),
            "ridge": asdict(ridge_config),
            "climbers": asdict(climber_config),
            "point_cloud": asdict(cloud_config),
            "support_geometry": asdict(support_config),
            "distance": asdict(distance_config),
            "marginal_copula": True,
            "ordered_context_lanes": 3,
            "context_topology": {
                "uses_left_phone": use_left_context,
                "uses_right_phone": use_right_context,
                "silence_label": SILENCE_LABEL,
            },
            "boundary_transport_quantiles": 128,
            "relational_channel_scales": scales,
            "conservative_rank_fusion": {
                "fields": list(fused_fields),
                "operation": "maximum empirical mid-rank percentile",
                "uses_phone_labels": False,
            },
            "proposal_channels": list(proposal_rankings),
        },
        "summary": summary,
        "center_label_ranking": center_ranking,
        "support_label_ranking": support_ranking,
        "context_label_ranking": context_ranking,
        "relational_label_ranking": relational_ranking,
        "conservative_fusion_label_ranking": fused_ranking,
        "proposal_shortlists": proposal_shortlists,
        "instances": rows,
    }
    output = args.out / args.output_name
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
