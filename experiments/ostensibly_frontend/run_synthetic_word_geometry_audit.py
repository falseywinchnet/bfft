#!/usr/bin/env python3
"""Rank routed words by a leave-sentence-out synthetic occupation field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
    sliced_wasserstein_projection,
    sliced_wasserstein_projection_distance,
)
from .run_occupation_context_battery import _cache_name, _subset, _window_list
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks
from .word_lattice import (
    ARPABET_39,
    PronunciationProposalIndex,
    boundary_evidence_from_span_result,
    parse_cmudict,
    proposal_sequence_audit,
    provenance_evidence_from_context_result,
)


def build_phone_prototype(
    clouds: tuple[np.ndarray, ...],
    examples_per_phone: int,
    points_per_example: int,
) -> np.ndarray:
    """Build an equal-mass deterministic raw-cloud prototype."""

    if not clouds or examples_per_phone < 1 or points_per_example < 1:
        raise ValueError("phone prototype requires examples and positive counts")
    if len(clouds) >= examples_per_phone:
        indices = np.linspace(
            0, len(clouds) - 1, examples_per_phone, dtype=np.int64
        )
    else:
        indices = np.arange(examples_per_phone, dtype=np.int64) % len(clouds)
    chunks = []
    for index in indices:
        chunk = _subset(
            np.asarray(clouds[int(index)], dtype=np.float64), points_per_example
        ).copy()
        if chunk.ndim != 2 or chunk.shape[1] != 3 or not chunk.size:
            raise ValueError("prototype example is not a nonempty point cloud")
        chunk[:, 1] = (
            rankdata(chunk[:, 1], method="average") - 0.5
        ) / chunk.shape[0]
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def build_word_realizations(
    slot_clouds: tuple[tuple[np.ndarray, ...], ...],
    realization_count: int,
    points_per_slot: int,
) -> tuple[np.ndarray, ...]:
    """Build complete jointly gauged word realizations.

    Each realization chooses one raw occurrence independently in every phone
    slot. Frequency is canonicalized only after those occurrences form a
    complete word, retaining relative between-phone placement instead of
    pooling it away in independent phone prototypes.
    """

    if not slot_clouds or realization_count < 1 or points_per_slot < 1:
        raise ValueError("word ensemble requires slots and positive counts")
    if any(not pool for pool in slot_clouds):
        raise ValueError("word ensemble contains an empty phone pool")
    schedules = []
    for slot, pool in enumerate(slot_clouds):
        if len(pool) >= realization_count:
            schedule = np.linspace(
                0, len(pool) - 1, realization_count, dtype=np.int64
            )
        else:
            schedule = np.arange(realization_count, dtype=np.int64) % len(pool)
        schedules.append(np.roll(schedule, slot))
    realizations = []
    for realization in range(realization_count):
        chunks = tuple(
            pool[int(schedules[slot][realization])]
            for slot, pool in enumerate(slot_clouds)
        )
        realizations.append(
            lane_chunks(joint_gauge_chunks(chunks, points_per_slot))
        )
    return tuple(realizations)


def build_word_ensemble(
    slot_clouds: tuple[tuple[np.ndarray, ...], ...],
    realization_count: int,
    points_per_slot: int,
) -> np.ndarray:
    """Superimpose complete jointly gauged word realizations."""

    return np.concatenate(
        build_word_realizations(
            slot_clouds, realization_count, points_per_slot
        ),
        axis=0,
    )


def quantile_affine_registration(
    source_rows: np.ndarray, target_rows: np.ndarray
) -> tuple[float, float]:
    """Fit a positive affine chart map between two row distributions."""

    source = np.asarray(source_rows, dtype=np.float64).reshape(-1)
    target = np.asarray(target_rows, dtype=np.float64).reshape(-1)
    if not source.size or not target.size:
        raise ValueError("chart registration requires nonempty row samples")
    quantiles = np.linspace(0.01, 0.99, 99)
    left = np.quantile(source, quantiles)
    right = np.quantile(target, quantiles)
    centered = left - np.mean(left)
    denominator = float(np.dot(centered, centered))
    scale = (
        float(np.dot(centered, right - np.mean(right)) / denominator)
        if denominator > 1e-15
        else 1.0
    )
    scale = float(np.clip(scale, 0.25, 4.0))
    shift = float(np.median(right - scale * left))
    return scale, shift


def quantile_affine_residual(
    source_rows: np.ndarray, target_rows: np.ndarray
) -> float:
    """Return symmetric normalized error after the best affine chart match."""

    def directed(left_rows: np.ndarray, right_rows: np.ndarray) -> float:
        quantiles = np.linspace(0.01, 0.99, 99)
        left = np.quantile(np.asarray(left_rows, dtype=np.float64), quantiles)
        right = np.quantile(np.asarray(right_rows, dtype=np.float64), quantiles)
        scale, shift = quantile_affine_registration(left_rows, right_rows)
        residual = right - (scale * left + shift)
        spread = float(np.quantile(right, 0.9) - np.quantile(right, 0.1))
        return float(np.sqrt(np.mean(residual * residual)) / max(spread, 1e-12))

    return 0.5 * (
        directed(source_rows, target_rows)
        + directed(target_rows, source_rows)
    )


def select_coherent_pair_witnesses(
    options: tuple[tuple[tuple[str, float], ...], ...],
    cloud_loader,
    coherence_weight: float = 1.0,
) -> tuple[str, ...]:
    """Choose a minimum-cost diphone path with compatible overlap charts.

    Unary costs preserve each boundary's query ranking.  Pairwise costs measure
    the affine-registration residual between the two occurrences of the phone
    shared by adjacent diphones.
    """

    if not options or any(not pool for pool in options):
        raise ValueError("coherent witness selection requires nonempty pools")
    if coherence_weight < 0.0:
        raise ValueError("coherence weight cannot be negative")

    unary_costs = []
    for pool in options:
        distances = np.asarray([distance for _, distance in pool], dtype=np.float64)
        floor = float(np.min(distances))
        scale = float(np.quantile(distances, 0.75) - floor)
        if scale <= 1e-12:
            scale = max(float(np.max(distances) - floor), 1.0)
        unary_costs.append((distances - floor) / scale)

    costs = unary_costs[0].copy()
    backpointers: list[np.ndarray] = []
    for boundary in range(1, len(options)):
        previous = options[boundary - 1]
        current = options[boundary]
        next_costs = np.empty(len(current), dtype=np.float64)
        pointers = np.empty(len(current), dtype=np.int64)
        for right_index, (right_witness, _) in enumerate(current):
            right_left_cloud = cloud_loader(right_witness)[0]
            transition_costs = np.empty(len(previous), dtype=np.float64)
            for left_index, (left_witness, _) in enumerate(previous):
                left_right_cloud = cloud_loader(left_witness)[1]
                transition_costs[left_index] = quantile_affine_residual(
                    left_right_cloud[:, 0], right_left_cloud[:, 0]
                )
            candidates = costs + coherence_weight * transition_costs
            pointer = int(np.argmin(candidates))
            pointers[right_index] = pointer
            next_costs[right_index] = (
                candidates[pointer] + unary_costs[boundary][right_index]
            )
        backpointers.append(pointers)
        costs = next_costs

    index = int(np.argmin(costs))
    selected = [options[-1][index][0]]
    for boundary in range(len(options) - 2, -1, -1):
        index = int(backpointers[boundary][index])
        selected.append(options[boundary][index][0])
    selected.reverse()
    return tuple(selected)


def stitch_pair_witnesses(
    pair_clouds: tuple[tuple[np.ndarray, np.ndarray], ...],
    points_per_phone: int,
) -> np.ndarray:
    """Stitch overlapping diphone charts into one coherent word cloud."""

    if not pair_clouds or points_per_phone < 2:
        raise ValueError("stitching requires pair witnesses and phone mass")
    charts = [
        [chunk.copy() for chunk in joint_gauge_chunks(pair, points_per_phone)]
        for pair in pair_clouds
    ]
    for index in range(1, len(charts)):
        scale, shift = quantile_affine_registration(
            charts[index][0][:, 0], charts[index - 1][1][:, 0]
        )
        for chunk in charts[index]:
            chunk[:, 0] = scale * chunk[:, 0] + shift

    slots = [charts[0][0]]
    left_count = points_per_phone // 2
    right_count = points_per_phone - left_count
    for index in range(1, len(charts)):
        slots.append(
            np.concatenate(
                (
                    _subset(charts[index - 1][1], left_count),
                    _subset(charts[index][0], right_count),
                ),
                axis=0,
            )
        )
    slots.append(charts[-1][1])

    combined_rows = np.concatenate([slot[:, 0] for slot in slots])
    canonical_rows = (
        rankdata(combined_rows, method="average") - 0.5
    ) / combined_rows.size
    offset = 0
    for slot in slots:
        slot[:, 0] = canonical_rows[offset : offset + slot.shape[0]]
        offset += slot.shape[0]
    return lane_chunks(tuple(slots))


def ordered_lane_windows(
    cloud: np.ndarray, phone_count: int, window_width: int
) -> tuple[np.ndarray, ...]:
    """Extract contiguous lane windows without changing the shared row gauge."""

    points = np.asarray(cloud, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or phone_count < 1
        or window_width < 1
        or window_width > phone_count
    ):
        raise ValueError("lane windows require a valid ordered word cloud")
    output = []
    for start in range(phone_count - window_width + 1):
        lower = start / phone_count
        upper = (start + window_width) / phone_count
        if start + window_width == phone_count:
            mask = (points[:, 1] >= lower) & (points[:, 1] <= upper)
        else:
            mask = (points[:, 1] >= lower) & (points[:, 1] < upper)
        window = points[mask].copy()
        if not window.size:
            raise ValueError("ordered word cloud contains an empty lane window")
        window[:, 1] = (window[:, 1] - lower) / (upper - lower)
        output.append(window)
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("span_battery", type=Path)
    parser.add_argument("target_word")
    parser.add_argument("phone_results", nargs="+", type=Path)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--top-k-per-channel", type=int, default=8)
    parser.add_argument("--maximum-uncovered-phones", type=int, default=0)
    parser.add_argument("--maximum-boundary-rank", type=int)
    parser.add_argument("--examples-per-phone", type=int, default=8)
    parser.add_argument("--points-per-example", type=int, default=512)
    parser.add_argument(
        "--prototype-mode",
        choices=(
            "pooled-phone",
            "word-ensemble",
            "boundary-witness",
            "stitched-boundary-witness",
        ),
        default="pooled-phone",
    )
    parser.add_argument("--atlas-field", default="joint_boundary_pair_rankings")
    parser.add_argument("--projection-count", type=int, default=64)
    parser.add_argument("--skip-support", action="store_true")
    parser.add_argument("--score-realizations", action="store_true")
    parser.add_argument("--score-lane-windows", action="store_true")
    parser.add_argument("--report-count", type=int, default=20)
    parser.add_argument("--retain-all-scores", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    span_document = json.loads(args.span_battery.read_text())
    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(), frozenset(ARPABET_39)
    )
    phone_documents = [json.loads(path.read_text()) for path in args.phone_results]
    evidence = tuple(
        provenance_evidence_from_context_result(
            document, args.top_k_per_channel
        )
        for document in phone_documents
    )
    boundaries = boundary_evidence_from_span_result(
        span_document, args.atlas_field
    )
    witness_maps = tuple(
        {
            tuple(item["phones"]): str(item["witness"])
            for item in boundary["ranking"]
        }
        for boundary in span_document[args.atlas_field]
    )
    index_started = perf_counter()
    index = PronunciationProposalIndex(pronunciations)
    index_seconds = perf_counter() - index_started
    route_started = perf_counter()
    route = index.query(
        evidence,
        boundaries,
        maximum_uncovered_phones=args.maximum_uncovered_phones,
        maximum_boundary_rank=args.maximum_boundary_rank,
    )
    route_seconds = perf_counter() - route_started
    query_phones = tuple(str(phone) for phone in span_document["query"]["phones"])
    sequence_audit = proposal_sequence_audit(
        evidence,
        boundaries,
        query_phones,
        maximum_uncovered_phones=args.maximum_uncovered_phones,
        maximum_boundary_rank=args.maximum_boundary_rank,
    )
    target_class = next(
        (
            (ordinal, item)
            for ordinal, item in enumerate(index.classes)
            if item.phones == query_phones
        ),
        None,
    )

    point_config = PointCloudConfig(point_count=32768)
    payload = {
        "rows": 128,
        "noise_db": 8.0,
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
            raise FileNotFoundError(
                f"raw cloud missing for {window.key}; run the span battery with --raw-cache"
            )
        return np.asarray(np.load(path), dtype=np.float64)

    query = span_document["query"]
    query_windows = _window_list(
        args.arctic_root, query["speaker"], query["utterance"]
    )
    query_start = int(query["start_index"])
    phone_count = len(query["phones"])
    query_span = tuple(query_windows[query_start : query_start + phone_count])
    points_per_phone = args.examples_per_phone * args.points_per_example
    query_raw = tuple(load_raw(window) for window in query_span)
    query_realization = lane_chunks(
        joint_gauge_chunks(query_raw, args.points_per_example)
    )
    if args.prototype_mode == "word-ensemble":
        query_ordered = build_word_ensemble(
            tuple((cloud,) for cloud in query_raw),
            args.examples_per_phone,
            args.points_per_example,
        )
    else:
        query_ordered = lane_chunks(
            joint_gauge_chunks(query_raw, points_per_phone)
        )
    lane_width_names = {1: "phone", 2: "diphone", 3: "triphone"}
    query_lane_windows = (
        {
            width: ordered_lane_windows(query_ordered, phone_count, width)
            for width in range(1, min(3, phone_count) + 1)
        }
        if args.score_lane_windows
        else {}
    )

    reference = span_document["reference"]
    prototype_utterances = tuple(
        utterance
        for utterance in reference["utterances"]
        if utterance != query["utterance"]
    )
    examples: dict[str, list[np.ndarray]] = {}
    for utterance in prototype_utterances:
        for window in _window_list(
            args.arctic_root, reference["speaker"], utterance
        ):
            examples.setdefault(window.label, []).append(load_raw(window))
    prototypes = {
        phone: build_phone_prototype(
            tuple(clouds), args.examples_per_phone, args.points_per_example
        )
        for phone, clouds in examples.items()
    }

    window_cache = {}

    def witness_pair_clouds(
        phones: tuple[str, ...],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        pairs = []
        for boundary_index, pair in enumerate(zip(phones, phones[1:])):
            witness = witness_maps[boundary_index][pair]
            speaker, utterance, start_text = witness.split(":")
            cache_key = (speaker, utterance)
            if cache_key not in window_cache:
                window_cache[cache_key] = _window_list(
                    args.arctic_root, speaker, utterance
                )
            windows = window_cache[cache_key]
            start = int(start_text)
            pairs.append(
                (load_raw(windows[start]), load_raw(windows[start + 1]))
            )
        return tuple(pairs)

    def witness_slot_clouds(
        phones: tuple[str, ...]
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        slots: list[list[np.ndarray]] = [[] for _ in phones]
        for boundary_index, (left, right) in enumerate(
            witness_pair_clouds(phones)
        ):
            slots[boundary_index].append(left)
            slots[boundary_index + 1].append(right)
        return tuple(tuple(slot) for slot in slots)

    distance_config = CloudFitConfig(
        fit_point_count=points_per_phone * phone_count,
        row_metric_scale=1.0,
        frame_metric_scale=1.0,
        height_metric_scale=1e-6,
        distance_mode="sliced_wasserstein",
        sliced_projection_count=args.projection_count,
        global_iterations=1,
        population_size=4,
    )
    support_config = SupportGeometryConfig(minimum_cell_count=1)
    projection_started = perf_counter()
    query_projection = sliced_wasserstein_projection(query_ordered, distance_config)
    query_lane_projections = {
        width: tuple(
            sliced_wasserstein_projection(window, distance_config)
            for window in windows
        )
        for width, windows in query_lane_windows.items()
    }
    query_realization_projection = (
        sliced_wasserstein_projection(query_realization, distance_config)
        if args.score_realizations
        else None
    )
    query_projection_seconds = perf_counter() - projection_started
    score_started = perf_counter()
    swd_seconds = 0.0
    support_seconds = 0.0
    realization_seconds = 0.0
    lane_window_seconds = 0.0
    scored = []
    missing_prototype_candidates = 0
    for ordinal in route.boundary_candidates:
        item = index.classes[ordinal]
        if (
            args.prototype_mode
            not in ("boundary-witness", "stitched-boundary-witness")
            and any(phone not in prototypes for phone in item.phones)
        ):
            missing_prototype_candidates += 1
            continue
        if args.prototype_mode == "stitched-boundary-witness":
            candidate_ordered = stitch_pair_witnesses(
                witness_pair_clouds(item.phones), points_per_phone
            )
            candidate_realizations = ()
        elif args.prototype_mode == "boundary-witness":
            slot_clouds = witness_slot_clouds(item.phones)
            candidate_ordered = lane_chunks(
                joint_gauge_chunks(
                    tuple(
                        build_phone_prototype(
                            pool,
                            args.examples_per_phone,
                            args.points_per_example,
                        )
                        for pool in slot_clouds
                    ),
                    points_per_phone,
                )
            )
            candidate_realizations = ()
        elif args.prototype_mode == "word-ensemble":
            slot_clouds = tuple(tuple(examples[phone]) for phone in item.phones)
            candidate_realizations = build_word_realizations(
                slot_clouds,
                args.examples_per_phone,
                args.points_per_example,
            )
            candidate_ordered = np.concatenate(candidate_realizations, axis=0)
        else:
            candidate_realizations = ()
            candidate_ordered = lane_chunks(
                joint_gauge_chunks(
                    tuple(prototypes[phone] for phone in item.phones),
                    points_per_phone,
                )
            )
        swd_started = perf_counter()
        joint_swd = sliced_wasserstein_projection_distance(
            query_projection,
            sliced_wasserstein_projection(candidate_ordered, distance_config),
        )
        swd_seconds += perf_counter() - swd_started
        row = {
            "phones": list(item.phones),
            "words": list(item.words),
            "joint_swd": joint_swd,
        }
        if args.score_lane_windows:
            lane_started = perf_counter()
            all_local_distances = []
            for width, query_windows_for_width in query_lane_projections.items():
                candidate_windows = ordered_lane_windows(
                    candidate_ordered, phone_count, width
                )
                distances = np.asarray(
                    [
                        sliced_wasserstein_projection_distance(
                            left,
                            sliced_wasserstein_projection(right, distance_config),
                        )
                        for left, right in zip(
                            query_windows_for_width,
                            candidate_windows,
                            strict=True,
                        )
                    ],
                    dtype=np.float64,
                )
                all_local_distances.extend(distances.tolist())
                name = lane_width_names[width]
                row[f"{name}_worst_swd"] = float(np.max(distances))
                row[f"{name}_mean_swd"] = float(np.mean(distances))
            row["local_worst_swd"] = float(np.max(all_local_distances))
            row["local_mean_swd"] = float(np.mean(all_local_distances))
            lane_window_seconds += perf_counter() - lane_started
        if args.score_realizations:
            if not candidate_realizations:
                raise ValueError(
                    "realization scoring requires --prototype-mode word-ensemble"
                )
            realization_started = perf_counter()
            realization_distances = np.asarray(
                [
                    sliced_wasserstein_projection_distance(
                        query_realization_projection,
                        sliced_wasserstein_projection(realization, distance_config),
                    )
                    for realization in candidate_realizations
                ],
                dtype=np.float64,
            )
            realization_seconds += perf_counter() - realization_started
            row.update(
                {
                    "realization_min_swd": float(np.min(realization_distances)),
                    "realization_q25_swd": float(
                        np.quantile(realization_distances, 0.25)
                    ),
                    "realization_median_swd": float(
                        np.median(realization_distances)
                    ),
                    "realization_mean_swd": float(
                        np.mean(realization_distances)
                    ),
                }
            )
        if not args.skip_support:
            support_started = perf_counter()
            row["joint_support_distance"] = jittered_support_distance(
                query_ordered, candidate_ordered, support_config
            )
            support_seconds += perf_counter() - support_started
        scored.append(row)
    score_seconds = perf_counter() - score_started
    score_fields = ["joint_swd"]
    if args.score_realizations:
        score_fields.extend(
            (
                "realization_min_swd",
                "realization_q25_swd",
                "realization_median_swd",
                "realization_mean_swd",
            )
        )
    if args.score_lane_windows:
        for width in query_lane_windows:
            name = lane_width_names[width]
            score_fields.extend((f"{name}_worst_swd", f"{name}_mean_swd"))
        score_fields.extend(("local_worst_swd", "local_mean_swd"))
    if not args.skip_support:
        score_fields.append("joint_support_distance")
    for field in score_fields:
        order = sorted(scored, key=lambda item: (item[field], item["phones"]))
        for rank, item in enumerate(order, start=1):
            item[f"{field}_rank"] = rank
    target_word = args.target_word.lower()
    target = next(
        (item for item in scored if target_word in item["words"]), None
    )
    swd_ranking = sorted(scored, key=lambda item: (item["joint_swd"], item["phones"]))
    support_ranking = (
        sorted(
            scored,
            key=lambda item: (item["joint_support_distance"], item["phones"]),
        )
        if not args.skip_support
        else []
    )
    realization_rankings = {
        field: sorted(scored, key=lambda item: (item[field], item["phones"]))
        for field in score_fields
        if field.startswith("realization_")
    }
    lane_rankings = {
        field: sorted(scored, key=lambda item: (item[field], item["phones"]))
        for field in score_fields
        if "phone_" in field or field.startswith("local_")
    }
    result = {
        "method": "leave_sentence_out_synthetic_phone_prototype_word_geometry",
        "dictionary": {
            "path": str(args.cmudict),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
            "pronunciation_count": len(pronunciations),
        },
        "query": {
            "target_word": target_word,
            "phones": query["phones"],
            "speaker": query["speaker"],
            "utterance": query["utterance"],
        },
        "prototype_bank": {
            "speaker": reference["speaker"],
            "utterances": list(prototype_utterances),
            "excluded_query_utterance": query["utterance"],
            "examples_per_phone": args.examples_per_phone,
            "points_per_example": args.points_per_example,
            "projection_count": args.projection_count,
            "mode": args.prototype_mode,
            "phone_count": len(prototypes),
        },
        "route": {
            "top_k_per_channel": args.top_k_per_channel,
            "maximum_uncovered_phones": args.maximum_uncovered_phones,
            "maximum_boundary_rank": args.maximum_boundary_rank,
            "atlas_field": args.atlas_field,
            "phone_candidate_count": len(route.phone_candidates),
            "boundary_candidate_count": len(route.boundary_candidates),
            "missing_prototype_candidates": missing_prototype_candidates,
            "index_seconds": index_seconds,
            "route_seconds": route_seconds,
            "target_audit": {
                "exact_dictionary_anchor": target_class is not None,
                "dictionary_words": (
                    list(target_class[1].words) if target_class is not None else []
                ),
                "phone_channel_ranks": [
                    dict(ranks) for ranks in sequence_audit.phone_channel_ranks
                ],
                "phone_position_admitted": list(
                    sequence_audit.phone_position_admitted
                ),
                "admitted_phone_count": sequence_audit.admitted_phone_count,
                "phone_stage_admitted": sequence_audit.phone_stage_admitted,
                "boundary_ranks": list(sequence_audit.boundary_ranks),
                "boundary_position_admitted": list(
                    sequence_audit.boundary_position_admitted
                ),
                "boundary_stage_admitted": sequence_audit.boundary_stage_admitted,
                "present_in_phone_candidates": (
                    target_class is not None
                    and target_class[0] in route.phone_candidates
                ),
                "present_in_boundary_candidates": (
                    target_class is not None
                    and target_class[0] in route.boundary_candidates
                ),
            },
        },
        "score": {
            "candidate_count": len(scored),
            "elapsed_seconds": score_seconds,
            "query_projection_seconds": query_projection_seconds,
            "swd_seconds": swd_seconds,
            "support_seconds": support_seconds,
            "realization_seconds": realization_seconds,
            "lane_window_seconds": lane_window_seconds,
            "target": target,
            "top_joint_swd": swd_ranking[: args.report_count],
            "top_joint_support": support_ranking[: args.report_count],
            "top_realizations": {
                field: ranking[: args.report_count]
                for field, ranking in realization_rankings.items()
            },
            "top_lane_windows": {
                field: ranking[: args.report_count]
                for field, ranking in lane_rankings.items()
            },
            "all_scores": scored if args.retain_all_scores else None,
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "target_word": target_word,
                "target": target,
                "routed_candidates": len(route.boundary_candidates),
                "scored_candidates": len(scored),
                "score_seconds": score_seconds,
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
