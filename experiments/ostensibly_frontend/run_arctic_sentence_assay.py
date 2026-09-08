#!/usr/bin/env python3
"""Locate the acoustic-to-word bottleneck on timestamped held-out speech."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_boundary_atlas import load_boundary_atlas
from .compiled_phone_atlas import load_phone_atlas
from .conditional_ridge_atlas import (
    conditional_ridge_signature,
    load_conditional_ridge_atlas,
)
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .generic_word_template_bank import (
    load_generic_word_template_bank,
    word_signature_aggregation_scores,
)
from .generic_word_signature_cache import (
    GenericWordSignatureCache,
    file_sha256,
)
from .occupation_point_cloud import PointCloudConfig, marginal_copula_cloud
from .phone_duration_gate import load_phone_duration_gate
from .run_occupation_context_battery import _cache_name, _subset, _window_list
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks
from .run_oracle_lexicographic_sequence_audit import local_history_key
from .run_query_radio_boundary_atlas import endpoint_ranking, fuse_rankings
from .run_synthetic_word_geometry_audit import (
    build_phone_prototype,
    build_word_realizations,
)
from .transcript_phone_alignment import transcript_word_phone_spans, transcript_words
from .word_lattice import (
    ARPABET_39,
    BoundaryPairEvidence,
    PronunciationProposalIndex,
    ProvenancePhoneEvidence,
    parse_cmudict,
    proposal_sequence_audit,
)


def target_class_rank(
    rows: list[tuple], index: PronunciationProposalIndex, word: str, phones: tuple[str, ...]
) -> int | None:
    """Return the exact target pronunciation rank in an already sorted list."""

    return next(
        (
            rank
            for rank, row in enumerate(rows, start=1)
            if index.classes[row[-1]].phones == phones
            and word in index.classes[row[-1]].words
        ),
        None,
    )


def rank_summary(ranks: list[int | None]) -> dict[str, object]:
    finite = [rank for rank in ranks if rank is not None]
    return {
        "count": len(ranks),
        "finite": len(finite),
        "top1": sum(rank == 1 for rank in finite),
        "top5": sum(rank <= 5 for rank in finite),
        "top20": sum(rank <= 20 for rank in finite),
        "median_rank": float(np.median(finite)) if finite else None,
    }


def normalized_distance_map(ranking: list[dict]) -> dict[str, float]:
    """Normalize one query's distances without changing their label order."""

    values = np.asarray([float(item["distance"]) for item in ranking])
    minimum = float(np.min(values))
    scale = max(float(np.median(values)) - minimum, 1e-12)
    return {
        str(item["phone"]): (float(item["distance"]) - minimum) / scale
        for item in ranking
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("conditional_atlas", type=Path)
    parser.add_argument("copula_atlas", type=Path)
    parser.add_argument("boundary_atlas", type=Path)
    parser.add_argument("duration_gate", type=Path)
    parser.add_argument("--query-speaker", required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--deep-k", type=int, default=24)
    parser.add_argument("--duration-label-count", type=int, default=24)
    parser.add_argument("--maximum-uncovered-phones", type=int, default=1)
    parser.add_argument("--maximum-boundary-rank", type=int, default=128)
    parser.add_argument("--maximum-uncovered-boundaries", type=int, default=0)
    parser.add_argument("--report-count", type=int, default=5)
    parser.add_argument("--generic-verifier-candidates", type=int, default=0)
    parser.add_argument("--generic-time-bins", type=int, default=64)
    parser.add_argument("--generic-row-quantiles", type=int, default=24)
    parser.add_argument("--generic-realizations", type=int, default=0)
    parser.add_argument("--generic-bank", type=Path)
    parser.add_argument("--generic-context-conditioned", action="store_true")
    parser.add_argument("--generic-context-mixture", action="store_true")
    parser.add_argument("--generic-signature-cache", type=Path)
    parser.add_argument(
        "--generic-cache-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.generic_context_conditioned and args.generic_context_mixture:
        raise ValueError("choose maximum or mixed generic context, not both")
    generic_context_policy = (
        "mixed"
        if args.generic_context_mixture
        else "maximum"
        if args.generic_context_conditioned
        else "unconditional"
    )
    if not 1 <= args.top_k <= args.deep_k:
        raise ValueError("sentence assay requires 1 <= top-k <= deep-k")

    prompt_document = json.loads(args.prompts.read_text())
    prompts = prompt_document["utterances"]
    dictionary_entries = parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39)
    )
    pronunciations: dict[str, list[tuple[str, ...]]] = {}
    for entry in dictionary_entries:
        pronunciations.setdefault(entry.word, []).append(entry.phones)
    for options in pronunciations.values():
        options.sort()
    index = PronunciationProposalIndex(dictionary_entries)
    by_length: dict[int, list[int]] = {}
    for ordinal, item in enumerate(index.classes):
        by_length.setdefault(len(item.phones), []).append(ordinal)

    conditional_atlas = load_conditional_ridge_atlas(args.conditional_atlas)
    copula_atlas = load_phone_atlas(args.copula_atlas)
    boundary_atlas = load_boundary_atlas(args.boundary_atlas)
    duration_gate = load_phone_duration_gate(args.duration_gate)
    projection_points = int((copula_atlas.provenance or {}).get("projection_points", 4096))
    points_per_phone = int((boundary_atlas.provenance or {}).get("points_per_phone", 2048))
    cache_payload = {
        "rows": 128,
        "noise_db": 8.0,
        "morphology": asdict(DepthmapGeometryConfig()),
        "ridge": asdict(HessianRidgeConfig()),
        "climbers": asdict(CrazyClimberConfig()),
        "point_cloud": asdict(PointCloudConfig(point_count=32768)),
    }

    def load_raw(window) -> np.ndarray:
        path = args.raw_cache / _cache_name(
            window,
            {
                **cache_payload,
                "key": window.key,
                "representation": "raw_occupation_point_cloud_v1",
            },
        )
        if not path.exists():
            raise FileNotFoundError(f"raw cloud missing for {window.key}")
        return np.asarray(np.load(path), dtype=np.float64)

    generic_phone_prototypes = {}
    generic_template_cache: dict[
        tuple[str, ...], tuple[tuple[np.ndarray, np.ndarray], ...]
    ] = {}
    generic_phone_pools: dict[str, tuple[np.ndarray, ...]] = {}
    generic_bank = (
        load_generic_word_template_bank(args.generic_bank)
        if args.generic_bank is not None
        else None
    )
    generic_bank_labels = (
        set(generic_bank.labels.astype(str)) if generic_bank is not None else set()
    )
    persistent_generic_cache = None
    generic_cache_hits = 0
    generic_cache_misses = 0
    generic_cache_load_seconds = 0.0
    if args.generic_signature_cache:
        if not args.generic_verifier_candidates or generic_bank is None:
            raise ValueError(
                "generic signature cache requires bank-backed generic verification"
            )
        cache_started = perf_counter()
        bank_fingerprint = file_sha256(args.generic_bank)
        if args.generic_signature_cache.exists():
            persistent_generic_cache = GenericWordSignatureCache.load(
                args.generic_signature_cache
            )
        else:
            persistent_generic_cache = GenericWordSignatureCache(
                bank_fingerprint=bank_fingerprint,
                realization_count=max(args.generic_realizations, 1),
                time_bins=args.generic_time_bins,
                row_quantiles=args.generic_row_quantiles,
                storage_dtype=args.generic_cache_dtype,
                realization_policy=(
                    "contextual_mixture"
                    if args.generic_context_mixture
                    else "contextual"
                    if args.generic_context_conditioned
                    else "unconditional"
                ),
            )
        persistent_generic_cache.require_configuration(
            bank_fingerprint=bank_fingerprint,
            realization_count=max(args.generic_realizations, 1),
            time_bins=args.generic_time_bins,
            row_quantiles=args.generic_row_quantiles,
            storage_dtype=args.generic_cache_dtype,
            realization_policy=(
                "contextual_mixture"
                if args.generic_context_mixture
                else "contextual"
                if args.generic_context_conditioned
                else "unconditional"
            ),
        )
        generic_cache_load_seconds = perf_counter() - cache_started
    if args.generic_verifier_candidates:
        if generic_bank is None:
            reference_provenance = dict(boundary_atlas.provenance or {})
            reference_speaker = str(reference_provenance["speaker"])
            reference_utterances = tuple(reference_provenance["utterances"])
            pools: dict[str, list[np.ndarray]] = {}
            for utterance in reference_utterances:
                for window in _window_list(
                    args.arctic_root, reference_speaker, str(utterance)
                ):
                    pools.setdefault(window.label, []).append(load_raw(window))
            generic_phone_prototypes = {
                label: build_phone_prototype(tuple(clouds), 4, 256)
                for label, clouds in pools.items()
            }
            generic_phone_pools = {
                label: tuple(clouds) for label, clouds in pools.items()
            }

        def generic_template(
            phones: tuple[str, ...]
        ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
            nonlocal generic_cache_hits, generic_cache_misses
            cached = generic_template_cache.get(phones)
            if cached is not None:
                return cached
            if persistent_generic_cache is not None:
                cached = persistent_generic_cache.get(phones)
                if cached is None:
                    generic_cache_misses += 1
                else:
                    generic_cache_hits += 1
            if cached is not None:
                generic_template_cache[phones] = cached
                return cached
            if generic_bank is not None:
                signatures = generic_bank.word_signatures(
                    phones,
                    max(args.generic_realizations, 1),
                    args.generic_time_bins,
                    args.generic_row_quantiles,
                    context_conditioned=args.generic_context_conditioned,
                    context_policy=generic_context_policy,
                )
                if persistent_generic_cache is not None:
                    signatures = persistent_generic_cache.put(
                        phones, signatures
                    )
                generic_template_cache[phones] = signatures
                return signatures
            if args.generic_realizations:
                clouds = build_word_realizations(
                    tuple(generic_phone_pools[phone] for phone in phones),
                    args.generic_realizations,
                    512,
                )
            else:
                chunks = tuple(
                    generic_phone_prototypes[phone] for phone in phones
                )
                clouds = (lane_chunks(joint_gauge_chunks(chunks, 1024)),)
            signatures = tuple(
                conditional_ridge_signature(
                    cloud,
                    args.generic_time_bins,
                    args.generic_row_quantiles,
                )
                for cloud in clouds
            )
            generic_template_cache[phones] = signatures
            return signatures

    phone_channel_hits = {
        depth: {name: 0 for name in ("copula", "conditional", "duration", "boundary", "union")}
        for depth in (args.top_k, args.deep_k)
    }
    phone_count = 0
    word_rows = []
    utterance_rows = []
    started = perf_counter()
    for utterance, prompt in prompts.items():
        windows = _window_list(args.arctic_root, args.query_speaker, utterance)
        clouds = [load_raw(window) for window in windows]
        conditional_rankings = [conditional_atlas.rank(cloud) for cloud in clouds]
        copula_rankings = [
            copula_atlas.rank(_subset(marginal_copula_cloud(cloud), projection_points))
            for cloud in clouds
        ]
        duration_rankings = [
            duration_gate.filter_ranking(
                window.duration, ranking, args.duration_label_count
            )
            for window, ranking in zip(windows, conditional_rankings, strict=True)
        ]
        pair_rankings = []
        left_channels: list[list[dict] | None] = [None] * len(windows)
        right_channels: list[list[dict] | None] = [None] * len(windows)
        for position, pair in enumerate(zip(clouds, clouds[1:])):
            joint = lane_chunks(joint_gauge_chunks(pair, points_per_phone))
            ranking = boundary_atlas.rank(joint)
            pair_rankings.append(ranking)
            right_channels[position] = endpoint_ranking(ranking, 0)
            left_channels[position + 1] = endpoint_ranking(ranking, 1)
        boundary_phone_rankings = []
        for position in range(len(windows)):
            channels = {}
            if left_channels[position] is not None:
                channels["left"] = left_channels[position]
            if right_channels[position] is not None:
                channels["right"] = right_channels[position]
            boundary_phone_rankings.append(fuse_rankings(channels, "mean"))

        ranking_sets = {
            "copula": copula_rankings,
            "conditional": conditional_rankings,
            "duration": duration_rankings,
            "boundary": boundary_phone_rankings,
        }
        for position, window in enumerate(windows):
            phone_count += 1
            for depth in (args.top_k, args.deep_k):
                admitted = {
                    name: {
                        str(item["phone"])
                        for item in rankings[position][:depth]
                    }
                    for name, rankings in ranking_sets.items()
                }
                for name, values in admitted.items():
                    phone_channel_hits[depth][name] += window.label in values
                phone_channel_hits[depth]["union"] += window.label in set().union(
                    *admitted.values()
                )

        spans, missing = transcript_word_phone_spans(
            transcript_words(prompt), pronunciations
        )
        if missing:
            raise ValueError(f"prompt has missing dictionary words: {missing}")
        expected_count = sum(len(span.phones) for span in spans)
        if expected_count != len(windows):
            raise ValueError(
                f"{utterance} dictionary phones {expected_count} != labels {len(windows)}"
            )
        utterance_top1 = True
        for span in spans:
            region0, region1 = span.phone0, span.phone1
            target_phones = tuple(span.phones)
            evidence = []
            for position in range(region0, region1):
                channel_ranks = {
                    name: {
                        str(item["phone"]): rank
                        for rank, item in enumerate(rankings[position][: args.top_k], start=1)
                    }
                    for name, rankings in ranking_sets.items()
                }
                evidence.append(ProvenancePhoneEvidence(channel_ranks, args.top_k))
            boundaries = tuple(
                BoundaryPairEvidence(
                    {
                        tuple(str(phone) for phone in item["phones"]): int(item["rank"])
                        for item in pair_rankings[position]
                    },
                    len(pair_rankings[position]),
                )
                for position in range(region0, region1 - 1)
            )
            target_audit = proposal_sequence_audit(
                evidence,
                boundaries,
                target_phones,
                maximum_uncovered_phones=args.maximum_uncovered_phones,
                maximum_boundary_rank=args.maximum_boundary_rank,
                maximum_uncovered_boundaries=args.maximum_uncovered_boundaries,
            )
            route = index.query_aligned(
                evidence,
                boundaries,
                maximum_uncovered_phones=args.maximum_uncovered_phones,
                maximum_boundary_rank=args.maximum_boundary_rank,
                maximum_length_delta=0,
                maximum_uncovered_boundaries=args.maximum_uncovered_boundaries,
            )
            routed = sorted(
                (
                    route.relational_costs[ordinal],
                    route.costs[ordinal],
                    index.classes[ordinal].phones,
                    index.classes[ordinal].words,
                    ordinal,
                )
                for ordinal in route.candidates
            )
            route_rank = target_class_rank(routed, index, span.word, target_phones)
            generic_rank = None
            generic_candidate_count = 0
            generic_report = []
            generic_distance_statistics = None
            generic_context_depths = None
            generic_aggregation_ranks = {
                name: None
                for name in ("minimum", "mean", "median", "barycenter")
            }
            if args.generic_verifier_candidates:
                if (
                    generic_bank is not None
                    and generic_context_policy != "unconditional"
                    and all(phone in generic_bank_labels for phone in target_phones)
                ):
                    generic_context_depths = list(
                        generic_bank.context_depths(target_phones)
                    )
                query_word_cloud = lane_chunks(
                    joint_gauge_chunks(
                        tuple(clouds[region0:region1]), 1024
                    )
                )
                query_signature = conditional_ridge_signature(
                    query_word_cloud,
                    args.generic_time_bins,
                    args.generic_row_quantiles,
                )
                generic_rows_by_aggregation = {
                    name: []
                    for name in ("minimum", "mean", "median", "barycenter")
                }
                for route_row in routed[: args.generic_verifier_candidates]:
                    ordinal = route_row[-1]
                    item = index.classes[ordinal]
                    if generic_bank is not None and any(
                        phone not in generic_bank_labels for phone in item.phones
                    ):
                        continue
                    reference_signatures = generic_template(item.phones)
                    aggregation_scores = word_signature_aggregation_scores(
                        query_signature,
                        reference_signatures,
                        maximum_shift=max(2, args.generic_time_bins // 16),
                    )
                    for name, score in aggregation_scores.items():
                        generic_rows_by_aggregation[name].append(
                            (score, item.phones, item.words, ordinal)
                        )
                for rows in generic_rows_by_aggregation.values():
                    rows.sort()
                generic_rows = generic_rows_by_aggregation["minimum"]
                generic_candidate_count = len(generic_rows)
                for name, rows in generic_rows_by_aggregation.items():
                    generic_aggregation_ranks[name] = target_class_rank(
                        rows, index, span.word, target_phones
                    )
                generic_rank = generic_aggregation_ranks["minimum"]
                route_rank_by_ordinal = {
                    route_row[-1]: rank
                    for rank, route_row in enumerate(routed, start=1)
                }
                generic_report = [
                    {
                        "rank": rank,
                        "distance": float(row[0]),
                        "route_rank": route_rank_by_ordinal[row[-1]],
                        "phones": list(row[1]),
                        "words": list(row[2]),
                    }
                    for rank, row in enumerate(
                        generic_rows[: args.report_count], start=1
                    )
                ]
                distances = np.asarray(
                    [float(row[0]) for row in generic_rows], dtype=np.float64
                )
                if distances.size:
                    best = float(distances[0])
                    second = float(distances[min(1, distances.size - 1)])
                    median = float(np.median(distances))
                    generic_distance_statistics = {
                        "best": best,
                        "second": second,
                        "median": median,
                        "best_to_median_ratio": best / max(median, 1e-12),
                        "gap_to_median_range": (
                            (second - best) / max(median - best, 1e-12)
                        ),
                    }

            conditional_maps = [
                {str(item["phone"]): int(item["rank"]) for item in conditional_rankings[position]}
                for position in range(region0, region1)
            ]
            boundary_maps = [
                {
                    tuple(str(phone) for phone in item["phones"]): int(item["rank"])
                    for item in pair_rankings[position]
                }
                for position in range(region0, region1 - 1)
            ]
            conditional_distance_maps = [
                {
                    str(item["phone"]): float(item["distance"])
                    for item in conditional_rankings[position]
                }
                for position in range(region0, region1)
            ]
            conditional_normalized_maps = [
                normalized_distance_map(conditional_rankings[position])
                for position in range(region0, region1)
            ]
            lexicographic = []
            continuous_candidates = {
                "raw_sum": [],
                "normalized_sum": [],
                "normalized_worst_then_sum": [],
            }
            for ordinal in by_length.get(len(target_phones), ()): 
                item = index.classes[ordinal]
                local_ranks = [
                    conditional_maps[position][phone]
                    for position, phone in enumerate(item.phones)
                ]
                transition_ranks = [
                    boundary_maps[position].get(pair)
                    for position, pair in enumerate(zip(item.phones, item.phones[1:]))
                ]
                uncovered = sum(
                    rank is None or rank > args.maximum_boundary_rank
                    for rank in transition_ranks
                )
                lexicographic.append(
                    (
                        uncovered,
                        local_history_key(local_ranks),
                        sum(local_ranks),
                        item.phones,
                        item.words,
                        ordinal,
                    )
                )
                missed_phones = 0
                for position, phone in enumerate(item.phones):
                    if not any(
                        phone in channel
                        for channel in evidence[position].channel_ranks.values()
                    ):
                        missed_phones += 1
                admitted = (
                    missed_phones <= args.maximum_uncovered_phones
                    and uncovered <= args.maximum_uncovered_boundaries
                )
                if admitted:
                    raw_values = [
                        conditional_distance_maps[position][phone]
                        for position, phone in enumerate(item.phones)
                    ]
                    normalized_values = [
                        conditional_normalized_maps[position][phone]
                        for position, phone in enumerate(item.phones)
                    ]
                    continuous_candidates["raw_sum"].append(
                        (sum(raw_values), max(raw_values), item.phones, item.words, ordinal)
                    )
                    continuous_candidates["normalized_sum"].append(
                        (
                            sum(normalized_values),
                            max(normalized_values),
                            item.phones,
                            item.words,
                            ordinal,
                        )
                    )
                    continuous_candidates["normalized_worst_then_sum"].append(
                        (
                            max(normalized_values),
                            sum(normalized_values),
                            item.phones,
                            item.words,
                            ordinal,
                        )
                    )
            lexicographic.sort()
            lexical_rank = target_class_rank(
                lexicographic, index, span.word, target_phones
            )
            continuous_ranks = {}
            for name, candidates in continuous_candidates.items():
                candidates.sort()
                continuous_ranks[name] = target_class_rank(
                    candidates, index, span.word, target_phones
                )
            utterance_top1 &= route_rank == 1
            word_rows.append(
                {
                    "utterance": utterance,
                    "word": span.word,
                    "phones": list(target_phones),
                    "region0": region0,
                    "region1": region1,
                    "phone_stage_admitted": target_audit.phone_stage_admitted,
                    "boundary_stage_admitted": target_audit.boundary_stage_admitted,
                    "target_phone_channel_ranks": [
                        dict(values) for values in target_audit.phone_channel_ranks
                    ],
                    "target_boundary_ranks": list(target_audit.boundary_ranks),
                    "route_candidate_count": len(routed),
                    "route_target_rank": route_rank,
                    "generic_word_target_rank": generic_rank,
                    "generic_word_candidate_count": generic_candidate_count,
                    "generic_aggregation_target_ranks": generic_aggregation_ranks,
                    "generic_distance_statistics": generic_distance_statistics,
                    "generic_context_depths": generic_context_depths,
                    "lexicographic_target_rank": lexical_rank,
                    "continuous_conditional_target_ranks": continuous_ranks,
                    "continuous_candidate_counts": {
                        name: len(candidates)
                        for name, candidates in continuous_candidates.items()
                    },
                    "route_top": [
                        {
                            "rank": rank,
                            "phones": list(row[2]),
                            "words": list(row[3]),
                            "relational_cost": asdict(row[0]),
                            "provenance_cost": asdict(row[1]),
                        }
                        for rank, row in enumerate(routed[: args.report_count], start=1)
                    ],
                    "generic_word_top": generic_report,
                }
            )
        utterance_rows.append(
            {
                "utterance": utterance,
                "word_count": len(spans),
                "all_words_route_top1": utterance_top1,
            }
        )
        print(f"audited {utterance}: {len(windows)} phones, {len(spans)} words", flush=True)

    route_ranks = [row["route_target_rank"] for row in word_rows]
    lexical_ranks = [row["lexicographic_target_rank"] for row in word_rows]
    continuous_rank_summaries = {
        name: rank_summary(
            [
                row["continuous_conditional_target_ranks"][name]
                for row in word_rows
            ]
        )
        for name in ("raw_sum", "normalized_sum", "normalized_worst_then_sum")
    }
    generic_word_ranks = [row["generic_word_target_rank"] for row in word_rows]
    generic_aggregation_rank_summaries = {
        name: rank_summary(
            [
                row["generic_aggregation_target_ranks"][name]
                for row in word_rows
            ]
        )
        for name in ("minimum", "mean", "median", "barycenter")
    }
    generic_cache_save_seconds = 0.0
    if (
        persistent_generic_cache is not None
        and persistent_generic_cache.dirty
    ):
        cache_started = perf_counter()
        persistent_generic_cache.save(args.generic_signature_cache)
        generic_cache_save_seconds = perf_counter() - cache_started
    generic_cache_bytes = (
        args.generic_signature_cache.stat().st_size
        if args.generic_signature_cache
        and args.generic_signature_cache.exists()
        else 0
    )
    result = {
        "method": "timestamped_cross_speaker_acoustic_to_word_bottleneck_assay",
        "selection_status": "query labels and prompt text used only for oracle measurement",
        "query_speaker": args.query_speaker,
        "reference_atlases": {
            "conditional": str(args.conditional_atlas),
            "copula": str(args.copula_atlas),
            "boundary": str(args.boundary_atlas),
            "duration": str(args.duration_gate),
        },
        "parameters": {
            "top_k": args.top_k,
            "deep_k": args.deep_k,
            "duration_label_count": args.duration_label_count,
            "maximum_uncovered_phones": args.maximum_uncovered_phones,
            "maximum_boundary_rank": args.maximum_boundary_rank,
            "maximum_uncovered_boundaries": args.maximum_uncovered_boundaries,
            "generic_verifier_candidates": args.generic_verifier_candidates,
            "generic_realizations": args.generic_realizations,
            "generic_bank": str(args.generic_bank) if args.generic_bank else None,
            "generic_context_conditioned": args.generic_context_conditioned,
            "generic_context_policy": generic_context_policy,
            "generic_signature_cache": (
                str(args.generic_signature_cache)
                if args.generic_signature_cache
                else None
            ),
            "generic_cache_dtype": (
                args.generic_cache_dtype if args.generic_signature_cache else None
            ),
        },
        "elapsed_seconds": perf_counter() - started,
        "summary": {
            "phone_count": phone_count,
            "word_count": len(word_rows),
            "phone_channel_hits": phone_channel_hits,
            "phone_channel_recall": {
                str(depth): {
                    name: hits / phone_count for name, hits in values.items()
                }
                for depth, values in phone_channel_hits.items()
            },
            "word_phone_stage_admitted": sum(row["phone_stage_admitted"] for row in word_rows),
            "word_boundary_stage_admitted": sum(row["boundary_stage_admitted"] for row in word_rows),
            "route_rank": rank_summary(route_ranks),
            "lexicographic_rank": rank_summary(lexical_ranks),
            "continuous_conditional_rank": continuous_rank_summaries,
            "generic_word_rank": rank_summary(generic_word_ranks),
            "generic_aggregation_rank": generic_aggregation_rank_summaries,
            "generic_template_count": len(generic_template_cache),
            "generic_cache_hits": generic_cache_hits,
            "generic_cache_misses": generic_cache_misses,
            "generic_cache_entry_count": (
                len(persistent_generic_cache.entries)
                if persistent_generic_cache is not None
                else 0
            ),
            "generic_cache_load_seconds": generic_cache_load_seconds,
            "generic_cache_save_seconds": generic_cache_save_seconds,
            "generic_cache_bytes": generic_cache_bytes,
            "sentence_route_top1": sum(row["all_words_route_top1"] for row in utterance_rows),
        },
        "utterances": utterance_rows,
        "words": word_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
