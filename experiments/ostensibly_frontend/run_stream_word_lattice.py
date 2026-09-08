#!/usr/bin/env python3
"""Build streaming word edges from top-k phone and adjacent-pair evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .conditional_ridge_atlas import (
    conditional_ridge_signature,
)
from .generic_word_signature_cache import (
    GenericWordSignatureCache,
    file_sha256,
)
from .generic_pronunciation_catalog import GenericPronunciationCatalog
from .generic_word_template_bank import (
    load_generic_word_template_bank,
    word_signature_distance,
)
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks

from .word_lattice import (
    ARPABET_39,
    BoundaryPairEvidence,
    PronunciationProposalIndex,
    ProvenancePhoneEvidence,
    parse_cmudict,
)


def scalar_provenance_cost(cost, observed_length: int, top_k: int) -> float:
    scale = max(observed_length * top_k, 1)
    return float(
        2.0 * cost.uncovered
        + 0.5 * cost.edits
        + cost.rank_sum / scale
        + 0.1 * cost.worst_rank / max(top_k, 1)
        + 0.01 * cost.negative_channel_support / max(observed_length, 1)
    )


def scalar_relational_cost(cost, observed_length: int, boundary_cap: int) -> float:
    transition_count = max(observed_length - 1, 1)
    return float(
        cost.uncovered_transitions
        + cost.transition_rank_sum / max(transition_count * boundary_cap, 1)
        + 0.1 * cost.worst_transition_rank / max(boundary_cap, 1)
    )


def proposal_channel_union(
    route_order: list[tuple[str, ...]],
    generic_order: list[tuple[str, ...]],
    count: int,
) -> tuple[tuple[str, ...], ...]:
    """Retain bounded winners from both terminal proposal channels."""

    if count < 1:
        raise ValueError("proposal channel union count must be positive")
    output = []
    seen = set()
    for phones in route_order[:count] + generic_order[:count]:
        if phones not in seen:
            seen.add(phones)
            output.append(phones)
    return tuple(output)


def proposal_channels_union(
    channel_orders: list[list[tuple[str, ...]]], count: int
) -> tuple[tuple[str, ...], ...]:
    """Retain bounded winners from any number of independent channels."""

    if not channel_orders or count < 1:
        raise ValueError("proposal channels union needs channels and positive count")
    output = []
    seen = set()
    for order in channel_orders:
        for phones in order[:count]:
            if phones not in seen:
                seen.add(phones)
                output.append(phones)
    return tuple(output)


def policy_cache_path(
    base: Path, policy: str, multiple_policies: bool
) -> Path:
    """Derive stable per-policy cache paths while preserving single-policy ABI."""

    if not multiple_policies:
        return base
    return base.with_name(f"{base.stem}.{policy}{base.suffix}")


def cache_realization_policy(policy: str) -> str:
    return {
        "unconditional": "unconditional",
        "maximum": "contextual",
        "mixed": "contextual_mixture",
    }[policy]


def selected_proposal_channels(
    route_rank: int,
    generic_ranks: dict[str, int | None],
    count: int,
) -> list[str]:
    """Label only channels whose bounded winner set selected the candidate."""

    if route_rank < 1 or count < 1:
        raise ValueError("proposal channel ranks must be positive")
    channels = ["route"] if route_rank <= count else []
    channels.extend(
        f"generic_word:{policy}"
        for policy, rank in generic_ranks.items()
        if rank is not None and rank <= count
    )
    if not channels:
        raise ValueError("retained candidate has no selecting proposal channel")
    return channels


def proposal_evidence_stratum(relational_cost, phones: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Keep unlike marker-support regimes as separate proposal channels."""

    return (
        len(phones),
        int(relational_cost.uncovered_phones),
        int(relational_cost.uncovered_transitions),
        int(relational_cost.edits),
    )


def stratified_generic_ranks(
    rows: list[tuple[float, int, tuple[str, ...]]],
    strata: dict[tuple[str, ...], tuple[int, int, int, int]],
) -> tuple[
    dict[tuple[str, ...], int],
    dict[tuple[str, ...], int],
    list[tuple[float, int, tuple[str, ...]]],
]:
    """Rank geometry within marker-support strata while retaining global audit ranks."""

    ordered = sorted(rows)
    counts = {}
    local = {}
    global_ranks = {}
    for global_rank, (_, _, phones) in enumerate(ordered, start=1):
        stratum = strata[phones]
        counts[stratum] = counts.get(stratum, 0) + 1
        local[phones] = counts[stratum]
        global_ranks[phones] = global_rank
    return local, global_ranks, ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fusion_lattice", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--top-k-per-channel", type=int, default=24)
    parser.add_argument("--maximum-uncovered-phones", type=int, default=1)
    parser.add_argument("--maximum-boundary-rank", type=int, default=128)
    parser.add_argument("--maximum-uncovered-boundaries", type=int, default=0)
    parser.add_argument("--maximum-length-delta", type=int, default=1)
    parser.add_argument("--max-phone-span", type=int, default=18)
    parser.add_argument("--edge-candidates", type=int, default=8)
    parser.add_argument("--max-gap-seconds", type=float, default=0.30)
    parser.add_argument("--region-limit", type=int)
    parser.add_argument("--audit-words", default="")
    parser.add_argument("--region-clouds", type=Path)
    parser.add_argument("--generic-word-bank", type=Path)
    parser.add_argument("--generic-verifier-candidates", type=int, default=0)
    parser.add_argument("--generic-realizations", type=int, default=4)
    parser.add_argument("--generic-time-bins", type=int, default=64)
    parser.add_argument("--generic-row-quantiles", type=int, default=24)
    parser.add_argument(
        "--generic-realization-aggregation",
        choices=("minimum", "mean", "median", "barycenter"),
        default="mean",
    )
    parser.add_argument("--generic-context-conditioned", action="store_true")
    parser.add_argument("--generic-context-mixture", action="store_true")
    parser.add_argument(
        "--generic-context-policies",
        default="",
        help="Comma-separated independent policies: unconditional,maximum,mixed",
    )
    parser.add_argument("--generic-signature-cache", type=Path)
    parser.add_argument("--generic-pronunciation-catalog", type=Path)
    parser.add_argument(
        "--generic-catalog-candidates",
        type=int,
        default=0,
        help="Coarse catalog candidates retained per marker-support stratum",
    )
    parser.add_argument(
        "--generic-cache-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.generic_context_conditioned and args.generic_context_mixture:
        raise ValueError("choose maximum or mixed generic context, not both")
    requested_policies = tuple(
        value.strip()
        for value in args.generic_context_policies.split(",")
        if value.strip()
    )
    if requested_policies and (
        args.generic_context_conditioned or args.generic_context_mixture
    ):
        raise ValueError(
            "generic context policy list cannot be combined with legacy flags"
        )
    if requested_policies:
        if (
            len(set(requested_policies)) != len(requested_policies)
            or any(
                policy not in ("unconditional", "maximum", "mixed")
                for policy in requested_policies
            )
        ):
            raise ValueError("generic context policies are invalid or repeated")
        generic_context_policies = requested_policies
    else:
        generic_context_policies = (
            "mixed"
            if args.generic_context_mixture
            else "maximum"
            if args.generic_context_conditioned
            else "unconditional",
        )
    generic_context_policy = generic_context_policies[0]
    document = json.loads(args.fusion_lattice.read_text())
    phone_rows = document["phones"]
    if args.region_limit is not None:
        phone_rows = phone_rows[: args.region_limit]
    phone_count = len(phone_rows)
    if phone_count < 1:
        raise ValueError("stream word lattice requires phone regions")
    use_generic = args.generic_verifier_candidates > 0
    if use_generic != bool(args.generic_word_bank and args.region_clouds):
        raise ValueError(
            "generic verification requires its candidate count, bank, and region clouds"
        )
    if args.generic_signature_cache and not use_generic:
        raise ValueError("generic signature cache requires generic verification")
    use_generic_catalog = args.generic_catalog_candidates > 0
    if use_generic_catalog != bool(args.generic_pronunciation_catalog):
        raise ValueError(
            "generic catalog requires its path and positive candidate count"
        )
    if use_generic_catalog and not use_generic:
        raise ValueError("generic catalog requires full generic verification")
    generic_bank = (
        load_generic_word_template_bank(args.generic_word_bank)
        if use_generic
        else None
    )
    generic_catalog = (
        GenericPronunciationCatalog.load(args.generic_pronunciation_catalog)
        if use_generic_catalog
        else None
    )
    generic_bank_labels = (
        set(generic_bank.labels.astype(str)) if generic_bank is not None else set()
    )
    if use_generic:
        with np.load(args.region_clouds) as cloud_document:
            region_clouds = np.asarray(
                cloud_document["clouds"], dtype=np.float64
            )
        if region_clouds.shape[0] < phone_count:
            raise ValueError("generic verifier has fewer clouds than phone regions")
        region_clouds = region_clouds[:phone_count]
    else:
        region_clouds = None
    generic_template_cache = {}
    persistent_generic_caches = {}
    persistent_generic_cache_paths = {}
    generic_cache_load_seconds = 0.0
    generic_cache_hits = {policy: 0 for policy in generic_context_policies}
    generic_cache_misses = {policy: 0 for policy in generic_context_policies}
    if args.generic_signature_cache:
        cache_started = perf_counter()
        bank_fingerprint = file_sha256(args.generic_word_bank)
        multiple_policies = len(generic_context_policies) > 1
        for policy in generic_context_policies:
            cache_path = policy_cache_path(
                args.generic_signature_cache, policy, multiple_policies
            )
            if cache_path.exists():
                cache = GenericWordSignatureCache.load(cache_path)
            else:
                cache = GenericWordSignatureCache(
                    bank_fingerprint=bank_fingerprint,
                    realization_count=args.generic_realizations,
                    time_bins=args.generic_time_bins,
                    row_quantiles=args.generic_row_quantiles,
                    storage_dtype=args.generic_cache_dtype,
                    realization_policy=cache_realization_policy(policy),
                )
            cache.require_configuration(
                bank_fingerprint=bank_fingerprint,
                realization_count=args.generic_realizations,
                time_bins=args.generic_time_bins,
                row_quantiles=args.generic_row_quantiles,
                storage_dtype=args.generic_cache_dtype,
                realization_policy=cache_realization_policy(policy),
            )
            persistent_generic_caches[policy] = cache
            persistent_generic_cache_paths[policy] = cache_path
        generic_cache_load_seconds = perf_counter() - cache_started

    def generic_signatures(policy: str, phones: tuple[str, ...]):
        key = (policy, phones)
        cached = generic_template_cache.get(key)
        if cached is None:
            persistent_cache = persistent_generic_caches.get(policy)
            if persistent_cache is not None:
                cached = persistent_cache.get(phones)
                if cached is None:
                    generic_cache_misses[policy] += 1
                else:
                    generic_cache_hits[policy] += 1
            if cached is None:
                cached = generic_bank.word_signatures(
                    phones,
                    args.generic_realizations,
                    args.generic_time_bins,
                    args.generic_row_quantiles,
                    context_policy=policy,
                )
                if persistent_cache is not None:
                    # Score the quantized representation immediately so a cold
                    # build and a later warm load have exactly the same ABI.
                    cached = persistent_cache.put(phones, cached)
            generic_template_cache[key] = cached
        return cached

    boundary_rows = document["joint_boundary_pair_swd_rankings"][: max(0, phone_count - 1)]
    boundaries = tuple(
        BoundaryPairEvidence(
            {
                tuple(str(phone) for phone in item["phones"]): int(item["rank"])
                for item in boundary["ranking"]
            },
            max((int(item["rank"]) for item in boundary["ranking"]), default=0),
        )
        for boundary in boundary_rows
    )
    if len(boundaries) != phone_count - 1:
        raise ValueError("stream lattice needs one adjacent-pair ranking per region gap")

    evidence = []
    for row in phone_rows:
        channels = {}
        for channel, field in (
            ("center", "center_top5"),
            ("boundary_endpoint", "boundary_top5"),
        ):
            channels[channel] = {
                str(item["phone"]): rank
                for rank, item in enumerate(
                    row[field][: args.top_k_per_channel], start=1
                )
            }
        evidence.append(
            ProvenancePhoneEvidence(channels, args.top_k_per_channel)
        )

    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(),
        frozenset(ARPABET_39),
        max_phones=args.max_phone_span + args.maximum_length_delta,
    )
    index_started = perf_counter()
    index = PronunciationProposalIndex(pronunciations)
    index_seconds = perf_counter() - index_started
    audit_words = tuple(
        word.strip().lower() for word in args.audit_words.split(",") if word.strip()
    )
    audit_hits = {word: [] for word in audit_words}

    edges = []
    spans_considered = routed_spans = 0
    candidate_counts = []
    route_started = perf_counter()
    generic_query_seconds = 0.0
    for start in range(phone_count):
        for stop in range(start + 1, min(phone_count, start + args.max_phone_span) + 1):
            if stop > start + 1:
                gap = float(phone_rows[stop - 1]["seconds0"]) - float(
                    phone_rows[stop - 2]["seconds1"]
                )
                if gap > args.max_gap_seconds:
                    break
            spans_considered += 1
            route = index.query_grouped_aligned(
                evidence[start:stop],
                boundaries[start : stop - 1],
                maximum_uncovered_phones=args.maximum_uncovered_phones,
                maximum_boundary_rank=args.maximum_boundary_rank,
                maximum_length_delta=args.maximum_length_delta,
                maximum_uncovered_boundaries=args.maximum_uncovered_boundaries,
            )
            candidate_counts.append(len(route.candidates))
            if not route.candidates:
                continue
            routed_spans += 1
            ranked = []
            for ordinal in route.candidates:
                item = index.classes[ordinal]
                cost = route.costs[ordinal]
                relational_cost = route.relational_costs[ordinal]
                ranked.append((relational_cost, cost, item))
            ranked.sort(
                key=lambda row: (row[0], row[1], row[2].phones, row[2].words)
            )
            route_ranks = {
                item.phones: rank
                for rank, (_, _, item) in enumerate(ranked, start=1)
            }
            generic_ranks_by_policy = {}
            generic_global_ranks_by_policy = {}
            generic_distances_by_policy = {}
            generic_catalog_ranks = {}
            generic_catalog_global_ranks = {}
            generic_catalog_distances = {}
            selected_phones = tuple(
                item.phones for _, _, item in ranked[: args.edge_candidates]
            )
            if use_generic:
                generic_started = perf_counter()
                query_cloud = lane_chunks(
                    joint_gauge_chunks(
                        tuple(region_clouds[start:stop]), 1024
                    )
                )
                query_signature = conditional_ridge_signature(
                    query_cloud,
                    args.generic_time_bins,
                    args.generic_row_quantiles,
                )
                generic_rows_by_policy = {
                    policy: [] for policy in generic_context_policies
                }
                verifier_items = list(
                    enumerate(ranked[: args.generic_verifier_candidates], start=1)
                )
                route_strata = {
                    item.phones: proposal_evidence_stratum(relational_cost, item.phones)
                    for relational_cost, _, item in ranked
                }
                if generic_catalog is not None:
                    coarse_signature = conditional_ridge_signature(
                        query_cloud,
                        generic_catalog.time_bins,
                        generic_catalog.row_quantiles,
                    )
                    allowed_lengths = {
                        len(item.phones) for _, _, item in ranked
                    }
                    coarse_rows = generic_catalog.rank(
                        coarse_signature,
                        allowed_lengths,
                        count=int(generic_catalog.phone_keys.size),
                        maximum_shift=max(1, generic_catalog.time_bins // 16),
                    )
                    route_item_by_phones = {
                        item.phones: (route_rank, row)
                        for route_rank, row in enumerate(ranked, start=1)
                        for item in (row[2],)
                    }
                    stratum_counts = {}
                    for global_catalog_rank, (phones, distance) in enumerate(
                        coarse_rows, start=1
                    ):
                        routed = route_item_by_phones.get(phones)
                        if routed is None:
                            continue
                        stratum = route_strata[phones]
                        stratum_rank = stratum_counts.get(stratum, 0) + 1
                        stratum_counts[stratum] = stratum_rank
                        if stratum_rank > args.generic_catalog_candidates:
                            continue
                        generic_catalog_ranks[phones] = stratum_rank
                        generic_catalog_global_ranks[phones] = global_catalog_rank
                        generic_catalog_distances[phones] = distance
                        verifier_items.append(routed)
                unique_verifier_items = {}
                for route_rank, row in verifier_items:
                    unique_verifier_items.setdefault(row[2].phones, (route_rank, row))
                for route_rank, (_, _, item) in unique_verifier_items.values():
                    if any(
                        phone not in generic_bank_labels for phone in item.phones
                    ):
                        continue
                    for policy in generic_context_policies:
                        distance = word_signature_distance(
                            query_signature,
                            generic_signatures(policy, item.phones),
                            maximum_shift=max(2, args.generic_time_bins // 16),
                            aggregation=args.generic_realization_aggregation,
                        )
                        generic_rows_by_policy[policy].append(
                            (distance, route_rank, item.phones)
                        )
                for policy, rows in generic_rows_by_policy.items():
                    local_ranks, global_ranks, ordered = stratified_generic_ranks(
                        rows, route_strata
                    )
                    generic_ranks_by_policy[policy] = local_ranks
                    generic_global_ranks_by_policy[policy] = global_ranks
                    generic_distances_by_policy[policy] = {
                        phones: float(distance)
                        for distance, _, phones in ordered
                    }
                selected_phones = tuple(
                    dict.fromkeys(
                        [item.phones for _, _, item in ranked[: args.edge_candidates]]
                        + [
                            phones
                            for policy in generic_context_policies
                            for phones, rank in generic_ranks_by_policy[policy].items()
                            if rank <= args.edge_candidates
                        ]
                    )
                )
                generic_query_seconds += perf_counter() - generic_started
            if audit_words:
                for rank, (relational_cost, cost, item) in enumerate(ranked, start=1):
                    for word in audit_words:
                        if word in item.words:
                            audit_hits[word].append(
                                {
                                    "phone0": start,
                                    "phone1": stop,
                                    "rank": rank,
                                    "candidate_count": len(ranked),
                                    "phones": list(item.phones),
                                    "cost": asdict(cost),
                                    "relational_cost": asdict(relational_cost),
                                }
                            )
            classes = []
            selected = set(selected_phones)
            for relational_cost, cost, item in ranked:
                if item.phones not in selected:
                    continue
                generic_word_channels = {}
                candidate_policy_ranks = {}
                for policy in generic_context_policies:
                    policy_rank = generic_ranks_by_policy.get(policy, {}).get(
                        item.phones
                    )
                    policy_distance = generic_distances_by_policy.get(
                        policy, {}
                    ).get(item.phones)
                    if policy_rank is not None:
                        generic_word_channels[policy] = {
                            "rank": policy_rank,
                            "distance": policy_distance,
                        }
                    candidate_policy_ranks[policy] = policy_rank
                proposal_channels = selected_proposal_channels(
                    route_ranks[item.phones],
                    candidate_policy_ranks,
                    args.edge_candidates,
                )
                primary_rank = generic_ranks_by_policy.get(
                    generic_context_policy, {}
                ).get(item.phones)
                primary_distance = generic_distances_by_policy.get(
                    generic_context_policy, {}
                ).get(item.phones)
                classes.append(
                    {
                        "score": scalar_provenance_cost(
                            cost, stop - start, args.top_k_per_channel
                        )
                        + scalar_relational_cost(
                            relational_cost,
                            stop - start,
                            args.maximum_boundary_rank,
                        ),
                        "phones": list(item.phones),
                        "words": list(item.words),
                        "provenance_cost": asdict(cost),
                        "relational_cost": asdict(relational_cost),
                        "route_rank": route_ranks[item.phones],
                        "generic_word_rank": primary_rank,
                        "generic_word_distance": primary_distance,
                        "generic_word_channels": generic_word_channels,
                        "generic_word_global_ranks": {
                            policy: generic_global_ranks_by_policy.get(policy, {}).get(
                                item.phones
                            )
                            for policy in generic_context_policies
                            if item.phones
                            in generic_global_ranks_by_policy.get(policy, {})
                        },
                        "generic_catalog_rank": generic_catalog_ranks.get(
                            item.phones
                        ),
                        "generic_catalog_global_rank": generic_catalog_global_ranks.get(
                            item.phones
                        ),
                        "generic_catalog_distance": generic_catalog_distances.get(
                            item.phones
                        ),
                        "proposal_channels": proposal_channels,
                    }
                )
            edges.append(
                {
                    "phone0": start,
                    "phone1": stop,
                    "seconds0": float(phone_rows[start]["seconds0"]),
                    "seconds1": float(phone_rows[stop - 1]["seconds1"]),
                    "duration_seconds": float(phone_rows[stop - 1]["seconds1"])
                    - float(phone_rows[start]["seconds0"]),
                    "observed_phone_count": stop - start,
                    "route_candidate_count": len(route.candidates),
                    "ambiguity_classes": classes,
                }
            )
        if (start + 1) % 25 == 0:
            print(f"routed spans from {start + 1}/{phone_count} regions", flush=True)
    route_seconds = perf_counter() - route_started
    generic_cache_save_seconds_by_policy = {
        policy: 0.0 for policy in generic_context_policies
    }
    for policy, cache in persistent_generic_caches.items():
        if cache.dirty:
            cache_started = perf_counter()
            cache.save(persistent_generic_cache_paths[policy])
            generic_cache_save_seconds_by_policy[policy] = (
                perf_counter() - cache_started
            )
    generic_cache_save_seconds = sum(
        generic_cache_save_seconds_by_policy.values()
    )
    generic_cache_bytes_by_policy = {
        policy: (
            persistent_generic_cache_paths[policy].stat().st_size
            if persistent_generic_cache_paths[policy].exists()
            else 0
        )
        for policy in persistent_generic_caches
    }
    generic_cache_bytes = sum(generic_cache_bytes_by_policy.values())
    candidate_counts.sort()
    result = {
        "method": "aligned_positional_phone_and_boundary_proposal_word_lattice",
        "selection_status": (
            "route and generic-word policy proposals retained independently; no terminal fusion"
            if use_generic
            else "top-k provenance retained; no sentence path selected"
        ),
        "fusion_lattice": str(args.fusion_lattice),
        "dictionary": {
            "path": str(args.cmudict),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
            "pronunciation_count": len(pronunciations),
        },
        "parameters": {
            "top_k_per_channel": args.top_k_per_channel,
            "maximum_uncovered_phones": args.maximum_uncovered_phones,
            "maximum_boundary_rank": args.maximum_boundary_rank,
            "maximum_uncovered_boundaries": args.maximum_uncovered_boundaries,
            "maximum_length_delta": args.maximum_length_delta,
            "max_phone_span": args.max_phone_span,
            "edge_candidates": args.edge_candidates,
            "max_gap_seconds": args.max_gap_seconds,
            "generic_verifier_candidates": args.generic_verifier_candidates,
            "generic_realizations": args.generic_realizations if use_generic else 0,
            "generic_time_bins": args.generic_time_bins if use_generic else 0,
            "generic_row_quantiles": (
                args.generic_row_quantiles if use_generic else 0
            ),
            "generic_realization_aggregation": (
                args.generic_realization_aggregation if use_generic else None
            ),
            "generic_context_conditioned": (
                args.generic_context_conditioned if use_generic else False
            ),
            "generic_context_policy": (
                generic_context_policy if use_generic else None
            ),
            "generic_context_policies": (
                list(generic_context_policies) if use_generic else []
            ),
            "generic_signature_cache": (
                str(args.generic_signature_cache)
                if args.generic_signature_cache
                else None
            ),
            "generic_pronunciation_catalog": (
                str(args.generic_pronunciation_catalog)
                if args.generic_pronunciation_catalog
                else None
            ),
            "generic_catalog_candidates": args.generic_catalog_candidates,
            "generic_cache_dtype": (
                args.generic_cache_dtype if args.generic_signature_cache else None
            ),
        },
        "phone_count": phone_count,
        "edge_count": len(edges),
        "diagnostics": {
            "index_seconds": index_seconds,
            "route_seconds": route_seconds,
            "generic_query_seconds": generic_query_seconds,
            "generic_template_count": len(generic_template_cache),
            "generic_cache_load_seconds": generic_cache_load_seconds,
            "generic_cache_save_seconds": generic_cache_save_seconds,
            "generic_cache_hits": generic_cache_hits,
            "generic_cache_misses": generic_cache_misses,
            "generic_cache_entry_count": {
                policy: len(cache.entries)
                for policy, cache in persistent_generic_caches.items()
            },
            "generic_cache_bytes": generic_cache_bytes,
            "generic_cache_bytes_by_policy": generic_cache_bytes_by_policy,
            "generic_cache_save_seconds_by_policy": (
                generic_cache_save_seconds_by_policy
            ),
            "spans_considered": spans_considered,
            "routed_spans": routed_spans,
            "routed_fraction": routed_spans / max(spans_considered, 1),
            "candidate_count_median": (
                candidate_counts[len(candidate_counts) // 2]
                if candidate_counts
                else 0
            ),
            "candidate_count_maximum": max(candidate_counts, default=0),
        },
        "audit_words": audit_hits,
        "edges": edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "phone_count": phone_count,
                "edge_count": len(edges),
                "diagnostics": result["diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
