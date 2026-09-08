#!/usr/bin/env python3
"""Score one exact word anchor from independent phone-evidence channels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .word_lattice import (
    ARPABET_39,
    PronunciationProposalIndex,
    parse_cmudict,
    boundary_evidence_from_span_result,
    provenance_evidence_from_context_result,
    rank_pronunciation_oracle,
    relational_sequence_cost,
    provenance_edit_cost,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("target_word")
    parser.add_argument("phone_results", nargs="+", type=Path)
    parser.add_argument("--top-k-per-channel", type=int, default=8)
    parser.add_argument("--maximum-uncovered-phones", type=int, default=0)
    parser.add_argument("--maximum-boundary-rank", type=int)
    parser.add_argument("--maximum-length-delta", type=int, default=1)
    parser.add_argument("--report-count", type=int, default=20)
    parser.add_argument("--span-battery", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.report_count < 1:
        raise ValueError("report count must be positive")

    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(),
        frozenset(ARPABET_39),
    )
    documents = [json.loads(path.read_text()) for path in args.phone_results]
    evidence = tuple(
        provenance_evidence_from_context_result(
            document, args.top_k_per_channel
        )
        for document in documents
    )
    observed_targets = tuple(
        str(document["summary"]["target_phone"]) for document in documents
    )

    started = perf_counter()
    ranking = rank_pronunciation_oracle(
        evidence,
        pronunciations,
        maximum_length_delta=args.maximum_length_delta,
    )
    elapsed = perf_counter() - started
    target_word = args.target_word.lower()
    target_rank = next(
        (
            index + 1
            for index, item in enumerate(ranking)
            if target_word in item.words
        ),
        None,
    )
    target_item = ranking[target_rank - 1] if target_rank is not None else None
    target_pronunciations = sorted(
        {entry.phones for entry in pronunciations if entry.word == target_word}
    )

    def serialize(item) -> dict:
        return {
            "phones": list(item.phones),
            "words": list(item.words),
            "cost": asdict(item.cost),
        }

    relational = None
    routed = None
    if args.span_battery is not None:
        span_document = json.loads(args.span_battery.read_text())
        boundary_evidence = boundary_evidence_from_span_result(span_document)
        relational_ranking = [
            (
                relational_sequence_cost(item.cost, item.phones, boundary_evidence),
                item,
            )
            for item in ranking
        ]
        relational_ranking.sort(
            key=lambda pair: (pair[0], pair[1].phones, pair[1].words)
        )
        relational_target_rank = next(
            (
                index + 1
                for index, (_, item) in enumerate(relational_ranking)
                if target_word in item.words
            ),
            None,
        )
        relational_target = (
            relational_ranking[relational_target_rank - 1]
            if relational_target_rank is not None
            else None
        )
        relational = {
            "span_battery": str(args.span_battery),
            "target_word_rank": relational_target_rank,
            "target": {
                **serialize(relational_target[1]),
                "relational_cost": asdict(relational_target[0]),
            }
            if relational_target
            else None,
            "top": [
                {**serialize(item), "relational_cost": asdict(cost)}
                for cost, item in relational_ranking[: args.report_count]
            ],
        }
        index_started = perf_counter()
        proposal_index = PronunciationProposalIndex(pronunciations)
        index_elapsed = perf_counter() - index_started
        route_started = perf_counter()
        route = proposal_index.query(
            evidence,
            boundary_evidence,
            maximum_uncovered_phones=args.maximum_uncovered_phones,
            maximum_boundary_rank=args.maximum_boundary_rank,
        )
        route_elapsed = perf_counter() - route_started
        score_started = perf_counter()
        routed_ranking = []
        for ordinal in route.boundary_candidates:
            item = proposal_index.classes[ordinal]
            phone_cost = provenance_edit_cost(evidence, item.phones)
            routed_ranking.append(
                (relational_sequence_cost(phone_cost, item.phones, boundary_evidence), item, phone_cost)
            )
        routed_ranking.sort(key=lambda item: (item[0], item[1].phones, item[1].words))
        score_elapsed = perf_counter() - score_started
        routed_target_rank = next(
            (
                index + 1
                for index, (_, item, _) in enumerate(routed_ranking)
                if target_word in item.words
            ),
            None,
        )
        routed = {
            "index_build_seconds": index_elapsed,
            "route_seconds": route_elapsed,
            "score_seconds": score_elapsed,
            "phone_candidate_count": len(route.phone_candidates),
            "boundary_candidate_count": len(route.boundary_candidates),
            "phone_posting_reads": route.phone_posting_reads,
            "boundary_posting_reads": route.boundary_posting_reads,
            "target_word_rank": routed_target_rank,
            "top": [
                {
                    "phones": list(item.phones),
                    "words": list(item.words),
                    "cost": asdict(phone_cost),
                    "relational_cost": asdict(cost),
                }
                for cost, item, phone_cost in routed_ranking[: args.report_count]
            ],
        }

    result = {
        "method": "exhaustive_exact_anchor_provenance_evidence_semiring",
        "selection_status": "oracle_measurement_not_yet_compressed",
        "dictionary": {
            "path": str(args.cmudict),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
            "pronunciation_count": len(pronunciations),
        },
        "parameters": {
            "top_k_per_channel": args.top_k_per_channel,
            "maximum_uncovered_phones": args.maximum_uncovered_phones,
            "maximum_boundary_rank": args.maximum_boundary_rank,
            "maximum_length_delta": args.maximum_length_delta,
            "channel_order": list(evidence[0].channel_ranks) if evidence else [],
            "cost_order": [
                "uncovered",
                "edits",
                "worst_rank",
                "rank_sum",
                "negative_channel_support",
            ],
        },
        "query": {
            "target_word": target_word,
            "observed_target_phones": list(observed_targets),
            "phone_result_paths": [str(path) for path in args.phone_results],
            "dictionary_pronunciations": [
                list(phones) for phones in target_pronunciations
            ],
        },
        "oracle": {
            "scored_pronunciation_classes": len(ranking),
            "elapsed_seconds": elapsed,
            "classes_per_second": len(ranking) / max(elapsed, 1e-12),
            "target_word_rank": target_rank,
            "target": serialize(target_item) if target_item else None,
            "top": [serialize(item) for item in ranking[: args.report_count]],
        },
        "relational_oracle": relational,
        "indexed_route": routed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "target_word": target_word,
                "target_word_rank": target_rank,
                "relational_target_word_rank": (
                    relational["target_word_rank"] if relational else None
                ),
                "indexed_target_word_rank": (
                    routed["target_word_rank"] if routed else None
                ),
                "scored_classes": len(ranking),
                "elapsed_seconds": elapsed,
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
