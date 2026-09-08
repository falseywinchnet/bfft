#!/usr/bin/env python3
"""Rank dictionary words by continuous phone and diphone geometry costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .word_lattice import ARPABET_39, PronunciationProposalIndex, parse_cmudict


def normalized_cost_map(
    ranking: list[dict], label_key: str, floor_quantile: float = 0.75
) -> dict[object, float]:
    """Convert one distance ranking into a robust nonnegative local cost."""

    if not ranking:
        raise ValueError("geometry cost map requires a nonempty ranking")
    distances = np.asarray([float(item["distance"]) for item in ranking])
    floor = float(np.min(distances))
    scale = float(np.quantile(distances, floor_quantile) - floor)
    scale = max(scale, float(np.max(distances) - floor) * 0.1, 1e-12)
    output = {}
    for item in ranking:
        label = item[label_key]
        if isinstance(label, list):
            label = tuple(str(value) for value in label)
        else:
            label = str(label)
        output[label] = (float(item["distance"]) - floor) / scale
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fusion_lattice", type=Path)
    parser.add_argument("transcript_alignment", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--section", default="Gemini listener")
    parser.add_argument("--center-field", default="center_top5")
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--word-limit", type=int)
    parser.add_argument("--report-count", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.boundary_weight < 0.0:
        raise ValueError("boundary weight cannot be negative")

    fusion = json.loads(args.fusion_lattice.read_text())
    alignment = json.loads(args.transcript_alignment.read_text())
    spans = alignment["audits"][args.section]["word_region_spans"]
    phone_rows = fusion["phones"]
    boundaries = fusion["joint_boundary_pair_swd_rankings"]
    pronunciations = parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39)
    )
    index = PronunciationProposalIndex(pronunciations)
    by_length: dict[int, list[int]] = {}
    for ordinal, item in enumerate(index.classes):
        by_length.setdefault(len(item.phones), []).append(ordinal)

    center_costs = [
        normalized_cost_map(row[args.center_field], "phone")
        for row in phone_rows
    ]
    boundary_costs = [
        normalized_cost_map(item["ranking"], "phones")
        for item in boundaries
    ]
    boundary_missing = [
        max(costs.values(), default=0.0) + 1.0 for costs in boundary_costs
    ]

    results = []
    started = perf_counter()
    for span in spans:
        if args.word_limit is not None and len(results) >= args.word_limit:
            break
        if span["region0"] is None or span["region1"] is None:
            continue
        region0 = int(span["region0"])
        region1 = int(span["region1"])
        observed_count = region1 - region0
        target_phones = tuple(str(phone) for phone in span["phones"])
        if observed_count != len(target_phones):
            continue
        target_word = str(span["word"])
        candidates = by_length.get(observed_count, ())
        scored = []
        for ordinal in candidates:
            item = index.classes[ordinal]
            center = sum(
                center_costs[region0 + position].get(phone, 100.0)
                for position, phone in enumerate(item.phones)
            ) / observed_count
            if observed_count > 1:
                pair_values = []
                for position, pair in enumerate(zip(item.phones, item.phones[1:])):
                    boundary_index = region0 + position
                    pair_values.append(
                        boundary_costs[boundary_index].get(
                            pair, boundary_missing[boundary_index]
                        )
                    )
                boundary = float(np.mean(pair_values))
            else:
                boundary = 0.0
            scored.append(
                (
                    center,
                    boundary,
                    center + args.boundary_weight * boundary,
                    item.phones,
                    ordinal,
                )
            )

        def rank_for(field: int) -> tuple[int | None, list[dict]]:
            ordered = sorted(scored, key=lambda row: (row[field], row[3]))
            target_rank = next(
                (
                    rank
                    for rank, row in enumerate(ordered, start=1)
                    if row[3] == target_phones
                    and target_word in index.classes[row[4]].words
                ),
                None,
            )
            top = [
                {
                    "rank": rank,
                    "score": float(row[field]),
                    "phones": list(row[3]),
                    "words": list(index.classes[row[4]].words),
                }
                for rank, row in enumerate(ordered[: args.report_count], start=1)
            ]
            return target_rank, top

        center_rank, top_center = rank_for(0)
        boundary_rank, top_boundary = rank_for(1)
        combined_rank, top_combined = rank_for(2)
        results.append(
            {
                "word_index": int(span["word_index"]),
                "word": target_word,
                "phones": list(target_phones),
                "region0": region0,
                "region1": region1,
                "candidate_count": len(scored),
                "center_rank": center_rank,
                "boundary_rank": boundary_rank,
                "combined_rank": combined_rank,
                "top_center": top_center,
                "top_boundary": top_boundary,
                "top_combined": top_combined,
            }
        )
        print(
            f"ranked {len(results)} {target_word}: "
            f"center={center_rank} boundary={boundary_rank} combined={combined_rank}",
            flush=True,
        )

    def summary(field: str) -> dict[str, float | int | None]:
        ranks = [row[field] for row in results if row[field] is not None]
        return {
            "count": len(ranks),
            "top1": sum(rank == 1 for rank in ranks),
            "top5": sum(rank <= 5 for rank in ranks),
            "top20": sum(rank <= 20 for rank in ranks),
            "median_rank": float(np.median(ranks)) if ranks else None,
            "mean_reciprocal_rank": (
                float(np.mean([1.0 / rank for rank in ranks])) if ranks else None
            ),
        }

    result = {
        "method": "exact_cardinality_continuous_phone_diphone_lexical_geometry",
        "selection_status": "transcript used only for oracle rank measurement",
        "section": args.section,
        "parameters": {
            "center_field": args.center_field,
            "boundary_weight": args.boundary_weight,
        },
        "elapsed_seconds": perf_counter() - started,
        "summary": {
            "word_count": len(results),
            "center": summary("center_rank"),
            "boundary": summary("boundary_rank"),
            "combined": summary("combined_rank"),
        },
        "words": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
