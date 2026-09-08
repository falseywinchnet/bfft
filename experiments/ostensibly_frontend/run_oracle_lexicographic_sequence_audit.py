#!/usr/bin/env python3
"""Verify exact-length words by complete ordered local-rank histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .word_lattice import ARPABET_39, PronunciationProposalIndex, parse_cmudict


def rank_map(ranking: list[dict], label_field: str) -> dict[object, int]:
    output = {}
    for rank, item in enumerate(ranking, start=1):
        label = item[label_field]
        if isinstance(label, list):
            label = tuple(str(value) for value in label)
        else:
            label = str(label)
        output[label] = int(item.get("rank", rank))
    return output


def local_history_key(ranks: list[int]) -> tuple[int, ...]:
    """Minimax ordering with every next-worst local mismatch retained."""

    return tuple(sorted((int(rank) for rank in ranks), reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_lattice", type=Path)
    parser.add_argument("boundary_lattice", type=Path)
    parser.add_argument("transcript_alignment", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--section", default="Gemini listener")
    parser.add_argument("--phone-field", default="top5")
    parser.add_argument("--maximum-boundary-rank", type=int, default=128)
    parser.add_argument("--maximum-uncovered-boundaries", type=int, default=1)
    parser.add_argument(
        "--boundary-mode", choices=("admit", "ignore"), default="admit"
    )
    parser.add_argument("--report-count", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    phones_document = json.loads(args.phone_lattice.read_text())
    boundary_document = json.loads(args.boundary_lattice.read_text())
    alignment = json.loads(args.transcript_alignment.read_text())
    spans = alignment["audits"][args.section]["word_region_spans"]
    phone_maps = [
        rank_map(row[args.phone_field], "phone")
        for row in phones_document["phones"]
    ]
    boundary_maps = [
        rank_map(row["ranking"], "phones")
        for row in boundary_document["joint_boundary_pair_swd_rankings"]
    ]
    index = PronunciationProposalIndex(
        parse_cmudict(
            args.cmudict.read_text().splitlines(), frozenset(ARPABET_39)
        )
    )
    by_length = {}
    for ordinal, item in enumerate(index.classes):
        by_length.setdefault(len(item.phones), []).append(ordinal)

    results = []
    started = perf_counter()
    for span in spans:
        if span["region0"] is None or span["region1"] is None:
            continue
        region0 = int(span["region0"])
        region1 = int(span["region1"])
        target_phones = tuple(str(phone) for phone in span["phones"])
        observed_count = region1 - region0
        if len(target_phones) != observed_count:
            continue
        target_word = str(span["word"])
        scored = []
        for ordinal in by_length.get(observed_count, ()):
            item = index.classes[ordinal]
            local_ranks = [
                phone_maps[region0 + position][phone]
                for position, phone in enumerate(item.phones)
            ]
            boundary_ranks = [
                boundary_maps[region0 + position].get(pair)
                for position, pair in enumerate(zip(item.phones, item.phones[1:]))
            ]
            uncovered = sum(
                rank is None or rank > args.maximum_boundary_rank
                for rank in boundary_ranks
            )
            admissible = (
                args.boundary_mode == "ignore"
                or uncovered <= args.maximum_uncovered_boundaries
            )
            ordering_uncovered = (
                uncovered if args.boundary_mode == "admit" else 0
            )
            scored.append(
                (
                    not admissible,
                    ordering_uncovered,
                    local_history_key(local_ranks),
                    sum(local_ranks),
                    item.phones,
                    ordinal,
                    uncovered,
                )
            )
        scored.sort()
        target_rank = next(
            (
                rank
                for rank, row in enumerate(scored, start=1)
                if row[4] == target_phones
                and target_word in index.classes[row[5]].words
            ),
            None,
        )
        target_row = next(
            (
                row
                for row in scored
                if row[4] == target_phones
                and target_word in index.classes[row[5]].words
            ),
            None,
        )
        results.append(
            {
                "word_index": int(span["word_index"]),
                "word": target_word,
                "phones": list(target_phones),
                "region0": region0,
                "region1": region1,
                "candidate_count": len(scored),
                "target_rank": target_rank,
                "target_admissible": bool(target_row is not None and not target_row[0]),
                "target_uncovered_boundaries": (
                    int(target_row[6]) if target_row is not None else None
                ),
                "target_local_history": (
                    list(target_row[2]) if target_row is not None else None
                ),
                "top": [
                    {
                        "rank": rank,
                        "admissible": not row[0],
                        "uncovered_boundaries": row[6],
                        "local_history": list(row[2]),
                        "phones": list(row[4]),
                        "words": list(index.classes[row[5]].words),
                    }
                    for rank, row in enumerate(scored[: args.report_count], start=1)
                ],
            }
        )
        print(
            f"verified {len(results)} {target_word}: rank={target_rank} "
            f"admissible={results[-1]['target_admissible']}",
            flush=True,
        )
    ranks = [row["target_rank"] for row in results if row["target_rank"] is not None]
    result = {
        "method": "boundary_admission_then_lexicographic_conditional_phone_history",
        "selection_status": "transcript used only for oracle rank measurement",
        "parameters": {
            "maximum_boundary_rank": args.maximum_boundary_rank,
            "maximum_uncovered_boundaries": args.maximum_uncovered_boundaries,
            "boundary_mode": args.boundary_mode,
        },
        "elapsed_seconds": perf_counter() - started,
        "summary": {
            "word_count": len(results),
            "target_admitted": sum(row["target_admissible"] for row in results),
            "top1": sum(rank == 1 for rank in ranks),
            "top5": sum(rank <= 5 for rank in ranks),
            "top20": sum(rank <= 20 for rank in ranks),
            "median_rank": float(np.median(ranks)) if ranks else None,
            "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])) if ranks else None,
        },
        "words": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
