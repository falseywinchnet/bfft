#!/usr/bin/env python3
"""Audit uncertain phone regions against listener-supported transcript sections."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .transcript_phone_alignment import (
    align_phone_proposals,
    align_phone_proposals_trace,
    markdown_section,
    transcript_phone_sequence,
    transcript_word_phone_spans,
    transcript_words,
)
from .word_lattice import ARPABET_39, parse_cmudict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_lattice", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("transcript_markdown", type=Path)
    parser.add_argument(
        "--sections", default="Gemini listener,SOL cloud listener"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ranking-field", default="top5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("transcript phone audit top-k must be positive")

    dictionary_entries = parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39)
    )
    pronunciations: dict[str, list[tuple[str, ...]]] = {}
    for entry in dictionary_entries:
        pronunciations.setdefault(entry.word, []).append(entry.phones)
    for options in pronunciations.values():
        options.sort()

    phone_document = json.loads(args.phone_lattice.read_text())
    proposals = tuple(
        tuple(
            str(item["phone"])
            for item in row[args.ranking_field][: args.top_k]
        )
        for row in phone_document["phones"]
    )
    transcript = args.transcript_markdown.read_text()
    audits = {}
    for heading in (item.strip() for item in args.sections.split(",")):
        words = transcript_words(markdown_section(transcript, heading))
        word_spans, missing_words = transcript_word_phone_spans(words, pronunciations)
        phones = tuple(phone for span in word_spans for phone in span.phones)
        alignment, phone_mapping = align_phone_proposals_trace(phones, proposals)
        record = asdict(alignment)
        record.update(
            {
                "transcript_word_count": len(words),
                "dictionary_word_count": len(words) - len(missing_words),
                "missing_dictionary_words": list(missing_words),
                "top1_recall_over_aligned": (
                    alignment.top1_hits / alignment.aligned_count
                ),
                "topk_recall_over_aligned": (
                    alignment.topk_hits / alignment.aligned_count
                ),
                "normalized_edit_cost": (
                    alignment.edit_cost
                    / max(alignment.reference_phone_count, alignment.proposal_region_count)
                ),
                "word_region_spans": [
                    {
                        "word_index": index,
                        "word": span.word,
                        "phones": list(span.phones),
                        "reference_phone0": span.phone0,
                        "reference_phone1": span.phone1,
                        "region0": (
                            min(mapped) if mapped else None
                        ),
                        "region1": (
                            max(mapped) + 1 if mapped else None
                        ),
                        "aligned_phone_count": len(mapped),
                    }
                    for index, span in enumerate(word_spans)
                    for mapped in [
                        [
                            region
                            for region in phone_mapping[span.phone0 : span.phone1]
                            if region is not None
                        ]
                    ]
                ],
            }
        )
        audits[heading] = record
    result = {
        "method": "listener_transcript_to_uncertain_phone_region_global_alignment",
        "phone_lattice": str(args.phone_lattice),
        "dictionary": str(args.cmudict),
        "transcript": str(args.transcript_markdown),
        "top_k": args.top_k,
        "ranking_field": args.ranking_field,
        "audits": audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
