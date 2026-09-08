#!/usr/bin/env python3
"""Measure listener-phone recall of provenance-preserving proposal unions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .transcript_phone_alignment import (
    align_phone_proposals_trace,
    markdown_section,
    transcript_phone_sequence,
    transcript_words,
)
from .word_lattice import ARPABET_39, parse_cmudict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_lattice", type=Path)
    parser.add_argument("copula_lattice", type=Path)
    parser.add_argument("conditional_lattice", type=Path)
    parser.add_argument("boundary_lattice", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("transcript_markdown", type=Path)
    parser.add_argument("--base-field", default="mean_rank_top5")
    parser.add_argument("--top-k-base", type=int, default=24)
    parser.add_argument("--top-k-channel", type=int, default=5)
    parser.add_argument(
        "--extra-channel",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "LATTICE", "FIELD"),
        help="add a provenance-preserving proposal channel",
    )
    parser.add_argument("--sections", default="Gemini listener,SOL cloud listener")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    documents = {
        "base": json.loads(args.base_lattice.read_text()),
        "copula": json.loads(args.copula_lattice.read_text()),
        "conditional": json.loads(args.conditional_lattice.read_text()),
        "boundary": json.loads(args.boundary_lattice.read_text()),
    }
    channel_fields = {
        "copula": "top5",
        "conditional": "top5",
        "boundary": "boundary_top5",
    }
    for name, path, field in args.extra_channel:
        if name in documents:
            raise ValueError(f"duplicate proposal channel {name!r}")
        documents[name] = json.loads(Path(path).read_text())
        channel_fields[name] = field
    counts = {len(document["phones"]) for document in documents.values()}
    if len(counts) != 1:
        raise ValueError("proposal union lattices have different region counts")
    pronunciations = {}
    for entry in parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39)
    ):
        pronunciations.setdefault(entry.word, []).append(entry.phones)
    for values in pronunciations.values():
        values.sort()
    base_proposals = tuple(
        tuple(str(item["phone"]) for item in row[args.base_field][: args.top_k_base])
        for row in documents["base"]["phones"]
    )
    channel_proposals = {
        name: tuple(
            frozenset(str(item["phone"]) for item in row[field][: args.top_k_channel])
            for row in documents[name]["phones"]
        )
        for name, field in channel_fields.items()
    }
    transcript = args.transcript_markdown.read_text()
    audits = {}
    for section in (value.strip() for value in args.sections.split(",")):
        words = transcript_words(markdown_section(transcript, section))
        phones, missing = transcript_phone_sequence(words, pronunciations)
        alignment, mapping = align_phone_proposals_trace(phones, base_proposals)
        hits = {name: 0 for name in channel_proposals}
        union_hits = 0
        union_sizes = []
        aligned = 0
        for phone, region in zip(phones, mapping, strict=True):
            if region is None:
                continue
            aligned += 1
            union = frozenset().union(
                *(values[region] for values in channel_proposals.values())
            )
            union_sizes.append(len(union))
            union_hits += phone in union
            for name, values in channel_proposals.items():
                hits[name] += phone in values[region]
        audits[section] = {
            "reference_phone_count": len(phones),
            "missing_dictionary_words": list(missing),
            "aligned_count": aligned,
            "channel_hits": hits,
            "channel_recall": {name: value / aligned for name, value in hits.items()},
            "union_hits": union_hits,
            "union_recall": union_hits / aligned,
            "union_size_median": float(np.median(union_sizes)),
            "union_size_p90": float(np.quantile(union_sizes, 0.9)),
            "union_size_maximum": max(union_sizes),
            "base_alignment_edit_cost": alignment.edit_cost,
        }
    result = {
        "method": "fixed_alignment_provenance_preserving_phone_proposal_union",
        "parameters": {
            "top_k_base": args.top_k_base,
            "top_k_channel": args.top_k_channel,
            "channels": channel_fields,
        },
        "audits": audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
