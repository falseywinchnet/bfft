#!/usr/bin/env python3
"""Build an ambiguity-preserving CMUdict word lattice from phone top-k data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

from experiments.ostensibly_frontend.word_lattice import (
    ARPABET_39,
    Pronunciation,
    PronunciationIndex,
    acoustic_edit_cost,
    k_best_phone_sequences,
    normalized_phone_evidence,
    parse_cmudict,
)


def _mutations(
    phones: tuple[str, ...],
    alphabet: Sequence[str],
    selector: int,
) -> tuple[tuple[str, ...], ...]:
    if not phones:
        return ()
    position = selector % len(phones)
    replacement = alphabet[(alphabet.index(phones[position]) + 1) % len(alphabet)]
    substitution = phones[:position] + (replacement,) + phones[position + 1 :]
    deletion = phones[:position] + phones[position + 1 :]
    insertion_phone = alphabet[(selector * 7 + 3) % len(alphabet)]
    insertion = phones[:position] + (insertion_phone,) + phones[position:]
    return substitution, deletion, insertion


def audit_index(
    index: PronunciationIndex,
    sample_count: int = 4096,
) -> dict[str, object]:
    rng = random.Random(0x057E451B1E)
    ordinals = list(range(len(index.pronunciations)))
    rng.shuffle(ordinals)
    ordinals = ordinals[: min(sample_count, len(ordinals))]
    hits = {"exact": 0, "substitution": 0, "deletion": 0, "insertion": 0}
    fanouts = []
    for ordinal in ordinals:
        phones = index.pronunciations[ordinal].phones
        exact = index.query(phones)
        hits["exact"] += ordinal in exact.candidates
        fanouts.append(len(exact.candidates))
        for kind, mutation in zip(
            ("substitution", "deletion", "insertion"),
            _mutations(phones, tuple(sorted(index.atom_codes)), ordinal),
            strict=True,
        ):
            audit = index.query(mutation)
            hits[kind] += ordinal in audit.candidates
            fanouts.append(len(audit.candidates))
    denominator = max(len(ordinals), 1)
    fanouts.sort()
    return {
        "sample_count": len(ordinals),
        "recall": {kind: count / denominator for kind, count in hits.items()},
        "candidate_fanout": {
            "median": fanouts[len(fanouts) // 2] if fanouts else 0,
            "q95": fanouts[int(0.95 * (len(fanouts) - 1))] if fanouts else 0,
            "maximum": max(fanouts, default=0),
        },
    }


def _ambiguity_classes(
    candidates: set[int],
    index: PronunciationIndex,
    evidence,
    count: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], set[str]] = {}
    for ordinal in candidates:
        pronunciation = index.pronunciations[ordinal]
        grouped.setdefault(pronunciation.phones, set()).add(pronunciation.word)
    ranked = sorted(
        (
            acoustic_edit_cost(evidence, phones),
            phones,
            tuple(sorted(words)),
        )
        for phones, words in grouped.items()
    )
    return [
        {"score": score, "phones": list(phones), "words": list(words)}
        for score, phones, words in ranked[:count]
    ]


def build_edges(
    phone_rows: Sequence[dict[str, object]],
    index: PronunciationIndex,
    max_phone_span: int,
    beam_size: int,
    edge_candidates: int,
    max_gap_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evidence = [normalized_phone_evidence(row["top5"]) for row in phone_rows]
    edges = []
    spans_considered = 0
    spans_with_candidates = 0
    proposal_fanouts = []
    for start in range(len(phone_rows)):
        span_evidence = []
        for stop in range(start + 1, min(len(phone_rows), start + max_phone_span) + 1):
            if stop > start + 1:
                gap = (
                    float(phone_rows[stop - 1]["seconds0"])
                    - float(phone_rows[stop - 2]["seconds1"])
                )
                if gap > max_gap_seconds:
                    break
            span_evidence.append(evidence[stop - 1])
            spans_considered += 1
            candidate_ids: set[int] = set()
            beams = k_best_phone_sequences(span_evidence, beam_size)
            for sequence, _ in beams:
                candidate_ids.update(index.query(sequence).candidates)
            proposal_fanouts.append(len(candidate_ids))
            if not candidate_ids:
                continue
            spans_with_candidates += 1
            classes = _ambiguity_classes(
                candidate_ids, index, span_evidence, edge_candidates
            )
            edges.append({
                "phone0": start,
                "phone1": stop,
                "seconds0": float(phone_rows[start]["seconds0"]),
                "seconds1": float(phone_rows[stop - 1]["seconds1"]),
                "duration_seconds": (
                    float(phone_rows[stop - 1]["seconds1"])
                    - float(phone_rows[start]["seconds0"])
                ),
                "observed_phone_count": stop - start,
                "route_beam": [
                    {"phones": list(sequence), "cost": cost}
                    for sequence, cost in beams[:5]
                ],
                "ambiguity_classes": classes,
            })
        if (start + 1) % 25 == 0:
            print(f"word spans from {start + 1}/{len(phone_rows)} phones", flush=True)
    proposal_fanouts.sort()
    return edges, {
        "spans_considered": spans_considered,
        "spans_with_candidates": spans_with_candidates,
        "candidate_coverage": spans_with_candidates / max(spans_considered, 1),
        "candidate_fanout": {
            "median": proposal_fanouts[len(proposal_fanouts) // 2]
            if proposal_fanouts else 0,
            "q95": proposal_fanouts[int(0.95 * (len(proposal_fanouts) - 1))]
            if proposal_fanouts else 0,
            "maximum": max(proposal_fanouts, default=0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_lattice_json", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-phone-span", type=int, default=18)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--edge-candidates", type=int, default=8)
    parser.add_argument("--max-gap-seconds", type=float, default=0.30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(),
        frozenset(ARPABET_39),
        max_phones=args.max_phone_span,
    )
    print(f"building index for {len(pronunciations)} pronunciations", flush=True)
    index = PronunciationIndex(pronunciations, ARPABET_39)
    gate = audit_index(index)
    print(json.dumps({"history_occupancy_gate": gate}, indent=2), flush=True)

    phone_lattice = json.loads(args.phone_lattice_json.read_text())
    edges, edge_diagnostics = build_edges(
        phone_lattice["phones"],
        index,
        args.max_phone_span,
        args.beam_size,
        args.edge_candidates,
        args.max_gap_seconds,
    )
    result = {
        "method": (
            "coupled_history_occupancy_candidate_routing_then_"
            "exact_anchor_acoustic_dynamic_programming"
        ),
        "selection_status": (
            "ambiguity_preserving_word_edges; no sentence path selected"
        ),
        "phone_lattice": str(args.phone_lattice_json),
        "dictionary": {
            "path": str(args.cmudict),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
            "pronunciation_count": len(pronunciations),
            "maximum_phones": args.max_phone_span,
        },
        "parameters": {
            "phone_inventory": list(ARPABET_39),
            "beam_size": args.beam_size,
            "edge_candidates": args.edge_candidates,
            "max_gap_seconds": args.max_gap_seconds,
        },
        "history_occupancy_gate": gate,
        "edge_diagnostics": edge_diagnostics,
        "edge_count": len(edges),
        "edges": edges,
    }
    output = args.out / "word_candidate_lattice.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"edge_count": len(edges), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
