#!/usr/bin/env python3
"""Measure an optimistic phone top-k ceiling against listener transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from experiments.ostensibly_frontend.word_lattice import (
    ARPABET_39,
    normalized_phone_evidence,
    parse_cmudict,
)


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SECTIONS = {
    "gemini": ("## Gemini listener", "## SOL cloud listener"),
    "sol": ("## SOL cloud listener", "## Evaluation policy"),
}


def _section_words(markdown: str, start: str, stop: str) -> tuple[str, ...]:
    if start not in markdown or stop not in markdown:
        raise ValueError(f"transcript section markers missing: {start!r}, {stop!r}")
    body = markdown.split(start, 1)[1].split(stop, 1)[0]
    return tuple(token.lower() for token in TOKEN_RE.findall(body))


def _reference_phones(words, lexicon):
    reference = []
    missing = []
    for word_index, word in enumerate(words):
        pronunciations = lexicon.get(word)
        if not pronunciations:
            missing.append(word)
            continue
        # The audit is already deliberately optimistic in its time alignment.
        # A deterministic shortest declared pronunciation avoids adding a
        # second hidden pronunciation search to the reported ceiling.
        phones = min(pronunciations, key=lambda item: (len(item), item))
        reference.extend((word_index, word, phone) for phone in phones)
    return tuple(reference), tuple(missing)


def optimistic_alignment_audit(reference, phone_rows, skip_penalty=1.1):
    evidence = [normalized_phone_evidence(row["top5"]) for row in phone_rows]
    rows = len(reference)
    columns = len(evidence)
    costs = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    backtrace = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        costs[row][0] = row * skip_penalty
        backtrace[row][0] = 1
    for column in range(1, columns + 1):
        costs[0][column] = column * skip_penalty
        backtrace[0][column] = 2
    for row in range(1, rows + 1):
        phone = reference[row - 1][2]
        for column in range(1, columns + 1):
            observed = evidence[column - 1]
            alternatives = (
                costs[row - 1][column - 1]
                + observed.costs.get(phone, observed.unseen_cost),
                costs[row - 1][column] + skip_penalty,
                costs[row][column - 1] + skip_penalty,
            )
            choice = min(range(3), key=lambda index: (alternatives[index], index))
            costs[row][column] = alternatives[choice]
            backtrace[row][column] = choice

    row = rows
    column = columns
    pairs = []
    reference_skips = 0
    observation_skips = 0
    while row or column:
        choice = backtrace[row][column]
        if row and column and choice == 0:
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and (not column or choice == 1):
            reference_skips += 1
            row -= 1
        else:
            observation_skips += 1
            column -= 1
    pairs.reverse()
    hits = {1: 0, 3: 0, 5: 0}
    for reference_index, observation_index in pairs:
        target = reference[reference_index][2]
        ranked = [
            str(item["phone"])
            for item in phone_rows[observation_index]["top5"]
        ]
        for count in hits:
            hits[count] += int(target in ranked[:count])
    matched = len(pairs)
    return {
        "interpretation": (
            "optimistic monotone forced-alignment ceiling; not independent accuracy"
        ),
        "reference_phone_count": rows,
        "observation_count": columns,
        "matched_pairs": matched,
        "reference_skips": reference_skips,
        "observation_skips": observation_skips,
        "top1_ceiling": hits[1] / max(matched, 1),
        "top3_ceiling": hits[3] / max(matched, 1),
        "top5_ceiling": hits[5] / max(matched, 1),
        "cost_per_reference_phone": costs[rows][columns] / max(rows, 1),
        "skip_penalty": skip_penalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_lattice_json", type=Path)
    parser.add_argument("transcript_markdown", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    phone_rows = json.loads(args.phone_lattice_json.read_text())["phones"]
    markdown = args.transcript_markdown.read_text()
    entries = parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39), 18
    )
    lexicon = {}
    for entry in entries:
        lexicon.setdefault(entry.word, []).append(entry.phones)

    listeners = {}
    for name, markers in SECTIONS.items():
        words = _section_words(markdown, *markers)
        reference, missing = _reference_phones(words, lexicon)
        listeners[name] = {
            "word_count": len(words),
            "missing_dictionary_words": list(missing),
            **optimistic_alignment_audit(reference, phone_rows),
        }
    result = {
        "phone_lattice": str(args.phone_lattice_json),
        "transcript_witnesses": str(args.transcript_markdown),
        "listeners": listeners,
    }
    output = args.out / "reference_phone_ceiling.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
