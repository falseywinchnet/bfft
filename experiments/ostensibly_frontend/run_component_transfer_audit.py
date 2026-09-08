#!/usr/bin/env python3
"""Score baseline components on one fixed listener-aligned phone pairing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from experiments.ostensibly_frontend.phone_match import (
    WholePatchFingerprint,
    whole_patch_distance,
)
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


def _fingerprint(field: np.ndarray) -> WholePatchFingerprint:
    return WholePatchFingerprint(
        anchor_row=0,
        source_centroid_row=0.0,
        source_centroid_frame=0.0,
        centroid_row=0.5 * (field.shape[0] - 1),
        centroid_frame=0.5 * (field.shape[1] - 1),
        duration_frames=field.shape[1],
        harmonicity=0.0,
        field=np.asarray(field, dtype=np.float64),
    )


def _reference(markdown, markers, lexicon):
    body = markdown.split(markers[0], 1)[1].split(markers[1], 1)[0]
    words = [token.lower() for token in TOKEN_RE.findall(body)]
    phones = []
    for word in words:
        declarations = lexicon[word]
        phones.extend(min(declarations, key=lambda item: (len(item), item)))
    return tuple(phones)


def _fixed_pairs(reference, phone_rows, skip_penalty=1.1):
    evidence = [normalized_phone_evidence(row["top5"]) for row in phone_rows]
    rows = len(reference)
    columns = len(evidence)
    previous = [column * skip_penalty for column in range(columns + 1)]
    backtrace = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    backtrace[0, 1:] = 2
    for row, phone in enumerate(reference, 1):
        current = [row * skip_penalty] + [0.0] * columns
        backtrace[row, 0] = 1
        for column, observed in enumerate(evidence, 1):
            alternatives = (
                previous[column - 1]
                + observed.costs.get(phone, observed.unseen_cost),
                previous[column] + skip_penalty,
                current[column - 1] + skip_penalty,
            )
            choice = min(
                range(3), key=lambda index: (alternatives[index], index)
            )
            current[column] = alternatives[choice]
            backtrace[row, column] = choice
        previous = current
    row = rows
    column = columns
    pairs = []
    while row or column:
        choice = int(backtrace[row, column])
        if row and column and choice == 0:
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and (not column or choice == 1):
            row -= 1
        else:
            column -= 1
    return tuple(reversed(pairs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("templates_npz", type=Path)
    parser.add_argument("phone_lattice_json", type=Path)
    parser.add_argument("transcript_markdown", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument(
        "--query", action="append", required=True,
        help="component=/path/to/radio_query_fingerprints.npz",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved = np.load(args.templates_npz)
    labels = [str(value) for value in saved["labels"]]
    label_index = {label: index for index, label in enumerate(labels)}
    component_names = [str(value) for value in saved["component_names"]]
    template_fields = np.asarray(saved["fields"], dtype=np.float64)
    templates = {
        component: [
            _fingerprint(field) for field in template_fields[index]
        ]
        for index, component in enumerate(component_names)
    }
    query_paths = {}
    for declaration in args.query:
        component, path = declaration.split("=", 1)
        query_paths[component] = Path(path)
    if set(query_paths) != set(component_names):
        raise ValueError("one query fingerprint file is required per component")
    queries = {
        component: [
            _fingerprint(field)
            for field in np.load(path)["fields"]
        ]
        for component, path in query_paths.items()
    }

    phone_rows = json.loads(args.phone_lattice_json.read_text())["phones"]
    markdown = args.transcript_markdown.read_text()
    lexicon = defaultdict(list)
    for entry in parse_cmudict(
        args.cmudict.read_text().splitlines(), frozenset(ARPABET_39), 18
    ):
        lexicon[entry.word].append(entry.phones)

    listeners = {}
    for listener, markers in SECTIONS.items():
        reference = _reference(markdown, markers, lexicon)
        pairs = _fixed_pairs(reference, phone_rows)
        component_results = {}
        for component in component_names:
            hits = {1: 0, 3: 0, 5: 0}
            attractors = Counter()
            for reference_index, observation_index in pairs:
                route = phone_rows[observation_index]["route"]
                ranking = sorted(
                    (
                        whole_patch_distance(
                            queries[component][observation_index],
                            templates[component][label_index[label]],
                        ),
                        label,
                    )
                    for label in route
                )
                ranked_labels = [label for _, label in ranking]
                attractors[ranked_labels[0]] += 1
                target = reference[reference_index]
                for count in hits:
                    hits[count] += int(target in ranked_labels[:count])
            denominator = max(len(pairs), 1)
            component_results[component] = {
                "top1": hits[1] / denominator,
                "top3": hits[3] / denominator,
                "top5": hits[5] / denominator,
                "top1_attractors": attractors.most_common(10),
            }
            print(listener, component, component_results[component], flush=True)
        listeners[listener] = {
            "fixed_pair_count": len(pairs),
            "components": component_results,
        }
    result = {
        "interpretation": (
            "component metrics evaluated on fixed pairs aligned by merged baseline"
        ),
        "listeners": listeners,
    }
    output = args.out / "component_transfer_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
