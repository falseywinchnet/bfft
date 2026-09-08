#!/usr/bin/env python3
"""Compose exact CMUdict anchors through a coverage-bounded state phone DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .state_word_lattice import (
    PronunciationTrie,
    compose_state_word_lattice,
    word_candidate_to_dict,
)
from .word_lattice import ARPABET_39, parse_cmudict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_dag", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--maximum-phones", type=int, default=18)
    parser.add_argument("--candidates-per-span", type=int, default=256)
    parser.add_argument(
        "--duration-admission-rank",
        type=int,
        default=0,
        help="Prioritize this duration-envelope stratum; zero disables it",
    )
    parser.add_argument(
        "--state-admission-rank",
        type=int,
        default=0,
        help="Prioritize this Cleanup/SHARK state-profile stratum; zero disables it",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    if (
        args.maximum_phones < 1
        or args.candidates_per_span < 1
        or args.duration_admission_rank < 0
        or args.state_admission_rank < 0
    ):
        raise ValueError("state word lattice limits must be positive")
    phone_dag = json.loads(args.phone_dag.read_text(encoding="utf-8"))
    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(),
        frozenset(ARPABET_39),
        max_phones=args.maximum_phones,
    )
    trie = PronunciationTrie(pronunciations)
    retained = phone_dag["retained_coverage"]
    phone_retention_lower = float(retained["lower_95"])
    output_crops = []
    total_classes = 0
    retained_classes = 0
    for crop in phone_dag["crops"]:
        candidates = compose_state_word_lattice(
            crop,
            trie,
            maximum_phones=args.maximum_phones,
            maximum_phone_rank=int(phone_dag["retained_phone_count"]),
            maximum_duration_rank=(
                args.duration_admission_rank
                if args.duration_admission_rank > 0
                else None
            ),
            maximum_state_rank=(
                args.state_admission_rank
                if args.state_admission_rank > 0
                else None
            ),
        )
        grouped: dict[tuple[int, int], list[object]] = {}
        for candidate in candidates:
            grouped.setdefault((candidate.node0, candidate.node1), []).append(candidate)
        spans = []
        landmarks = [int(value) for value in crop["landmarks"]]
        for (node0, node1), values in sorted(grouped.items()):
            values.sort(key=lambda item: item.objective)
            total_classes += len(values)
            retained_values = values[: args.candidates_per_span]
            retained_classes += len(retained_values)
            rows = []
            for rank, candidate in enumerate(retained_values, start=1):
                row = word_candidate_to_dict(candidate)
                row["rank"] = rank
                row["phone_retention_lower_95_bonferroni"] = max(
                    0.0,
                    1.0
                    - candidate.cost.phone_count * (1.0 - phone_retention_lower),
                )
                rows.append(row)
            spans.append(
                {
                    "node0": node0,
                    "node1": node1,
                    "frame0": landmarks[node0],
                    "frame1": landmarks[node1],
                    "seconds0": landmarks[node0]
                    * int(phone_dag["hop_length"])
                    / int(phone_dag["sample_rate"]),
                    "seconds1": landmarks[node1]
                    * int(phone_dag["hop_length"])
                    / int(phone_dag["sample_rate"]),
                    "candidate_count": len(values),
                    "retained_candidate_count": len(rows),
                    "candidates": rows,
                }
            )
        output_crops.append(
            {
                "crop_index": int(crop["crop_index"]),
                "frame0": int(crop["frame0"]),
                "frame1": int(crop["frame1"]),
                "landmarks": landmarks,
                "span_count": len(spans),
                "spans": spans,
            }
        )
    result = {
        "method": "exact_cmudict_trie_dynamic_programming_over_state_phone_dag",
        "selection_status": "acoustic dictionary ambiguity retained; no language prior or sentence path",
        "phone_dag": str(args.phone_dag),
        "dictionary": {
            "path": str(args.cmudict),
            "sha256": hashlib.sha256(dictionary_bytes).hexdigest(),
            "pronunciation_count": len(pronunciations),
            "pronunciation_class_count": len(trie.classes),
            "maximum_phones": args.maximum_phones,
        },
        "parameters": {
            "candidates_per_span": args.candidates_per_span,
            "maximum_phone_rank": int(phone_dag["retained_phone_count"]),
            "duration_admission_rank": (
                args.duration_admission_rank
                if args.duration_admission_rank > 0
                else None
            ),
            "state_admission_rank": (
                args.state_admission_rank
                if args.state_admission_rank > 0
                else None
            ),
            "phone_retention_lower_95": phone_retention_lower,
            "word_cost": (
                "duration/state proposal-union uncovered stratum, worst phone rank, "
                "mean rank, worst geometry, mean geometry"
            ),
            "edit_tolerance": 0,
            "language_prior": None,
        },
        "crop_count": len(output_crops),
        "candidate_class_count": total_classes,
        "retained_candidate_class_count": retained_classes,
        "crops": output_crops,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "dictionary_pronunciations": len(pronunciations),
                "pronunciation_classes": len(trie.classes),
                "crop_count": len(output_crops),
                "candidate_class_count": total_classes,
                "retained_candidate_class_count": retained_classes,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
