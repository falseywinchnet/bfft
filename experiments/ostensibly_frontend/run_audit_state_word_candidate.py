#!/usr/bin/env python3
"""Audit a named word after label-free state-DAG dictionary composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .state_word_lattice import (
    PronunciationTrie,
    compose_state_word_lattice,
    word_candidate_to_dict,
)
from .word_lattice import ARPABET_39, parse_cmudict


def _thresholds(raw: str) -> tuple[int | None, ...]:
    values = tuple(int(value) for value in raw.split(","))
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("admission ranks must be nonnegative")
    return tuple(None if value == 0 else value for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_dag", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("word")
    parser.add_argument("--crop-index", type=int, required=True)
    parser.add_argument("--node0", type=int, default=0)
    parser.add_argument("--node1", type=int)
    parser.add_argument("--maximum-phones", type=int, default=18)
    parser.add_argument("--duration-ranks", type=_thresholds, default=(None,))
    parser.add_argument("--state-ranks", type=_thresholds, default=(None,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.crop_index < 0 or args.node0 < 0 or args.maximum_phones < 1:
        raise ValueError("audit indices and limits must be nonnegative")

    started = perf_counter()
    phone_dag = json.loads(args.phone_dag.read_text(encoding="utf-8"))
    crops = [
        crop
        for crop in phone_dag["crops"]
        if int(crop["crop_index"]) == args.crop_index
    ]
    if len(crops) != 1:
        raise ValueError("audit crop is absent or ambiguous")
    crop = crops[0]
    node1 = len(crop["landmarks"]) - 1 if args.node1 is None else args.node1
    if not args.node0 < node1 < len(crop["landmarks"]):
        raise ValueError("audit node span is outside the crop")

    pronunciations = parse_cmudict(
        args.cmudict.read_text(encoding="utf-8").splitlines(),
        frozenset(ARPABET_39),
        max_phones=args.maximum_phones,
    )
    trie = PronunciationTrie(pronunciations)
    experiments = []
    target = args.word.lower()
    for duration_rank in args.duration_ranks:
        for state_rank in args.state_ranks:
            candidates = compose_state_word_lattice(
                crop,
                trie,
                maximum_phones=args.maximum_phones,
                maximum_phone_rank=int(phone_dag["retained_phone_count"]),
                maximum_duration_rank=duration_rank,
                maximum_state_rank=state_rank,
            )
            span = sorted(
                (
                    candidate
                    for candidate in candidates
                    if (candidate.node0, candidate.node1) == (args.node0, node1)
                ),
                key=lambda candidate: candidate.objective,
            )
            target_rows = []
            for rank, candidate in enumerate(span, start=1):
                if target in candidate.words:
                    row = word_candidate_to_dict(candidate)
                    row["rank"] = rank
                    target_rows.append(row)
            experiments.append(
                {
                    "duration_admission_rank": duration_rank,
                    "state_admission_rank": state_rank,
                    "candidate_class_count": len(span),
                    "fully_admitted_class_count": sum(
                        candidate.cost.proposal_uncovered == 0
                        for candidate in span
                    ),
                    "target_candidates": target_rows,
                    "leaders": [
                        {
                            "rank": rank,
                            **word_candidate_to_dict(candidate),
                        }
                        for rank, candidate in enumerate(span[:10], start=1)
                    ],
                }
            )

    result = {
        "method": "posthoc_named_word_audit_after_label_free_dag_composition",
        "phone_dag": str(args.phone_dag),
        "cmudict": str(args.cmudict),
        "target_word": target,
        "crop_index": args.crop_index,
        "node_span": [args.node0, node1],
        "experiments": experiments,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "target_word": target,
                "crop_index": args.crop_index,
                "node_span": [args.node0, node1],
                "experiments": [
                    {
                        "duration_admission_rank": experiment[
                            "duration_admission_rank"
                        ],
                        "state_admission_rank": experiment[
                            "state_admission_rank"
                        ],
                        "candidate_class_count": experiment[
                            "candidate_class_count"
                        ],
                        "fully_admitted_class_count": experiment[
                            "fully_admitted_class_count"
                        ],
                        "target_ranks": [
                            candidate["rank"]
                            for candidate in experiment["target_candidates"]
                        ],
                    }
                    for experiment in experiments
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
