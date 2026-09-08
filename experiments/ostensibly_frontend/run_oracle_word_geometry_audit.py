#!/usr/bin/env python3
"""Measure exact word identity with true boundaries but no language prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrated_geometry_fusion import calibration_objective, rank_summary
from .oracle_word_geometry import (
    POLICIES,
    compile_pronunciation_catalogs,
    learned_rank_log_likelihood,
    query_rank_rows,
    rank_pronunciation_catalog,
    target_catalog_rank,
)
from .phone_rank_matrix import load_phone_rank_matrix
from .word_lattice import ARPABET_39, parse_cmudict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_matrix", type=Path)
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("assays", nargs="+", type=Path)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_candidates < 1:
        raise ValueError("oracle word audit needs positive top-candidate depth")
    matrix = load_phone_rank_matrix(args.rank_matrix)
    dictionary_bytes = args.cmudict.read_bytes()
    pronunciations = parse_cmudict(
        dictionary_bytes.decode("utf-8").splitlines(),
        frozenset(ARPABET_39),
        max_phones=18,
    )
    catalogs = compile_pronunciation_catalogs(
        pronunciations, tuple(map(str, matrix.labels))
    )
    catalog_indices = {
        length: {phones: index for index, phones in enumerate(catalog.phones)}
        for length, catalog in catalogs.items()
    }
    assay_words = []
    for path in args.assays:
        document = json.loads(path.read_text(encoding="utf-8"))
        query = str(document["query_speaker"])
        reference = "slt" if query == "bdl" else "bdl"
        for word in document["words"]:
            assay_words.append((reference, query, dict(word)))
    utterances = tuple(sorted({str(word[2]["utterance"]) for word in assay_words}))
    likelihoods = {
        utterance: learned_rank_log_likelihood(
            matrix, excluded_utterance=utterance
        )
        for utterance in utterances
    }
    rows = []
    for completed, (reference, query, word) in enumerate(assay_words, start=1):
        phones = tuple(map(str, word["phones"]))
        catalog = catalogs[len(phones)]
        target_index = catalog_indices[len(phones)].get(phones)
        if target_index is None:
            rows.append({
                "reference_speaker": reference,
                "query_speaker": query,
                "utterance": str(word["utterance"]),
                "word": str(word["word"]),
                "phones": list(phones),
                "target_in_dictionary": False,
            })
            continue
        rank_rows = query_rank_rows(
            matrix,
            reference_speaker=reference,
            query_speaker=query,
            utterance=str(word["utterance"]),
            ordinal0=int(word["region0"]),
            phone_count=len(phones),
        )
        policy_rows = {}
        for policy in POLICIES:
            order, candidate_ranks = rank_pronunciation_catalog(
                rank_rows,
                catalog,
                policy=policy,
                rank_log_likelihood=(
                    likelihoods[str(word["utterance"])]
                    if policy == "learned_rank_likelihood"
                    else None
                ),
            )
            leaders = []
            for index in order[: args.top_candidates]:
                leaders.append({
                    "phones": list(catalog.phones[int(index)]),
                    "words": list(catalog.words[int(index)]),
                    "phone_ranks": candidate_ranks[int(index)].tolist(),
                })
            policy_rows[policy] = {
                "target_rank": target_catalog_rank(order, target_index),
                "target_phone_ranks": candidate_ranks[target_index].tolist(),
                "leaders": leaders,
            }
        rows.append({
            "reference_speaker": reference,
            "query_speaker": query,
            "utterance": str(word["utterance"]),
            "word": str(word["word"]),
            "phones": list(phones),
            "region0": int(word["region0"]),
            "region1": int(word["region1"]),
            "candidate_class_count": len(catalog.phones),
            "target_in_dictionary": True,
            "policies": policy_rows,
        })
        if completed % 32 == 0 or completed == len(assay_words):
            print(json.dumps({"completed": completed, "total": len(assay_words)}), flush=True)
    valid = [row for row in rows if row["target_in_dictionary"]]
    summaries = {
        policy: rank_summary(
            int(row["policies"][policy]["target_rank"]) for row in valid
        )
        for policy in POLICIES
    }
    folds = []
    crossfit_ranks = []
    for held_out in utterances:
        training = [row for row in valid if row["utterance"] != held_out]
        validation = [row for row in valid if row["utterance"] == held_out]
        training_summaries = {
            policy: rank_summary(
                int(row["policies"][policy]["target_rank"]) for row in training
            )
            for policy in POLICIES
        }
        selected = min(
            POLICIES,
            key=lambda policy: (calibration_objective(training_summaries[policy]), policy),
        )
        held_ranks = [
            int(row["policies"][selected]["target_rank"]) for row in validation
        ]
        crossfit_ranks.extend(held_ranks)
        folds.append({
            "held_out_utterance": held_out,
            "selected_policy": selected,
            "training_summaries": training_summaries,
            "validation": rank_summary(held_ranks),
        })
    result = {
        "method": "oracle_word_boundaries_exact_dictionary_rank_from_phone_geometry",
        "selection_status": (
            "word labels and boundaries used only for audit; no language prior"
        ),
        "rank_matrix": str(args.rank_matrix),
        "assays": [str(path) for path in args.assays],
        "dictionary": {
            "path": str(args.cmudict),
            "pronunciation_count": len(pronunciations),
            "class_counts_by_phone_length": {
                str(length): len(catalog.phones)
                for length, catalog in sorted(catalogs.items())
            },
        },
        "word_count": len(rows),
        "dictionary_target_count": len(valid),
        "summaries": summaries,
        "leave_one_utterance_out": {
            "selection_counts": {
                policy: sum(fold["selected_policy"] == policy for fold in folds)
                for policy in POLICIES
            },
            "summary": rank_summary(crossfit_ranks),
            "folds": folds,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "word_count": len(rows),
        "dictionary_target_count": len(valid),
        "summaries": summaries,
        "leave_one_utterance_out": {
            "selection_counts": result["leave_one_utterance_out"][
                "selection_counts"
            ],
            "summary": result["leave_one_utterance_out"]["summary"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
