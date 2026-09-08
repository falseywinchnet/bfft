#!/usr/bin/env python3
"""Compile and cross-check top-k coverage for a frozen phone ranking rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .phone_rank_coverage import fit_phone_rank_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_calibration", type=Path)
    parser.add_argument("--configuration", default="geometric:1")
    parser.add_argument("--coverage-targets", default="0.5,0.8,0.9,0.95")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.geometry_calibration.read_text(encoding="utf-8"))
    rows = source["rows"]
    labels = tuple(sorted({str(row["phone"]) for row in rows}))
    ranks = np.asarray(
        [int(row["fused_ranks"][args.configuration]) for row in rows],
        dtype=np.int64,
    )
    utterances = np.asarray([str(row["utterance"]) for row in rows])
    targets = tuple(float(value) for value in args.coverage_targets.split(",") if value)
    if not targets or any(not 0.0 < value < 1.0 for value in targets):
        raise ValueError("coverage targets must lie strictly within (0, 1)")
    coverage = fit_phone_rank_coverage(
        ranks,
        len(labels),
        provenance={
            "geometry_calibration": str(args.geometry_calibration),
            "configuration": args.configuration,
            "selection": source.get("selection_status"),
        },
    )
    crossfit = []
    unique_utterances = tuple(sorted(set(utterances)))
    for rank in range(1, len(labels) + 1):
        predicted = np.empty(ranks.size, dtype=np.float64)
        for held_out in unique_utterances:
            train = utterances != held_out
            test = ~train
            fold = fit_phone_rank_coverage(ranks[train], len(labels))
            predicted[test] = float(fold.coverage(rank)["estimate"])
        observed = ranks <= rank
        crossfit.append(
            {
                "rank": rank,
                "predicted_coverage": float(np.mean(predicted)),
                "observed_coverage": float(np.mean(observed)),
                "absolute_calibration_error": float(
                    abs(np.mean(predicted) - np.mean(observed))
                ),
                "brier_score": float(np.mean((predicted - observed) ** 2)),
            }
        )
    atlas_path = args.out.with_suffix(".npz")
    coverage.save(atlas_path)
    result = {
        "method": "exchangeable_cross_speaker_phone_rank_coverage",
        "selection_status": "does not alter geometry order; no radio queries used",
        "source": str(args.geometry_calibration),
        "configuration": args.configuration,
        "query_count": coverage.query_count,
        "label_count": coverage.label_count,
        "counts": coverage.counts.tolist(),
        "coverage_curve": [
            coverage.coverage(rank) for rank in range(1, coverage.label_count + 1)
        ],
        "crossfit": crossfit,
        "minimum_rank_for_conservative_coverage": {
            str(target): coverage.minimum_rank_for_lower_coverage(target)
            for target in targets
        },
        "atlas": str(atlas_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "atlas": str(atlas_path),
                "top_k": {
                    str(rank): coverage.coverage(rank)
                    for rank in (1, 5, 10, 16, 24, 32, coverage.label_count)
                },
                "minimum_rank_for_conservative_coverage": result[
                    "minimum_rank_for_conservative_coverage"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
