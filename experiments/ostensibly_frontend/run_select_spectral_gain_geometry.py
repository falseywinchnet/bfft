#!/usr/bin/env python3
"""Select Cleanup spectral gain against the exact raw-geometry baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibrated_geometry_fusion import (
    calibration_objective,
    rank_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_audit", type=Path)
    parser.add_argument("spectral_gain_audit", type=Path)
    parser.add_argument("--gain-floor", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    denoised = json.loads(args.spectral_gain_audit.read_text(encoding="utf-8"))
    gain_key = f"{args.gain_floor:g}"

    baseline_rows = {
        (
            str(row["reference_speaker"]),
            str(row["query_speaker"]),
            str(row["utterance"]),
            int(row["ordinal"]),
        ): row
        for row in baseline["rows"]
    }
    rows = []
    for row in denoised["rows"]:
        key = (
            str(row["reference_speaker"]),
            str(row["query_speaker"]),
            str(row["utterance"]),
            int(row["ordinal"]),
        )
        baseline_row = baseline_rows.get(key)
        if baseline_row is None or baseline_row["phone"] != row["phone"]:
            raise ValueError("baseline and denoised query rows do not align")
        raw_rank = int(baseline_row["fused_ranks"]["geometric:1"])
        gain_rank = int(row["target_ranks"][gain_key])
        rows.append(
            {
                "reference_speaker": key[0],
                "query_speaker": key[1],
                "utterance": key[2],
                "ordinal": key[3],
                "phone": str(row["phone"]),
                "baseline_rank": raw_rank,
                "gain_rank": gain_rank,
                "rank_delta": gain_rank - raw_rank,
            }
        )
    if len(rows) != len(baseline_rows):
        raise ValueError("baseline and denoised audits have different query counts")

    utterances = tuple(sorted({str(row["utterance"]) for row in rows}))
    fold_rows = []
    crossfit_ranks = []
    for heldout in utterances:
        training = [row for row in rows if row["utterance"] != heldout]
        baseline_training = rank_summary(row["baseline_rank"] for row in training)
        gain_training = rank_summary(row["gain_rank"] for row in training)
        selected = (
            "spectral_gain"
            if calibration_objective(gain_training)
            < calibration_objective(baseline_training)
            else "baseline"
        )
        heldout_rows = [row for row in rows if row["utterance"] == heldout]
        heldout_ranks = [
            row["gain_rank"] if selected == "spectral_gain" else row["baseline_rank"]
            for row in heldout_rows
        ]
        crossfit_ranks.extend(heldout_ranks)
        fold_rows.append(
            {
                "heldout_utterance": heldout,
                "selected": selected,
                "baseline_training_summary": baseline_training,
                "gain_training_summary": gain_training,
                "heldout_summary": rank_summary(heldout_ranks),
            }
        )

    phone_rows = []
    for phone in sorted({str(row["phone"]) for row in rows}):
        selected = [row for row in rows if row["phone"] == phone]
        phone_rows.append(
            {
                "phone": phone,
                "count": len(selected),
                "baseline_mean_rank": sum(row["baseline_rank"] for row in selected)
                / len(selected),
                "gain_mean_rank": sum(row["gain_rank"] for row in selected)
                / len(selected),
                "mean_rank_delta": sum(row["rank_delta"] for row in selected)
                / len(selected),
                "improved": sum(row["rank_delta"] < 0 for row in selected),
                "equal": sum(row["rank_delta"] == 0 for row in selected),
                "worsened": sum(row["rank_delta"] > 0 for row in selected),
            }
        )

    result = {
        "method": "utterance_held_out_cleanup_spectral_gain_selection",
        "selection_status": "generic paired audit; no radio query used",
        "gain_floor": args.gain_floor,
        "query_count": len(rows),
        "baseline_summary": rank_summary(row["baseline_rank"] for row in rows),
        "gain_summary": rank_summary(row["gain_rank"] for row in rows),
        "paired_counts": {
            "improved": sum(row["rank_delta"] < 0 for row in rows),
            "equal": sum(row["rank_delta"] == 0 for row in rows),
            "worsened": sum(row["rank_delta"] > 0 for row in rows),
        },
        "leave_one_utterance_out": fold_rows,
        "crossfit_summary": rank_summary(crossfit_ranks),
        "phone_summary": phone_rows,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "baseline_summary": result["baseline_summary"],
                "gain_summary": result["gain_summary"],
                "paired_counts": result["paired_counts"],
                "selected": [row["selected"] for row in fold_rows],
                "crossfit_summary": result["crossfit_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
