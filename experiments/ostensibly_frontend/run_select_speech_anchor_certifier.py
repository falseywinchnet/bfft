#!/usr/bin/env python3
"""Select Cleanup certifier strength by utterance-held-out precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .run_audit_speech_anchors import _add_counts, _boundary_summary
from .speech_anchor_audit import summarize_counts, wilson_lower_bound


def _frame_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    counts: dict[str, int] = {}
    for row in rows:
        _add_counts(counts, row["frame_counts"])
    result = summarize_counts(counts)
    predicted = int(result["true_positive"]) + int(result["false_positive"])
    result["precision_lower_95"] = wilson_lower_bound(
        int(result["true_positive"]), predicted
    )
    return result


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "frames": _frame_summary(rows),
        "crop_edges": _boundary_summary(rows, "crop"),
        "manner_edges": _boundary_summary(rows, "manner"),
    }


def _total_strength(name: str) -> float:
    match = re.fullmatch(r"certifier_v([0-9.]+)_u([0-9.]+)", name)
    if match is None:
        raise ValueError(f"invalid certifier variant {name}")
    return float(match.group(1)) + float(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--minimum-precision-lower-95", type=float, default=0.99)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.minimum_precision_lower_95 < 1.0:
        raise ValueError("precision lower bound must lie in (0, 1)")
    document = json.loads(args.audit.read_text(encoding="utf-8"))
    variants = {
        name: rows
        for name, rows in document["rows"].items()
        if name.startswith("certifier_")
    }
    if not variants:
        raise ValueError("audit contains no certifier-strength variants")
    utterances = tuple(document["utterances"])
    folds = []
    selected_rows = []
    for held_out in utterances:
        training = {
            name: _summary([row for row in rows if row["utterance"] != held_out])
            for name, rows in variants.items()
        }
        eligible = [
            name
            for name, summary in training.items()
            if summary["frames"]["precision_lower_95"]
            >= args.minimum_precision_lower_95
        ]
        pool = eligible or list(variants)
        selected = max(
            pool,
            key=lambda name: (
                training[name]["frames"]["recall"] if eligible
                else training[name]["frames"]["precision_lower_95"],
                training[name]["frames"]["f1"],
                training[name]["manner_edges"]["f1"],
                -_total_strength(name),
            ),
        )
        held_rows = [
            row for row in variants[selected] if row["utterance"] == held_out
        ]
        selected_rows.extend(held_rows)
        folds.append({
            "held_out_utterance": held_out,
            "selected_variant": selected,
            "eligible_variants": eligible,
            "training_summaries": training,
            "held_out_summary": _summary(held_rows),
        })
    result = {
        "method": "utterance_held_out_cleanup_certifier_selection",
        "selection_rule": (
            "maximize training recall among variants whose one-sided 95% Wilson "
            "precision lower bound meets the declared floor"
        ),
        "minimum_precision_lower_95": args.minimum_precision_lower_95,
        "folds": folds,
        "selection_counts": {
            name: sum(fold["selected_variant"] == name for fold in folds)
            for name in variants
        },
        "crossfit_summary": _summary(selected_rows),
        "full_summaries": {name: _summary(rows) for name, rows in variants.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "selection_counts": result["selection_counts"],
        "crossfit_summary": result["crossfit_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
