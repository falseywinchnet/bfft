#!/usr/bin/env python3
"""Aggregate response-enhancement accuracy, tail, and timing comparisons."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ML_experiment.response_enhanced import (
    BASELINE_CFF,
    ORIGINAL_SELF_CONTEXT,
    RELATIONAL_CFF_DEEP,
    RELATIONAL_SCL,
)


METRICS = (
    "validation_score",
    "score",
    "tail_score",
    "learning_auc",
    "train_seconds",
    "inference_ms",
    "training_examples_per_second",
    "allocation_entropy",
    "allocation_max_weight",
    "jacobian_variability",
)


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    runs = payload["runs"]
    for row in runs:
        row.setdefault("tail_score", row["score"])

    variants = payload["configuration"]["variants"]
    tasks = list(dict.fromkeys(row["task"] for row in runs))
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["task"], row["variant"]].append(row)

    by_task = []
    for task in tasks:
        original_by_seed = {
            row["seed"]: row for row in grouped[task, ORIGINAL_SELF_CONTEXT]
        }
        scl_by_seed = {row["seed"]: row for row in grouped[task, RELATIONAL_SCL]}
        for variant in variants:
            rows = grouped[task, variant]
            item = {
                "task": task,
                "variant": variant,
                "kind": rows[0]["kind"],
                "input_dim": rows[0]["input_dim"],
                "output_dim": rows[0]["output_dim"],
                "parameters": rows[0]["parameters"],
                "seeds": len(rows),
            }
            for metric in METRICS:
                item[metric] = mean(rows, metric)
                if len(rows) > 1:
                    item[metric + "_sd"] = statistics.stdev(
                        row[metric] for row in rows
                    )
            if variant != ORIGINAL_SELF_CONTEXT:
                for metric in ("score", "tail_score", "learning_auc"):
                    item[metric + "_vs_original"] = statistics.fmean(
                        row[metric] - original_by_seed[row["seed"]][metric]
                        for row in rows
                    )
            if variant != RELATIONAL_SCL:
                for metric in ("score", "tail_score", "learning_auc"):
                    item[metric + "_vs_scl"] = statistics.fmean(
                        row[metric] - scl_by_seed[row["seed"]][metric]
                        for row in rows
                    )
            by_task.append(item)

    overall = []
    for variant in variants:
        rows = [row for row in by_task if row["variant"] == variant]
        item = {"variant": variant, "tasks": len(rows)}
        for metric in METRICS:
            item[metric] = mean(rows, metric)
        item["parameters"] = mean(rows, "parameters")
        item["score_wins_vs_original"] = sum(
            row.get("score_vs_original", 0) > 0 for row in rows
        )
        item["score_wins_vs_scl"] = sum(
            row.get("score_vs_scl", 0) > 0 for row in rows
        )
        overall.append(item)

    task_lookup = {(row["task"], row["variant"]): row for row in by_task}
    pareto = []
    for task in tasks:
        scl = task_lookup[task, RELATIONAL_SCL]
        cff = task_lookup[task, BASELINE_CFF]
        deep = task_lookup[task, RELATIONAL_CFF_DEEP]
        pareto.append(
            {
                "task": task,
                "scl_score": scl["score"],
                "cff_score": cff["score"],
                "deep_cff_score": deep["score"],
                "scl_minus_cff": scl["score"] - cff["score"],
                "scl_train_speedup_vs_cff": cff["train_seconds"] / scl["train_seconds"],
                "scl_inference_speedup_vs_cff": cff["inference_ms"] / scl["inference_ms"],
                "scl_within_one_point_of_cff": scl["score"] >= cff["score"] - 0.01,
                "scl_within_one_point_of_deep_cff": scl["score"] >= deep["score"] - 0.01,
            }
        )

    result = {
        "source": str(args.results),
        "configuration": payload["configuration"],
        "tasks": tasks,
        "variants": variants,
        "by_task": by_task,
        "overall": overall,
        "pareto": pareto,
    }
    out = args.out or args.results.with_name("summary.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(overall, indent=2))
    print(
        json.dumps(
            {
                "scl_within_one_point_of_cff": sum(
                    row["scl_within_one_point_of_cff"] for row in pareto
                ),
                "scl_within_one_point_of_deep_cff": sum(
                    row["scl_within_one_point_of_deep_cff"] for row in pareto
                ),
                "tasks": len(pareto),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
