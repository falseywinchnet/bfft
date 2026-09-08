#!/usr/bin/env python3
"""Summarize paired Anchor acquisition against a configurable optimizer set."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_OPTIMIZERS = ("adamw", "sgd", "anchor")
DEFAULT_BASELINES = ("adamw", "sgd")
METRICS = ("learning_auc", "validation_score", "score", "tail_score", "seconds")


def mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key, row.get("score") if key == "tail_score" else None)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return statistics.fmean(values) if values else float("nan")


def first_step(history, target):
    return next(
        (int(row["step"]) for row in history if float(row["score"]) >= target - 1e-12),
        None,
    )


def aggregate_curve(rows):
    by_step = defaultdict(list)
    for row in rows:
        for point in row["history"]:
            score = float(point["score"])
            if math.isfinite(score):
                by_step[int(point["step"])].append(score)
    return [
        {
            "step": step,
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
        for step, values in sorted(by_step.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--web-out", type=Path,
        help="also write the browser exhibit's frozen evidence payload",
    )
    parser.add_argument(
        "--optimizers", default=",".join(DEFAULT_OPTIMIZERS),
        help="comma-separated paired optimizer arms to include",
    )
    parser.add_argument(
        "--baselines", default=",".join(DEFAULT_BASELINES),
        help="comma-separated arms whose best result Anchor must beat",
    )
    args = parser.parse_args()
    optimizers = tuple(name.strip() for name in args.optimizers.split(",") if name.strip())
    baselines = tuple(name.strip() for name in args.baselines.split(",") if name.strip())
    if "anchor" not in optimizers:
        raise ValueError("the primary 'anchor' arm must be included in --optimizers")
    if not baselines or any(name not in optimizers for name in baselines):
        raise ValueError("--baselines must be a nonempty subset of --optimizers")
    if "anchor" in baselines:
        raise ValueError("the primary 'anchor' arm cannot also be a baseline")
    payload = json.loads(args.results.read_text())
    runs = payload["runs"]
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["task"], row["variant"], row["optimizer"]].append(row)

    tasks = list(dict.fromkeys(row["task"] for row in runs))
    variants = list(dict.fromkeys(row["variant"] for row in runs))
    pairs = []
    curves = []
    for task in tasks:
        for variant in variants:
            optimizer_rows = {
                optimizer: sorted(
                    grouped[task, variant, optimizer], key=lambda row: row["seed"]
                )
                for optimizer in optimizers
            }
            if not all(optimizer_rows.values()):
                continue
            summary = {"task": task, "variant": variant, "kind": optimizer_rows["anchor"][0]["kind"]}
            for optimizer, rows in optimizer_rows.items():
                summary[optimizer] = {metric: mean(rows, metric) for metric in METRICS}
                summary[optimizer]["failures"] = sum(row["status"] != "complete" for row in rows)
                curves.append({
                    "task": task,
                    "variant": variant,
                    "optimizer": optimizer,
                    "values": aggregate_curve(rows),
                })

            anchor = summary["anchor"]
            summary["auc_margin"] = anchor["learning_auc"] - max(
                summary[name]["learning_auc"] for name in baselines
            )
            summary["validation_margin"] = anchor["validation_score"] - max(
                summary[name]["validation_score"] for name in baselines
            )
            summary["heldout_margin"] = anchor["score"] - max(
                summary[name]["score"] for name in baselines
            )

            paired_auc_wins = 0
            paired_validation_wins = 0
            speedups = []
            reached = 0
            seeds = sorted(row["seed"] for row in optimizer_rows["anchor"])
            by_seed = {
                optimizer: {row["seed"]: row for row in rows}
                for optimizer, rows in optimizer_rows.items()
            }
            for seed in seeds:
                anchor_row = by_seed["anchor"][seed]
                baseline_rows = [by_seed[name][seed] for name in baselines]
                paired_auc_wins += anchor_row["learning_auc"] > max(
                    row["learning_auc"] for row in baseline_rows
                )
                paired_validation_wins += anchor_row["validation_score"] > max(
                    row["validation_score"] for row in baseline_rows
                )
                target = max(row["validation_score"] for row in baseline_rows)
                baseline_step = min(
                    first_step(row["history"], target)
                    for row in baseline_rows
                    if first_step(row["history"], target) is not None
                )
                anchor_step = first_step(anchor_row["history"], target)
                if anchor_step is not None:
                    reached += 1
                    speedups.append(baseline_step / max(anchor_step, 1))
            summary["paired_auc_wins"] = paired_auc_wins
            summary["paired_validation_wins"] = paired_validation_wins
            summary["baseline_target_reached"] = reached
            summary["median_speedup_to_best_baseline"] = (
                statistics.median(speedups) if speedups else None
            )
            summary["speedups_to_best_baseline"] = speedups
            pairs.append(summary)

    overall = []
    for optimizer in optimizers:
        rows = [row for row in runs if row["optimizer"] == optimizer]
        overall.append({
            "optimizer": optimizer,
            "runs": len(rows),
            **{metric: mean(rows, metric) for metric in METRICS},
            "failures": sum(row["status"] != "complete" for row in rows),
        })

    seeds = int(payload["configuration"]["seeds"])
    winners = sorted(
        (
            pair for pair in pairs
            if pair["auc_margin"] > 0
            and pair["paired_auc_wins"] >= math.ceil(seeds / 2)
        ),
        key=lambda pair: pair["auc_margin"],
        reverse=True,
    )
    result = {
        "source": str(args.results),
        "configuration": payload["configuration"],
        "tasks": tasks,
        "variants": variants,
        "optimizers": list(optimizers),
        "baselines": list(baselines),
        "overall": overall,
        "pairs": pairs,
        "curves": curves,
        "winner_pairs": [
            {key: value for key, value in pair.items() if key not in optimizers}
            for pair in winners
        ],
        "counts": {
            "runs": len(runs),
            "task_variant_pairs": len(pairs),
            "anchor_auc_pair_wins": sum(pair["auc_margin"] > 0 for pair in pairs),
            "anchor_validation_pair_wins": sum(pair["validation_margin"] > 0 for pair in pairs),
            "anchor_heldout_pair_wins": sum(pair["heldout_margin"] > 0 for pair in pairs),
            "robust_anchor_winners": len(winners),
        },
    }
    out = args.out or args.results.with_name("summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if args.web_out:
        web_result = {
            "status": "complete",
            "source": result["source"],
            "configuration": result["configuration"],
            "tasks": result["tasks"],
            "variants": result["variants"],
            "optimizers": result["optimizers"],
            "overall": result["overall"],
            "winner_pairs": result["winner_pairs"],
            "curves": result["curves"],
            "counts": result["counts"],
        }
        args.web_out.parent.mkdir(parents=True, exist_ok=True)
        args.web_out.write_text(json.dumps(web_result, separators=(",", ":")))
    print(json.dumps({"overall": overall, "counts": result["counts"], "top_winners": result["winner_pairs"][:10]}, indent=2))


if __name__ == "__main__":
    main()
