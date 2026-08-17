#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def fmt(value):
    return "" if value is None else f"{value:.8g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.results.parent
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.results.read_text())
    runs = payload["runs"]
    tasks = list(dict.fromkeys(row["task"] for row in runs))
    rows = []
    for task in tasks:
        for width in sorted({row["width"] for row in runs if row["task"] == task}):
            baseline = [row for row in runs if row["task"] == task and row["width"] == width and row["model"] == "ordinary_mlp"]
            soft = [row for row in runs if row["task"] == task and row["width"] == width and row["model"] == "soft_eikonal"]
            baseline_fit = statistics.mean(max(point["score"] for point in row["history"]) for row in baseline)
            soft_fit = mean(soft, "matched_score")
            row = {
                "task": task, "width": width, "parameters": baseline[0]["parameters"],
                "mlp_fit": baseline_fit, "soft_fit": soft_fit, "fit_advantage": soft_fit - baseline_fit,
                "mlp_test": mean(baseline, "score"), "soft_test": mean(soft, "score"),
                "mlp_auc": mean(baseline, "learning_auc"), "soft_auc": mean(soft, "learning_auc"),
                "mlp_seconds": mean(baseline, "seconds"), "soft_seconds": mean(soft, "seconds"),
                "mlp_tail": mean(baseline, "tail_score"), "soft_tail": mean(soft, "tail_score"),
                "base_only_drop": mean(soft, "base_only_drop"), "uniform_drop": mean(soft, "uniform_drop"),
                "mismatched_drop": mean(soft, "mismatched_drop"),
                "mlp_jacobian_variability": mean(baseline, "jacobian_variability"),
                "soft_jacobian_variability": mean(soft, "jacobian_variability"),
                "soft_jacobian_change_rank": mean(soft, "jacobian_change_rank"),
            }
            row["test_advantage"] = row["soft_test"] - row["mlp_test"]
            row["tail_advantage"] = None if row["soft_tail"] is None else row["soft_tail"] - row["mlp_tail"]
            rows.append(row)
    with (out / "task_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in rows)
    compact = {
        "source": str(args.results), "runs": len(runs), "seeds": len({row["seed"] for row in runs}),
        "widths": sorted({row["width"] for row in runs}), "tasks": tasks, "comparisons": rows,
    }
    (out / "analysis.json").write_text(json.dumps(compact, indent=2))
    print(json.dumps({"runs": len(runs), "tasks": len(tasks), "comparisons": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
