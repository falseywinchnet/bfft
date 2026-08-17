#!/usr/bin/env python3
"""Aggregate named frame-refinement configurations with paired deltas."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("score", "tail_score", "learning_auc", "seconds")


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    runs = payload["runs"]
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["task"], row["configuration"]].append(row)
    tasks = list(dict.fromkeys(row["task"] for row in runs))
    configurations = list(dict.fromkeys(row["configuration"] for row in runs))

    by_task = []
    for task in tasks:
        reference = {
            row["seed"]: row for row in grouped[task, "frame_reference"]
        }
        for configuration in configurations:
            rows = grouped[task, configuration]
            item = {
                "task": task,
                "configuration": configuration,
                "variant": rows[0]["variant"],
                "optimizer": rows[0]["optimizer"],
                "width": rows[0]["width"],
                "parameters": rows[0]["parameters"],
                "seeds": len(rows),
            }
            for metric in METRICS:
                item[metric] = mean(row.get(metric) for row in rows)
                if configuration != "frame_reference":
                    paired = [
                        row.get(metric) - reference[row["seed"]].get(metric)
                        for row in rows
                        if row.get(metric) is not None
                        and reference[row["seed"]].get(metric) is not None
                    ]
                    item[f"{metric}_delta"] = mean(paired)
            by_task.append(item)

    overall = []
    for configuration in configurations:
        rows = [row for row in by_task if row["configuration"] == configuration]
        item = {
            "configuration": configuration,
            "width": rows[0]["width"],
            "optimizer": rows[0]["optimizer"],
            "parameters": rows[0]["parameters"],
            "tasks": len(rows),
        }
        for metric in METRICS:
            item[metric] = mean(row[metric] for row in rows)
            if configuration != "frame_reference":
                item[f"{metric}_delta"] = mean(row[f"{metric}_delta"] for row in rows)
                if metric != "seconds":
                    item[f"{metric}_wins"] = sum(
                        row[f"{metric}_delta"] is not None
                        and row[f"{metric}_delta"] > 0
                        for row in rows
                    )
        overall.append(item)

    result = {
        "source": str(args.results),
        "tasks": tasks,
        "configurations": configurations,
        "reference": "frame_reference",
        "overall": overall,
        "by_task": by_task,
    }
    out = args.out or args.results.with_name("summary.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
