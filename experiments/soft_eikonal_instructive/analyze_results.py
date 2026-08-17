#!/usr/bin/env python3
"""Aggregate paired-seed Eikonal variant results without inventing a composite score."""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


METRICS = ("validation_score", "score", "tail_score", "learning_auc", "seconds")


def mean(values):
    return stats.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", default="soft_eikonal")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text()); runs = payload["runs"]
    for row in runs:
        row.setdefault("tail_score", row["score"])
    groups = defaultdict(list)
    for row in runs:
        groups[row["task"], row["variant"]].append(row)
    tasks = sorted({row["task"] for row in runs}); variants = sorted({row["variant"] for row in runs})
    table = []
    for task in tasks:
        reference = {row["seed"]: row for row in groups[task, args.reference]}
        for variant in variants:
            rows = groups[task, variant]
            record = {"task": task, "variant": variant, "seeds": len(rows),
                      "parameters": rows[0]["parameters"]}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                record[metric] = mean(values)
                record[metric + "_sd"] = stats.stdev(values) if len(values) > 1 else 0.0
                if metric != "seconds" and variant != args.reference:
                    differences = [row[metric] - reference[row["seed"]][metric] for row in rows]
                    record[metric + "_delta"] = mean(differences)
            table.append(record)
    overall = []
    for variant in variants:
        records = [row for row in table if row["variant"] == variant]
        item = {"variant": variant, "tasks": len(records), "parameters": sorted({r["parameters"] for r in records})}
        for metric in METRICS:
            item[metric] = mean([row[metric] for row in records])
        if variant != args.reference:
            for metric in METRICS[:-1]:
                item[metric + "_delta"] = mean([row[metric + "_delta"] for row in records])
                item[metric + "_wins"] = sum(row[metric + "_delta"] > 0 for row in records)
        overall.append(item)
    result = {"source": str(args.results), "reference": args.reference,
              "tasks": tasks, "variants": variants, "by_task": table, "overall": overall}
    out = args.out or args.results.with_name("summary.json"); out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
