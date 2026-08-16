#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

METRICS = ("validation_score", "score", "tail_score", "learning_auc", "seconds")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("results", type=Path); parser.add_argument("--out", type=Path)
    args = parser.parse_args(); payload = json.loads(args.results.read_text()); runs = payload["runs"]
    for row in runs: row.setdefault("tail_score", row["score"])
    grouped = defaultdict(list)
    for row in runs: grouped[row["task"], row["variant"]].append(row)
    tasks = list(dict.fromkeys(row["task"] for row in runs)); variants = list(dict.fromkeys(row["variant"] for row in runs)); baseline = "self_context"
    by_task = []
    for task in tasks:
        base = {row["seed"]: row for row in grouped[task, baseline]}
        for variant in variants:
            rows = grouped[task, variant]; item = {"task": task, "variant": variant, "seeds": len(rows),
                                                    "parameters": rows[0]["parameters"], "kind": rows[0]["kind"],
                                                    "input_dim": rows[0]["input_dim"], "output_dim": rows[0]["output_dim"]}
            for metric in METRICS:
                values = [r[metric] for r in rows]; item[metric] = stats.fmean(values)
                item[metric + "_sd"] = stats.stdev(values) if len(values) > 1 else 0.0
                if variant != baseline and metric != "seconds":
                    delta = [r[metric] - base[r["seed"]][metric] for r in rows]; item[metric + "_delta"] = stats.fmean(delta)
            by_task.append(item)
    overall = []
    for variant in variants:
        rows = [r for r in by_task if r["variant"] == variant]; item = {"variant": variant, "tasks": len(rows)}
        for metric in METRICS: item[metric] = stats.fmean(r[metric] for r in rows)
        if variant != baseline:
            for metric in METRICS[:-1]:
                ds = [r[metric + "_delta"] for r in rows]; item[metric + "_delta"] = stats.fmean(ds)
                item[metric + "_wins"] = sum(d > 0 for d in ds); item[metric + "_meaningful_wins"] = sum(d > .005 for d in ds)
                item[metric + "_meaningful_losses"] = sum(d < -.005 for d in ds)
        overall.append(item)
    result = {"source": str(args.results), "configuration": payload["configuration"], "baseline": baseline,
              "tasks": tasks, "variants": variants, "by_task": by_task, "overall": overall}
    out = args.out or args.results.with_name("summary.json"); out.write_text(json.dumps(result, indent=2)); print(json.dumps(overall, indent=2))


if __name__ == "__main__": main()
