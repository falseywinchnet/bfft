#!/usr/bin/env python3
"""Build the 8-turn spiral capacity/optimization diagnostic viewer."""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_stiefel_flow_curvature",
)
CONFIGURATIONS = (
    (38, 500),
    (38, 1000),
    (38, 2000),
    (54, 500),
    (76, 500),
    (54, 1000),
    (76, 2000),
)
FIELD_CONFIGURATIONS = ((38, 500), (38, 2000), (76, 500), (76, 2000))


def mean(values):
    return stats.fmean(values)


def sd(values):
    return stats.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results_spiral_capacity_sweep")
    parser.add_argument("--baseline-dir", type=Path, default=HERE / "results_spiral_evidence_horizon")
    parser.add_argument("--fragment", type=Path, default=HERE / "spiral_capacity_sweep.fragment.html")
    args = parser.parse_args()
    sweep = json.loads((args.results_dir / "results.json").read_text())
    sweep_probes = json.loads((args.results_dir / "probes.json").read_text())["probes"]
    baseline = json.loads((args.baseline_dir / "results.json").read_text())
    baseline_probes = json.loads((args.baseline_dir / "probes.json").read_text())["probes"]

    runs = [
        {**row, "width": 38, "steps": 500}
        for row in baseline["runs"]
        if row["visible_turns"] == 8
    ] + sweep["runs"]
    probes = [
        {**row, "width": 38, "steps": 500}
        for row in baseline_probes
        if row["visible_turns"] == 8
    ] + sweep_probes
    grouped = defaultdict(list)
    probe_grouped = defaultdict(list)
    for row in runs:
        grouped[row["width"], row["steps"], row["variant"]].append(row)
    for row in probes:
        probe_grouped[row["width"], row["steps"], row["variant"]].append(row)

    metrics = []
    for width, steps in CONFIGURATIONS:
        values = {}
        for variant in VARIANTS:
            rows = grouped[width, steps, variant]
            values[variant] = {
                "observed": round(mean([row["validation_score"] for row in rows]), 5),
                "observed_sd": round(sd([row["validation_score"] for row in rows]), 5),
                "withheld": round(mean([row["test_score"] for row in rows]), 5),
                "withheld_sd": round(sd([row["test_score"] for row in rows]), 5),
                "auc": round(mean([row["learning_auc"] for row in rows]), 5),
                "parameters": rows[0]["parameters"],
            }
        metrics.append({"width": width, "steps": steps, "values": values})

    fields = []
    for width, steps in FIELD_CONFIGURATIONS:
        probabilities = {}
        scores = {}
        truth = None
        size = None
        limits = None
        for variant in VARIANTS:
            rows = probe_grouped[width, steps, variant]
            size = rows[0]["size"]
            truth = rows[0]["truth"]
            limits = rows[0]["limits"]
            probabilities[variant] = [
                round(mean([row["probability"][index] for row in rows]), 4)
                for index in range(size * size)
            ]
            value = next(row for row in metrics if row["width"] == width and row["steps"] == steps)["values"][variant]
            scores[variant] = value
        fields.append({
            "width": width,
            "steps": steps,
            "size": size,
            "limits": limits,
            "truth": truth,
            "probabilities": probabilities,
            "scores": scores,
        })

    data = {"variants": VARIANTS, "metrics": metrics, "fields": fields}
    (args.results_dir / "summary.json").write_text(json.dumps({
        "source": str(args.results_dir / "results.json"),
        "baseline_source": str(args.baseline_dir / "results.json"),
        "metrics": metrics,
    }, indent=2))
    template = (HERE / "spiral_capacity_sweep.template.html").read_text()
    args.fragment.write_text(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(args.fragment)
    print(f"{args.fragment.stat().st_size} bytes; {len(metrics)} configurations")


if __name__ == "__main__":
    main()
