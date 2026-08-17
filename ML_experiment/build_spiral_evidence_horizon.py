#!/usr/bin/env python3
"""Build the 2/4/8-turn dual-spiral evidence-horizon viewer."""
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


def mean(values):
    return stats.fmean(values)


def sd(values):
    return stats.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "results_spiral_evidence_horizon",
    )
    parser.add_argument(
        "--fragment",
        type=Path,
        default=HERE / "spiral_evidence_horizon.fragment.html",
    )
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    probes = json.loads((args.results_dir / "probes.json").read_text())["probes"]
    grouped = defaultdict(list)
    probe_grouped = defaultdict(list)
    for row in results["runs"]:
        grouped[row["visible_turns"], row["variant"]].append(row)
    for row in probes:
        probe_grouped[row["visible_turns"], row["variant"]].append(row)

    turns_values = sorted({row["visible_turns"] for row in results["runs"]})
    metrics = []
    fields = []
    for turns in turns_values:
        values = {}
        probabilities = {}
        scores = {}
        truth = None
        size = None
        for variant in VARIANTS:
            rows = grouped[turns, variant]
            values[variant] = {
                "validation": round(mean([row["validation_score"] for row in rows]), 5),
                "validation_sd": round(sd([row["validation_score"] for row in rows]), 5),
                "test": round(mean([row["test_score"] for row in rows]), 5),
                "test_sd": round(sd([row["test_score"] for row in rows]), 5),
                "final": round(mean([row["final_turn_score"] for row in rows]), 5),
                "final_sd": round(sd([row["final_turn_score"] for row in rows]), 5),
                "auc": round(mean([row["learning_auc"] for row in rows]), 5),
                "parameters": rows[0]["parameters"],
                "tail_profile": [
                    round(mean([row["tail_scores"][index] for row in rows]), 5)
                    for index in range(turns)
                ],
            }
            probe_rows = probe_grouped[turns, variant]
            size = probe_rows[0]["size"]
            truth = probe_rows[0]["truth"]
            probabilities[variant] = [
                round(mean([row["probability"][index] for row in probe_rows]), 4)
                for index in range(size * size)
            ]
            scores[variant] = {
                "test": values[variant]["test"],
                "test_sd": values[variant]["test_sd"],
            }
        metrics.append({"turns": turns, "values": values})
        fields.append({
            "turns": turns,
            "size": size,
            "limits": probe_grouped[turns, VARIANTS[0]][0]["limits"],
            "truth": truth,
            "probabilities": probabilities,
            "scores": scores,
        })

    data = {
        "variants": VARIANTS,
        "metrics": metrics,
        "fields": fields,
    }
    (args.results_dir / "summary.json").write_text(json.dumps({
        "source": str(args.results_dir / "results.json"),
        "configuration": results["configuration"],
        "metrics": metrics,
    }, indent=2))
    template = (HERE / "spiral_evidence_horizon.template.html").read_text()
    args.fragment.write_text(
        template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    print(args.fragment)
    print(f"{args.fragment.stat().st_size} bytes; {len(turns_values)} horizons")


if __name__ == "__main__":
    main()
