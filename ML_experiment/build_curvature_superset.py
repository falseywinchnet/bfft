#!/usr/bin/env python3
"""Build the direct 22-task self-context/curvature-context comparison."""
from __future__ import annotations

import json
from pathlib import Path

from ML_experiment.build_problem_atlas import (
    DESCRIPTIONS,
    compact_curve,
    compact_field,
    compact_parity,
    compact_scatter,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_curvature_superset/results.json"
SUMMARY = HERE / "results_curvature_superset/summary.json"
PROBES = HERE / "results_curvature_superset/probes.json"
TEMPLATE = HERE / "curvature_superset.template.html"
FRAGMENT = HERE / "curvature_superset.fragment.html"

VARIANTS = ("self_context", "self_context_jet_curvature_context")
LEARNING_TASKS = ("checkerboard", "nd_spiral_high_rank", "chirp_1d", "multiscale_1d")


def mean_histories(runs):
    result = {}
    for task in LEARNING_TASKS:
        result[task] = {}
        for variant in VARIANTS:
            rows = [run["history"] for run in runs if run["task"] == task and run["variant"] == variant]
            steps = [row["step"] for row in rows[0]]
            result[task][variant] = {
                "step": steps,
                "score": [
                    round(sum(history[i]["score"] for history in rows) / len(rows), 5)
                    for i in range(len(steps))
                ],
            }
    return result


def main():
    results = json.loads(RESULTS.read_text())
    summary = json.loads(SUMMARY.read_text())
    probe_rows = json.loads(PROBES.read_text())["probes"]
    by_probe = {(row["task"], row["variant"]): row for row in probe_rows}
    by_metric = {(row["task"], row["variant"]): row for row in summary["by_task"]}
    kind = {row["task"]: row["kind"] for row in summary["by_task"]}

    tasks = []
    for task in summary["tasks"]:
        rows = [by_probe[task, variant] for variant in VARIANTS]
        probe_type = rows[0]["type"]
        if probe_type == "field":
            geometry = compact_field(rows, kind[task])
        elif probe_type == "scatter":
            geometry = compact_scatter(rows)
        elif probe_type == "parity":
            geometry = compact_parity(rows)
        elif probe_type == "curve3d":
            geometry = compact_curve(rows, 3)
        else:
            geometry = compact_curve(rows)
        curvature = by_metric[task, VARIANTS[1]]
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": {
                variant: round(by_metric[task, variant]["score"], 4)
                for variant in VARIANTS
            },
            "deltas": {
                "learning_auc": round(curvature.get("learning_auc_delta", 0.0), 5),
                "score": round(curvature.get("score_delta", 0.0), 5),
                "tail": round(curvature.get("tail_score_delta", 0.0), 5),
            },
            **geometry,
        })

    curvature_overall = next(row for row in summary["overall"] if row["variant"] == VARIANTS[1])
    data = {
        "variants": VARIANTS,
        "overall": {
            "learning_auc_delta": round(curvature_overall["learning_auc_delta"], 5),
            "learning_auc_wins": curvature_overall["learning_auc_wins"],
            "score_delta": round(curvature_overall["score_delta"], 5),
            "tail_delta": round(curvature_overall["tail_score_delta"], 5),
            "runtime_ratio": round(curvature_overall["seconds"] / next(
                row["seconds"] for row in summary["overall"] if row["variant"] == VARIANTS[0]
            ), 2),
        },
        "learning": mean_histories(results["runs"]),
        "tasks": tasks,
    }
    payload = json.dumps(data, separators=(",", ":"))
    FRAGMENT.write_text(TEMPLATE.read_text().replace("__DATA__", payload))
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes; {len(tasks)} tasks")


if __name__ == "__main__":
    main()
