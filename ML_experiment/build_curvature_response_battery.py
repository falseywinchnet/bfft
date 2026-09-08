#!/usr/bin/env python3
"""Build the self-context versus continuous-curvature fitted-function atlas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML_experiment.build_problem_atlas import (
    DESCRIPTIONS, compact_curve, compact_field, compact_parity, compact_scatter,
)


HERE = Path(__file__).resolve().parent
SELF = "self_context"
CURVATURE = "self_context_curvature_response"
VARIANTS = (SELF, CURVATURE)
LEARNING_TASKS = (
    "checkerboard", "nd_spiral_high_rank", "radial_stripes", "ripple",
    "complex_spiral_3d", "multiscale_1d",
)


def histories(runs):
    result = {}
    for task in LEARNING_TASKS:
        result[task] = {}
        for variant in VARIANTS:
            history = next(
                row["history"] for row in runs
                if row["task"] == task and row["variant"] == variant
            )
            result[task][variant] = {
                "step": [row["step"] for row in history],
                "score": [round(row["score"], 5) for row in history],
            }
    return result


def template(task_count: int, width: int):
    source = (HERE / "curvature_superset.template.html").read_text()
    replacements = {
        "Self-context versus curvature self-context":
            "Self-context versus continuous curvature response",
        "Where curvature changes acquisition, endpoint, and tails":
            "Where one continuous even frame response helps—and where it harms",
        "Complete 22-problem suite · identical parameters · two paired seeds · width 24 · 500 steps · M4 CPU":
            f"Complete {task_count}-problem suite · +1 parameter · one paired seed · width {width} · 500 steps · M4 CPU",
        "Curvature self-context metric differences versus self-context for 22 tasks":
            f"Continuous curvature-response differences versus self-context for {task_count} tasks",
        "Task-level metric differences between curvature self-context and self-context":
            "Task-level metric differences between continuous curvature response and self-context",
        "Curvature self-context − self-context":
            "Continuous curvature response − self-context",
        "Curvature self-context</span>":
            "Continuous curvature response</span>",
        "const names={self_context:'Self-context',self_context_jet_curvature_context:'Curvature self-context'};":
            "const names={self_context:'Self-context',self_context_curvature_response:'Continuous curvature response'};",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:90]}")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=HERE / "results_curvature_response_full",
    )
    parser.add_argument(
        "--fragment", type=Path,
        default=HERE / "curvature_response_battery.fragment.html",
    )
    parser.add_argument("--width", type=int, default=38)
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    summary = json.loads((args.results_dir / "summary.json").read_text())
    probes = json.loads((args.results_dir / "probes.json").read_text())["probes"]
    by_probe = {(row["task"], row["variant"]): row for row in probes}
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
        candidate = by_metric[task, CURVATURE]
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": {
                variant: round(by_metric[task, variant]["score"], 4)
                for variant in VARIANTS
            },
            "deltas": {
                "learning_auc": round(candidate["learning_auc_delta"], 5),
                "score": round(candidate["score_delta"], 5),
                "tail": round(candidate["tail_score_delta"], 5),
            },
            **geometry,
        })

    overall = {row["variant"]: row for row in summary["overall"]}
    candidate = overall[CURVATURE]
    baseline = overall[SELF]
    data = {
        "variants": VARIANTS,
        "overall": {
            "learning_auc_delta": round(candidate["learning_auc_delta"], 5),
            "learning_auc_wins": candidate["learning_auc_wins"],
            "score_delta": round(candidate["score_delta"], 5),
            "tail_delta": round(candidate["tail_score_delta"], 5),
            "runtime_ratio": round(candidate["seconds"] / baseline["seconds"], 2),
        },
        "learning": histories(results["runs"]),
        "tasks": tasks,
    }
    args.fragment.write_text(
        template(len(tasks), args.width).replace(
            "__DATA__", json.dumps(data, separators=(",", ":"))
        )
    )
    print(f"{args.fragment} ({args.fragment.stat().st_size} bytes; {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
