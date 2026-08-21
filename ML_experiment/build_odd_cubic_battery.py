#!/usr/bin/env python3
"""Build the five-column truth/reference/odd-cubic fitted-function atlas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML_experiment.build_continuous_frame_full import transformed_template
from ML_experiment.build_problem_atlas import (
    DESCRIPTIONS, compact_curve, compact_field, compact_parity, compact_scatter)


VANILLA = "ordinary_mlp"
SELF = "self_context"
FLOW = "self_context_stiefel_flow_curvature"
ODD = "shallow_odd_cubic"
VARIANTS = (VANILLA, SELF, FLOW, ODD)
LEARNING_TASKS = ("checkerboard", "nd_spiral_high_rank", "radial_stripes", "multiscale_1d")


def mean_histories(runs):
    result = {}
    for task in LEARNING_TASKS:
        result[task] = {}
        for variant in VARIANTS:
            histories = [row["history"] for row in runs
                         if row["task"] == task and row["variant"] == variant]
            steps = [row["step"] for row in histories[0]]
            result[task][variant] = {
                "step": steps,
                "score": [round(sum(history[index]["score"] for history in histories)
                                / len(histories), 5) for index in range(len(steps))],
            }
    return result


def template(width, task_count):
    source = transformed_template(width=width, task_count=task_count)
    replacements = {
        "Continuous frame flow: four-way fitted-function atlas":
            "Odd third-order baseline: five-way fitted-function atlas",
        "Where continuous frame flow changes acquisition, endpoint, and tails":
            "Where the shallow odd-cubic baseline changes acquisition, endpoint, and tails",
        f"Complete {task_count}-problem suite · identical parameters within each task · two paired seeds · width {width} · 500 steps · M4 CPU":
            f"Complete {task_count}-problem suite · identical optimization budget · two paired seeds · width {width} references · 500 steps · M4 CPU",
        f"Continuous frame flow metric differences versus self-context for {task_count} tasks":
            f"Shallow odd-cubic metric differences versus self-context for {task_count} tasks",
        "Task-level metric differences between continuous frame flow and self-context":
            "Task-level metric differences between shallow odd-cubic and self-context",
        "Continuous frame flow − self-context": "Shallow odd-cubic − self-context",
        '<span class="key"><span class="dot" style="--key:var(--viz-series-1)"></span>Vanilla MLP</span><span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>Self-context</span><span class="key"><span class="dot" style="--key:var(--viz-series-3)"></span>Continuous frame flow</span>':
            '<span class="key"><span class="dot" style="--key:var(--viz-series-1)"></span>Vanilla MLP</span><span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>Self-context</span><span class="key"><span class="dot" style="--key:var(--viz-series-3)"></span>Continuous frame flow</span><span class="key"><span class="dot" style="--key:var(--viz-series-4)"></span>Shallow odd-cubic</span>',
        "const names={ordinary_mlp:'Vanilla LELU MLP',self_context:'Self-context',self_context_stiefel_flow_curvature:'Continuous frame flow (AdamW)'};":
            "const names={ordinary_mlp:'Vanilla LELU MLP',self_context:'Self-context',self_context_stiefel_flow_curvature:'Continuous frame flow (AdamW)',shallow_odd_cubic:'Shallow odd-cubic'};",
        "grid-template-columns:repeat(4,minmax(0,1fr))":
            "grid-template-columns:repeat(5,minmax(0,1fr))",
        "function panelData(){return[{variant:'truth',title:'Truth'},{variant:'ordinary_mlp',title:'Vanilla LELU MLP'},{variant:'self_context',title:'Self-context'},{variant:'self_context_stiefel_flow_curvature',title:'Continuous frame flow (AdamW)'}]}":
            "function panelData(){return[{variant:'truth',title:'Truth'},{variant:'ordinary_mlp',title:'Vanilla LELU MLP'},{variant:'self_context',title:'Self-context'},{variant:'self_context_stiefel_flow_curvature',title:'Continuous frame flow (AdamW)'},{variant:'shallow_odd_cubic',title:'Shallow odd-cubic'}]}",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:100]}")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path("ML_experiment/results_odd_cubic_full"))
    parser.add_argument("--fragment", type=Path,
                        default=Path("ML_experiment/odd_cubic_battery.fragment.html"))
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
        if probe_type == "field": geometry = compact_field(rows, kind[task])
        elif probe_type == "scatter": geometry = compact_scatter(rows)
        elif probe_type == "parity": geometry = compact_parity(rows)
        elif probe_type == "curve3d": geometry = compact_curve(rows, 3)
        else: geometry = compact_curve(rows)
        odd = by_metric[task, ODD]
        tasks.append({
            "task": task, "kind": kind[task], "description": DESCRIPTIONS[task],
            "scores": {variant: round(by_metric[task, variant]["score"], 4)
                       for variant in VARIANTS},
            "deltas": {"learning_auc": round(odd["learning_auc_delta"], 5),
                       "score": round(odd["score_delta"], 5),
                       "tail": round(odd["tail_score_delta"], 5)},
            **geometry,
        })
    overall = {row["variant"]: row for row in summary["overall"]}
    odd = overall[ODD]
    data = {
        "variants": VARIANTS,
        "overall": {
            "learning_auc_delta": round(odd["learning_auc_delta"], 5),
            "learning_auc_wins": odd["learning_auc_wins"],
            "score_delta": round(odd["score_delta"], 5),
            "tail_delta": round(odd["tail_score_delta"], 5),
            "runtime_ratio": round(odd["seconds"] / overall[SELF]["seconds"], 2),
        },
        "learning": mean_histories(results["runs"]), "tasks": tasks,
    }
    args.fragment.write_text(template(args.width, len(tasks)).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"{args.fragment} ({args.fragment.stat().st_size} bytes; {len(tasks)} tasks)")


if __name__ == "__main__": main()
