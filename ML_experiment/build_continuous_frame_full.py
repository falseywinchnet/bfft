#!/usr/bin/env python3
"""Build a matched vanilla/self-context/continuous-flow report."""
from __future__ import annotations

import argparse
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
RESULTS = HERE / "results_continuous_frame_full/results.json"
SUMMARY = HERE / "results_continuous_frame_full/summary.json"
PROBES = HERE / "results_continuous_frame_full/probes.json"
SOURCE_TEMPLATE = HERE / "curvature_superset.template.html"
FRAGMENT = HERE / "continuous_frame_full.fragment.html"

VANILLA = "ordinary_mlp"
SELF = "self_context"
FLOW = "self_context_stiefel_flow_curvature"
VARIANTS = (VANILLA, SELF, FLOW)
LEARNING_TASKS = (
    "checkerboard",
    "nd_spiral_high_rank",
    "radial_stripes",
    "chirp_1d",
)


def mean_histories(runs):
    result = {}
    for task in LEARNING_TASKS:
        result[task] = {}
        for variant in VARIANTS:
            histories = [
                run["history"]
                for run in runs
                if run["task"] == task and run["variant"] == variant
            ]
            steps = [row["step"] for row in histories[0]]
            result[task][variant] = {
                "step": steps,
                "score": [
                    round(sum(history[i]["score"] for history in histories) / len(histories), 5)
                    for i in range(len(steps))
                ],
            }
    return result


def transformed_template(*, width: int, task_count: int):
    source = SOURCE_TEMPLATE.read_text()
    replacements = {
        "Self-context versus curvature self-context": "Continuous frame flow: four-way fitted-function atlas",
        "Where curvature changes acquisition, endpoint, and tails": "Where continuous frame flow changes acquisition, endpoint, and tails",
        "Complete 22-problem suite · identical parameters · two paired seeds · width 24 · 500 steps · M4 CPU": f"Complete {task_count}-problem suite · identical parameters within each task · two paired seeds · width {width} · 500 steps · M4 CPU",
        "Curvature self-context metric differences versus self-context for 22 tasks": f"Continuous frame flow metric differences versus self-context for {task_count} tasks",
        "Task-level metric differences between curvature self-context and self-context": "Task-level metric differences between continuous frame flow and self-context",
        "Curvature self-context − self-context": "Continuous frame flow − self-context",
        "<span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-1)\"></span>Self-context</span><span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-2)\"></span>Curvature self-context</span>": "<span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-1)\"></span>Vanilla MLP</span><span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-2)\"></span>Self-context</span><span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-3)\"></span>Continuous frame flow</span>",
        "const names={self_context:'Self-context',self_context_jet_curvature_context:'Curvature self-context'};": "const names={ordinary_mlp:'Vanilla LELU MLP',self_context:'Self-context',self_context_stiefel_flow_curvature:'Continuous frame flow (AdamW)'};",
        "grid-template-columns:repeat(3,minmax(0,1fr))": "grid-template-columns:repeat(4,minmax(0,1fr))",
        "variant===variants[0]?'var(--viz-series-1)':'var(--viz-series-2)'": "`var(--viz-series-${variants.indexOf(variant)+1})`",
        "function panelData(){return[{variant:'truth',title:'Truth'},...variants.map(variant=>({variant,title:names[variant]}))]}": "function panelData(){return[{variant:'truth',title:'Truth'},{variant:'ordinary_mlp',title:'Vanilla LELU MLP'},{variant:'self_context',title:'Self-context'},{variant:'self_context_stiefel_flow_curvature',title:'Continuous frame flow (AdamW)'}]}",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:80]}")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS.parent)
    parser.add_argument("--fragment", type=Path, default=FRAGMENT)
    parser.add_argument("--width", type=int, default=24)
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
        flow = by_metric[task, FLOW]
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": {
                variant: round(by_metric[task, variant]["score"], 4)
                for variant in VARIANTS
            },
            "deltas": {
                "learning_auc": round(flow["learning_auc_delta"], 5),
                "score": round(flow["score_delta"], 5),
                "tail": round(flow["tail_score_delta"], 5),
            },
            **geometry,
        })

    overall = {row["variant"]: row for row in summary["overall"]}
    flow = overall[FLOW]
    data = {
        "variants": VARIANTS,
        "overall": {
            "learning_auc_delta": round(flow["learning_auc_delta"], 5),
            "learning_auc_wins": flow["learning_auc_wins"],
            "score_delta": round(flow["score_delta"], 5),
            "tail_delta": round(flow["tail_score_delta"], 5),
            "runtime_ratio": round(flow["seconds"] / overall[SELF]["seconds"], 2),
        },
        "learning": mean_histories(results["runs"]),
        "tasks": tasks,
    }
    args.fragment.write_text(
        transformed_template(width=args.width, task_count=len(tasks)).replace(
            "__DATA__", json.dumps(data, separators=(",", ":"))
        )
    )
    print(args.fragment)
    print(f"{args.fragment.stat().st_size} bytes; {len(tasks)} tasks")


if __name__ == "__main__":
    main()
