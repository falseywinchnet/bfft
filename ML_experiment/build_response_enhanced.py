#!/usr/bin/env python3
"""Build the relational-response four-way fitted-function report."""
from __future__ import annotations

import argparse
import html
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
DEFAULT_RESULTS = HERE / "results_response_enhanced"
TEMPLATE = HERE / "curvature_superset.template.html"
DEFAULT_FRAGMENT = HERE / "response_enhanced.fragment.html"

ORIGINAL = "self_context"
SCL = "relational_scl"
CFF = "cff"
DEEP_CFF = "relational_cff_deep"
VARIANTS = (ORIGINAL, SCL, CFF, DEEP_CFF)
NAMES = {
    ORIGINAL: "Original self-context",
    SCL: "Relational SCL",
    CFF: "Baseline CFF",
    DEEP_CFF: "Relational deep CFF",
}
LEARNING_TASKS = (
    "spiral",
    "checkerboard",
    "nd_spiral_high_rank",
    "radial_stripes",
    "chirp_1d",
    "fourier_mix_1d",
)


def mean_histories(runs):
    output = {}
    for task in LEARNING_TASKS:
        output[task] = {}
        for variant in VARIANTS:
            histories = [
                row["history"]
                for row in runs
                if row["task"] == task and row["variant"] == variant
            ]
            steps = [point["step"] for point in histories[0]]
            output[task][variant] = {
                "step": steps,
                "score": [
                    round(
                        sum(history[index]["score"] for history in histories)
                        / len(histories),
                        5,
                    )
                    for index in range(len(steps))
                ],
            }
    return output


def cost_table(overall):
    rows = []
    for variant in VARIANTS:
        row = overall[variant]
        rows.append(
            "<tr>"
            f"<th>{html.escape(NAMES[variant])}</th>"
            f"<td>{row['parameters']:.0f}</td>"
            f"<td>{row['train_seconds']:.2f} s</td>"
            f"<td>{row['inference_ms']:.2f} ms</td>"
            f"<td>{row['learning_auc']:.3f}</td>"
            f"<td>{row['score']:.3f}</td>"
            f"<td>{row['tail_score']:.3f}</td>"
            f"<td>{row['allocation_entropy']:.3f}</td>"
            "</tr>"
        )
    return (
        '<div class="cost-table"><table><caption>Mean across the 23-task battery</caption>'
        '<thead><tr><th>Model</th><th>Parameters</th><th>Train / task</th>'
        '<th>Inference / 256</th><th>Learning AUC</th><th>Held-out</th>'
        '<th>Tail</th><th>Allocation entropy</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def transformed_template(width, task_count, seeds, table):
    source = TEMPLATE.read_text()
    source = source.replace("curvature-superset-viz", "response-enhanced-viz")
    replacements = {
        "Self-context versus curvature self-context": "Relational response enhancement: fitted geometry and cost",
        '<p class="lede">Complete 22-problem suite · identical parameters · two paired seeds · width 24 · 500 steps · M4 CPU</p>': (
            f'<p class="lede">Complete {task_count}-problem suite · paired initialization · '
            f'{seeds} seeds · width {width} · 500 steps · AdamW · M4 CPU</p>{table}'
        ),
        "Where curvature changes acquisition, endpoint, and tails": "Where relational SCL changes acquisition, endpoint, and tails",
        "Curvature self-context metric differences versus self-context for 22 tasks": f"Relational SCL metric differences versus original self-context for {task_count} tasks",
        "Task-level metric differences between curvature self-context and self-context": "Task-level metric differences between relational SCL and original self-context",
        "Curvature self-context − self-context": "Relational SCL − original self-context",
        '<span class="key"><span class="dot" style="--key:var(--viz-series-1)"></span>Self-context</span><span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>Curvature self-context</span>': (
            '<span class="key"><span class="dot" style="--key:var(--viz-series-1)"></span>Original self-context</span>'
            '<span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>Relational SCL</span>'
            '<span class="key"><span class="dot" style="--key:var(--viz-series-3)"></span>Baseline CFF</span>'
            '<span class="key"><span class="dot" style="--key:var(--viz-series-4)"></span>Relational deep CFF</span>'
        ),
        "const names={self_context:'Self-context',self_context_jet_curvature_context:'Curvature self-context'};": (
            "const names={self_context:'Original self-context',relational_scl:'Relational SCL',"
            "cff:'Baseline CFF',relational_cff_deep:'Relational deep CFF'};"
        ),
        "grid-template-columns:repeat(3,minmax(0,1fr))": "grid-template-columns:repeat(5,minmax(0,1fr))",
        "variant===variants[0]?'var(--viz-series-1)':'var(--viz-series-2)'": "`var(--viz-series-${variants.indexOf(variant)+1})`",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:100]}")
        source = source.replace(old, new)
    source = source.replace(
        "#response-enhanced-viz .tooltip{",
        "#response-enhanced-viz .cost-table{overflow-x:auto;margin:8px 0 22px}"
        "#response-enhanced-viz table{border-collapse:collapse;width:100%;font-size:12px}"
        "#response-enhanced-viz caption{text-align:left;color:var(--muted-foreground);padding-bottom:6px}"
        "#response-enhanced-viz th,#response-enhanced-viz td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}"
        "#response-enhanced-viz th:first-child{text-align:left}"
        "#response-enhanced-viz .tooltip{",
    )
    source = source.replace(
        "@media(max-width:760px){#response-enhanced-viz .learning-grid",
        "@media(max-width:1100px){#response-enhanced-viz .panels{grid-template-columns:repeat(3,minmax(0,1fr))}}"
        "@media(max-width:760px){#response-enhanced-viz .learning-grid",
    )
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    args = parser.parse_args()

    results = json.loads((args.results_dir / "results.json").read_text())
    summary = json.loads((args.results_dir / "summary.json").read_text())
    probes = json.loads((args.results_dir / "probes.json").read_text())["probes"]
    width = results["configuration"]["width"]
    seeds = results["configuration"]["seeds"]

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
        original = by_metric[task, ORIGINAL]
        scl = by_metric[task, SCL]
        tasks.append(
            {
                "task": task,
                "kind": kind[task],
                "description": DESCRIPTIONS[task],
                "scores": {
                    variant: round(by_metric[task, variant]["score"], 4)
                    for variant in VARIANTS
                },
                "deltas": {
                    "learning_auc": round(scl["learning_auc"] - original["learning_auc"], 5),
                    "score": round(scl["score"] - original["score"], 5),
                    "tail": round(scl["tail_score"] - original["tail_score"], 5),
                },
                **geometry,
            }
        )

    overall = {row["variant"]: row for row in summary["overall"]}
    data = {
        "variants": VARIANTS,
        "overall": overall,
        "learning": mean_histories(results["runs"]),
        "tasks": tasks,
    }
    args.fragment.write_text(
        transformed_template(width, len(tasks), seeds, cost_table(overall)).replace(
            "__DATA__", json.dumps(data, separators=(",", ":"))
        )
    )
    print(args.fragment)
    print(f"{args.fragment.stat().st_size} bytes; {len(tasks)} tasks")


if __name__ == "__main__":
    main()
