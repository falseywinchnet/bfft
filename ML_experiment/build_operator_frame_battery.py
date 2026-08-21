#!/usr/bin/env python3
"""Build the truth-plus-seven-model operator-frame fitted-function atlas."""
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
    DESCRIPTIONS, compact_curve, compact_field, compact_parity, compact_scatter,
)


MLP_SELF = "ordinary_mlp_self_budget"
MLP_CONE = "ordinary_mlp_cone_budget"
SELF = "self_context"
FLOW = "cff"
CONE = "self_contextual_full_learned_cone"
DIRECT = "self_contextual_operator_sphere_global_r2"
NESTED = "self_contextual_nested_operator_r2"
CELL = "operator_sphere_hermite_cells"
VARIANTS = (MLP_SELF, MLP_CONE, SELF, FLOW, CONE, DIRECT, NESTED)
LEARNING_TASKS = (
    "nd_spiral_high_rank", "radial_stripes", "ripple", "multiscale_1d",
    "chirp_1d", "fourier_mix_1d",
)
NAMES = {
    MLP_SELF: "MLP · 9k budget",
    MLP_CONE: "MLP · 26k budget",
    SELF: "Self-context",
    FLOW: "Relational CFF",
    CONE: "Learned cone",
    DIRECT: "Direct operator sphere",
    NESTED: "Nested operator chart",
    CELL: "Operator sphere · Hermite cells",
}


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


def template(width: int, task_count: int):
    source = transformed_template(width=width, task_count=task_count)
    legend = "".join(
        f'<span class="key"><span class="dot" style="--key:var(--viz-series-{index % 6 + 1})"></span>{NAMES[name]}</span>'
        for index, name in enumerate(VARIANTS)
    )
    legend += (
        '<span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>'
        'Hermite cells · sparse-sine acquisition only</span>'
    )
    names = "const names=" + json.dumps(NAMES, separators=(",", ":")) + ";"
    panels = (
        "function panelData(task){const result=[{variant:'truth',title:'Truth'},"
        "...variants.map(variant=>({variant,title:names[variant]}))];"
        f"if(task.extra_variant){{const index=result.findIndex(d=>d.variant==='{DIRECT}');"
        "result.splice(index+1,0,{variant:task.extra_variant,title:names[task.extra_variant]})}"
        "return result}"
    )
    replacements = {
        "Continuous frame flow: four-way fitted-function atlas":
            "Continuous operator frame: complete fitted-function battery",
        "Where continuous frame flow changes acquisition, endpoint, and tails":
            "Where nested operator transport changes acquisition, endpoint, and tails",
        f"Complete {task_count}-problem suite · identical parameters within each task · two paired seeds · width {width} · 500 steps · M4 CPU":
            f"Complete {task_count}-problem suite · two exact MLP parameter controls · one paired seed · width {width} · 500 steps · M4 CPU",
        f"Continuous frame flow metric differences versus self-context for {task_count} tasks":
            f"Nested operator metric differences versus self-context for {task_count} tasks",
        "Task-level metric differences between continuous frame flow and self-context":
            "Task-level metric differences between nested operator and self-context",
        "Continuous frame flow − self-context": "Nested operator − self-context",
        '<span class="key"><span class="dot" style="--key:var(--viz-series-1)"></span>Vanilla MLP</span><span class="key"><span class="dot" style="--key:var(--viz-series-2)"></span>Self-context</span><span class="key"><span class="dot" style="--key:var(--viz-series-3)"></span>Continuous frame flow</span>': legend,
        "const names={ordinary_mlp:'Vanilla LELU MLP',self_context:'Self-context',self_context_stiefel_flow_curvature:'Continuous frame flow (AdamW)'};": names,
        "function panelData(){return[{variant:'truth',title:'Truth'},{variant:'ordinary_mlp',title:'Vanilla LELU MLP'},{variant:'self_context',title:'Self-context'},{variant:'self_context_stiefel_flow_curvature',title:'Continuous frame flow (AdamW)'}]}": panels,
        ".attr('stroke',(d,i)=>`var(--viz-series-${i+1})`)":
            ".attr('stroke',(d,i)=>token(i))",
        ".attr('stroke',`var(--viz-series-${variants.indexOf(variant)+1})`)":
            ".attr('stroke',token(variants.indexOf(variant)))",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:100]}")
        source = source.replace(old, new)
    source = source.replace(
        "const variants=data.variants;",
        "const variants=data.variants;const allVariants=[...variants,data.extra_variant];",
    )
    source = source.replace("variants.indexOf(variant)", "allVariants.indexOf(variant)")
    source = source.replace(".data(panelData())", ".data(panelData(task))")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("ML_experiment/results_operator_frame_battery_merged"),
    )
    parser.add_argument(
        "--fragment", type=Path,
        default=Path("ML_experiment/operator_frame_battery.fragment.html"),
    )
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument(
        "--sparse-geometry", type=Path,
        default=Path("ML_experiment/results_sparse_sine_operator_best_curve/results.json"),
    )
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    summary = json.loads((args.results_dir / "summary.json").read_text())
    probes = json.loads((args.results_dir / "probes.json").read_text())["probes"]
    by_probe = {(row["task"], row["variant"]): row for row in probes}
    by_metric = {(row["task"], row["variant"]): row for row in summary["by_task"]}
    kind = {row["task"]: row["kind"] for row in summary["by_task"]}
    sparse_geometry = json.loads(args.sparse_geometry.read_text())["runs"][0]
    tasks = []
    for task in summary["tasks"]:
        rows = [by_probe[task, variant] for variant in VARIANTS]
        if task == "sparse_sine_1d":
            probe = sparse_geometry["probe"]
            rows.append({
                "task": task,
                "variant": CELL,
                "type": "curve",
                "x": probe["x"],
                "truth": probe["truth"],
                "prediction": probe["prediction"],
                "train_limits": [0.0, probe["observed_limit"]],
            })
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
        nested = by_metric[task, NESTED]
        scores = {
            variant: round(by_metric[task, variant]["score"], 4)
            for variant in VARIANTS
        }
        if task == "sparse_sine_1d":
            scores[CELL] = round(sparse_geometry["score"], 4)
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": scores,
            "extra_variant": CELL if task == "sparse_sine_1d" else None,
            "deltas": {
                "learning_auc": round(nested["learning_auc_delta"], 5),
                "score": round(nested["score_delta"], 5),
                "tail": round(nested["tail_score_delta"], 5),
            },
            **geometry,
        })
    overall = {row["variant"]: row for row in summary["overall"]}
    nested = overall[NESTED]
    data = {
        "variants": VARIANTS,
        "extra_variant": CELL,
        "overall": {
            "learning_auc_delta": round(nested["learning_auc_delta"], 5),
            "learning_auc_wins": nested["learning_auc_wins"],
            "score_delta": round(nested["score_delta"], 5),
            "tail_delta": round(nested["tail_score_delta"], 5),
            "runtime_ratio": round(nested["seconds"] / overall[SELF]["seconds"], 2),
        },
        "learning": histories(results["runs"]),
        "tasks": tasks,
    }
    args.fragment.write_text(
        template(args.width, len(tasks)).replace(
            "__DATA__", json.dumps(data, separators=(",", ":"))
        )
    )
    print(f"{args.fragment} ({args.fragment.stat().st_size} bytes; {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
