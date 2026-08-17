#!/usr/bin/env python3
"""Build the optimizer/capacity/speed frame-flow report fragment."""
from __future__ import annotations

import json
from pathlib import Path

from ML_experiment.build_problem_atlas import (
    DESCRIPTIONS,
    compact_curve,
    compact_field,
    compact_scatter,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_frame_refinement/results.json"
SUMMARY = HERE / "results_frame_refinement/summary.json"
PROBES = HERE / "results_frame_refinement/probes.json"
TEMPLATE = HERE / "curvature_superset.template.html"
FRAGMENT = HERE / "frame_refinement.fragment.html"

CONFIGURATIONS = (
    "ordinary_mlp",
    "self_context",
    "frame_reference",
    "frame_muon",
    "frame_capacity",
    "frame_fast",
)
NAMES = {
    "ordinary_mlp": "Ordinary MLP",
    "self_context": "Self-context",
    "frame_reference": "Frame reference",
    "frame_muon": "Frame + Muon",
    "frame_capacity": "Frame width 32",
    "frame_fast": "Frame two-probe",
}
LEARNING_TASKS = (
    "radial_stripes",
    "nd_spiral_high_rank",
    "poly_drifted_chirp_1d",
    "complex_spiral_3d",
)


def mean_histories(runs):
    result = {}
    for task in LEARNING_TASKS:
        result[task] = {}
        for configuration in CONFIGURATIONS:
            histories = [
                row["history"] for row in runs
                if row["task"] == task and row["configuration"] == configuration
            ]
            result[task][configuration] = {
                "step": [point["step"] for point in histories[0]],
                "score": [
                    round(sum(history[index]["score"] for history in histories) / len(histories), 5)
                    for index in range(len(histories[0]))
                ],
            }
    return result


def transformed_template():
    source = TEMPLATE.read_text().replace("curvature-superset-viz", "frame-refinement-viz")
    names = "const names=" + json.dumps(NAMES, separators=(",", ":")) + ";"
    legend = "".join(
        f'<span class="key"><span class="dot" style="--key:var(--viz-series-{index + 1})"></span>{NAMES[name]}</span>'
        for index, name in enumerate(CONFIGURATIONS)
    )
    replacements = {
        "Self-context versus curvature self-context": "Continuous frame flow: optimizer, capacity, and speed",
        "Complete 22-problem suite · identical parameters · two paired seeds · width 24 · 500 steps · M4 CPU":
            "Seven diagnostic problems · two paired seeds · 500 steps · M4 CPU · no task-aligned features",
        "Where curvature changes acquisition, endpoint, and tails": "What Muon changes relative to the frame reference",
        "Curvature self-context metric differences versus self-context for 22 tasks": "Muon metric differences versus the AdamW frame reference",
        "Task-level metric differences between curvature self-context and self-context": "Paired Muon differences by task",
        "Curvature self-context − self-context": "Frame + Muon − frame reference",
        "<span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-1)\"></span>Self-context</span><span class=\"key\"><span class=\"dot\" style=\"--key:var(--viz-series-2)\"></span>Curvature self-context</span>": legend,
        "Matched fitted-function atlas": "Fitted geometry (seed 0; shaded or solid segment is observed support)",
        "const names={self_context:'Self-context',self_context_jet_curvature_context:'Curvature self-context'};": names,
        "grid-template-columns:repeat(3,minmax(0,1fr))": "grid-template-columns:repeat(4,minmax(0,1fr))",
        "variant===variants[0]?'var(--viz-series-1)':'var(--viz-series-2)'": "`var(--viz-series-${variants.indexOf(variant)+1})`",
        "metrics.map(([key,label,color,shape])=>({task:r.task,key,label,color,shape,value:r.deltas[key]})))":
            "metrics.map(([key,label,color,shape])=>({task:r.task,key,label,color,shape,value:r.deltas[key]}))).filter(d=>d.value!==null)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"template marker missing: {old[:90]}")
        source = source.replace(old, new)

    old_3d = "function renderCurve3d(node,task,variant){const {w,h,m}=dims(node),svg=d3.select(node).attr('viewBox',`0 0 ${w} ${h}`).attr('height',h),{x,y}=lineScales(task,w,h,m);frame(svg,w,h,m,'t','coordinate');for(let dim=0;dim<3;dim++){const line=d3.line().x((d,i)=>x(task.x[i])).y(d=>y(d[dim]));svg.append('path').datum(task.truth).attr('fill','none').attr('stroke',token(dim)).attr('stroke-width',variant==='truth'?1.8:1).attr('opacity',variant==='truth'?1:.25).attr('d',line);if(variant!=='truth')svg.append('path').datum(task.predictions[variant]).attr('fill','none').attr('stroke',token(dim)).attr('stroke-width',1.8).attr('d',line)}}"
    new_3d = """function renderCurve3d(node,task,variant){
      const {w,h,m}=dims(node,true),svg=d3.select(node).attr('viewBox',`0 0 ${w} ${h}`).attr('height',h);
      frame(svg,w,h,m,'oblique x','oblique z');
      const vectors=[task.truth,...Object.values(task.predictions)],all=vectors.flat();
      const center=[0,1,2].map(j=>d3.mean(all,d=>d[j]));
      const az=-.72,el=.48;
      const project=p=>{const a=p[0]-center[0],b=p[1]-center[1],c=p[2]-center[2],u=Math.cos(az)*a-Math.sin(az)*b,v=Math.sin(az)*a+Math.cos(az)*b;return[u,Math.cos(el)*c-Math.sin(el)*v]};
      const projected=all.map(project),xe=d3.extent(projected,d=>d[0]),ye=d3.extent(projected,d=>d[1]),pad=.06;
      const xp=d3.scaleLinear([xe[0]-(xe[1]-xe[0])*pad,xe[1]+(xe[1]-xe[0])*pad],[m.l,w-m.r]);
      const yp=d3.scaleLinear([ye[0]-(ye[1]-ye[0])*pad,ye[1]+(ye[1]-ye[0])*pad],[h-m.b,m.t]);
      const line=d3.line().x(d=>xp(project(d)[0])).y(d=>yp(project(d)[1]));
      const boundary=d3.bisector(d=>d).right(task.x,task.train_limits[1]);
      const draw=(values,stroke,width,opacity)=>{svg.append('path').datum(values.slice(0,boundary+1)).attr('fill','none').attr('stroke',stroke).attr('stroke-width',width).attr('opacity',opacity).attr('d',line);svg.append('path').datum(values.slice(Math.max(0,boundary-1))).attr('fill','none').attr('stroke',stroke).attr('stroke-width',width).attr('stroke-dasharray','4 3').attr('opacity',opacity).attr('d',line)};
      draw(task.truth,'var(--foreground)',variant==='truth'?2.2:1,variant==='truth'?1:.28);
      if(variant!=='truth')draw(task.predictions[variant],`var(--viz-series-${variants.indexOf(variant)+1})`,2,1);
      const mark=project(task.truth[Math.min(boundary,task.truth.length-1)]);svg.append('circle').attr('cx',xp(mark[0])).attr('cy',yp(mark[1])).attr('r',3).attr('fill','var(--foreground)');
    }"""
    if old_3d not in source:
        raise RuntimeError("3-D renderer marker missing")
    return source.replace(old_3d, new_3d)


def main():
    results = json.loads(RESULTS.read_text())
    summary = json.loads(SUMMARY.read_text())
    probes = json.loads(PROBES.read_text())["probes"]
    by_probe = {(row["task"], row["configuration"]): row for row in probes}
    by_metric = {(row["task"], row["configuration"]): row for row in summary["by_task"]}
    kind = {row["task"]: row["kind"] for row in results["runs"]}

    tasks = []
    for task in summary["tasks"]:
        rows = []
        for configuration in CONFIGURATIONS:
            row = dict(by_probe[task, configuration])
            row["variant"] = configuration
            rows.append(row)
        probe_type = rows[0]["type"]
        if probe_type == "field":
            geometry = compact_field(rows, kind[task])
        elif probe_type == "scatter":
            geometry = compact_scatter(rows)
        elif probe_type == "curve3d":
            geometry = compact_curve(rows, 3)
            geometry["train_limits"] = rows[0]["train_limits"]
        else:
            geometry = compact_curve(rows)
        muon = by_metric[task, "frame_muon"]
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": {
                configuration: round(by_metric[task, configuration]["score"], 4)
                for configuration in CONFIGURATIONS
            },
            "deltas": {
                "learning_auc": round(muon["learning_auc_delta"], 5),
                "score": round(muon["score_delta"], 5),
                "tail": None if muon["tail_score_delta"] is None else round(muon["tail_score_delta"], 5),
            },
            **geometry,
        })

    data = {
        "variants": CONFIGURATIONS,
        "learning": mean_histories(results["runs"]),
        "tasks": tasks,
    }
    FRAGMENT.write_text(
        transformed_template().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes; {len(tasks)} tasks")


if __name__ == "__main__":
    main()
