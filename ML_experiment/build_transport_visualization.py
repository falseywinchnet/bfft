#!/usr/bin/env python3
"""Build the compact self-context transport graphical report."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
CORE = HERE / "results_transport_study/results.json"
EXTRA = HERE / "results_transport_study/odd_superset_results.json"
PROBES = HERE / "results_transport_study/probes.json"
BASIS = HERE / "results_transport_study/eikonal_basis_ray_core.json"
TEMPLATE = HERE / "transport_study.template.html"
FRAGMENT = HERE / "transport_study.fragment.html"

SELF = "self_context"
ODD = "self_context_transport_self_ray_odd"
CORE_TASKS = ("radial_stripes", "multiscale_1d", "nd_spiral_high_rank", "fourier_mix_1d")
PAIRED_VARIANTS = (
    SELF,
    "self_context_iterated",
    "self_context_transport_heun",
    ODD,
    "self_context_transport_basis_ray_odd",
    "self_context_transport_self_ray_even",
    "self_context_jet_curvature_context",
    "self_context_jet_curvature_bounded",
    "self_context_jet_curvature_detached",
)
LEARNING_VARIANTS = (
    SELF,
    ODD,
    "self_context_jet_curvature_context",
    "self_context_jet_curvature_detached",
)
PROBE_VARIANTS = (
    SELF,
    ODD,
    "self_context_jet_curvature_context",
    "self_context_jet_curvature_bounded",
)


def avg(rows, key):
    values = [row[key] for row in rows if key in row]
    return mean(values) if values else None


def indices(length, maximum):
    if length <= maximum:
        return list(range(length))
    return [round(index * (length - 1) / (maximum - 1)) for index in range(maximum)]


def main():
    core = json.loads(CORE.read_text())["runs"] + json.loads(BASIS.read_text())["runs"]
    extra = json.loads(EXTRA.read_text())["runs"]
    full = [row for row in core if row["task"] in CORE_TASKS and row["variant"] in {SELF, ODD}] + extra

    task_rows = []
    for task in sorted({row["task"] for row in full}):
        baseline = [row for row in full if row["task"] == task and row["variant"] == SELF]
        candidate = [row for row in full if row["task"] == task and row["variant"] == ODD]
        record = {"task": task}
        for metric in ("score", "tail_score", "learning_auc"):
            a, b = avg(baseline, metric), avg(candidate, metric)
            record[metric] = {"self": a, "odd": b, "delta": None if a is None or b is None else b - a}
        task_rows.append(record)

    core_summary = []
    for variant in PAIRED_VARIANTS:
        selected = [row for row in core if row["task"] in CORE_TASKS and row["variant"] == variant]
        per_task_score = [avg([row for row in selected if row["task"] == task], "score") for task in CORE_TASKS]
        core_summary.append({
            "variant": variant,
            "score": mean(per_task_score),
            "learning_auc": mean(avg([row for row in selected if row["task"] == task], "learning_auc") for task in CORE_TASKS),
            "seconds": avg(selected, "seconds"),
            "chart_points": avg(selected, "up_chart_points"),
        })

    learning = {}
    for task in ("radial_stripes", "multiscale_1d"):
        learning[task] = {}
        for variant in LEARNING_VARIANTS:
            histories = [row["history"] for row in core if row["task"] == task and row["variant"] == variant]
            steps = [point["step"] for point in histories[0]]
            learning[task][variant] = {
                "step": steps,
                "score": [round(mean(history[index]["score"] for history in histories), 5)
                          for index in range(len(steps))],
            }

    mechanism = []
    for task in CORE_TASKS:
        for variant in ("self_context_jet_curvature_context", "self_context_jet_curvature_detached"):
            selected = [row for row in core if row["task"] == task and row["variant"] == variant]
            mechanism.append({"task": task, "variant": variant, "score": avg(selected, "score"),
                              "learning_auc": avg(selected, "learning_auc")})

    probes = json.loads(PROBES.read_text())["probes"]
    probe_map = {(row["task"], row["variant"]): row for row in probes}
    radial_rows = [probe_map["radial_stripes", variant] for variant in PROBE_VARIANTS]
    old_size = radial_rows[0]["size"]
    positions = list(range(0, old_size, 2))
    sample_field = lambda values: [int(values[y * old_size + x]) for y in positions for x in positions]
    radial = {
        "size": len(positions),
        "truth": sample_field(radial_rows[0]["truth"]),
        "predictions": {row["variant"]: sample_field(row["prediction"]) for row in radial_rows},
    }

    curve_rows = [probe_map["multiscale_1d", variant] for variant in PROBE_VARIANTS]
    chosen = indices(len(curve_rows[0]["x"]), 241)
    select = lambda values: [round(float(values[index]), 4) for index in chosen]
    multiscale = {
        "x": select(curve_rows[0]["x"]),
        "truth": select(curve_rows[0]["truth"]),
        "predictions": {row["variant"]: select(row["prediction"]) for row in curve_rows},
        "train_limits": curve_rows[0]["train_limits"],
    }

    payload = {
        "task_rows": task_rows,
        "core_summary": core_summary,
        "learning": learning,
        "mechanism": mechanism,
        "radial": radial,
        "multiscale": multiscale,
        "learning_variants": LEARNING_VARIANTS,
        "probe_variants": PROBE_VARIANTS,
    }
    FRAGMENT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
