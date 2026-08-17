#!/usr/bin/env python3
"""Build the compact curvature-state comparison from confirmed results."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results_curvature_confirm/summary.json"
PROBES = HERE / "results_curvature_confirm/probes.json"
TEMPLATE = HERE / "curvature_state.template.html"
FRAGMENT = HERE / "curvature_state.fragment.html"

VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_jet_factor",
    "self_context_jet_curvature_context",
    "self_context_nested",
)

TASK_LABELS = {
    "spiral": "Spiral",
    "checkerboard": "Checkerboard",
    "nd_spiral_low_rank": "N-D spiral · low rank",
    "nd_spiral_high_rank": "N-D spiral · high rank",
    "radial_stripes": "Radial stripes",
    "swiss_cheese": "Swiss cheese",
    "ripple": "Ripple",
    "multiscale_1d": "Multiscale 1-D",
    "chirp_1d": "Chirp",
    "localized_steps_1d": "Localized steps",
    "fourier_mix_1d": "Fourier mixture",
}


def indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    return [round(i * (length - 1) / (maximum - 1)) for i in range(maximum)]


def rounded(values, digits: int = 4):
    return [round(float(value), digits) for value in values]


def compact_field(rows):
    old = rows[0]["size"]
    positions = list(range(0, old, 2))

    def sample(values):
        return [int(values[y * old + x]) for y in positions for x in positions]

    return {
        "type": "field",
        "size": len(positions),
        "truth": sample(rows[0]["truth"]),
        "predictions": {row["variant"]: sample(row["prediction"]) for row in rows},
    }


def compact_scatter(rows):
    chosen = indices(len(rows[0]["xy"]), 420)
    return {
        "type": "scatter",
        "xy": [[round(float(v), 3) for v in rows[0]["xy"][i]] for i in chosen],
        "truth": [int(rows[0]["truth"][i]) for i in chosen],
        "predictions": {
            row["variant"]: [int(row["prediction"][i]) for i in chosen] for row in rows
        },
    }


def compact_curve(rows):
    chosen = indices(len(rows[0]["x"]), 241)
    return {
        "type": "curve",
        "x": rounded([rows[0]["x"][i] for i in chosen]),
        "truth": rounded([rows[0]["truth"][i] for i in chosen]),
        "predictions": {
            row["variant"]: rounded([row["prediction"][i] for i in chosen])
            for row in rows
        },
        "train_limits": rows[0].get("train_limits"),
    }


def main():
    summary = json.loads(SUMMARY.read_text())
    probes = json.loads(PROBES.read_text())["probes"]
    by_probe = {(row["task"], row["variant"]): row for row in probes}

    overall = []
    for row in summary["overall"]:
        if row["variant"] not in VARIANTS:
            continue
        overall.append({
            "variant": row["variant"],
            "score_delta": round(row.get("score_delta", 0.0), 5),
            "tail_score_delta": round(row.get("tail_score_delta", 0.0), 5),
            "learning_auc_delta": round(row.get("learning_auc_delta", 0.0), 5),
            "seconds": round(row["seconds"], 3),
        })

    tasks = []
    for row in summary["by_task"]:
        if row["variant"] not in VARIANTS:
            continue
        tasks.append({
            "task": row["task"],
            "label": TASK_LABELS[row["task"]],
            "variant": row["variant"],
            "score": round(row["score"], 4),
            "score_delta": round(row.get("score_delta", 0.0), 4),
            "tail_score_delta": round(row.get("tail_score_delta", 0.0), 4),
            "learning_auc_delta": round(row.get("learning_auc_delta", 0.0), 4),
        })

    geometry = {}
    for task in ("spiral", "nd_spiral_high_rank", "radial_stripes",
                 "multiscale_1d", "localized_steps_1d", "fourier_mix_1d"):
        rows = [by_probe[task, variant] for variant in VARIANTS]
        if rows[0]["type"] == "field":
            geometry[task] = compact_field(rows)
        elif rows[0]["type"] == "scatter":
            geometry[task] = compact_scatter(rows)
        else:
            geometry[task] = compact_curve(rows)

    payload = json.dumps(
        {"variants": VARIANTS, "overall": overall, "tasks": tasks, "geometry": geometry},
        separators=(",", ":"),
    )
    FRAGMENT.write_text(TEMPLATE.read_text().replace("__DATA__", payload))
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
