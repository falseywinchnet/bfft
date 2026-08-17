#!/usr/bin/env python3
"""Build a genuine x/y/z complex-spiral comparison from retained probes."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROBES = HERE / "results_frame_refinement/probes.json"
SUMMARY = HERE / "results_frame_refinement/summary.json"
TEMPLATE = HERE / "complex_spiral_3d.template.html"
FRAGMENT = HERE / "complex_spiral_3d.fragment.html"

CONFIGURATIONS = (
    ("truth", "Ground truth"),
    ("ordinary_mlp", "Ordinary MLP"),
    ("self_context", "Self-context"),
    ("frame_reference", "Continuous frame flow (AdamW)"),
    ("frame_muon", "Continuous frame flow (Muon)"),
    ("frame_capacity", "Continuous frame flow (width 32)"),
    ("frame_fast", "Continuous frame flow (two-probe)"),
)


def indices(length: int, maximum: int = 900):
    if length <= maximum:
        return list(range(length))
    return [round(index * (length - 1) / (maximum - 1)) for index in range(maximum)]


def main():
    probes = json.loads(PROBES.read_text())["probes"]
    rows = {
        row["configuration"]: row for row in probes
        if row["task"] == "complex_spiral_3d"
    }
    summary = json.loads(SUMMARY.read_text())["by_task"]
    scores = {
        row["configuration"]: row["score"] for row in summary
        if row["task"] == "complex_spiral_3d"
    }
    source = rows["ordinary_mlp"]
    chosen = indices(len(source["input"]))
    parameter = [(source["input"][index] + 1) / 2 for index in chosen]

    series = []
    for configuration, name in CONFIGURATIONS:
        values = (source["truth"] if configuration == "truth"
                  else rows[configuration]["prediction"])
        points = [[round(float(value), 5) for value in values[index]] for index in chosen]
        bounds = [
            [min(point[axis] for point in points), max(point[axis] for point in points)]
            for axis in range(3)
        ]
        series.append({
            "configuration": configuration,
            "name": name,
            "score": None if configuration == "truth" else round(scores[configuration], 5),
            "points": points,
            "bounds": bounds,
        })
    data = {"parameter": parameter, "series": series}
    FRAGMENT.write_text(
        TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes; {len(chosen)} points per panel")


if __name__ == "__main__":
    main()
