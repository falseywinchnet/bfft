#!/usr/bin/env python3
"""Build the compact graphical report for the rapid nested-chart check."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_nested_chart_check/results.json"
PROBES = HERE / "results_nested_chart_check/probes.json"
TEMPLATE = HERE / "nested_chart_check.template.html"
FRAGMENT = HERE / "nested_chart_check.fragment.html"

CONFIGURATIONS = (
    "self_context",
    "curvature_context",
    "nested_chart_scratch",
    "nested_chart_staged",
)


def indices(length: int, maximum: int):
    if length <= maximum:
        return list(range(length))
    return [round(i * (length - 1) / (maximum - 1)) for i in range(maximum)]


def main():
    results = json.loads(RESULTS.read_text())
    probes = json.loads(PROBES.read_text())["probes"]
    by_probe = {(row["task"], row["configuration"]): row for row in probes}

    learning = {}
    for task in ("radial_stripes", "multiscale_1d"):
        learning[task] = {}
        for configuration in CONFIGURATIONS:
            histories = [
                row["history"] for row in results["runs"]
                if row["task"] == task and row["configuration"] == configuration
            ]
            steps = [row["step"] for row in histories[0]]
            learning[task][configuration] = {
                "step": steps,
                "score": [round(sum(h[i]["score"] for h in histories) / len(histories), 5)
                          for i in range(len(steps))],
            }

    radial_rows = [by_probe["radial_stripes", configuration] for configuration in CONFIGURATIONS]
    old = radial_rows[0]["size"]
    positions = list(range(0, old, 2))
    sample = lambda values: [int(values[y * old + x]) for y in positions for x in positions]
    radial = {
        "size": len(positions),
        "truth": sample(radial_rows[0]["truth"]),
        "predictions": {row["configuration"]: sample(row["prediction"]) for row in radial_rows},
    }

    multiscale_rows = [by_probe["multiscale_1d", configuration] for configuration in CONFIGURATIONS]
    chosen = indices(len(multiscale_rows[0]["x"]), 241)
    rounded = lambda values: [round(float(values[i]), 4) for i in chosen]
    multiscale = {
        "x": rounded(multiscale_rows[0]["x"]),
        "truth": rounded(multiscale_rows[0]["truth"]),
        "predictions": {
            row["configuration"]: rounded(row["prediction"])
            for row in multiscale_rows
        },
        "train_limits": multiscale_rows[0]["train_limits"],
    }

    payload = json.dumps({
        "configurations": CONFIGURATIONS,
        "summary": results["summary"],
        "learning": learning,
        "radial": radial,
        "multiscale": multiscale,
    }, separators=(",", ":"))
    FRAGMENT.write_text(TEMPLATE.read_text().replace("__DATA__", payload))
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
