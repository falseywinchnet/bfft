#!/usr/bin/env python3
"""Build the compact, all-task visualization from stored fitted probes."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROBES = HERE / "results_confirm/probes.json"
SUMMARY = HERE / "results_confirm/summary.json"
TEMPLATE = HERE / "problem_atlas.template.html"
FRAGMENT = HERE / "problem_atlas.fragment.html"

VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_hard",
    "self_context_chart",
)

DESCRIPTIONS = {
    "spiral": "2-D continuation · outer turns withheld",
    "checkerboard": "2-D localized alternation",
    "two_moons": "2-D curved classes",
    "pinwheel": "2-D rotating three-class arms",
    "nd_spiral_low_rank": "16-D · one latent spiral plane · PCA view",
    "nd_spiral_high_rank": "16-D · eight harmonic planes · PCA view",
    "hypercube_checker": "16-D rotated orthant parity · PCA view",
    "xor_quads": "2-D compositional XOR",
    "sinusoid_bounds": "2-D oscillatory decision bounds",
    "radial_stripes": "2-D concentric periodic classes",
    "swiss_cheese": "2-D localized holes",
    "lorenz_lobes": "3-D lobe classes · PCA view",
    "periodic_wells": "1-D periodic localized wells",
    "ripple": "1-D damped oscillation",
    "ring_sdf": "2-D signed distance field",
    "complex_spiral_3d": "1-D to 3-D spiral continuation",
    "periodic_nd": "10-D periodic regression · parity plot",
    "hyperchecker": "10-D parity classes · PCA view",
    "multiscale_1d": "1-D smooth, local, and high-frequency structure",
    "chirp_1d": "1-D continuously changing frequency",
    "poly_drifted_chirp_1d": "1-D polynomial amplitude and phase drift · outer bands withheld",
    "localized_steps_1d": "1-D piecewise localized steps",
    "fourier_mix_1d": "1-D mixed incommensurate frequencies",
}


def indices(length: int, maximum: int):
    if length <= maximum:
        return list(range(length))
    return [round(i * (length - 1) / (maximum - 1)) for i in range(maximum)]


def rounded(values, digits=4):
    return [round(float(value), digits) for value in values]


def compact_field(rows, kind):
    old = rows[0]["size"]
    positions = list(range(0, old, 2))

    def sample(values):
        result = [values[y * old + x] for y in positions for x in positions]
        return [int(value) for value in result] if kind == "classification" else rounded(result)

    return {
        "type": "field",
        "size": len(positions),
        "limits": rows[0]["limits"],
        "truth": sample(rows[0]["truth"]),
        "predictions": {row["variant"]: sample(row["prediction"]) for row in rows},
    }


def compact_scatter(rows):
    chosen = indices(len(rows[0]["xy"]), 500)
    return {
        "type": "scatter",
        "xy": [[round(float(v), 3) for v in rows[0]["xy"][i]] for i in chosen],
        "truth": [int(rows[0]["truth"][i]) for i in chosen],
        "predictions": {
            row["variant"]: [int(row["prediction"][i]) for i in chosen] for row in rows
        },
    }


def compact_parity(rows):
    chosen = indices(len(rows[0]["truth"]), 450)
    return {
        "type": "parity",
        "truth": rounded([rows[0]["truth"][i] for i in chosen]),
        "predictions": {
            row["variant"]: rounded([row["prediction"][i] for i in chosen]) for row in rows
        },
    }


def compact_curve(rows, dimensions=1):
    chosen = indices(len(rows[0]["input"] if dimensions == 3 else rows[0]["x"]), 121)
    x_key = "input" if dimensions == 3 else "x"

    def sample(values):
        if dimensions == 1:
            return rounded([values[i] for i in chosen])
        return [[round(float(v), 4) for v in values[i]] for i in chosen]

    result = {
        "type": "curve3d" if dimensions == 3 else "curve",
        "x": rounded([rows[0][x_key][i] for i in chosen]),
        "truth": sample(rows[0]["truth"]),
        "predictions": {row["variant"]: sample(row["prediction"]) for row in rows},
    }
    if dimensions == 1:
        result["train_limits"] = rows[0].get("train_limits")
    return result


def main():
    probe_rows = json.loads(PROBES.read_text())["probes"]
    summary = json.loads(SUMMARY.read_text())
    by_probe = {(row["task"], row["variant"]): row for row in probe_rows}
    by_score = {(row["task"], row["variant"]): row["score"] for row in summary["by_task"]}
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
        tasks.append({
            "task": task,
            "kind": kind[task],
            "description": DESCRIPTIONS[task],
            "scores": {variant: round(by_score[task, variant], 3) for variant in VARIANTS},
            **geometry,
        })
    data = json.dumps({"variants": VARIANTS, "tasks": tasks}, separators=(",", ":"))
    source = TEMPLATE.read_text().replace("__DATA__", data)
    FRAGMENT.write_text(source)
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes; {len(tasks)} tasks")


if __name__ == "__main__":
    main()
