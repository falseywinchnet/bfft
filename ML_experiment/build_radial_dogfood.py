#!/usr/bin/env python3
"""Build the topology-first radial dogfood report."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_radial_dogfood"
TEMPLATE = HERE / "radial_dogfood.template.html"
FRAGMENT = HERE / "radial_dogfood.fragment.html"

SPECS = (
    ("self_context", "value_separation.json", "self_context"),
    ("raw_curvature", "differentials.json", "self_context_jet_curvature_context"),
    ("tight_curvature", "tight_frame.json", "self_context_tight_frame_curvature"),
    ("continuous_flow", "stiefel_flow.json", "self_context_stiefel_flow_curvature"),
)


def selected_run(filename: str, variant: str):
    payload = json.loads((RESULTS / filename).read_text())
    return next(row for row in payload["runs"] if row["variant"] == variant)


def main():
    models = []
    truth = None
    for key, filename, variant in SPECS:
        row = selected_run(filename, variant)
        field = row["field"]
        if truth is None:
            truth = field["truth"]
        models.append({
            "key": key,
            "variant": variant,
            "score": round(row["validation_score"], 5),
            "profile_accuracy": round(row["profile"]["profile_accuracy"], 5),
            "angular_consistency": round(row["profile"]["angular_consistency"], 5),
            "central_class0_probability": round(row["profile"]["central_class0_probability"], 5),
            "radial_transitions": row["profile"]["radial_transitions"],
            "field": field["prediction"],
            "profile": row["profile"]["class1_probability"],
            "history": [{"step": point["step"], "score": round(point["score"], 5)}
                        for point in row["history"]],
        })
    first = selected_run(SPECS[0][1], SPECS[0][2])
    payload = {
        "size": first["field"]["size"],
        "limits": first["field"]["limits"],
        "truth": truth,
        "radii": first["profile"]["radius"],
        "profile_truth": first["profile"]["truth"],
        "models": models,
    }
    FRAGMENT.write_text(
        TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    print(FRAGMENT)
    print(f"{FRAGMENT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
