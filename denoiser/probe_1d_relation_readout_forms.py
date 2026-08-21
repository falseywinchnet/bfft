"""Matched 1-D gate for mean, median, and maximum characteristic branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import relation_scale_readout_forms
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_forms, _clean_diagnostic = relation_scale_readout_forms(truth)
        clean_rows.append({
            "preset": preset,
            **{
                name: metrics(value, truth)
                for name, value in clean_forms.items()
            },
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=8100 + seed,
                )
                forms, diagnostic = relation_scale_readout_forms(observation)
                rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    **{name: metrics(value, truth) for name, value in forms.items()},
                    "collision_population": diagnostic[
                        "mean_collision_population"],
                })
    methods = ("mean", "median", "maximum_branch")
    summary = {
        method: {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in rows[0][method]
        }
        for method in methods
    }
    clean_summary = {
        method: {
            key: float(np.mean([row[method][key] for row in clean_rows]))
            for key in clean_rows[0][method]
        }
        for method in methods
    }
    by_condition = {
        condition: {
            method: {
                key: float(np.mean([
                    row[method][key] for row in rows
                    if row["condition"] == condition
                ]))
                for key in rows[0][method]
            }
            for method in methods
        }
        for condition, *_rest in CONDITIONS
    }
    return {
        "purpose": "separate 1-D relation transport from its scalar readout",
        "size": int(size),
        "seeds": int(seeds),
        "summary": summary,
        "clean_summary": clean_summary,
        "by_condition": by_condition,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean_summary": result["clean_summary"],
        "summary": result["summary"],
        "heavy": {
            condition: result["by_condition"][condition]
            for condition in (
                "replacement 0.25", "replacement 0.40",
                "mixed 0.25", "mixed 0.40",
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
