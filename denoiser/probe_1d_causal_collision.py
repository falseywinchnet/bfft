"""Gate target-excluded affine collisions against the current 1-D laws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    causal_collision_readout_forms,
    causal_crossfit_readout_forms,
    relation_scale_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    methods = (
        "validated_mean", "validated_median",
        "collision_mean", "collision_median", "collision_maximum",
        "crossfit_mean", "crossfit_median", "crossfit_maximum",
    )

    def evaluate(value):
        validated = relation_scale_readout_forms(value)[0]
        collision = causal_collision_readout_forms(value)[0]
        crossfit = causal_crossfit_readout_forms(value)[0]
        return {
            "validated_mean": validated["mean"],
            "validated_median": validated["median"],
            "collision_mean": collision["mean"],
            "collision_median": collision["median"],
            "collision_maximum": collision["maximum_branch"],
            "crossfit_mean": crossfit["mean"],
            "crossfit_median": crossfit["median"],
            "crossfit_maximum": crossfit["maximum_branch"],
        }

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_rows.append({
            "preset": preset,
            **{name: metrics(value, truth)
               for name, value in evaluate(truth).items()},
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=9100 + seed)
                rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    **{name: metrics(value, truth)
                       for name, value in evaluate(observation).items()},
                })

    def summarize(selected):
        return {
            method: {
                key: float(np.mean([row[method][key] for row in selected]))
                for key in selected[0][method]
            }
            for method in methods
        }

    return {
        "purpose": (
            "test target-excluded characteristic collision action before "
            "residual continuation"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition, *_rest in CONDITIONS
        },
        "clean_rows": clean_rows,
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
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
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
