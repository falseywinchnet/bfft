"""Full battery for one-pass, mixed, and fully W1 residual continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    denoise_cross_predictive_mixed_transport,
    denoise_cross_predictive_w1_transport,
    relation_scale_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        one_pass = relation_scale_readout_forms(truth)[0]["median"]
        mixed, mixed_diagnostic = denoise_cross_predictive_mixed_transport(truth)
        w1, w1_diagnostic = denoise_cross_predictive_w1_transport(truth)
        clean_rows.append({
            "preset": preset,
            "w1_one_pass": metrics(one_pass, truth),
            "mixed_continuation": metrics(mixed, truth),
            "w1_continuation": metrics(w1, truth),
            "mixed_steps": mixed_diagnostic["accepted_continuations"],
            "w1_steps": w1_diagnostic["accepted_continuations"],
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
                one_pass = relation_scale_readout_forms(observation)[0]["median"]
                mixed, mixed_diagnostic = (
                    denoise_cross_predictive_mixed_transport(observation))
                w1, w1_diagnostic = denoise_cross_predictive_w1_transport(
                    observation)
                rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    "w1_one_pass": metrics(one_pass, truth),
                    "mixed_continuation": metrics(mixed, truth),
                    "w1_continuation": metrics(w1, truth),
                    "mixed_steps": mixed_diagnostic["accepted_continuations"],
                    "w1_steps": w1_diagnostic["accepted_continuations"],
                    "mixed_ceiling": mixed_diagnostic[
                        "continuation_ceiling_hit"],
                    "w1_ceiling": w1_diagnostic["continuation_ceiling_hit"],
                })
    methods = ("w1_one_pass", "mixed_continuation", "w1_continuation")

    def summarize(selected):
        return {
            method: {
                key: float(np.mean([row[method][key] for row in selected]))
                for key in selected[0][method]
            }
            for method in methods
        }

    by_condition = {
        condition: summarize([
            row for row in rows if row["condition"] == condition])
        for condition, *_rest in CONDITIONS
    }
    return {
        "purpose": (
            "test whether the absolute-action relation law requires W1 "
            "barycenters in both the initial and residual sections"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(rows),
        "by_condition": by_condition,
        "mixed_ceiling_hits": int(sum(row["mixed_ceiling"] for row in rows)),
        "w1_ceiling_hits": int(sum(row["w1_ceiling"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
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
        "mixed_ceiling_hits": result["mixed_ceiling_hits"],
        "w1_ceiling_hits": result["w1_ceiling_hits"],
    }, indent=2))


if __name__ == "__main__":
    main()
