"""Gate branch-particle covariance against scalar residual barycenters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    denoise_cross_predictive_mixed_transport,
    denoise_cross_predictive_particle_transport,
    denoise_cross_predictive_w1_transport,
    relation_scale_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(size: int, seeds: int) -> dict:
    rows = []
    clean_rows = []
    methods = ("one_pass", "w1", "mixed", "particle")
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]

        def evaluate(value):
            return {
                "one_pass": relation_scale_readout_forms(value)[0]["median"],
                "w1": denoise_cross_predictive_w1_transport(value)[0],
                "mixed": denoise_cross_predictive_mixed_transport(value)[0],
                "particle": denoise_cross_predictive_particle_transport(value)[0],
            }

        clean_rows.append({
            "preset": preset,
            **{name: metrics(value, truth) for name, value in evaluate(truth).items()},
        })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density, seed=8100 + seed)
                forms = evaluate(observation)
                particle_diagnostic = denoise_cross_predictive_particle_transport(
                    observation)[1]
                rows.append({
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    **{name: metrics(value, truth) for name, value in forms.items()},
                    "particle_steps": particle_diagnostic[
                        "accepted_continuations"],
                    "particle_ceiling": particle_diagnostic[
                        "continuation_ceiling_hit"],
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
            "test residual covariance before collapsing the aligned lag/path law"
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
        "particle_ceiling_hits": int(sum(
            row["particle_ceiling"] for row in rows)),
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
        "particle_ceiling_hits": result["particle_ceiling_hits"],
    }, indent=2))


if __name__ == "__main__":
    main()
