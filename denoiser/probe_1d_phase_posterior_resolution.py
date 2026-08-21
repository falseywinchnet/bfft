"""Resolution gate for phase-collision connection-posterior transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cross_predictive_transport import (
    action_contracting_connection_readout_forms,
)
from .run_1d_cross_predictive_battery import PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


CONDITIONS = (
    ("uniform 0.15", "uniform additive", 0.15, 0.25),
    ("Gaussian 0.15", "Gaussian additive", 0.15, 0.25),
    ("replacement 0.25", "random-value replacement", 0.15, 0.25),
    ("salt-pepper 0.10", "salt and pepper", 0.15, 0.10),
    ("mixed 0.25", "mixed replacement + uniform", 0.15, 0.25),
)


def evaluate(value):
    retained, _ = action_contracting_connection_readout_forms(
        value,
        fuse_population_phase_odds=True,
        newton_optimize_connection=True,
    )
    candidate, diagnostic = action_contracting_connection_readout_forms(
        value,
        fuse_population_phase_odds=True,
        phase_coherent_connection_posterior=True,
    )
    return {
        "zero_connection": candidate["baseline_collision_mean"],
        "retained_newton_phase": retained["collision_mean"],
        "phase_collision_posterior": candidate["collision_mean"],
    }, diagnostic


def run(sizes: tuple[int, ...], seeds: int) -> dict:
    rows = []
    for size in sizes:
        for preset in PRESET_NAMES:
            truth = compose_series(size, PRESETS[preset])[1]
            forms, diagnostic = evaluate(truth)
            rows.append({
                "size": size,
                "preset": preset,
                "condition": "clean",
                "seed": None,
                **{name: metrics(value, truth) for name, value in forms.items()},
                "mean_phase_connection_order": diagnostic[
                    "mean_phase_connection_order"],
            })
            for condition, kind, amount, density in CONDITIONS:
                for seed in range(seeds):
                    observation = corrupt(
                        truth,
                        kind,
                        amount=amount,
                        density=density,
                        seed=20100 + seed,
                    )
                    forms, diagnostic = evaluate(observation)
                    rows.append({
                        "size": size,
                        "preset": preset,
                        "condition": condition,
                        "seed": seed,
                        **{
                            name: metrics(value, truth)
                            for name, value in forms.items()
                        },
                        "mean_phase_connection_order": diagnostic[
                            "mean_phase_connection_order"],
                    })

    methods = (
        "zero_connection",
        "retained_newton_phase",
        "phase_collision_posterior",
    )

    def summary(selected):
        return {
            method: {
                key: sum(row[method][key] for row in selected) / len(selected)
                for key in selected[0][method]
            }
            for method in methods
        }

    return {
        "purpose": (
            "test phase-collision posterior transport across sampling "
            "resolutions without changing its law"
        ),
        "sizes": list(sizes),
        "seeds": seeds,
        "conditions": [condition for condition, *_ in CONDITIONS],
        "by_size": {
            str(size): {
                "clean": summary([
                    row for row in rows
                    if row["size"] == size and row["condition"] == "clean"
                ]),
                "noisy": summary([
                    row for row in rows
                    if row["size"] == size and row["condition"] != "clean"
                ]),
            }
            for size in sizes
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="96,160")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if any(size < 32 for size in sizes) or args.seeds < 1:
        raise ValueError("sizes must be >= 32 and seeds must be positive")
    result = run(sizes, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["by_size"], indent=2))


if __name__ == "__main__":
    main()
