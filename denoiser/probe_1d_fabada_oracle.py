"""Matched oracle-noise PFABADA comparison on the 1-D transport battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fabada_oracle import denoise_oracle_fabada_from_corruption_1d
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


def run(
    size: int,
    seeds: int,
    transport_record: Path | None = None,
) -> dict:
    transport_clean: dict[str, dict] = {}
    transport_noisy: dict[tuple[str, str, int], dict] = {}
    if transport_record is not None:
        source = json.loads(transport_record.read_text())
        if int(source["size"]) != size or int(source["seeds"]) != seeds:
            raise ValueError("transport record size/seeds do not match probe")
        transport_clean = {
            row["preset"]: row["action_contracted_collision_mean"]
            for row in source["clean_rows"]
        }
        transport_noisy = {
            (row["preset"], row["condition"], int(row["seed"])):
            row["action_contracted_collision_mean"]
            for row in source["rows"]
        }

    clean_rows = []
    noisy_rows = []
    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_forms, clean_diagnostic = (
            denoise_oracle_fabada_from_corruption_1d(
                truth, truth, "none", amount=0.0, density=0.0)
        )
        clean_row = {
            "preset": preset,
            "observation": metrics(truth, truth),
            "fabada_oracle_global": metrics(clean_forms["global"], truth),
            "fabada_oracle_local": metrics(clean_forms["local"], truth),
            "fabada_effective_dimension": clean_diagnostic.get(
                "effective_dimension", float(size)),
        }
        if preset in transport_clean:
            clean_row["phase_collision_posterior"] = transport_clean[preset]
        clean_rows.append(clean_row)

        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=20100 + seed,
                )
                forms, diagnostic = denoise_oracle_fabada_from_corruption_1d(
                    observation,
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                )
                row = {
                    "preset": preset,
                    "condition": condition,
                    "seed": seed,
                    "observation": metrics(observation, truth),
                    "fabada_oracle_global": metrics(forms["global"], truth),
                    "fabada_oracle_local": metrics(forms["local"], truth),
                    "fabada_effective_dimension": diagnostic[
                        "global_aggregate_effective_dimension"],
                    "fabada_minimum_risk_dimension": diagnostic[
                        "minimum_risk_effective_dimension"],
                }
                key = (preset, condition, seed)
                if key in transport_noisy:
                    row["phase_collision_posterior"] = transport_noisy[key]
                noisy_rows.append(row)

    methods = [
        "observation",
        "fabada_oracle_global",
        "fabada_oracle_local",
    ]
    if transport_noisy:
        methods.append("phase_collision_posterior")

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([row[method][key] for row in rows]))
                for key in rows[0][method]
            }
            for method in methods
        }

    result = {
        "purpose": (
            "explicitly unfair PFABADA comparison using the exact generating "
            "corruption family and conditional moments; clean truth is used "
            "only to evaluate those moments, never to score or select a "
            "candidate"
        ),
        "method": (
            "continuous Cesaro heat family indexed by effective degrees of "
            "freedom and aggregated by known-covariance unbiased risk"
        ),
        "removed_from_pyitd": (
            "boundary mass leak, reused-data variance collapse, malformed "
            "evidence denominator, chi-square weighting/stopping"
        ),
        "size": size,
        "seeds": seeds,
        "transport_record": str(transport_record) if transport_record else None,
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "by_condition": {
            condition: summarize([
                row for row in noisy_rows if row["condition"] == condition
            ])
            for condition, *_ in CONDITIONS
        },
        "mean_fabada_effective_dimension": float(np.mean([
            row["fabada_effective_dimension"] for row in noisy_rows
        ])),
        "clean_rows": clean_rows,
        "rows": noisy_rows,
    }
    if transport_noisy:
        result["mse_case_wins"] = {
            "fabada_oracle_global": int(sum(
                row["fabada_oracle_global"]["mse"]
                < row["phase_collision_posterior"]["mse"]
                for row in noisy_rows
            )),
            "phase_collision_posterior": int(sum(
                row["phase_collision_posterior"]["mse"]
                < row["fabada_oracle_global"]["mse"]
                for row in noisy_rows
            )),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--transport-record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds, args.transport_record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean_summary": result["clean_summary"],
        "noisy_summary": result["noisy_summary"],
        "mse_case_wins": result.get("mse_case_wins"),
        "mean_fabada_effective_dimension": result[
            "mean_fabada_effective_dimension"],
    }, indent=2))


if __name__ == "__main__":
    main()
