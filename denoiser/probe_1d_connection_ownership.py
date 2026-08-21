"""Broad gate for joint signal-branch and connection-ownership transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    connection_ownership_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "fused_collision_mean",
    "ownership_mean",
    "ownership_collision_mean",
    "ownership_path_collision_mean",
)


def evaluate(
    value: np.ndarray,
    ownership_measure: str,
) -> tuple[dict[str, np.ndarray], dict]:
    ownership, diagnostic = connection_ownership_readout_forms(
        value, ownership_measure=ownership_measure)
    return {
        "fused_collision_mean": ownership["baseline_collision_mean"],
        "ownership_mean": ownership["mean"],
        "ownership_collision_mean": ownership["collision_mean"],
        "ownership_path_collision_mean": ownership["path_collision_mean"],
    }, diagnostic


def run(size: int, seeds: int, ownership_measure: str) -> dict:
    clean_rows: list[dict] = []
    noisy_rows: list[dict] = []

    def row(
        preset: str,
        truth: np.ndarray,
        value: np.ndarray,
        *,
        condition: str | None = None,
        seed: int | None = None,
    ) -> dict:
        forms, diagnostic = evaluate(value, ownership_measure)
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            "mean_drift_ownership": diagnostic["mean_drift_ownership"],
            "mean_mode_population": diagnostic["mean_mode_population"],
            "mean_mode_transition_population": diagnostic[
                "mean_mode_transition_population"],
            "mean_root_authority": diagnostic["mean_root_authority"],
            "mean_context_authority": diagnostic[
                "mean_context_authority"],
            "mean_sparse_bypass_survival": diagnostic[
                "mean_sparse_bypass_survival"],
            "mean_two_history_bypass_survival": diagnostic[
                "mean_two_history_bypass_survival"],
        }
        if condition is not None:
            result["condition"] = condition
        if seed is not None:
            result["seed"] = seed
        return result

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_rows.append(row(preset, truth, truth))
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=18100 + seed,
                )
                noisy_rows.append(row(
                    preset,
                    truth,
                    observation,
                    condition=condition,
                    seed=seed,
                ))

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([entry[method][key] for entry in rows]))
                for key in rows[0][method]
            }
            for method in METHODS
        }

    def uncertainty(rows: list[dict]) -> dict:
        return {
            key: float(np.mean([entry[key] for entry in rows]))
            for key in (
                "mean_drift_ownership",
                "mean_mode_population",
                "mean_mode_transition_population",
                "mean_root_authority",
                "mean_context_authority",
                "mean_sparse_bypass_survival",
                "mean_two_history_bypass_survival",
            )
        }

    return {
        "purpose": (
            "falsify joint connection ownership as a transported latent path "
            "variable before signal-branch marginalization"
        ),
        "joint_state": "(connection ownership, signal branch)",
        "connection_modes": (
            "fused zero-defect information connection",
            "bidirectional two-history drift connection",
        ),
        "ownership_measure": ownership_measure,
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "clean_uncertainty": uncertainty(clean_rows),
        "noisy_uncertainty": uncertainty(noisy_rows),
        "by_condition": {
            condition: summarize([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "uncertainty_by_condition": {
            condition: uncertainty([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "clean_rows": clean_rows,
        "rows": noisy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--ownership-measure",
        choices=(
            "root_context",
            "connection_hotelling",
            "connection_hellinger",
            "transported_hellinger_contrast",
            "transported_covariance_contrast",
            "transported_gaussian_law_contrast",
        ),
        default="root_context",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds, args.ownership_measure)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
        "uncertainty": {
            "clean": result["clean_uncertainty"],
            "noisy": result["noisy_uncertainty"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
