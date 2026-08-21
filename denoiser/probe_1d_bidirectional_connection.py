"""Broad gate for bidirectional two-history connection survival."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    _scalar_lineage_readouts,
    ancestry_connection_lineage_transport_1d,
    bidirectional_collision_connection_lineage_transport_1d,
    lineage_branch_transport_1d,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "fused_collision_mean",
    "bidirectional_connection_collision_mean",
    "bidirectional_connection_path_fidelity_mean",
    "bidirectional_connection_self_consistent_transport_mean",
    "ancestry_connection_collision_mean",
    "ancestry_connection_path_fidelity_mean",
    "ancestry_connection_self_consistent_transport_mean",
)


def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    fused_law, fused_diagnostic = lineage_branch_transport_1d(value)
    candidate_law, candidate_diagnostic = (
        bidirectional_collision_connection_lineage_transport_1d(value))
    fused = _scalar_lineage_readouts(fused_law)
    candidate = _scalar_lineage_readouts(candidate_law)
    ancestry_law, ancestry_diagnostic = ancestry_connection_lineage_transport_1d(
        value)
    ancestry = _scalar_lineage_readouts(ancestry_law)
    return {
        "fused_collision_mean": fused["collision_mean"],
        "bidirectional_connection_collision_mean": candidate[
            "collision_mean"],
        "bidirectional_connection_path_fidelity_mean": candidate[
            "path_fidelity_mean"],
        "bidirectional_connection_self_consistent_transport_mean": candidate[
            "self_consistent_transport_mean"],
        "ancestry_connection_collision_mean": ancestry["collision_mean"],
        "ancestry_connection_path_fidelity_mean": ancestry[
            "path_fidelity_mean"],
        "ancestry_connection_self_consistent_transport_mean": ancestry[
            "self_consistent_transport_mean"],
    }, {
        "fused": fused_diagnostic,
        "candidate": candidate_diagnostic,
        "ancestry": ancestry_diagnostic,
    }


def run(size: int, seeds: int) -> dict:
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
        forms, diagnostic = evaluate(value)
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            "mean_connection_authority": diagnostic["candidate"][
                "mean_connection_authority"],
            "candidate_log_evidence_advantage": (
                diagnostic["candidate"]["log_path_evidence"]
                - diagnostic["fused"]["log_path_evidence"]),
            "mean_ancestry_connection_authority": diagnostic["ancestry"][
                "mean_connection_authority"],
            "mean_connection_family_disagreement": diagnostic["ancestry"][
                "mean_connection_family_disagreement"],
            "mean_connection_family_fidelity": diagnostic["ancestry"][
                "mean_connection_family_fidelity"],
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
                    seed=17100 + seed,
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
                "mean_connection_authority",
                "candidate_log_evidence_advantage",
                "mean_ancestry_connection_authority",
                "mean_connection_family_disagreement",
                "mean_connection_family_fidelity",
            )
        }

    return {
        "purpose": (
            "broad falsification of bidirectional two-history survival for "
            "posterior connection drift"
        ),
        "connection_action": "s = mu^T (Cov_left + Cov_right)^-1 mu",
        "single_history_authority": "rho = s / (1 + s)",
        "bidirectional_survival": "rho^(2 histories * 2 endpoints)",
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
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
