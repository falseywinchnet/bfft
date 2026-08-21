"""Gate complete support algebra against the fused information metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    _scalar_lineage_readouts,
    complete_participation_lineage_transport_1d,
    lineage_branch_transport_1d,
    transport_distribution_lineage_transport_1d,
    transport_covariance_lineage_transport_1d,
    self_consistent_connection_lineage_transport_1d,
    collision_consistent_connection_lineage_transport_1d,
    bidirectional_collision_connection_lineage_transport_1d,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "fused_collision_mean",
    "fused_self_consistent_transport_mean",
    "complete_collision_mean",
    "complete_path_collision_mean",
    "complete_path_fidelity_mean",
    "complete_self_consistent_transport_mean",
    "complete_two_history_action_transport_mean",
    "connection_collision_mean",
    "connection_path_fidelity_mean",
    "connection_self_consistent_transport_mean",
    "connection_two_history_action_transport_mean",
    "covariance_collision_mean",
    "covariance_path_fidelity_mean",
    "covariance_self_consistent_transport_mean",
    "covariance_two_history_action_transport_mean",
    "self_consistent_connection_collision_mean",
    "self_consistent_connection_path_fidelity_mean",
    "self_consistent_connection_self_consistent_transport_mean",
    "collision_connection_collision_mean",
    "collision_connection_path_fidelity_mean",
    "collision_connection_self_consistent_transport_mean",
    "bidirectional_connection_collision_mean",
    "bidirectional_connection_path_fidelity_mean",
    "bidirectional_connection_self_consistent_transport_mean",
)


def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    fused_law, _fused_diagnostic = lineage_branch_transport_1d(value)
    complete_law, complete_diagnostic = (
        complete_participation_lineage_transport_1d(value))
    connection_law, connection_diagnostic = (
        transport_distribution_lineage_transport_1d(value))
    covariance_law, covariance_diagnostic = (
        transport_covariance_lineage_transport_1d(value))
    self_connection_law, self_connection_diagnostic = (
        self_consistent_connection_lineage_transport_1d(value))
    collision_connection_law, collision_connection_diagnostic = (
        collision_consistent_connection_lineage_transport_1d(value))
    bidirectional_connection_law, bidirectional_connection_diagnostic = (
        bidirectional_collision_connection_lineage_transport_1d(value))
    fused = _scalar_lineage_readouts(fused_law)
    complete = _scalar_lineage_readouts(complete_law)
    connection = _scalar_lineage_readouts(connection_law)
    covariance = _scalar_lineage_readouts(covariance_law)
    self_connection = _scalar_lineage_readouts(self_connection_law)
    collision_connection = _scalar_lineage_readouts(
        collision_connection_law)
    bidirectional_connection = _scalar_lineage_readouts(
        bidirectional_connection_law)
    return {
        "fused_collision_mean": fused["collision_mean"],
        "fused_self_consistent_transport_mean": fused[
            "self_consistent_transport_mean"],
        "complete_collision_mean": complete["collision_mean"],
        "complete_path_collision_mean": complete["path_collision_mean"],
        "complete_path_fidelity_mean": complete["path_fidelity_mean"],
        "complete_self_consistent_transport_mean": complete[
            "self_consistent_transport_mean"],
        "complete_two_history_action_transport_mean": complete[
            "two_history_action_transport_mean"],
        "connection_collision_mean": connection["collision_mean"],
        "connection_path_fidelity_mean": connection["path_fidelity_mean"],
        "connection_self_consistent_transport_mean": connection[
            "self_consistent_transport_mean"],
        "connection_two_history_action_transport_mean": connection[
            "two_history_action_transport_mean"],
        "covariance_collision_mean": covariance["collision_mean"],
        "covariance_path_fidelity_mean": covariance["path_fidelity_mean"],
        "covariance_self_consistent_transport_mean": covariance[
            "self_consistent_transport_mean"],
        "covariance_two_history_action_transport_mean": covariance[
            "two_history_action_transport_mean"],
        "self_consistent_connection_collision_mean": self_connection[
            "collision_mean"],
        "self_consistent_connection_path_fidelity_mean": self_connection[
            "path_fidelity_mean"],
        "self_consistent_connection_self_consistent_transport_mean": (
            self_connection["self_consistent_transport_mean"]),
        "collision_connection_collision_mean": collision_connection[
            "collision_mean"],
        "collision_connection_path_fidelity_mean": collision_connection[
            "path_fidelity_mean"],
        "collision_connection_self_consistent_transport_mean": (
            collision_connection["self_consistent_transport_mean"]),
        "bidirectional_connection_collision_mean": bidirectional_connection[
            "collision_mean"],
        "bidirectional_connection_path_fidelity_mean": (
            bidirectional_connection["path_fidelity_mean"]),
        "bidirectional_connection_self_consistent_transport_mean": (
            bidirectional_connection["self_consistent_transport_mean"]),
    }, {
        "fused": _fused_diagnostic,
        "complete": complete_diagnostic,
        "connection": connection_diagnostic,
        "covariance": covariance_diagnostic,
        "self_connection": self_connection_diagnostic,
        "collision_connection": collision_connection_diagnostic,
        "bidirectional_connection": bidirectional_connection_diagnostic,
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
            "mean_lineage_population": diagnostic["connection"][
                "mean_lineage_population"],
            "mean_transport_edge_fidelity": diagnostic["connection"][
                "transport_plan_fidelity"]["mean_edge_fidelity"],
            "connection_log_evidence_advantage": (
                diagnostic["connection"]["log_path_evidence"]
                - diagnostic["fused"]["log_path_evidence"]),
            "mean_connection_authority": diagnostic["self_connection"][
                "mean_connection_authority"],
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
                    seed=16100 + seed,
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

    return {
        "purpose": (
            "falsify complete unit-multiplicity support algebra against a "
            "single fused value/jet/residual transport metric"
        ),
        "identity": "(1 + K_v)(1 + K_j)(1 + K_r) - 1",
        "size": int(size),
        "seeds": int(seeds),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "by_condition": {
            condition: summarize([
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
    }, indent=2))


if __name__ == "__main__":
    main()
