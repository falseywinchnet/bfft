"""Gate target-free authority against uncertainty in the transport itself."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    _energy_root_authority,
    _representation_floor,
    lineage_branch_transport_1d,
    nested_midpoint_lineage_transport_1d,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "information_lineage_collision_mean",
    "self_consistent_transport_mean",
    "transport_fidelity_mean",
    "authority_competition_mean",
    "authority_competition_linear_control",
    "action_contractor_mean",
    "phase_action_contractor_mean",
)


def _mean(mass: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sum(mass * prediction, axis=1)


def evaluate(value: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    information, _information_diagnostic = lineage_branch_transport_1d(value)
    nested, nested_diagnostic = nested_midpoint_lineage_transport_1d(value)
    authority, _root_action, _population_action = _energy_root_authority(
        value, nested["prediction"], nested["mass"])
    participation = information["path_fidelity_participation"]
    participation_survival = participation * participation
    authority_survival = authority * authority
    denominator = participation_survival + authority_survival
    distributed_coordinate = np.divide(
        participation_survival,
        denominator,
        out=np.full_like(denominator, 0.5),
        where=denominator > np.finfo(float).tiny,
    )

    reference_mass = information["reference_mass"]
    reference = (
        reference_mass[None, :]
        if reference_mass.ndim == 1 else reference_mass)
    coherent_mass = information["self_consistent_transport_mass"]
    distributed_mass = information["transport_fidelity_mass"]
    coherent_log_density = np.log(np.maximum(
        coherent_mass / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    distributed_log_density = np.log(np.maximum(
        distributed_mass / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    fused_log_density = (
        (1.0 - distributed_coordinate[:, None]) * coherent_log_density
        + distributed_coordinate[:, None] * distributed_log_density
    )
    fused_log_density -= np.max(fused_log_density, axis=1, keepdims=True)
    fused_mass = reference * np.exp(fused_log_density)
    fused_mass /= np.sum(fused_mass, axis=1, keepdims=True)

    prediction = information["prediction"]
    marginal = information["mass"]
    local_collision = marginal * marginal / np.maximum(
        reference, np.finfo(float).tiny)
    local_collision /= np.sum(local_collision, axis=1, keepdims=True)
    coherent_mean = _mean(coherent_mass, prediction)
    distributed_mean = _mean(distributed_mass, prediction)
    nested_prediction = nested["prediction"]
    nested_mass = nested["mass"]
    coherent_action = np.sum(
        nested_mass * np.abs(
            coherent_mean[:, None] - nested_prediction),
        axis=1,
    )
    distributed_action = np.sum(
        nested_mass * np.abs(
            distributed_mean[:, None] - nested_prediction),
        axis=1,
    )
    action_floor = np.finfo(float).tiny
    inverse_action = 1.0 / np.maximum(
        np.column_stack((coherent_action, distributed_action)),
        action_floor,
    )
    action_weight = inverse_action / np.sum(
        inverse_action, axis=1, keepdims=True)
    action_contractor_mean = (
        action_weight[:, 0] * coherent_mean
        + action_weight[:, 1] * distributed_mean)
    baseline_history_action = 0.5 * (
        coherent_action + distributed_action)
    contracted_history_action = np.sum(
        action_weight * np.column_stack((
            coherent_action, distributed_action)),
        axis=1,
    )
    if np.any(
        contracted_history_action
        > baseline_history_action + 32.0 * np.finfo(float).eps
    ):
        raise RuntimeError("reciprocal estimator action did not contract")

    coherent_jet = np.gradient(coherent_mean)
    distributed_jet = np.gradient(distributed_mean)
    nested_jet = nested["jet"]
    phase_action = np.empty((value.size, 2), dtype=np.float64)
    covariance_floor = _representation_floor(value) ** 2
    for index in range(value.size):
        state = np.column_stack((
            nested_prediction[index], nested_jet[index]))
        weight = nested_mass[index]
        center = weight @ state
        centered = state - center
        covariance = (centered * weight[:, None]).T @ centered
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        precision_eigenvalue = 1.0 / np.maximum(
            eigenvalue, covariance_floor)
        precision_eigenvalue /= np.sqrt(np.prod(precision_eigenvalue))
        precision = (
            eigenvector * precision_eigenvalue[None, :]
        ) @ eigenvector.T
        candidates = np.asarray((
            (coherent_mean[index], coherent_jet[index]),
            (distributed_mean[index], distributed_jet[index]),
        ))
        defect = candidates[:, None, :] - state[None, :, :]
        distance = np.sqrt(np.maximum(np.einsum(
            "hka,ab,hkb->hk", defect, precision, defect), 0.0))
        phase_action[index] = distance @ weight
    inverse_phase_action = 1.0 / np.maximum(phase_action, action_floor)
    phase_action_weight = inverse_phase_action / np.sum(
        inverse_phase_action, axis=1, keepdims=True)
    phase_action_contractor_mean = (
        phase_action_weight[:, 0] * coherent_mean
        + phase_action_weight[:, 1] * distributed_mean)
    baseline_phase_history_action = np.mean(phase_action, axis=1)
    contracted_phase_history_action = np.sum(
        phase_action_weight * phase_action, axis=1)
    if np.any(
        contracted_phase_history_action
        > baseline_phase_history_action + 32.0 * np.finfo(float).eps
    ):
        raise RuntimeError("reciprocal phase action did not contract")
    return {
        "information_lineage_collision_mean": _mean(
            local_collision, prediction),
        "self_consistent_transport_mean": coherent_mean,
        "transport_fidelity_mean": distributed_mean,
        "authority_competition_mean": _mean(fused_mass, prediction),
        "authority_competition_linear_control": (
            (1.0 - distributed_coordinate) * coherent_mean
            + distributed_coordinate * distributed_mean
        ),
        "action_contractor_mean": action_contractor_mean,
        "phase_action_contractor_mean": phase_action_contractor_mean,
    }, {
        "mean_target_free_authority": float(np.mean(authority)),
        "mean_fibre_participation": float(np.mean(participation)),
        "mean_distributed_coordinate": float(np.mean(
            distributed_coordinate)),
        "target_value_enters_context_action": bool(
            nested_diagnostic["target_value_enters_local_action"]),
        "mean_baseline_history_action": float(np.mean(
            baseline_history_action)),
        "mean_contracted_history_action": float(np.mean(
            contracted_history_action)),
        "maximum_history_action_increase": float(np.max(
            contracted_history_action - baseline_history_action)),
        "mean_baseline_phase_history_action": float(np.mean(
            baseline_phase_history_action)),
        "mean_contracted_phase_history_action": float(np.mean(
            contracted_phase_history_action)),
        "maximum_phase_history_action_increase": float(np.max(
            contracted_phase_history_action
            - baseline_phase_history_action)),
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
        forms, uncertainty = evaluate(value)
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            **uncertainty,
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
                    seed=15100 + seed,
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
                "mean_target_free_authority",
                "mean_fibre_participation",
                "mean_distributed_coordinate",
                "mean_baseline_history_action",
                "mean_contracted_history_action",
                "maximum_history_action_increase",
                "mean_baseline_phase_history_action",
                "mean_contracted_phase_history_action",
                "maximum_phase_history_action_increase",
            )
        }

    return {
        "purpose": (
            "falsify the order-two competition between target-free root "
            "authority and uncertainty about transport"
        ),
        "equation": "delta = p^2 / (p^2 + a^2)",
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
