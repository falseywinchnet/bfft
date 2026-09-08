"""Compare parameter-free barycentres of two transported action measures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lifted_endpoint_action_transport_2d import (
    _endpoint_fractions,
    _hadamard_endpoint_intersection,
    denoise_lifted_endpoint_action_transport_2d,
)
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first) - np.asarray(second)) ** 2))


def _barycentres(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return smooth positive temporal means determined by two raw moments."""
    left = np.maximum(np.asarray(first, dtype=np.float64), 0.0)
    right = np.maximum(np.asarray(second, dtype=np.float64), 0.0)
    mu = 0.5 * (left + right)
    var = 0.25 * (left - right) ** 2
    floor = np.finfo(float).eps * max(float(np.max(mu * mu + var)), 1.0)
    second_moment = mu * mu + var
    product = np.maximum(mu * mu - var, 0.0)
    coherence = np.divide(
        mu * mu,
        second_moment,
        out=np.ones_like(mu),
        where=second_moment > floor,
    )
    harmonic = np.divide(
        product,
        mu,
        out=np.zeros_like(mu),
        where=mu > np.sqrt(floor),
    )
    pairwise_projective = np.zeros_like(mu)
    for start in (0, 2):
        left_pair = left[start:start + 2]
        right_pair = right[start:start + 2]
        left_mass = np.sum(left_pair, axis=0)
        right_mass = np.sum(right_pair, axis=0)
        left_probability = np.divide(
            left_pair,
            left_mass[None, ...],
            out=np.zeros_like(left_pair),
            where=left_mass[None, ...] > np.sqrt(floor),
        )
        right_probability = np.divide(
            right_pair,
            right_mass[None, ...],
            out=np.zeros_like(right_pair),
            where=right_mass[None, ...] > np.sqrt(floor),
        )
        left_probability[0, left_mass <= np.sqrt(floor)] = 1.0
        right_probability[0, right_mass <= np.sqrt(floor)] = 1.0
        pairwise_projective[start:start + 2] = 0.5 * (
            left_probability + right_probability)

    left_norm = np.sqrt(np.sum(left * left, axis=0))
    right_norm = np.sqrt(np.sum(right * right, axis=0))
    cosine = np.divide(
        np.sum(left * right, axis=0),
        left_norm * right_norm,
        out=np.zeros_like(left_norm),
        where=left_norm * right_norm > np.sqrt(floor),
    )
    causal_cosine = left + cosine[None, ...] * right
    return {
        "arithmetic": mu,
        "coherence_contracted": mu * coherence,
        "geometric": np.sqrt(product),
        "harmonic": harmonic,
        "rms": np.sqrt(second_moment),
        "pairwise_projective": pairwise_projective,
        "causal_cosine": causal_cosine,
    }


def run(size: int) -> dict[str, Any]:
    catalogue = sources(size)
    selected = ("cameraman", "tapered hair", "woven chirps")
    corruptions = (
        ("clean", None, 0.0, 0.0),
        ("Gaussian additive 0.10", "Gaussian additive", 0.10, 0.0),
        ("uniform additive 0.10", "uniform additive", 0.10, 0.0),
        ("salt and pepper 0.20", "salt and pepper", 0.0, 0.20),
        (
            "mixed replacement + uniform 0.25",
            "mixed replacement + uniform", 0.10, 0.25,
        ),
    )
    rows: list[dict[str, Any]] = []
    for source in selected:
        truth = catalogue[source]
        for condition, kind, amount, density in corruptions:
            observation = (
                truth.copy()
                if kind is None
                else corrupt(
                    truth, kind, amount=amount, density=density, seed=9100)
            )
            _baseline, diagnostic = (
                denoise_lifted_endpoint_action_transport_2d(
                    observation, return_temporal_ablation=True))
            posterior = np.asarray(
                diagnostic["posterior_before_endpoint_action"])
            fine_basis = np.asarray(diagnostic["fine_basis"])
            coarse_basis = np.asarray(diagnostic["coarse_basis"])
            phase = np.asarray(diagnostic["observation_phase_authority"])
            context = float(diagnostic["observation_phase"][
                "action_weighted_authority"])
            scale = np.asarray(diagnostic["transported_scale_support"])
            first_action = np.asarray(
                diagnostic["carried_first_endpoint_action"])
            second_action = np.asarray(
                diagnostic["second_observed_endpoint_action"])
            candidate_actions = _barycentres(first_action, second_action)
            temporal_support = phase * context * scale
            temporal_rejection = (
                (1.0 - phase) * (1.0 - context) * (1.0 - scale))
            temporal_total = temporal_support + temporal_rejection
            temporal_floor = np.finfo(float).eps * max(
                float(np.max(temporal_total)), 1.0)
            structural_authority = np.divide(
                temporal_support,
                temporal_total,
                out=np.ones_like(temporal_total),
                where=temporal_total > temporal_floor,
            )
            first_norm = np.sqrt(np.sum(first_action * first_action, axis=0))
            second_norm = np.sqrt(np.sum(
                second_action * second_action, axis=0))
            norm_product = first_norm * second_norm
            temporal_cosine = np.divide(
                np.sum(first_action * second_action, axis=0),
                norm_product,
                out=np.zeros_like(norm_product),
                where=norm_product > np.sqrt(temporal_floor),
            )
            component_overlap = np.divide(
                2.0 * first_action * second_action,
                first_action * first_action + second_action * second_action,
                out=np.zeros_like(first_action),
                where=(first_action * first_action
                       + second_action * second_action) > temporal_floor,
            )
            first_mass = np.sum(first_action, axis=0)
            second_mass = np.sum(second_action, axis=0)
            aligned_second = second_action * np.divide(
                first_mass,
                second_mass,
                out=np.zeros_like(first_mass),
                where=second_mass > np.sqrt(temporal_floor),
            )[None, ...]
            aligned_square_sum = first_action * first_action + (
                aligned_second * aligned_second)
            aligned_component_overlap = np.divide(
                2.0 * first_action * aligned_second,
                aligned_square_sum,
                out=np.zeros_like(first_action),
                where=aligned_square_sum > temporal_floor,
            )
            candidate_actions["structure_causal"] = (
                first_action
                + structural_authority[None, ...] * second_action
            )
            candidate_actions["structure_coherent_causal"] = (
                first_action
                + (structural_authority * temporal_cosine)[None, ...]
                * second_action
            )
            candidate_actions["rejection_causal"] = (
                first_action
                + (1.0 - structural_authority)[None, ...] * second_action
            )
            candidate_actions["rejection_coherent_causal"] = (
                first_action
                + ((1.0 - structural_authority) * temporal_cosine)[None, ...]
                * second_action
            )
            candidate_actions["rejection_overlap_causal"] = (
                first_action
                + (1.0 - structural_authority)[None, ...]
                * component_overlap * second_action
            )
            candidate_actions["rejection_overlap_coherent_causal"] = (
                first_action
                + ((1.0 - structural_authority)
                   * temporal_cosine)[None, ...]
                * component_overlap * second_action
            )
            candidate_actions[
                "projective_rejection_overlap_coherent_causal"
            ] = (
                first_action
                + ((1.0 - structural_authority)
                   * temporal_cosine)[None, ...]
                * aligned_component_overlap * aligned_second
            )
            candidate_actions["rejection_interpolation"] = (
                structural_authority[None, ...] * first_action
                + (1.0 - structural_authority)[None, ...] * second_action
            )
            estimates: dict[str, float] = {}
            for name, action in candidate_actions.items():
                normal_fine, normal_coarse, _fraction = (
                    _endpoint_fractions(action))
                fine = _hadamard_endpoint_intersection(
                    normal_fine, phase, context, scale)
                coarse = _hadamard_endpoint_intersection(
                    normal_coarse, phase, context, scale)
                estimate = (
                    posterior + fine * fine_basis + coarse * coarse_basis)
                estimates[name] = _mse(estimate, truth)
            first_estimate = np.asarray(diagnostic["first_cycle_estimate"])
            second_estimate = np.asarray(
                diagnostic["unstopped_second_cycle_estimate"])
            disagreement = np.mean(
                0.25 * (first_action - second_action) ** 2, axis=0)
            endpoint_variance = np.asarray(
                diagnostic["endpoint_action_variance"])
            transport_variance = np.asarray(
                diagnostic["lifted"]["transport_uncertainty"])
            action_second_moment = np.mean(
                0.5 * (first_action * first_action
                       + second_action * second_action),
                axis=0,
            )
            references = {
                "endpoint_fourth": endpoint_variance * endpoint_variance,
                "transport_fourth": transport_variance * transport_variance,
                "joint_fourth": (
                    endpoint_variance + transport_variance) ** 2,
                "action_second_moment": action_second_moment,
            }
            stopping_mse: dict[str, float] = {}
            mean_continuation: dict[str, float] = {}
            for name, reference in references.items():
                total = reference + disagreement
                stop_floor = np.finfo(float).eps * max(
                    float(np.max(total)), 1.0)
                continuation = np.divide(
                    reference,
                    total,
                    out=np.ones_like(total),
                    where=total > stop_floor,
                )
                stopped = first_estimate + continuation * (
                    second_estimate - first_estimate)
                stopping_mse[name] = _mse(stopped, truth)
                mean_continuation[name] = float(np.mean(continuation))
            rows.append({
                "source": source,
                "condition": condition,
                "observation_mse": _mse(observation, truth),
                "first_cycle_mse": _mse(
                    diagnostic["first_cycle_estimate"], truth),
                "candidate_mse": estimates,
                "stopping_mse": stopping_mse,
                "mean_continuation": mean_continuation,
            })

    rules = tuple(rows[0]["candidate_mse"])
    groups = {
        "clean": [row for row in rows if row["condition"] == "clean"],
        "additive": [
            row for row in rows
            if row["condition"].startswith(("Gaussian", "uniform"))
        ],
        "replacement": [
            row for row in rows
            if row["condition"].startswith(("salt", "mixed"))
        ],
    }
    summary: dict[str, Any] = {}
    for rule in rules:
        summary[rule] = {
            group: {
                "improves_first_count": sum(
                    row["candidate_mse"][rule] < row["first_cycle_mse"]
                    for row in members
                ),
                "case_count": len(members),
                "mean_mse": float(np.mean([
                    row["candidate_mse"][rule] for row in members
                ])),
            }
            for group, members in groups.items()
        }
    stopping_summary: dict[str, Any] = {}
    for rule in rows[0]["stopping_mse"]:
        stopping_summary[rule] = {
            group: {
                "improves_first_count": sum(
                    row["stopping_mse"][rule] < row["first_cycle_mse"]
                    for row in members
                ),
                "case_count": len(members),
                "mean_mse": float(np.mean([
                    row["stopping_mse"][rule] for row in members
                ])),
                "mean_continuation": float(np.mean([
                    row["mean_continuation"][rule] for row in members
                ])),
            }
            for group, members in groups.items()
        }
    return {
        "purpose": (
            "separate temporal action geometry from the downstream endpoint "
            "and compare smooth parameter-free two-observation means"
        ),
        "size": int(size),
        "summary": summary,
        "stopping_summary": stopping_summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "barycentres": result["summary"],
        "stopping": result["stopping_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
