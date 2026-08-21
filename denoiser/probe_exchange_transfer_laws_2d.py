"""Compare lawful residual-return maps inside conservative exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .causal_scale_transport_2d import _screened_transport
from .conservative_exchange_transport_2d import (
    _phase_action_authority,
    _phase_screened_smooth,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .residual_erosion_transport_2d import _cavity_residual_relation
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


_LAWS = (
    "whole_smooth",
    "no_return",
    "phase_action",
    "phase_action_no_joint",
    "phase_action_phase_joint",
    "phase_action_intersection_joint",
    "phase_action_relational_joint",
    "phase_curvature_sum",
)


def _return_candidate(
    law: str,
    posterior: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    phase, phase_diagnostic = _phase_action_authority(residual)
    smoothed, _smoothing = _phase_screened_smooth(
        residual, posterior, residual, phase)
    if law == "whole_smooth":
        donation = smoothed
        cavity_action = 0.0
    elif law == "no_return":
        donation = np.zeros_like(residual)
        cavity_action = 0.0
    elif law in {
        "phase_action",
        "phase_action_no_joint",
        "phase_action_phase_joint",
        "phase_action_intersection_joint",
        "phase_action_relational_joint",
    }:
        donation = phase * smoothed
        cavity_action = 0.0
    elif law == "phase_curvature_sum":
        metric = continual_transport_metric(posterior, residual * residual)
        laplacian, _markov, stencil = _continual_flux_laplacian(
            metric, 1.0 - phase)
        maximum_degree = float(stencil["maximum_degree"])
        raw_cavity, _relation = _cavity_residual_relation(
            posterior, residual, laplacian, maximum_degree)
        cavity = (
            _screened_transport(
                laplacian,
                1.0 / maximum_degree,
                raw_cavity[None, ...],
            )[0]
            if maximum_degree > 0.0 else raw_cavity
        )
        # The curvature channel acts only on the complement not already
        # claimed by reciprocal phase.  This is a partition, not a union of
        # two probabilities and not a fitted blend.
        donation = phase * smoothed + (1.0 - phase) * cavity
        cavity_action = float(np.mean(cavity * cavity))
    else:
        raise ValueError(f"unknown return law: {law}")
    return donation, {
        "donation_action": float(np.mean(donation * donation)),
        "smoothed_residual_action": float(np.mean(smoothed * smoothed)),
        "cavity_action": cavity_action,
        "residual_phase_action_authority": float(
            phase_diagnostic["action_weighted_authority"]),
    }


def _orbit(
    observation: np.ndarray,
    initial: np.ndarray,
    law: str,
    cycles: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]]]:
    posterior = initial.copy()
    residual = observation - posterior
    observation_phase, _observation_phase_diagnostic = (
        _phase_action_authority(observation))
    trajectory = [posterior.copy()]
    ledger = []
    for cycle in range(cycles):
        posterior_phase, _ = _phase_action_authority(posterior)
        smooth_posterior, _ = _phase_screened_smooth(
            posterior, posterior, residual, posterior_phase)
        shed = posterior - smooth_posterior
        posterior = smooth_posterior
        residual = residual + shed

        donation, transfer = _return_candidate(
            law, posterior, residual)
        posterior = posterior + donation
        residual = residual - donation

        if law != "phase_action_no_joint":
            joint, _ = _phase_screened_smooth(
                observation, posterior, residual, observation_phase)
            if law in {
                "phase_action_phase_joint",
                "phase_action_intersection_joint",
                "phase_action_relational_joint",
            }:
                correction = joint - posterior
                residual_phase, _ = _phase_action_authority(residual)
                joint_authority = residual_phase
                if law in {
                    "phase_action_intersection_joint",
                    "phase_action_relational_joint",
                }:
                    metric = continual_transport_metric(
                        posterior, residual * residual)
                    laplacian, _markov, stencil = _continual_flux_laplacian(
                        metric, 1.0 - residual_phase)
                    _raw, relation = _cavity_residual_relation(
                        posterior,
                        correction,
                        laplacian,
                        float(stencil["maximum_degree"]),
                    )
                    explained = np.asarray(relation["explained_action"])
                    if law == "phase_action_intersection_joint":
                        # Independent necessary witnesses intersect through
                        # their parameter-free Hellinger product.
                        joint_authority = np.sqrt(
                            residual_phase * explained)
                    else:
                        joint_authority = (
                            residual_phase
                            + (1.0 - residual_phase) * explained
                        )
                posterior = posterior + joint_authority * correction
                residual = observation - posterior
            else:
                posterior = joint
                residual = observation - posterior
        trajectory.append(posterior.copy())
        ledger.append({
            "cycle": int(cycle + 1),
            "shed_action": float(np.mean(shed * shed)),
            **transfer,
            "conservation_error": float(np.max(np.abs(
                posterior + residual - observation))),
        })
    return posterior, trajectory, ledger


def run(size: int, selected: tuple[str, ...], cycles: int) -> dict[str, Any]:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        cases = (
            ("clean", truth),
            (
                "mixed replacement + uniform 0.25",
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=9100,
                ),
            ),
        )
        from .causal_scale_transport_2d import (
            causal_scale_transport_observation_2d,
        )
        for condition, observation in cases:
            _base, _residual, diagnostic = (
                causal_scale_transport_observation_2d(observation))
            initial = np.asarray(
                diagnostic["readouts"]["phase_susceptibility"])
            law_rows = {}
            for law in _LAWS:
                _terminal, trajectory, ledger = _orbit(
                    observation, initial, law, cycles)
                measured = [metrics(member, truth) for member in trajectory]
                law_rows[law] = {
                    "trajectory": [
                        {"cycle": int(index), **row}
                        for index, row in enumerate(measured)
                    ],
                    "oracle_best_mse_cycle": int(np.argmin([
                        row["mse"] for row in measured])),
                    "ledger": ledger,
                }
            rows.append({
                "source": source,
                "condition": condition,
                "laws": law_rows,
            })
    return {
        "purpose": (
            "falsify residual amplitude-return laws while preserving the "
            "same posterior shedding and joint closure"
        ),
        "size": int(size),
        "cycles": int(cycles),
        "laws": list(_LAWS),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.cycles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f"{row['source']} | {row['condition']}")
        for law, record in row["laws"].items():
            print(law, [
                (round(item["mse"], 6), round(item["edge_retention"], 4))
                for item in record["trajectory"]
            ])


if __name__ == "__main__":
    main()
