#!/usr/bin/env python3
"""Search deterministic primal escape modes exposed by the BFFT contact stress."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chip_transport import (
    legalize_capacity,
    legalize_pose_capacity,
    polish_pose_capacity_slsqp,
    solve_soft_transport,
    write_svg,
)
from geometry import SQUARE_COUNT, capacity_state, wrap_square_phase
from reference_chart import REFERENCE_SIDE, reference_chart
from rigidity_analysis import coherent_contact_linearization
from topology_search import contact_signature


@dataclass(frozen=True)
class StressCutMode:
    active_index: int
    label: dict[str, int | str]
    direction: np.ndarray
    direction_norm: float
    dropped_response: float


@dataclass(frozen=True)
class StressCutRequest:
    mode: StressCutMode
    target_side: float
    scale: float
    iterations: int
    seed: int


def stress_cut_modes() -> list[StressCutMode]:
    """Rank all one-contact cuts that destroy the coherent shrink stress."""

    from scipy.optimize import LinearConstraint, linprog, minimize

    chart = coherent_contact_linearization()
    operator = chart.operator
    side = chart.side_response
    modes: list[StressCutMode] = []
    for dropped in range(len(operator)):
        retained = np.ones(len(operator), dtype=bool)
        retained[dropped] = False
        active = operator[retained]
        response = side[retained]
        feasible = linprog(
            np.zeros(operator.shape[1]),
            A_ub=-active,
            b_ub=-response,
            bounds=[(-1.0e4, 1.0e4)] * operator.shape[1],
            method="highs",
        )
        if not feasible.success:
            continue
        minimum_norm = minimize(
            lambda value: (0.5 * float(np.dot(value, value)), value),
            feasible.x,
            jac=True,
            method="SLSQP",
            constraints=LinearConstraint(active, response, np.inf),
            options={"ftol": 1.0e-12, "maxiter": 4000, "disp": False},
        )
        direction = np.asarray(minimum_norm.x)
        if np.min(active @ direction - response) < -2.0e-9:
            continue
        modes.append(
            StressCutMode(
                active_index=dropped,
                label=chart.labels[dropped],
                direction=direction,
                direction_norm=float(np.linalg.norm(direction)),
                dropped_response=float(
                    operator[dropped] @ direction - side[dropped]
                ),
            )
        )
    modes.sort(key=lambda mode: mode.direction_norm)
    return modes


def lift_coherent_direction(direction: np.ndarray) -> np.ndarray:
    """Lift the 36-coordinate physical chart back to 17 full square poses."""

    lifted = np.zeros((SQUARE_COUNT, 3), dtype=np.float64)
    for square in range(SQUARE_COUNT):
        lifted[square, :2] = direction[2 * square:2 * square + 2]
    lifted[8:14, 2] = direction[34]
    lifted[15, 2] = direction[35]
    return lifted


def run_cut(request: StressCutRequest) -> dict[str, object]:
    shrink = REFERENCE_SIDE - request.target_side
    initial = reference_chart() + request.scale * shrink * lift_coherent_direction(
        request.mode.direction
    )
    initial[:, 2] = wrap_square_phase(initial[:, 2])
    initial_state = capacity_state(initial, request.target_side)
    settled = legalize_pose_capacity(initial, request.target_side)
    polished, polish_audit = polish_pose_capacity_slsqp(
        settled, request.target_side
    )
    transport_trace: dict[str, object] | None = None
    if not polish_audit["success"] and request.iterations > 0:
        soft = solve_soft_transport(
            initial,
            request.target_side,
            iterations=request.iterations,
            seed=request.seed,
        )
        transported = legalize_pose_capacity(soft.poses, request.target_side)
        polished, polish_audit = polish_pose_capacity_slsqp(
            transported, request.target_side
        )
        transport_trace = {
            "soft_minimum_clearance": soft.minimum_clearance,
            "continuation_steps": soft.continuation_steps,
            "net_count": soft.audit.net_count,
            "paired_axis_count": soft.audit.paired_axis_count,
        }
    final = legalize_capacity(polished, request.target_side, iterations=480)
    state = capacity_state(final, request.target_side)
    return {
        "active_index": request.mode.active_index,
        "cut": request.mode.label,
        "direction_norm": request.mode.direction_norm,
        "dropped_linear_response": request.mode.dropped_response,
        "scale": request.scale,
        "target_side": request.target_side,
        "initial_minimum_clearance": initial_state.minimum_clearance,
        "initial_overlap_residual": initial_state.overlap_residual,
        "minimum_clearance": state.minimum_clearance,
        "overlap_residual": state.overlap_residual,
        "feasible": state.minimum_clearance >= -1.0e-8,
        "polish_audit": polish_audit,
        "transport_trace": transport_trace,
        "contact_signature": contact_signature(final, request.target_side),
        "poses": final.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-side", type=float, default=4.67)
    parser.add_argument("--cuts", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    modes = stress_cut_modes()[: args.cuts]
    requests = [
        StressCutRequest(mode, args.target_side, args.scale, args.iterations, args.seed + index)
        for index, mode in enumerate(modes)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        trials = list(executor.map(run_cut, requests))
    trials.sort(
        key=lambda trial: (
            float(trial["overlap_residual"]),
            -float(trial["minimum_clearance"]),
        )
    )
    result = {
        "status": "floating_point_stress_cut_search_not_global_proof",
        "method": "bfft_dual_stress_cut_primal_transport",
        "target_side": args.target_side,
        "cut_count": len(trials),
        "feasible_count": sum(bool(trial["feasible"]) for trial in trials),
        "best": trials[0],
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(
            args.svg,
            np.asarray(result["best"]["poses"]),
            float(result["target_side"]),
        )
    print(
        json.dumps(
            {
                "target_side": result["target_side"],
                "feasible_count": result["feasible_count"],
                "best_cut": result["best"]["cut"],
                "best_minimum_clearance": result["best"]["minimum_clearance"],
                "best_overlap_residual": result["best"]["overlap_residual"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
