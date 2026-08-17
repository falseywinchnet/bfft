#!/usr/bin/env python3
"""Stress-guided contact-topology search for the 17-square transport chart."""

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
    quantile_emit,
    solve_soft_transport,
    transport_chart,
    write_svg,
)
from geometry import PAIR_I, PAIR_J, SQUARE_COUNT, capacity_state, wrap_square_phase
from reference_chart import REFERENCE_SIDE, reference_chart
from rigidity_analysis import coherent_contact_linearization, nonnegative_shrink_stress


@dataclass(frozen=True)
class TrialRequest:
    seed: int
    target_side: float
    displacement: float
    phase_displacement: float
    iterations: int
    passes: int
    polisher: str = "numpy"


def stressed_contact_laplacian(tolerance: float = 1.0e-9) -> np.ndarray:
    """Return the stress-weighted cell graph of the jammed reference chart."""

    linearization = coherent_contact_linearization(tolerance)
    stress = nonnegative_shrink_stress(linearization)
    adjacency = np.zeros((SQUARE_COUNT, SQUARE_COUNT), dtype=np.float64)
    for label, weight in zip(linearization.labels, stress):
        if label["kind"] != "pair":
            continue
        first = int(label["first"])
        second = int(label["second"])
        # Zero-stress incidences still carry topology, but only weakly.
        edge_weight = 0.025 + float(weight)
        adjacency[first, second] += edge_weight
        adjacency[second, first] += edge_weight
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def transported_fracture(request: TrialRequest) -> np.ndarray:
    """Open a low-frequency contact cut and break only its square-phase cusp."""

    rng = np.random.default_rng(request.seed)
    poses = transport_chart(reference_chart(), REFERENCE_SIDE, request.target_side)
    laplacian = stressed_contact_laplacian()
    eigenvalues, modes = np.linalg.eigh(laplacian)
    nontrivial = np.flatnonzero(eigenvalues > 1.0e-10)
    retained = nontrivial[: min(9, len(nontrivial))]

    # The inverse square-root spectrum is the discrete transport Green kernel:
    # it favors coherent packet motion but retains several possible graph cuts.
    coefficients = rng.normal(size=(len(retained), 2))
    coefficients /= np.sqrt(eigenvalues[retained, None])
    displacement = modes[:, retained] @ coefficients
    displacement -= np.mean(displacement, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
    if rms > 1.0e-15:
        displacement *= request.displacement / rms
    poses[:, :2] += displacement

    phase_coefficients = rng.normal(size=len(retained)) / np.sqrt(eigenvalues[retained])
    phase = modes[:, retained] @ phase_coefficients
    phase -= np.mean(phase)
    phase_rms = float(np.sqrt(np.mean(phase * phase)))
    if phase_rms > 1.0e-15:
        phase *= request.phase_displacement / phase_rms
    # Add a small independent odd channel.  It is what lets equal-angle
    # contacts choose opposite sides of the nonsmooth relative-phase cusp.
    phase += rng.normal(0.0, 0.18 * request.phase_displacement, SQUARE_COUNT)
    poses[:, 2] = wrap_square_phase(poses[:, 2] + phase)
    return poses


def contact_signature(poses: np.ndarray, side: float, tolerance: float = 0.012) -> dict[str, object]:
    state = capacity_state(poses, side)
    boundary = []
    for square in range(SQUARE_COUNT):
        for face in range(4):
            if state.boundary_clearance[square, face] <= tolerance:
                boundary.append([square, face])
    pairs = [
        [int(PAIR_I[index]), int(PAIR_J[index])]
        for index in np.flatnonzero(state.pair_clearance <= tolerance)
    ]
    return {"boundary": boundary, "pairs": pairs}


def run_trial(request: TrialRequest) -> dict[str, object]:
    initial = transported_fracture(request)
    current = initial
    pass_trace: list[dict[str, float | int]] = []
    for pass_index in range(request.passes):
        soft = solve_soft_transport(
            current,
            request.target_side,
            iterations=request.iterations,
            seed=request.seed + pass_index * 1_000_003,
        )
        emitted = quantile_emit(current, soft.poses)
        soft_state = capacity_state(soft.poses, request.target_side)
        emitted_state = capacity_state(emitted, request.target_side)
        # Residual-gated self-distillation: retain the transported chart only
        # when the hard CDF inverse loses capacity information.
        current = (
            soft.poses
            if soft_state.overlap_residual <= emitted_state.overlap_residual
            else emitted
        )
        pass_trace.append(
            {
                "pass": pass_index + 1,
                "soft_minimum_clearance": soft_state.minimum_clearance,
                "soft_overlap_residual": soft_state.overlap_residual,
                "hard_minimum_clearance": emitted_state.minimum_clearance,
                "hard_overlap_residual": emitted_state.overlap_residual,
                "net_count": soft.audit.net_count,
                "paired_axis_count": soft.audit.paired_axis_count,
            }
        )
        if soft_state.minimum_clearance >= -1.0e-8:
            break
    chart_settled = legalize_pose_capacity(current, request.target_side)
    polish_audit: dict[str, object] | None = None
    if request.polisher == "scipy":
        chart_settled, polish_audit = polish_pose_capacity_slsqp(
            chart_settled, request.target_side
        )
    final = legalize_capacity(chart_settled, request.target_side, iterations=480)
    final_state = capacity_state(final, request.target_side)
    return {
        "seed": request.seed,
        "target_side": request.target_side,
        "displacement": request.displacement,
        "phase_displacement": request.phase_displacement,
        "minimum_clearance": final_state.minimum_clearance,
        "overlap_residual": final_state.overlap_residual,
        "feasible": final_state.minimum_clearance >= -1.0e-8,
        "poses": final.tolist(),
        "contact_signature": contact_signature(final, request.target_side),
        "polisher": request.polisher,
        "polish_audit": polish_audit,
        "passes": pass_trace,
    }


def search(
    *,
    seed_start: int,
    trials: int,
    target_side: float,
    displacement: float,
    phase_displacement: float,
    iterations: int,
    passes: int,
    workers: int,
    polisher: str,
) -> dict[str, object]:
    requests = [
        TrialRequest(
            seed=seed_start + trial,
            target_side=target_side,
            displacement=displacement,
            phase_displacement=phase_displacement,
            iterations=iterations,
            passes=passes,
            polisher=polisher,
        )
        for trial in range(trials)
    ]
    if workers == 1:
        results = [run_trial(request) for request in requests]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_trial, requests))
    results.sort(key=lambda result: (float(result["overlap_residual"]), -float(result["minimum_clearance"])))
    return {
        "status": "floating_point_topology_search_not_global_proof",
        "method": "stress_weighted_graph_transport_with_interval_stalk_legalization",
        "target_side": target_side,
        "trial_count": trials,
        "feasible_count": sum(bool(result["feasible"]) for result in results),
        "best": results[0],
        "trials": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--target-side", type=float, default=4.67)
    parser.add_argument("--displacement", type=float, default=0.055)
    parser.add_argument("--phase-displacement", type=float, default=0.025)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--polisher", choices=("numpy", "scipy"), default="numpy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    result = search(
        seed_start=args.seed_start,
        trials=args.trials,
        target_side=args.target_side,
        displacement=args.displacement,
        phase_displacement=args.phase_displacement,
        iterations=args.iterations,
        passes=args.passes,
        workers=args.workers,
        polisher=args.polisher,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        best = result["best"]
        write_svg(args.svg, np.asarray(best["poses"]), float(best["target_side"]))
    summary = {
        "target_side": result["target_side"],
        "feasible_count": result["feasible_count"],
        "best_seed": result["best"]["seed"],
        "best_minimum_clearance": result["best"]["minimum_clearance"],
        "best_overlap_residual": result["best"]["overlap_residual"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
