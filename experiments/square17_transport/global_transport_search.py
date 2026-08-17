#!/usr/bin/env python3
"""Continue independently learned relaxed square charts toward their frontier."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chip_transport import (
    initial_population,
    legalize_capacity,
    legalize_pose_capacity,
    polish_pose_capacity_slsqp,
    solve_soft_transport,
    transport_chart,
    write_svg,
)
from geometry import capacity_state
from topology_search import contact_signature


@dataclass(frozen=True)
class BasinRequest:
    seed: int
    family: int
    stop_side: float
    initial_step: float
    minimum_step: float
    iterations: int


ROW_FAMILIES = (
    (4, 4, 5, 4),
    (5, 4, 4, 4),
    (4, 5, 4, 4),
    (4, 4, 4, 5),
    (3, 4, 3, 4, 3),
    (4, 3, 4, 3, 3),
    (3, 3, 4, 3, 4),
    (5, 3, 3, 3, 3),
)


def diverse_population(seed: int, family: int, side: float = 5.0) -> np.ndarray:
    """A relaxed ownership chart from one sparse four/five-row family."""

    if family < 0 or family >= len(ROW_FAMILIES):
        raise ValueError("unknown relaxed row family")
    rng = np.random.default_rng(seed)
    counts = ROW_FAMILIES[family]
    y_values = np.linspace(0.53, side - 0.53, len(counts))
    rows = []
    for row, (count, y_value) in enumerate(zip(counts, y_values)):
        x_values = np.linspace(0.53, side - 0.53, count)
        if count < 5:
            offset = rng.uniform(-0.16, 0.16)
            x_values = np.clip(x_values + offset, 0.51, side - 0.51)
        rows.extend((value, y_value) for value in x_values)
    centers = np.asarray(rows, dtype=np.float64)
    centers += rng.normal(0.0, 0.025, centers.shape)
    theta = rng.uniform(-0.18, 0.18, len(centers))
    return np.column_stack((centers, theta))


def emit_exact_preimage(poses: np.ndarray, side: float) -> tuple[np.ndarray, dict[str, object]]:
    settled = legalize_pose_capacity(poses, side)
    polished, audit = polish_pose_capacity_slsqp(settled, side, iterations=2400)
    final = legalize_capacity(polished, side, iterations=480)
    state = capacity_state(final, side)
    audit = dict(audit)
    audit["minimum_clearance_after_fixed_phase"] = state.minimum_clearance
    audit["overlap_residual_after_fixed_phase"] = state.overlap_residual
    audit["success"] = bool(audit["success"] and state.minimum_clearance >= -1.0e-8)
    return final, audit


def continue_basin(request: BasinRequest) -> dict[str, object]:
    side = 5.0
    soft = solve_soft_transport(
        diverse_population(request.seed, request.family, side),
        side,
        iterations=max(request.iterations, 8000),
        seed=request.seed,
    )
    source, initial_audit = emit_exact_preimage(soft.poses, side)
    if not initial_audit["success"]:
        return {
            "seed": request.seed,
            "family": request.family,
            "status": "initial_5x5_preimage_failed",
            "side": side,
            "audit": initial_audit,
            "poses": source.tolist(),
            "trace": [],
        }

    trace: list[dict[str, object]] = []
    step = request.initial_step
    attempt = 0
    while side > request.stop_side + 1.0e-12 and step >= request.minimum_step - 1.0e-15:
        trial_side = max(request.stop_side, side - step)
        initial = transport_chart(source, side, trial_side)
        transported = solve_soft_transport(
            initial,
            trial_side,
            iterations=request.iterations,
            seed=request.seed + 104729 * (attempt + 1),
        )
        final, audit = emit_exact_preimage(transported.poses, trial_side)
        state = capacity_state(final, trial_side)
        feasible = bool(audit["success"] and state.minimum_clearance >= -1.0e-8)
        trace.append(
            {
                "attempt": attempt + 1,
                "from_side": side,
                "trial_side": trial_side,
                "step": step,
                "soft_minimum_clearance": transported.minimum_clearance,
                "soft_continuation_steps": transported.continuation_steps,
                "minimum_clearance": state.minimum_clearance,
                "overlap_residual": state.overlap_residual,
                "feasible": feasible,
                "polish_status": audit["status"],
                "polish_iterations": audit["iterations"],
            }
        )
        attempt += 1
        if feasible:
            side = trial_side
            source = final
            if side <= request.stop_side + 1.0e-12:
                break
        else:
            step *= 0.5

    state = capacity_state(source, side)
    return {
        "seed": request.seed,
        "family": request.family,
        "status": "feasible_frontier",
        "side": side,
        "minimum_clearance": state.minimum_clearance,
        "overlap_residual": state.overlap_residual,
        "contact_signature": contact_signature(source, side),
        "poses": source.tolist(),
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--family-start", type=int, default=0)
    parser.add_argument("--basins", type=int, default=4)
    parser.add_argument("--stop-side", type=float, default=4.65)
    parser.add_argument("--initial-step", type=float, default=0.025)
    parser.add_argument("--minimum-step", type=float, default=1.0e-4)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    requests = [
        BasinRequest(
            args.seed_start + index,
            (args.family_start + index) % len(ROW_FAMILIES),
            args.stop_side,
            args.initial_step,
            args.minimum_step,
            args.iterations,
        )
        for index in range(args.basins)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        basins = list(executor.map(continue_basin, requests))
    basins.sort(key=lambda basin: float(basin["side"]))
    best = basins[0]
    payload = {
        "status": "floating_point_basin_frontiers_not_global_proof",
        "method": "bfft_support_preserving_adaptive_continuation",
        "basin_count": len(basins),
        "best": best,
        "basins": basins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(args.svg, np.asarray(best["poses"]), float(best["side"]))
    print(
        json.dumps(
            {
                "best_seed": best["seed"],
                "best_side": best["side"],
                "best_minimum_clearance": best.get("minimum_clearance"),
                "basin_sides": {str(basin["seed"]): basin["side"] for basin in basins},
                "families": {str(basin["seed"]): basin["family"] for basin in basins},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
