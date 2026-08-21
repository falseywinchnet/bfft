"""Measure the full conservative exchange trajectory without oracle stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .conservative_exchange_transport_2d import (
    denoise_conservative_exchange_transport_2d,
)
from .run_2d_denoiser_battery import CONDITIONS, metrics, sources
from .sample_series import corrupt


def _cases(
    truth: np.ndarray,
    all_corruptions: bool,
) -> tuple[tuple[str, np.ndarray], ...]:
    if not all_corruptions:
        return (
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
    cases = [("clean", truth)]
    for condition, kind, amount, density in CONDITIONS:
        cases.append((
            condition,
            corrupt(
                truth,
                kind,
                amount=amount,
                density=density,
                seed=9100,
            ),
        ))
    return tuple(cases)


def run(
    size: int,
    selected: tuple[str, ...],
    numerical_cycle_ceiling: int,
    all_corruptions: bool = False,
) -> dict[str, Any]:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        for condition, observation in _cases(truth, all_corruptions):
            _estimate, diagnostic = denoise_conservative_exchange_transport_2d(
                observation,
                numerical_cycle_ceiling=numerical_cycle_ceiling,
            )
            trajectory = np.asarray(diagnostic["posterior_trajectory"])
            measured = [metrics(member, truth) for member in trajectory]
            mse = np.asarray([row["mse"] for row in measured])
            edge = np.asarray([row["edge_retention"] for row in measured])
            rows.append({
                "source": source,
                "condition": condition,
                "trajectory": [
                    {"cycle": int(index), **row}
                    for index, row in enumerate(measured)
                ],
                # These are retrospective audit coordinates only.  The
                # algorithm never sees truth and does not select either one.
                "oracle_best_mse_cycle": int(np.argmin(mse)),
                "oracle_best_edge_cycle": int(np.argmax(edge)),
                "initial": measured[0],
                "terminal": measured[-1],
                "completed_cycles": diagnostic["completed_cycles"],
                "equilibrium": diagnostic["equilibrium"],
                "maximum_conservation_error": diagnostic[
                    "maximum_conservation_error"],
                "cycle_actions": [{
                    key: value
                    for key, value in cycle.items()
                    if key in (
                        "cycle",
                        "posterior_displacement_action",
                        "posterior_shed_action",
                        "residual_donation_action",
                        "residual_refusal_action",
                        "joint_reassignment_action",
                    )
                } for cycle in diagnostic["cycles"]],
                "residual_phase_action_authority": [
                    float(cycle["residual_phase"][
                        "action_weighted_authority"])
                    for cycle in diagnostic["cycles"]
                ],
            })

    clean = [row for row in rows if row["condition"] == "clean"]
    mixed = [row for row in rows if row["condition"] != "clean"]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "mean_initial_mse": float(np.mean([
                row["initial"]["mse"] for row in group])),
            "mean_terminal_mse": float(np.mean([
                row["terminal"]["mse"] for row in group])),
            "mean_initial_edge_retention": float(np.mean([
                row["initial"]["edge_retention"] for row in group])),
            "mean_terminal_edge_retention": float(np.mean([
                row["terminal"]["edge_retention"] for row in group])),
            "oracle_best_mse_cycles": [
                row["oracle_best_mse_cycle"] for row in group],
            "equilibrium_count": int(sum(
                bool(row["equilibrium"]) for row in group)),
            "mse_improvement_count": int(sum(
                row["terminal"]["mse"] < row["initial"]["mse"]
                for row in group
            )),
            "edge_improvement_count": int(sum(
                row["terminal"]["edge_retention"]
                > row["initial"]["edge_retention"]
                for row in group
            )),
        }

    return {
        "purpose": (
            "trace posterior shedding, residual donation, and joint closure "
            "without truth-based stopping"
        ),
        "size": int(size),
        "sources": list(selected),
        "numerical_cycle_ceiling": int(numerical_cycle_ceiling),
        "all_corruptions": bool(all_corruptions),
        "clean_summary": summarize(clean),
        "corruption_summary": summarize(mixed),
        "mixed_summary": summarize(mixed) if not all_corruptions else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--numerical-cycle-ceiling", type=int, default=6)
    parser.add_argument("--all-corruptions", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.numerical_cycle_ceiling,
        args.all_corruptions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean_summary": result["clean_summary"],
        "corruption_summary": result["corruption_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
