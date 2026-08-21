"""Measure cavity residual erosion across scenes and unknown corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .causal_scale_transport_2d import causal_scale_transport_observation_2d
from .residual_erosion_transport_2d import (
    denoise_cavity_residual_erosion_2d,
)
from .run_2d_denoiser_battery import CONDITIONS, metrics, sources
from .sample_series import corrupt


def _differences(
    candidate: dict[str, float],
    base: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - base[key])
        for key in candidate
    }


def run(size: int, selected: tuple[str, ...], seeds: int) -> dict[str, Any]:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        cases = [("clean", truth)]
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                cases.append((
                    f"{condition} seed {seed}",
                    corrupt(
                        truth,
                        kind,
                        amount=amount,
                        density=density,
                        seed=9100 + seed,
                    ),
                ))
        for condition, observation in cases:
            _provisional, _residual, scale = (
                causal_scale_transport_observation_2d(observation))
            base = np.asarray(
                scale["readouts"]["phase_susceptibility"])
            estimate, diagnostic = denoise_cavity_residual_erosion_2d(
                observation,
                initial_state=base,
            )
            base_metrics = metrics(base, truth)
            candidate_metrics = metrics(estimate, truth)
            rows.append({
                "source": source,
                "condition": condition,
                "base": base_metrics,
                "cavity_residual_erosion": candidate_metrics,
                "difference_candidate_minus_base": _differences(
                    candidate_metrics, base_metrics),
                "accepted_continuations": diagnostic[
                    "accepted_continuations"],
                "continuation_guard_hit": diagnostic[
                    "continuation_guard_hit"],
                "initial_residual_action": diagnostic[
                    "initial_residual_action"],
                "final_residual_action": diagnostic[
                    "final_residual_action"],
                "terminal_status": diagnostic["status"],
            })
    noisy = [row for row in rows if row["condition"] != "clean"]
    clean = [row for row in rows if row["condition"] == "clean"]

    def summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "mean_base": {
                key: float(np.mean([row["base"][key] for row in group]))
                for key in group[0]["base"]
            },
            "mean_cavity_residual_erosion": {
                key: float(np.mean([
                    row["cavity_residual_erosion"][key] for row in group
                ]))
                for key in group[0]["base"]
            },
            "mse_improvement_count": int(sum(
                row["difference_candidate_minus_base"]["mse"] < 0.0
                for row in group
            )),
            "edge_improvement_count": int(sum(
                row["difference_candidate_minus_base"]["edge_retention"] > 0.0
                for row in group
            )),
            "mean_accepted_continuations": float(np.mean([
                row["accepted_continuations"] for row in group
            ])),
            "guard_hit_count": int(sum(
                row["continuation_guard_hit"] for row in group
            )),
        }

    return {
        "purpose": (
            "test target-excluded residual/curvature admission, screened "
            "erosion, and intrinsic action equilibrium"
        ),
        "size": int(size),
        "sources": list(selected),
        "seeds": int(seeds),
        "clean_summary": summary(clean),
        "unknown_corruption_summary": summary(noisy),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean_summary": result["clean_summary"],
        "unknown_corruption_summary": result[
            "unknown_corruption_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
