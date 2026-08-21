"""Test descent and equilibrium of the shared-label horizontal fixed point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_predictive_geometry import causal_predictive_fixed_point
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def run(size: int, quantiles: int, continuations: int) -> dict:
    rows = []
    for source, truth in sources(size).items():
        observations = {
            "clean": truth,
            "mixed 0.25": corrupt(
                truth,
                "mixed replacement + uniform",
                amount=0.10,
                density=0.25,
                seed=16000,
            ),
        }
        for condition, observation in observations.items():
            _particles, diagnostic = causal_predictive_fixed_point(
                observation,
                quantile_count=quantiles,
                maximum_continuations=continuations,
            )
            action = [
                record["self_consistency_action"]
                for record in diagnostic["continuations"]
            ]
            rows.append({
                "source": source,
                "condition": condition,
                "equilibrium": diagnostic["equilibrium"],
                "noncontractive": diagnostic["noncontractive"],
                "continuation_ceiling_hit": diagnostic[
                    "continuation_ceiling_hit"],
                "initial_action": action[0],
                "final_action": action[-1],
                "relative_action": action[-1] / max(action[0], 1e-30),
                "continuations": diagnostic["continuations"],
            })
    return {
        "purpose": (
            "test whether determinant-normalized line-search descent reaches "
            "a causal horizontal predictive fixed point"
        ),
        "size": int(size),
        "quantile_count": int(quantiles),
        "maximum_continuations": int(continuations),
        "summary": {
            "equilibria": int(sum(row["equilibrium"] for row in rows)),
            "noncontractive_stops": int(sum(
                row["noncontractive"] for row in rows)),
            "ceiling_hits": int(sum(
                row["continuation_ceiling_hit"] for row in rows)),
            "mean_relative_action": float(np.mean([
                row["relative_action"] for row in rows])),
            "maximum_final_action": float(max(
                row["final_action"] for row in rows)),
        },
        "rows": rows,
        "verdict_rule": (
            "the fixed point is not promoted unless every represented scene "
            "reaches numerical equilibrium with monotonically decreasing "
            "self-consistency action"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.quantiles, args.continuations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
