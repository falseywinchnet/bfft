"""Focused operator witness for transported, non-elementwise AdamW."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ML_experiment.muon_primacy import make_operator_problem
from ML_experiment.run_optimizer_operator_battery import _float_problem, run


RATES = {
    "adamw": .003,
    "adamw_transport": .003,
    "anchor": 1.0,
    "lepton_transport_guarded": .003,
    "muon": .003,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("/tmp/transported_adam_operator_screen.json"),
    )
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=731)
    args = parser.parse_args()
    rows = []
    for seed_index in range(args.seeds):
        problem = _float_problem(make_operator_problem(
            args.dimension, args.condition, args.seed_base + seed_index
        ))
        covariances = [problem.covariance] * args.steps
        for optimizer, learning_rate in RATES.items():
            row = run(problem, covariances, optimizer, learning_rate)
            row.update({
                "seed": seed_index,
                "problem_seed": args.seed_base + seed_index,
                "scenario": "population",
            })
            rows.append(row)
            print(json.dumps({
                key: value for key, value in row.items() if key != "history"
            }), flush=True)
    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "rates": RATES,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
