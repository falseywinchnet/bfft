#!/usr/bin/env python3
"""Separate optimization, capacity, and joint limits on the 8-turn spiral."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML_experiment.models import parameter_count
from ML_experiment.run_benchmark import auc, train_variant
from ML_experiment.run_spiral_evidence_horizon import (
    VARIANTS,
    accuracy,
    field_probe,
    make_task,
)


CONFIGURATIONS = (
    (38, 1000),
    (38, 2000),
    (54, 500),
    (76, 500),
    (54, 1000),
    (76, 2000),
)
PROBE_CONFIGURATIONS = {(38, 2000), (76, 500), (76, 2000)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/spiral_capacity_sweep"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_partial = args.out / "runs.partial.json"
    probe_partial = args.out / "probes.partial.json"
    runs = (json.loads(run_partial.read_text())["runs"]
            if args.resume and run_partial.exists() else [])
    probes = (json.loads(probe_partial.read_text())["probes"]
              if args.resume and probe_partial.exists() else [])
    done = {(row["width"], row["steps"], row["seed"], row["variant"]) for row in runs}

    for width, steps in CONFIGURATIONS:
        for seed in range(args.seeds):
            task = make_task(8, seed)
            for variant in VARIANTS:
                key = (width, steps, seed, variant)
                if key in done:
                    continue
                model, history, seconds, best_step = train_variant(
                    variant,
                    task,
                    width,
                    seed,
                    steps,
                    args.batch,
                    args.lr,
                    args.eval_every,
                    optimizer_name="adamw",
                )
                tail_scores = [
                    accuracy(model, x_tail, y_tail)
                    for x_tail, y_tail in zip(task.tail_x, task.tail_y)
                ]
                row = {
                    "visible_turns": 8,
                    "withheld_turns": 8,
                    "width": width,
                    "steps": steps,
                    "seed": seed,
                    "variant": variant,
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    "best_step": best_step,
                    "validation_score": accuracy(model, task.x_val, task.y_val),
                    "test_score": accuracy(model, task.x_test, task.y_test),
                    "final_turn_score": tail_scores[-1],
                    "tail_scores": tail_scores,
                    "learning_auc": auc(history, steps),
                    "history": history,
                }
                runs.append(row)
                run_partial.write_text(json.dumps({"runs": runs}, indent=2))
                if (width, steps) in PROBE_CONFIGURATIONS:
                    probes.append({
                        "width": width,
                        "steps": steps,
                        "seed": seed,
                        "variant": variant,
                        **field_probe(model, 8, args.grid),
                    })
                    probe_partial.write_text(json.dumps({"probes": probes}))
                print(json.dumps({
                    "width": width,
                    "steps": steps,
                    "seed": seed,
                    "variant": variant,
                    "parameters": row["parameters"],
                    "validation": row["validation_score"],
                    "test": row["test_score"],
                    "auc": row["learning_auc"],
                    "seconds": seconds,
                }), flush=True)

    configuration = {**vars(args), "out": str(args.out), "variants": VARIANTS,
                     "configurations": CONFIGURATIONS}
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": runs,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({"complete": True, "runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()
