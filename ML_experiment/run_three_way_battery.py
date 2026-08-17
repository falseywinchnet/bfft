#!/usr/bin/env python3
"""Run vanilla/self-context/continuous-flow once and retain viewer probes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.run_benchmark import auc, threshold, train_variant
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.tasks import TASK_BUILDERS


VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_stiefel_flow_curvature",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/continuous_frame_2x"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS))
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--grid", type=int, default=71)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_partial = args.out / "runs.partial.json"
    probe_partial = args.out / "probes.partial.json"
    runs = (json.loads(run_partial.read_text())["runs"]
            if args.resume and run_partial.exists() else [])
    probes = (json.loads(probe_partial.read_text())["probes"]
              if args.resume and probe_partial.exists() else [])
    done = {(row["task"], row["seed"], row["variant"]) for row in runs}

    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for variant in VARIANTS:
                if (task_name, seed, variant) in done:
                    continue
                model, history, seconds, best_step = train_variant(
                    variant, task, args.width, seed, args.steps, args.batch,
                    args.lr, args.eval_every, optimizer_name="adamw",
                )
                test = evaluate(model, task)
                tails = tail_metrics(model, task)
                variability, rank = jacobian_variability(model, task.x_val)
                row = {
                    "task": task_name,
                    "kind": task.kind,
                    "input_dim": task.input_dim,
                    "output_dim": task.output_dim,
                    "width": args.width,
                    "seed": seed,
                    "variant": variant,
                    "optimizer": "adamw",
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    "best_step": best_step,
                    "learning_auc": auc(history, args.steps),
                    "steps_to_80": threshold(history, .8),
                    "steps_to_90": threshold(history, .9),
                    "validation_score": max(point["score"] for point in history),
                    **test,
                    **tails,
                    "jacobian_variability": variability,
                    "jacobian_change_rank": rank,
                    "history": history,
                }
                runs.append(row)
                run_partial.write_text(json.dumps({"runs": runs}, indent=2))
                if seed == 0:
                    probes.append({
                        "task": task_name,
                        "variant": variant,
                        "seconds": seconds,
                        "best_step": best_step,
                        **make_probe(task, model, args.grid),
                    })
                    probe_partial.write_text(json.dumps({"probes": probes}))
                print(json.dumps({
                    "task": task_name,
                    "seed": seed,
                    "variant": variant,
                    "parameters": row["parameters"],
                    "score": row["score"],
                    "tail_score": row.get("tail_score"),
                    "learning_auc": row["learning_auc"],
                    "seconds": seconds,
                }), flush=True)

    configuration = {**vars(args), "out": str(args.out), "variants": VARIANTS}
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
