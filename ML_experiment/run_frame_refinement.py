#!/usr/bin/env python3
"""Focused continuous-frame refinement study.

Unlike the broad benchmark, this script compares named configurations.  Each
configuration changes one axis from the reference: optimizer, capacity, or
shell sampling cost.  The ordinary MLP and first-order self-context are kept
as interpretive anchors, not mixed into a parameter-count ranking.
"""
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
from ML_experiment.run_benchmark import auc, train_variant
from ML_experiment.run_probes import curve_probe, field_probe, scatter_probe
from ML_experiment.tasks import TASK_BUILDERS


CONFIGURATIONS = (
    {"name": "ordinary_mlp", "variant": "ordinary_mlp", "width": 24, "optimizer": "adamw"},
    {"name": "self_context", "variant": "self_context", "width": 24, "optimizer": "adamw"},
    {"name": "frame_reference", "variant": "self_context_stiefel_flow_curvature", "width": 24, "optimizer": "adamw"},
    {"name": "frame_muon", "variant": "self_context_stiefel_flow_curvature", "width": 24, "optimizer": "muon"},
    {"name": "frame_capacity", "variant": "self_context_stiefel_flow_curvature", "width": 32, "optimizer": "adamw"},
    {"name": "frame_fast", "variant": "self_context_stiefel_flow_curvature_hutch2", "width": 24, "optimizer": "adamw"},
)


def make_probe(task, model, grid):
    if task.input_dim == 1:
        return curve_probe(task, model)
    if task.input_dim == 2 and task.truth is not None:
        return field_probe(task, model, grid)
    return scatter_probe(task, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/frame_refinement"))
    parser.add_argument(
        "--tasks",
        default="radial_stripes,nd_spiral_high_rank,multiscale_1d,chirp_1d,"
                "poly_drifted_chirp_1d,localized_steps_1d,complex_spiral_3d",
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--grid", type=int, default=71)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.json"
    probe_partial = args.out / "probes.partial.json"
    runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    probes = (json.loads(probe_partial.read_text())["probes"]
              if args.resume and probe_partial.exists() else [])
    done = {(row["task"], row["seed"], row["configuration"]) for row in runs}

    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for configuration in CONFIGURATIONS:
                key = (task_name, seed, configuration["name"])
                if key in done:
                    continue
                model, history, seconds, best_step = train_variant(
                    configuration["variant"], task, configuration["width"], seed,
                    args.steps, args.batch, args.lr, args.eval_every,
                    optimizer_name=configuration["optimizer"],
                )
                test = evaluate(model, task)
                tails = tail_metrics(model, task)
                variability, rank = jacobian_variability(model, task.x_val)
                row = {
                    "task": task_name,
                    "kind": task.kind,
                    "input_dim": task.input_dim,
                    "output_dim": task.output_dim,
                    "seed": seed,
                    "configuration": configuration["name"],
                    **configuration,
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    "best_step": best_step,
                    "learning_auc": auc(history, args.steps),
                    "validation_score": max(point["score"] for point in history),
                    **test,
                    **tails,
                    "jacobian_variability": variability,
                    "jacobian_change_rank": rank,
                    "history": history,
                }
                runs.append(row)
                partial.write_text(json.dumps({"runs": runs}, indent=2))
                if seed == 0:
                    probes.append({
                        "task": task_name,
                        "configuration": configuration["name"],
                        **make_probe(task, model, args.grid),
                    })
                    probe_partial.write_text(json.dumps({"probes": probes}))
                print(json.dumps({
                    "task": task_name,
                    "seed": seed,
                    "configuration": configuration["name"],
                    "score": row["score"],
                    "tail_score": row.get("tail_score"),
                    "learning_auc": row["learning_auc"],
                    "seconds": seconds,
                }), flush=True)

    config = {**vars(args), "out": str(args.out), "configurations": CONFIGURATIONS}
    (args.out / "results.json").write_text(json.dumps({"configuration": config, "runs": runs}, indent=2))
    (args.out / "probes.json").write_text(json.dumps({"configuration": config, "probes": probes}))
    print(json.dumps({"complete": True, "runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()
