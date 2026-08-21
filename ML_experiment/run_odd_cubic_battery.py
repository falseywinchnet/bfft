#!/usr/bin/env python3
"""Run the shallow real odd-cubic baseline on the extant 23-task battery."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.nd_spiral_wall import ShallowOddCubicNet
from ML_experiment.run_benchmark import auc, threshold
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.tasks import TASK_BUILDERS


VARIANT = "shallow_odd_cubic"
torch.set_num_threads(8)


def train(task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(10_000 + seed)
    model = ShallowOddCubicNet(task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190_000 + seed)
    history, best = [], None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = (F.cross_entropy(output, y) if task.kind == "classification"
                else F.mse_loss(output, y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            history.append({"step": step, "loss": float(loss.detach()), **metrics})
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    return model, history, seconds, best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/odd_cubic_battery"))
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
    run_partial, probe_partial = args.out / "runs.partial.json", args.out / "probes.partial.json"
    runs = json.loads(run_partial.read_text())["runs"] if args.resume and run_partial.exists() else []
    probes = json.loads(probe_partial.read_text())["probes"] if args.resume and probe_partial.exists() else []
    done = {(row["task"], row["seed"]) for row in runs}
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            if (task_name, seed) in done:
                continue
            task = TASK_BUILDERS[task_name](seed)
            model, history, seconds, best_step = train(
                task, args.width, seed, args.steps, args.batch, args.lr, args.eval_every)
            test = evaluate(model, task)
            tails = tail_metrics(model, task)
            variability, rank = jacobian_variability(model, task.x_val)
            row = {
                "task": task_name, "kind": task.kind, "input_dim": task.input_dim,
                "output_dim": task.output_dim, "width": args.width, "seed": seed,
                "variant": VARIANT, "optimizer": "adamw",
                "parameters": parameter_count(model), "fixed_scalars": 0,
                "seconds": seconds, "best_step": best_step,
                "learning_auc": auc(history, args.steps),
                "steps_to_80": threshold(history, .8), "steps_to_90": threshold(history, .9),
                "validation_score": max(point["score"] for point in history),
                **test, **tails, "jacobian_variability": variability,
                "jacobian_change_rank": rank, "history": history,
            }
            runs.append(row)
            run_partial.write_text(json.dumps({"runs": runs}, indent=2))
            if seed == 0:
                probes.append({"task": task_name, "variant": VARIANT, "seconds": seconds,
                               "best_step": best_step, **make_probe(task, model, args.grid)})
                probe_partial.write_text(json.dumps({"probes": probes}))
            print(json.dumps({"task": task_name, "seed": seed, "parameters": row["parameters"],
                              "score": row["score"], "tail_score": row.get("tail_score"),
                              "learning_auc": row["learning_auc"], "seconds": seconds}), flush=True)
    configuration = {**vars(args), "out": str(args.out), "variant": VARIANT}
    (args.out / "results.json").write_text(json.dumps({"configuration": configuration, "runs": runs}, indent=2))
    (args.out / "probes.json").write_text(json.dumps({"configuration": configuration, "probes": probes}))
    print(json.dumps({"complete": True, "runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()

