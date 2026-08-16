#!/usr/bin/env python3
"""Rapid scratch-versus-staged nested-chart acquisition check."""
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

import numpy as np
import torch

from ML_experiment.metrics import evaluate, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.run_benchmark import task_loss
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import make_variant


torch.set_num_threads(8)

CONFIGURATIONS = {
    "self_context": "self_context",
    "curvature_context": "self_context_jet_curvature_context",
    "nested_chart_scratch": "self_context_nested_chart",
    "nested_chart_staged": "self_context_nested_chart",
}


def set_nested_chart(model, enabled: bool):
    mode = "nested_chart" if enabled else "none"
    model.up.jet_mode = mode
    model.down.jet_mode = mode


def area_under_history(history, steps):
    return float(np.trapz(
        [row["score"] for row in history],
        [row["step"] for row in history],
    ) / steps)


def train(configuration, task, width, seed, steps, switch_step, batch, lr, eval_every):
    torch.manual_seed(10000 + seed)
    model = make_variant(CONFIGURATIONS[configuration], task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190000 + seed)
    history, best = [], None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        if configuration == "nested_chart_staged":
            set_nested_chart(model, step > switch_step)
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model(x), y, task.kind)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            history.append({"step": step, "loss": float(loss.detach()), **metrics})
            eligible = configuration != "nested_chart_staged" or step > switch_step
            if eligible and (best is None or metrics["score"] > best[0]):
                best = (metrics["score"], copy.deepcopy(model.state_dict()), step)

    if configuration == "nested_chart_staged":
        set_nested_chart(model, True)
    model.load_state_dict(best[1])
    test = evaluate(model, task)
    tails = tail_metrics(model, task)
    tails.setdefault("tail_score", test["score"])
    return {
        "configuration": configuration,
        "parameters": parameter_count(model),
        "seconds": time.perf_counter() - started,
        "best_step": best[2],
        "validation_score": best[0],
        "learning_auc": area_under_history(history, steps),
        **test,
        **tails,
        "history": history,
    }


def summarize(runs):
    rows = []
    tasks = list(dict.fromkeys(row["task"] for row in runs))
    for task in tasks:
        baseline = [row for row in runs if row["task"] == task and row["configuration"] == "self_context"]
        baseline_mean = {
            key: sum(row[key] for row in baseline) / len(baseline)
            for key in ("validation_score", "score", "tail_score", "learning_auc", "seconds")
        }
        for configuration in CONFIGURATIONS:
            selected = [row for row in runs if row["task"] == task and row["configuration"] == configuration]
            mean = {
                key: sum(row[key] for row in selected) / len(selected)
                for key in ("validation_score", "score", "tail_score", "learning_auc", "seconds")
            }
            rows.append({
                "task": task,
                "configuration": configuration,
                **mean,
                **({} if configuration == "self_context" else {
                    f"{key}_delta": mean[key] - baseline_mean[key]
                    for key in ("validation_score", "score", "tail_score", "learning_auc")
                }),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/nested_chart_check.json"))
    parser.add_argument("--tasks", default="radial_stripes,multiscale_1d")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--switch-step", type=int, default=150)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    runs = []
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for configuration in CONFIGURATIONS:
                row = train(
                    configuration, task, args.width, seed, args.steps,
                    args.switch_step, args.batch, args.lr, args.eval_every,
                )
                row.update({"task": task_name, "seed": seed})
                runs.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in {"history", "tail_bins"}}), flush=True)

    payload = {"configuration": dict(vars(args)), "summary": summarize(runs), "runs": runs}
    payload["configuration"]["out"] = str(args.out)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "fits": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
