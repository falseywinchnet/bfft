#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metrics import classification_metrics, predict, shape_metrics, tail_metrics
from models import MODEL_BUILDERS, make_model, parameter_count
from render import decision_atlas, learning_curves, surface_atlas
from tasks import TASK_BUILDERS


torch.set_num_threads(8)


def train_one(model, task, seed, steps, batch, learning_rate, evaluation_interval):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(40000 + seed)
    history, best = [], None; started = time.perf_counter()
    for step in range(1, steps + 1):
        order = torch.randperm(len(task.x_train), generator=generator)
        if model.contextual:
            context_size = min(256, max(64, len(order) // 3))
            context = order[:context_size]; query = order[context_size:context_size + batch]
        else:
            query = order[:batch]; context = order[:min(256, len(order))]
        optimizer.zero_grad(set_to_none=True)
        logits = model(task.x_train[query], task.x_train[context], task.y_train[context])
        loss = F.cross_entropy(logits, task.y_train[query])
        auxiliary = model.auxiliary_loss(task.x_train[query])
        total = loss + .05 * auxiliary
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step % evaluation_interval == 0 or step == 1 or step == steps:
            validation = classification_metrics(predict(model, task.x_val, task.x_train, task.y_train), task.y_val)
            row = {"step": step, "loss": float(loss.detach()), "auxiliary_loss": float(auxiliary.detach()), **validation}
            history.append(row)
            score = validation["balanced_accuracy"] - .002 * float(loss.detach())
            if best is None or score > best[0]:
                best = (score, {key: value.detach().clone() for key, value in model.state_dict().items()}, step)
    elapsed = time.perf_counter() - started; model.load_state_dict(best[1])
    return history, elapsed, best[2]


def threshold_step(history, threshold):
    for row in history:
        if row["balanced_accuracy"] >= threshold: return row["step"]
    return None


def learning_auc(history, steps):
    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["balanced_accuracy"] for row in history], dtype=float)
    return float(np.trapz(ys, xs) / max(1, steps))


def flatten_row(row):
    return {key: value for key, value in row.items() if key not in {"tail_bins", "history"}}


def main():
    parser = argparse.ArgumentParser(description="LELU-only omni-inducement benchmark")
    parser.add_argument("--out", type=Path, default=Path("/tmp/omni_inducement_benchmark"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS))
    parser.add_argument("--models", default=",".join(MODEL_BUILDERS))
    parser.add_argument("--widths", default="16,36")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="continue from runs.partial.json and aggregate completed runs")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    task_names = args.tasks.split(","); model_names = args.models.split(","); widths = [int(value) for value in args.widths.split(",")]
    partial = args.out / "runs.partial.json"
    runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    completed = {(row["task"], row["seed"], row["width"], row["model"]) for row in runs}
    representatives, representative_histories = {}, {}; started = time.perf_counter()
    for task_name in task_names:
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for width in widths:
                for model_name in model_names:
                    if (task_name, seed, width, model_name) in completed: continue
                    torch.manual_seed(5000 + seed)
                    model = make_model(model_name, task.input_dim, width)
                    history, elapsed, best_step = train_one(model, task, seed, args.steps, args.batch, args.lr, args.eval_every)
                    validation = classification_metrics(predict(model, task.x_val, task.x_train, task.y_train), task.y_val)
                    tails = tail_metrics(model, task); shape = shape_metrics(model, task)
                    row = {
                        "task": task_name, "intrinsic_rank": task.intrinsic_rank, "ambient_dim": task.input_dim,
                        "model": model_name, "width": width, "seed": seed, "parameters": parameter_count(model),
                        "training_seconds": elapsed, "best_step": best_step,
                        "steps_to_85pct": threshold_step(history, .85), "steps_to_95pct": threshold_step(history, .95),
                        "learning_auc": learning_auc(history, args.steps), **validation, **tails, **shape, "history": history,
                    }
                    runs.append(row); print(json.dumps(flatten_row(row)), flush=True)
                    (args.out / "runs.partial.json").write_text(json.dumps({"runs": runs}, indent=2))
                    if seed == 0 and width == max(widths):
                        representatives.setdefault(task_name, {})[model_name] = model
                        representative_histories.setdefault(task_name, {})[model_name] = history
    summary = []
    group_keys = sorted({(row["task"], row["model"], row["width"]) for row in runs})
    for task_name, model_name, width in group_keys:
        selected = [row for row in runs if (row["task"], row["model"], row["width"]) == (task_name, model_name, width)]
        def mean(key): return float(np.mean([row[key] for row in selected]))
        def optional_mean(key):
            values = [row.get(key) for row in selected if row.get(key) is not None]
            return float(np.mean(values)) if values else None
        summary.append({
            "task": task_name, "model": model_name, "width": width, "parameters": selected[0]["parameters"],
            "validation_balanced_accuracy": mean("balanced_accuracy"), "validation_probability_mse": mean("probability_mse"),
            "learning_auc": mean("learning_auc"), "steps_to_85pct": optional_mean("steps_to_85pct"),
            "steps_to_95pct": optional_mean("steps_to_95pct"), "training_seconds": mean("training_seconds"),
            "tail_accuracy": mean("tail_accuracy"), "tail_min_class_recall": mean("tail_min_class_recall"),
            "frontier_min_class_recall": mean("frontier_min_class_recall"),
            "survival_bins_at_80pct": mean("survival_bins_at_80pct"), "retention_auc": mean("retention_auc"),
            "grid_accuracy": optional_mean("grid_accuracy"), "grid_probability_mse": optional_mean("grid_probability_mse"),
            "boundary_f1": optional_mean("boundary_f1"), "component_count_error": optional_mean("component_count_error"),
        })
    payload = {"configuration": vars(args) | {"out": str(args.out)}, "runtime_seconds": time.perf_counter() - started,
               "runs": runs, "summary": summary}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    for task_name, models in representatives.items():
        learning_curves(args.out / f"learning_{task_name}.svg", representative_histories[task_name])
        task = TASK_BUILDERS[task_name](0)
        if task.visual_limits is not None:
            decision_atlas(args.out / f"decision_{task_name}.svg", task, models)
            surface_atlas(args.out / f"surface3d_{task_name}.svg", task, models)
    print(json.dumps({"complete": True, "runtime_seconds": payload["runtime_seconds"], "runs": len(runs)}), flush=True)


if __name__ == "__main__": main()
