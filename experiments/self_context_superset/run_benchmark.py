#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.self_context_superset.models import VARIANTS, make_variant
from experiments.soft_eikonal_matched.metrics import evaluate, jacobian_variability, tail_metrics
from experiments.soft_eikonal_matched.models import parameter_count
from experiments.soft_eikonal_matched.tasks import TASK_BUILDERS

torch.set_num_threads(8)


def task_loss(output, target, kind):
    return F.cross_entropy(output, target) if kind == "classification" else F.mse_loss(output, target)


def secant_loss(model, x, y, kind, generator):
    output = model(x); order = torch.randperm(len(x), generator=generator)
    direct = task_loss(output, y, kind)
    if kind == "regression":
        relational = F.mse_loss(output - output[order], y - y[order])
    else:
        probability = torch.softmax(output, 1); target = F.one_hot(y, output.shape[1]).float()
        relational = F.mse_loss(probability - probability[order], target - target[order]) * output.shape[1]
    return direct + .5 * relational


def chart_loss(model, x, y, kind, generator, scale):
    output = model(x); center = tuple(model.allocation_weights())
    delta = torch.randn(x.shape, generator=generator) * scale
    _ = model(x + delta); plus = tuple(model.allocation_weights())
    _ = model(x - delta); minus = tuple(model.allocation_weights())
    curvature = sum((a + b - 2 * c).square().mean() for a, b, c in zip(plus, minus, center))
    return task_loss(output, y, kind) + 8.0 * curvature


def train_variant(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(10000 + seed); model = make_variant(name, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190000 + seed)
    scale = task.x_train.std(0, keepdim=True).clamp_min(1e-3) * .055
    history, best = [], None; started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator); x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        if name == "self_context_secant": loss = secant_loss(model, x, y, task.kind, generator)
        elif name == "self_context_chart": loss = chart_loss(model, x, y, task.kind, generator, scale)
        else: loss = task_loss(model(x), y, task.kind)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val); history.append({"step": step, "loss": float(loss.detach()), **metrics})
            if best is None or metrics["score"] > best[0]: best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(best[1]); return model, history, time.perf_counter() - started, best[2]


def auc(history, steps):
    return float(np.trapz([row["score"] for row in history], [row["step"] for row in history]) / steps)


def threshold(history, value):
    return next((row["step"] for row in history if row["score"] >= value), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/self_context_superset"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS)); parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--widths", default="16"); parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400); parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3); parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.json"; runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    done = {(r["task"], r["width"], r["seed"], r["variant"]) for r in runs}
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for width in map(int, args.widths.split(",")):
                for name in args.variants.split(","):
                    if (task_name, width, seed, name) in done: continue
                    model, history, seconds, best_step = train_variant(name, task, width, seed, args.steps, args.batch, args.lr, args.eval_every)
                    test = evaluate(model, task); tails = tail_metrics(model, task); variability, rank = jacobian_variability(model, task.x_val)
                    row = {"task": task_name, "kind": task.kind, "input_dim": task.input_dim, "output_dim": task.output_dim,
                           "width": width, "seed": seed, "variant": name, "parameters": parameter_count(model), "seconds": seconds,
                           "best_step": best_step, "learning_auc": auc(history, args.steps), "steps_to_80": threshold(history, .8),
                           "steps_to_90": threshold(history, .9), "validation_score": max(p["score"] for p in history),
                           **test, **tails, "jacobian_variability": variability, "jacobian_change_rank": rank, "history": history}
                    runs.append(row); partial.write_text(json.dumps({"runs": runs}, indent=2));
                    print(json.dumps({k: v for k, v in row.items() if k not in {"history", "tail_bins"}}), flush=True)
    payload = {"configuration": {**vars(args), "out": str(args.out)}, "runs": runs}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2)); print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__": main()
