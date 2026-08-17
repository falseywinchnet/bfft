#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.soft_eikonal_instructive.models import (
    PairZeroEvaluation,
    VARIANTS,
    make_variant,
)
from experiments.soft_eikonal_matched.metrics import evaluate, jacobian_variability, tail_metrics
from experiments.soft_eikonal_matched.models import SoftEikonalNet, parameter_count
from experiments.soft_eikonal_matched.tasks import TASK_BUILDERS


torch.set_num_threads(8)


def task_loss(output, target, kind):
    return F.cross_entropy(output, target) if kind == "classification" else F.mse_loss(output, target)


def garnish_loss(model, x, y, kind, generator, scale):
    delta1 = torch.randn(x.shape, generator=generator) * scale
    delta2 = torch.randn(x.shape, generator=generator) * scale
    true_output = model(x)
    mean_garnish = (model(x + delta1) + model(x + delta2)) / 2
    garnish_objective = task_loss(mean_garnish, y, kind)
    instruction = torch.autograd.grad(garnish_objective, mean_garnish, retain_graph=True)[0].detach()
    # The true-input stream receives the error derivative computed in the
    # garnish stream, but never a direct target loss of its own.
    return garnish_objective + (true_output * instruction).sum()


def secant_loss(model, x, y, kind, generator):
    output = model(x); order = torch.randperm(len(x), generator=generator)
    other_output, other_target = output[order], y[order]
    direct = task_loss(output, y, kind)
    if kind == "regression":
        relational = F.mse_loss(output - other_output, y - other_target)
    else:
        probability = torch.softmax(output, 1); other_probability = probability[order]
        target = F.one_hot(y, output.shape[1]).float(); other = target[order]
        relational = F.mse_loss(probability - other_probability, target - other) * output.shape[1]
    return direct + .5 * relational


def allocation_secant_loss(model, x, y, kind, generator, scale):
    output = model(x); center = tuple(model.allocation_weights())
    delta = torch.randn(x.shape, generator=generator) * scale
    _ = model(x + delta); plus = tuple(model.allocation_weights())
    _ = model(x - delta); minus = tuple(model.allocation_weights())
    curvature = sum((a + b - 2 * c).square().mean() for a, b, c in zip(plus, minus, center))
    return task_loss(output, y, kind) + 8.0 * curvature


def pair_loss(model, x, y, kind, generator, unpaired_fraction=.35):
    order = torch.randperm(len(x), generator=generator); paired = torch.rand(len(x), generator=generator) > unpaired_fraction
    other_x = x[order].clone(); other_x[~paired] = 0
    output = model(torch.cat((x, other_x), 1)).view(len(x), 2, model.output_dim)
    first = task_loss(output[:, 0], y, kind)
    second = task_loss(output[paired, 1], y[order][paired], kind) if paired.any() else 0
    return first + second


def train_variant(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(10000 + seed)
    model = make_variant(name, task.input_dim, task.output_dim, width)
    evaluator = PairZeroEvaluation(model) if name == "paired_zero" else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190000 + seed)
    scale = task.x_train.std(0, keepdim=True).clamp_min(1e-3) * .055
    history, best = [], None; started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        if name == "garnish_instructive":
            loss = garnish_loss(model, x, y, task.kind, generator, scale)
        elif name == "paired_zero":
            loss = pair_loss(model, x, y, task.kind, generator)
        elif name == "secant_relational":
            loss = secant_loss(model, x, y, task.kind, generator)
        elif name == "allocation_secant":
            loss = allocation_secant_loss(model, x, y, task.kind, generator, scale)
        else:
            loss = task_loss(model(x), y, task.kind)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(evaluator, task, task.x_val, task.y_val)
            history.append({"step": step, "loss": float(loss.detach()), **metrics})
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(best[1])
    return model, evaluator, history, time.perf_counter() - started, best[2]


def auc(history, steps):
    return float(np.trapz([row["score"] for row in history], [row["step"] for row in history]) / steps)


def threshold(history, value):
    return next((row["step"] for row in history if row["score"] >= value), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/soft_eikonal_instructive_screen"))
    parser.add_argument("--tasks", default="spiral,checkerboard,radial_stripes,ripple,multiscale_1d,localized_steps_1d")
    parser.add_argument("--variants", default=",".join(VARIANTS)); parser.add_argument("--widths", default="16")
    parser.add_argument("--seeds", type=int, default=2); parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=256); parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.json"
    runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    done = {(row["task"], row["width"], row["seed"], row["variant"]) for row in runs}
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for width in map(int, args.widths.split(",")):
                for name in args.variants.split(","):
                    if (task_name, width, seed, name) in done: continue
                    model, evaluator, history, seconds, best_step = train_variant(
                        name, task, width, seed, args.steps, args.batch, args.lr, args.eval_every)
                    test = evaluate(evaluator, task); tails = tail_metrics(evaluator, task)
                    variability, rank = jacobian_variability(evaluator, task.x_val)
                    row = {"task": task_name, "kind": task.kind, "width": width, "seed": seed,
                           "variant": name, "parameters": parameter_count(model), "seconds": seconds,
                           "best_step": best_step, "learning_auc": auc(history, args.steps),
                           "steps_to_80": threshold(history, .8), "steps_to_90": threshold(history, .9),
                           "validation_score": max(point["score"] for point in history), **test, **tails,
                           "jacobian_variability": variability, "jacobian_change_rank": rank, "history": history}
                    if name == "paired_zero":
                        row.update({"pair_width": model.pair_width, "budget_remainder": model.extra.numel()})
                    runs.append(row); partial.write_text(json.dumps({"runs": runs}, indent=2))
                    print(json.dumps({k: v for k, v in row.items() if k not in {"history", "tail_bins"}}), flush=True)
    payload = {"configuration": {**vars(args), "out": str(args.out)}, "runs": runs}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
