#!/usr/bin/env python3
"""Focused study of endogenous self-context transport.

The study separates trainable degrees of freedom from chart observations and
records acquisition, extrapolation, gradient routing, and internal chart-state
diagnostics.  Every compared model has the same trainable parameter count.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import TRANSPORT_VARIANTS, make_variant

torch.set_num_threads(8)


def task_loss(output, target, kind):
    return F.cross_entropy(output, target) if kind == "classification" else F.mse_loss(output, target)


def gradient_groups(model):
    totals = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if ".metric." in name or ".response." in name:
            group = "allocator"
        elif ".base." in name:
            group = "base"
        elif ".shared" in name or ".scale" in name:
            group = "correction"
        else:
            group = "envelope"
        totals[group] += float(parameter.grad.detach().square().sum())
    return {f"gradient_{key}": value ** .5 for key, value in totals.items()}


@torch.no_grad()
def state_snapshot(model, sample):
    model.eval()
    _ = model(sample)
    values = {}
    for layer, state in model.diagnostics().items():
        for key in ("allocation_js", "chart_points", "context_raw_ratio",
                    "innovation_rms_ratio", "context_cosine", "turn_fraction",
                    "basis_coordinate_rms",
                    "shell_allocation_js",
                    "curvature_raw_ratio", "curvature_authority"):
            if key in state:
                values[f"{layer}_{key}"] = float(state[key].float().mean())
        values[f"{layer}_allocation_entropy"] = float(state["entropy"].float().mean())
        values[f"{layer}_correction_ratio"] = float(
            (state["correction_norm"] / (state["base_norm"] + 1e-8)).mean()
        )
    return values


def auc(history, steps):
    return float(np.trapz([row["score"] for row in history], [row["step"] for row in history]) / steps)


def own_threshold(history, fraction):
    target = max(row["score"] for row in history) * fraction
    return next(row["step"] for row in history if row["score"] >= target)


def train(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(10000 + seed)
    model = make_variant(name, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190000 + seed)
    history, best = [], None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model(x), y, task.kind)
        loss.backward()
        gradients = gradient_groups(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            state = state_snapshot(model, task.x_val[:512])
            row = {"step": step, "loss": float(loss.detach()), **metrics, **gradients, **state}
            history.append(row)
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    final_state = state_snapshot(model, task.x_val[:512])
    return model, history, final_state, seconds, best[2]


def summarize(runs):
    numeric = ("validation_score", "score", "tail_score", "learning_auc",
               "steps_to_90pct_own", "seconds", "jacobian_variability")
    rows = []
    for task in sorted({row["task"] for row in runs}):
        task_rows = [row for row in runs if row["task"] == task]
        baseline = [row for row in task_rows if row["variant"] == "self_context"]
        baseline_mean = {}
        for key in numeric:
            values = [row[key] for row in baseline if key in row]
            if values:
                baseline_mean[key] = float(np.mean(values))
        for variant in TRANSPORT_VARIANTS:
            selected = [row for row in task_rows if row["variant"] == variant]
            if not selected:
                continue
            summary = {"task": task, "variant": variant, "parameters": selected[0]["parameters"]}
            for key in numeric:
                values = [row[key] for row in selected if key in row]
                if values:
                    summary[key] = float(np.mean(values))
                    if key in baseline_mean:
                        summary[f"{key}_delta"] = summary[key] - baseline_mean[key]
            rows.append(summary)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/transport_study"))
    parser.add_argument("--tasks", default="radial_stripes,multiscale_1d,nd_spiral_high_rank,fourier_mix_1d")
    parser.add_argument("--variants", default=",".join(TRANSPORT_VARIANTS))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.json"
    runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    done = {(row["task"], row["seed"], row["variant"]) for row in runs}
    variants = args.variants.split(",")
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for variant in variants:
                if (task_name, seed, variant) in done:
                    continue
                model, history, state, seconds, best_step = train(
                    variant, task, args.width, seed, args.steps, args.batch, args.lr, args.eval_every
                )
                test = evaluate(model, task)
                tails = tail_metrics(model, task)
                variability, rank = jacobian_variability(model, task.x_val)
                row = {
                    "task": task_name,
                    "kind": task.kind,
                    "seed": seed,
                    "variant": variant,
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    "best_step": best_step,
                    "validation_score": max(point["score"] for point in history),
                    "learning_auc": auc(history, args.steps),
                    "steps_to_90pct_own": own_threshold(history, .9),
                    "jacobian_variability": variability,
                    "jacobian_change_rank": rank,
                    **test,
                    **tails,
                    **state,
                    "history": history,
                }
                runs.append(row)
                partial.write_text(json.dumps({"runs": runs}, indent=2))
                print(json.dumps({key: value for key, value in row.items()
                                  if key not in {"history", "tail_bins"}}), flush=True)
    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "summary": summarize(runs),
        "runs": runs,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "fits": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
