#!/usr/bin/env python3
"""Compare original SC, relational SCL, CFF, and relational/deep CFF."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.response_enhanced import (
    RESPONSE_VARIANTS,
    allocation_summary,
    make_response_variant,
)
from ML_experiment.run_benchmark import auc, threshold
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.tasks import TASK_BUILDERS


torch.set_num_threads(8)


def task_loss(output, target, kind):
    return (
        F.cross_entropy(output, target)
        if kind == "classification"
        else F.mse_loss(output, target)
    )


@torch.no_grad()
def inference_timing(model, sample, batch, repeats=15):
    model.eval()
    if len(sample) < batch:
        sample = sample.repeat((batch + len(sample) - 1) // len(sample), 1)
    sample = sample[:batch]
    for _ in range(3):
        _ = model(sample)
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ = model(sample)
        measurements.append((time.perf_counter() - started) * 1000.0)
    return {
        "inference_ms": statistics.median(measurements),
        "inference_ms_p90": float(np.quantile(measurements, 0.9)),
        "inference_examples_per_second": batch / (statistics.median(measurements) / 1000),
    }


def train(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(10_000 + seed)
    model = make_response_variant(name, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190_000 + seed)
    history = []
    best = None
    train_seconds = 0.0
    total_started = time.perf_counter()

    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x = task.x_train[index]
        y = task.y_train[index]
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model(x), y, task.kind)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        train_seconds += time.perf_counter() - step_started

        if step == 1 or step % evaluate_every == 0 or step == steps:
            score = evaluate(model, task, task.x_val, task.y_val)
            history.append({"step": step, "loss": float(loss.detach()), **score})
            if best is None or score["score"] > best[0]:
                best = (score["score"], copy.deepcopy(model.state_dict()), step)

    total_seconds = time.perf_counter() - total_started
    model.load_state_dict(best[1])
    timing = inference_timing(model, task.x_val, batch)
    _ = model(task.x_val[:batch])
    return model, history, {
        "seconds": total_seconds,
        "train_seconds": train_seconds,
        "training_examples_per_second": steps * batch / train_seconds,
        "best_step": best[2],
        **timing,
        **allocation_summary(model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/response_enhanced_full"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS))
    parser.add_argument("--variants", default=",".join(RESPONSE_VARIANTS))
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=1)
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
    runs = (
        json.loads(run_partial.read_text())["runs"]
        if args.resume and run_partial.exists()
        else []
    )
    probes = (
        json.loads(probe_partial.read_text())["probes"]
        if args.resume and probe_partial.exists()
        else []
    )
    done = {(row["task"], row["seed"], row["variant"]) for row in runs}

    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for variant in args.variants.split(","):
                if (task_name, seed, variant) in done:
                    continue
                model, history, timing = train(
                    variant,
                    task,
                    args.width,
                    seed,
                    args.steps,
                    args.batch,
                    args.lr,
                    args.eval_every,
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
                    "learning_auc": auc(history, args.steps),
                    "steps_to_80": threshold(history, 0.8),
                    "steps_to_90": threshold(history, 0.9),
                    "validation_score": max(point["score"] for point in history),
                    **timing,
                    **test,
                    **tails,
                    "jacobian_variability": variability,
                    "jacobian_change_rank": rank,
                    "history": history,
                }
                runs.append(row)
                run_partial.write_text(json.dumps({"runs": runs}, indent=2))

                if seed == 0:
                    probes.append(
                        {
                            "task": task_name,
                            "variant": variant,
                            "seconds": timing["seconds"],
                            "best_step": timing["best_step"],
                            **make_probe(task, model, args.grid),
                        }
                    )
                    probe_partial.write_text(json.dumps({"probes": probes}))

                print(
                    json.dumps(
                        {
                            "task": task_name,
                            "variant": variant,
                            "parameters": row["parameters"],
                            "score": row["score"],
                            "tail_score": row.get("tail_score"),
                            "learning_auc": row["learning_auc"],
                            "train_seconds": row["train_seconds"],
                            "inference_ms": row["inference_ms"],
                            "allocation_entropy": row["allocation_entropy"],
                        }
                    ),
                    flush=True,
                )

    configuration = {
        **vars(args),
        "out": str(args.out),
        "variants": args.variants.split(","),
    }
    (args.out / "results.json").write_text(
        json.dumps({"configuration": configuration, "runs": runs}, indent=2)
    )
    (args.out / "probes.json").write_text(
        json.dumps({"configuration": configuration, "probes": probes})
    )
    print(json.dumps({"complete": True, "runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()
