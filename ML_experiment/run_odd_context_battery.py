#!/usr/bin/env python3
"""Full-task battery for chart-coupled odd self-context mechanisms."""
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

from ML_experiment.metrics import (
    evaluate, jacobian_variability, predict, regression_metrics, tail_metrics,
)
from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.run_benchmark import auc, threshold
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.tasks import TASK_BUILDERS


DEFAULT_VARIANTS = (
    "self_context",
    "self_capacity_match_full",
    "self_contextual_angular",
    "self_contextual_full_row_unit",
)
torch.set_num_threads(8)


def mechanism_diagnostics(model):
    bridge = getattr(model, "bridge", None)
    logits = getattr(bridge, "cone_logits", None)
    diagnostics = {}
    if logits is not None:
        cone = torch.sigmoid(logits.detach())
        diagnostics.update({
            "cone_source_a": float(cone[0]),
            "cone_source_b": float(cone[1]),
            "cone_chart": float(cone[2]),
        })
    degree_logits = getattr(bridge, "degree_logits", None)
    if degree_logits is not None:
        degree = 3.0 * torch.sigmoid(degree_logits.detach())
        diagnostics.update({
            "degree_mean": float(degree.mean()),
            "degree_std": float(degree.std(unbiased=False)),
            "degree_min": float(degree.min()),
            "degree_max": float(degree.max()),
        })
    log_row_gains = getattr(bridge, "log_row_gains", None)
    if log_row_gains is not None:
        gains = torch.cat([value.detach().exp() for value in log_row_gains])
        diagnostics.update({
            "gain_mean": float(gains.mean()),
            "gain_std": float(gains.std(unbiased=False)),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
        })
    operator_coordinates = getattr(model, "operator_coordinates", None)
    if operator_coordinates is not None:
        coordinates = F.normalize(operator_coordinates.detach(), dim=0)
        names = getattr(model, "operator_coordinate_names", None)
        if names is None:
            names = (["operator_odd", "operator_curvature"]
                     if len(coordinates) == 2 else
                     ["operator_odd", "operator_tangent", "operator_curvature"])
        diagnostics.update({
            name: float(coordinates[index].square().mean())
            for index, name in enumerate(names)
        })
    transport_logit = getattr(model, "transport_logit", None)
    if transport_logit is not None:
        diagnostics["transport_strength"] = float(
            0.5 * torch.tanh(transport_logit.detach())
        )
    return diagnostics


def task_geometry_diagnostics(model, task):
    """Task-owned diagnostics which ordinary aggregate metrics cannot expose."""
    if not hasattr(task, "segment_edges"):
        return {}
    prediction = predict(model, task.x_test)
    segments = []
    for index, (lo, hi) in enumerate(zip(
        task.segment_edges[:-1], task.segment_edges[1:]
    )):
        selected = (task.x_test[:, 0] >= lo) & (
            task.x_test[:, 0] <= hi if index + 1 == len(task.segment_counts)
            else task.x_test[:, 0] < hi
        )
        segments.append({
            "segment": index,
            "observation_count": task.segment_counts[index],
            **regression_metrics(prediction[selected], task.y_test[selected]),
        })
    return {
        "observed_segment_metrics": segments,
        "sparse_observed_r2": float(sum(row["r2"] for row in segments[-3:]) / 3),
        "minimum_observed_segment_r2": float(min(row["r2"] for row in segments)),
    }


def train(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(41000 + seed)
    model = make_hybrid(name, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42000 + seed)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
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
    parser.add_argument("--out", type=Path, default=Path("/tmp/odd_context_battery"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
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
    runs = (json.loads(run_partial.read_text())["runs"]
            if args.resume and run_partial.exists() else [])
    probes = (json.loads(probe_partial.read_text())["probes"]
              if args.resume and probe_partial.exists() else [])
    done = {(row["task"], row["variant"], row["seed"]) for row in runs}
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for name in args.variants.split(","):
                if (task_name, name, seed) in done:
                    continue
                print(f"START task={task_name} seed={seed} variant={name}", flush=True)
                model, history, seconds, best_step = train(
                    name, task, args.width, seed, args.steps, args.batch, args.lr,
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
                    "variant": name,
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
                    **task_geometry_diagnostics(model, task),
                    "jacobian_variability": variability,
                    "jacobian_change_rank": rank,
                    **mechanism_diagnostics(model),
                    "history": history,
                }
                runs.append(row)
                run_partial.write_text(json.dumps({"runs": runs}, indent=2))
                if seed == 0:
                    probes.append({
                        "task": task_name,
                        "variant": name,
                        "seconds": seconds,
                        "best_step": best_step,
                        **make_probe(task, model, args.grid),
                    })
                    probe_partial.write_text(json.dumps({"probes": probes}))
                print(json.dumps({
                    "task": task_name,
                    "variant": name,
                    "parameters": row["parameters"],
                    "score": row["score"],
                    "tail_score": row.get("tail_score"),
                    "learning_auc": row["learning_auc"],
                    "seconds": seconds,
                }), flush=True)
    configuration = {**vars(args), "out": str(args.out)}
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": runs,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({"complete": True, "runs": len(runs), "out": str(args.out)}))


if __name__ == "__main__":
    main()
