#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ML_experiment.context_backprop import (
    backward_with_context_split,
    clip_context_gradient_channels_,
)
from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.optimizers import make_optimizer
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import VARIANTS, make_variant

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


def post_nonlinearity_observer_modules(model):
    """Matrix contractions immediately consuming the model's main activation."""
    down = getattr(model, "down", None)
    if isinstance(down, torch.nn.Linear):
        return (down,)
    selected = tuple(
        module for name in ("base", "metric")
        if isinstance((module := getattr(down, name, None)), torch.nn.Linear)
    )
    if not selected:
        raise ValueError(
            "post_nonlinearity observer scope is not defined for this model"
        )
    return selected


def train_variant(name, task, width, seed, steps, batch, lr, evaluate_every,
                  optimizer_name="adamw", context_backward_mode="exact",
                  context_gradient_mode="blended", observer_scope="all_linear"):
    if context_gradient_mode == "split_trust" and name != "self_context":
        raise ValueError(
            "split_trust currently requires the deterministic self_context variant"
        )
    torch.manual_seed(10000 + seed); model = make_variant(name, task.input_dim, task.output_dim, width)
    if hasattr(model, "set_context_backward_mode"):
        model.set_context_backward_mode(context_backward_mode)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay=1e-4)
    if hasattr(optimizer, "attach_model"):
        if observer_scope == "post_nonlinearity":
            observed = {id(module) for module in post_nonlinearity_observer_modules(model)}
            optimizer.attach_model(
                model, module_filter=lambda module: id(module) in observed
            )
        else:
            optimizer.attach_model(model)
    generator = torch.Generator().manual_seed(190000 + seed)
    scale = task.x_train.std(0, keepdim=True).clamp_min(1e-3) * .055
    initial = copy.deepcopy(model.state_dict())
    history, best, failure = [], None, None; started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator); x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        context_gradient_diagnostic = {}
        split_context_gradient = False
        if name == "self_context_secant": loss = secant_loss(model, x, y, task.kind, generator)
        elif name == "self_context_chart": loss = chart_loss(model, x, y, task.kind, generator, scale)
        elif context_gradient_mode == "split_trust":
            loss, context_gradient_diagnostic = backward_with_context_split(
                model, lambda: task_loss(model(x), y, task.kind)
            )
            split_context_gradient = True
        else: loss = task_loss(model(x), y, task.kind)
        if not torch.isfinite(loss):
            failure = {"step": step, "stage": "loss"}
            history.append({"step": step, "loss": float(loss.detach()), "score": float("nan"), "failed": True})
            break
        if not split_context_gradient:
            loss.backward()
        if split_context_gradient:
            gradient_norm = clip_context_gradient_channels_(model.parameters(), 5)
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if not torch.isfinite(gradient_norm):
            failure = {"step": step, "stage": "gradient"}
            history.append({"step": step, "loss": float(loss.detach()), "score": float("nan"), "failed": True})
            break
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            failure = {"step": step, "stage": "parameters"}
            history.append({"step": step, "loss": float(loss.detach()), "score": float("nan"), "failed": True})
            break
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            diagnostic = {}
            if hasattr(optimizer, "last_ratio"):
                diagnostic = {
                    "anchor_ratio": float(optimizer.last_ratio),
                    "anchor_turn": float(optimizer.last_turn),
                    "anchor_alignment": float(optimizer.last_alignment),
                    "anchor_relative_step": float(optimizer.last_relative_step),
                    "anchor_lr_scale": float(optimizer.last_lr_scale),
                    "anchor_proposal_concentration": float(
                        optimizer.last_proposal_concentration
                    ),
                    "anchor_update_concentration": float(
                        optimizer.last_update_concentration
                    ),
                    "anchor_memory_ratio": float(
                        getattr(optimizer, "last_memory_ratio", 1.0)
                    ),
                }
            optional_diagnostics = {
                "optimizer_turn": "last_turn",
                "optimizer_transport_ratio": "last_transport_ratio",
                "optimizer_frame_residual": "last_frame_residual",
                "optimizer_history_alignment": "last_history_alignment",
                "optimizer_gradient_norm": "last_gradient_norm",
                "optimizer_update_norm": "last_update_norm",
                "optimizer_update_gradient_ratio": "last_update_gradient_ratio",
                "optimizer_certificate_scale": "last_certificate_scale",
                "optimizer_denominator_cv": "last_denominator_cv",
                "optimizer_ridge_fraction": "last_ridge_fraction",
                "optimizer_lr_scale": "last_lr_scale",
                "optimizer_covariance_rank": "last_covariance_rank",
                "optimizer_context_request_scale": "last_context_request_scale",
                "optimizer_context_request_ratio": "last_context_request_ratio",
                "optimizer_fixed_alignment": "last_fixed_alignment",
                "optimizer_reversal_fraction": "last_reversal_fraction",
                "optimizer_chart_only_fraction": "last_chart_only_fraction",
                "optimizer_transverse_retention": "last_transverse_retention",
                "optimizer_metric_condition": "last_metric_condition",
                "optimizer_root_refresh_fraction": "last_root_refresh_fraction",
                "optimizer_root_staleness": "last_root_staleness",
                "optimizer_gauge_transport_ratio": "last_gauge_transport_ratio",
                "optimizer_gauge_transport_deformation": "last_gauge_transport_deformation",
                "optimizer_gauge_turn_removed": "last_gauge_turn_removed",
                "optimizer_gauge_event_count": "last_gauge_event_count",
                "optimizer_longitudinal_closure": "last_longitudinal_closure",
                "optimizer_longitudinal_certificate": "last_longitudinal_certificate",
                "optimizer_transverse_momentum_ratio": "last_transverse_momentum_ratio",
                "optimizer_residual_entropy_rank": "last_residual_entropy_rank",
                "optimizer_polar_weight": "last_polar_weight",
            }
            for key, attribute in optional_diagnostics.items():
                if hasattr(optimizer, attribute):
                    diagnostic[key] = float(getattr(optimizer, attribute))
            diagnostic.update(context_gradient_diagnostic)
            history.append({"step": step, "loss": float(loss.detach()), **metrics, **diagnostic})
            if not math.isfinite(metrics["score"]):
                failure = {"step": step, "stage": "evaluation"}
                history[-1]["failed"] = True
                break
            if best is None or metrics["score"] > best[0]: best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    if best is None:
        model.load_state_dict(initial)
        best_step = 0
    else:
        model.load_state_dict(best[1])
        best_step = best[2]
    if hasattr(optimizer, "remove_observers"):
        optimizer.remove_observers()
    return model, history, time.perf_counter() - started, best_step, failure


def auc(history, steps):
    if not history or any(not math.isfinite(row["score"]) for row in history):
        return float("nan")
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(
        [row["score"] for row in history],
        [row["step"] for row in history],
    ) / steps)


def threshold(history, value):
    return next((row["step"] for row in history
                 if math.isfinite(row["score"]) and row["score"] >= value), None)


def _rounded(value, digits=5):
    return np.round(value.detach().cpu().numpy(), digits).tolist()


@torch.no_grad()
def fit_visualization(model, task, max_points=600):
    """Compact best-checkpoint predictions for the result atlas."""
    model.eval()

    def prediction(value):
        output = model(value)
        if task.kind == "classification":
            return torch.softmax(output, -1)
        if task.target_mean is not None:
            return output * task.target_std + task.target_mean
        return output

    if task.name.startswith("nd_spiral_"):
        observed = task.x_val[:max_points]
        continuation = task.x_test[:max_points]
        points = torch.cat((observed, continuation))
        labels = torch.cat((task.y_val[:max_points], task.y_test[:max_points]))
        region = torch.cat((torch.zeros(len(observed)), torch.ones(len(continuation))))
        centered = points - points.mean(0, keepdim=True)
        _, _, right = torch.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ right[:3].transpose(0, 1)
        return {
            "kind": "nd_spiral_3d",
            "coordinates": _rounded(coordinates),
            "label": labels.tolist(),
            "probability": _rounded(prediction(points)[:, 1]),
            "region": region.int().tolist(),
        }

    if task.name == "complex_spiral_3d":
        index = torch.linspace(0, len(task.x_test) - 1,
                               min(max_points, len(task.x_test))).long()
        value = task.x_test[index]
        target = task.y_test[index]
        if task.target_mean is not None:
            target = target * task.target_std + task.target_mean
        return {
            "kind": "curve_3d",
            "coordinate": _rounded(value[:, 0]),
            "target": _rounded(target),
            "prediction": _rounded(prediction(value)),
        }

    if task.input_dim == 1 and task.kind == "regression":
        index = torch.linspace(0, len(task.x_test) - 1,
                               min(max_points, len(task.x_test))).long()
        value = task.x_test[index]
        target = task.y_test[index]
        if task.target_mean is not None:
            target = target * task.target_std + task.target_mean
        return {
            "kind": "curve_1d",
            "coordinate": _rounded(value[:, 0]),
            "target": _rounded(target[:, 0]),
            "prediction": _rounded(prediction(value)[:, 0]),
        }

    if task.input_dim == 2 and task.visual_limits is not None:
        xmin, xmax, ymin, ymax = task.visual_limits
        side = 33
        gx, gy = torch.linspace(xmin, xmax, side), torch.linspace(ymin, ymax, side)
        xx, yy = torch.meshgrid(gx, gy, indexing="xy")
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)
        output = prediction(grid)
        payload = {
            "kind": "surface_2d", "side": side,
            "limits": [xmin, xmax, ymin, ymax],
            "sample_x": _rounded(task.x_val[:300]),
            "sample_y": task.y_val[:300].reshape(-1).tolist(),
        }
        if task.kind == "classification":
            payload["prediction"] = output.argmax(-1).tolist()
            payload["confidence"] = _rounded(output.max(-1).values)
            if task.truth is not None:
                payload["target"] = task.truth(grid).reshape(-1).tolist()
        else:
            payload["prediction"] = _rounded(output[:, 0])
            if task.truth is not None:
                payload["target"] = _rounded(task.truth(grid).reshape(-1))
        return payload

    value = task.x_val[:max_points]
    centered = value - value.mean(0, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right[:min(3, right.shape[0])].transpose(0, 1)
    output = prediction(value)
    payload = {
        "kind": "projected_fit",
        "coordinates": _rounded(coordinates),
    }
    if task.kind == "classification":
        payload["target"] = task.y_val[:max_points].tolist()
        payload["prediction"] = output.argmax(-1).tolist()
        payload["confidence"] = _rounded(output.max(-1).values)
    else:
        target = task.y_val[:max_points]
        if task.target_mean is not None:
            target = target * task.target_std + task.target_mean
        payload["target"] = _rounded(target)
        payload["prediction"] = _rounded(output)
    return payload


def parse_optimizer_lrs(specification):
    result = {}
    if not specification:
        return result
    for item in specification.split(","):
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/ml_experiment"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS)); parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--widths", default="16"); parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400); parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3); parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--optimizers", default="adamw")
    parser.add_argument("--optimizer-lrs", default="")
    parser.add_argument(
        "--observer-scope",
        choices=("all_linear", "post_nonlinearity"),
        default="all_linear",
    )
    parser.add_argument(
        "--context-backward-mode",
        choices=("exact", "detached", "nonexpansive"),
        default="exact",
    )
    parser.add_argument(
        "--context-gradient-mode", choices=("blended", "split_trust"),
        default="blended",
    )
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    optimizer_lrs = parse_optimizer_lrs(args.optimizer_lrs)
    partial = args.out / "runs.partial.json"; runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    done = {(r["task"], r["width"], r["seed"], r["variant"], r.get("optimizer", "adamw")) for r in runs}
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for width in map(int, args.widths.split(",")):
                for name in args.variants.split(","):
                    for optimizer_name in args.optimizers.split(","):
                        if (task_name, width, seed, name, optimizer_name) in done: continue
                        optimizer_lr = optimizer_lrs.get(optimizer_name, args.lr)
                        model, history, seconds, best_step, failure = train_variant(
                            name, task, width, seed, args.steps, args.batch, optimizer_lr,
                            args.eval_every, optimizer_name=optimizer_name,
                            context_backward_mode=args.context_backward_mode,
                            context_gradient_mode=args.context_gradient_mode,
                            observer_scope=args.observer_scope,
                        )
                        test = evaluate(model, task); tails = tail_metrics(model, task); variability, rank = jacobian_variability(model, task.x_val)
                        row = {"task": task_name, "kind": task.kind, "input_dim": task.input_dim, "output_dim": task.output_dim,
                               "width": width, "seed": seed, "variant": name, "optimizer": optimizer_name,
                               "optimizer_lr": optimizer_lr, "status": "failed" if failure else "complete",
                               "context_backward_mode": args.context_backward_mode,
                               "context_gradient_mode": args.context_gradient_mode,
                               "failure": failure,
                               "parameters": parameter_count(model), "seconds": seconds,
                               "best_step": best_step, "learning_auc": auc(history, args.steps), "steps_to_80": threshold(history, .8),
                               "steps_to_90": threshold(history, .9),
                               "validation_score": best["score"] if (best := max(
                                   (p for p in history if math.isfinite(p["score"])),
                                   key=lambda p: p["score"], default=None
                               )) is not None else float("nan"),
                               **test, **tails, "jacobian_variability": variability,
                               "jacobian_change_rank": rank,
                               "fit_visualization": fit_visualization(model, task) if seed == 0 else None,
                               "history": history}
                        runs.append(row); partial.write_text(json.dumps({"runs": runs}, indent=2));
                        print(json.dumps({k: v for k, v in row.items()
                                          if k not in {"history", "tail_bins", "fit_visualization"}}), flush=True)
    payload = {"configuration": {**vars(args), "out": str(args.out),
                                  "resolved_optimizer_lrs": optimizer_lrs}, "runs": runs}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2)); print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__": main()
