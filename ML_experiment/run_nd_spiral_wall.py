#!/usr/bin/env python3
"""Breadth-first wall screen on only the 16-D, eight-plane spiral."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from ML_experiment.nd_spiral_wall import WALL_MODELS, make_wall_model
from ML_experiment.optimizers import make_optimizer
from ML_experiment.tasks import TASK_BUILDERS, _orthogonal


LABELS = {
    "ordinary_mlp": "Ordinary LELU MLP",
    "self_context": "Relational self-context",
    "cff_fast": "Continuous frame flow",
    "cff_fast_muon": "Continuous frame flow + Muon",
    "dynamic_subspace": "Dynamic subspace conduits",
    "odd_cubic": "Odd cubic tensor sketch",
    "odd_cubic_smoothed": "Odd cubic + smoothed objective",
    "learned_bispectrum": "Learned complex bispectrum",
    "midpoint_hessian": "Midpoint Hessian/coset bank",
    "soft_hypotheses": "Soft affine hypotheses",
    "gaussian_derivatives": "Gaussian derivative bank",
    "moving_frame": "Tied moving-frame recurrence",
    "cayley_flow": "Input-conditioned Cayley flow",
    "living_graph": "Living hidden-coordinate graph",
    "shallow_odd_cubic": "Shallow real odd cubic",
    "shallow_bispectrum": "Shallow learned bispectrum",
    "fixed_random_bispectrum": "Fixed random bispectrum",
    "even_quadratic_control": "Even quadratic parity control",
}


CONFIGURATIONS = tuple(WALL_MODELS) + ("cff_fast_muon", "odd_cubic_smoothed")


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def accuracy(model, x, y):
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def diagnostics(model, task):
    logits = model(task.x_test)
    prediction = logits.argmax(1)
    probabilities = logits.softmax(1)[:, 1]
    flipped = model(-task.x_test).softmax(1)[:, 1]
    class_scores = [
        float((prediction[task.y_test == label] == label).float().mean())
        for label in (0, 1)
    ]
    # This is a diagnostic, not an objective: an antipodal classifier should
    # satisfy p(class 1|-x) = 1-p(class 1|x).
    antipodal_error = float((probabilities + flipped - 1).abs().mean())
    return {
        "validation_accuracy": accuracy(model, task.x_val, task.y_val),
        "tail_accuracy": float((prediction == task.y_test).float().mean()),
        "tail_class_0": class_scores[0],
        "tail_class_1": class_scores[1],
        "antipodal_error": antipodal_error,
        "tail_bins": [accuracy(model, x, y) for x, y in zip(task.tail_x, task.tail_y)],
    }


def train(configuration, task, width, seed, steps, batch, lr, evaluate_every):
    base_name = configuration.replace("_muon", "").replace("_smoothed", "")
    torch.manual_seed(12000 + seed)
    model = make_wall_model(base_name, task.input_dim, task.output_dim, width)
    optimizer_name = "muon" if configuration.endswith("_muon") else "adamw"
    optimizer = make_optimizer(model, optimizer_name, lr)
    generator = torch.Generator().manual_seed(13000 + seed)
    best_state, best_score, best_step = None, -1.0, 0
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        if configuration.endswith("_smoothed"):
            # Antithetic convolution smoothing of the observed objective only.
            progress = step / steps
            sigma = 0.08 * (1.0 - progress) + 0.005
            noise = torch.randn(x.shape, generator=generator) * sigma
            logits = (model(x + noise) + model(x - noise)) * 0.5
        else:
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            score = accuracy(model, task.x_val, task.y_val)
            history.append({"step": step, "accuracy": score, "loss": float(loss.detach())})
            if score > best_score:
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.state_dict())
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    with torch.no_grad():
        for _ in range(3):
            model(task.x_test)
        timing_started = time.perf_counter()
        for _ in range(20):
            model(task.x_test)
        milliseconds = 1000 * (time.perf_counter() - timing_started) / 20
    return model, {
        "configuration": configuration,
        "label": LABELS[configuration],
        "parameters": parameter_count(model),
        "fixed_scalars": sum(buffer.numel() for buffer in model.buffers()),
        "optimizer": optimizer_name,
        "seconds": seconds,
        "inference_ms_2000": milliseconds,
        "best_step": best_step,
        "history": history,
        **diagnostics(model, task),
    }


def exact_curve(seed, count=900):
    u = torch.linspace(0.015, 1.0, count)
    theta = 0.45 + 5.4 * math.pi * u
    radius = 0.12 + 0.88 * u
    rotation = _orthogonal(16, 700 + seed)
    branches = []
    for label in (0, 1):
        features = []
        for plane in range(8):
            frequency = plane + 1
            phase = frequency * theta + 0.37 * plane + label * math.pi
            amplitude = radius * (1 + 0.08 * torch.sin((plane + 2) * theta)) / math.sqrt(8)
            features.extend((amplitude * torch.cos(phase), amplitude * torch.sin(phase)))
        branches.append(torch.stack(features, 1) @ rotation)
    return u, branches


@torch.no_grad()
def visual_probe(model, task, seed):
    u, branches = exact_curve(seed)
    center = task.x_train.mean(0)
    _, _, vectors = torch.pca_lowrank(task.x_train - center, q=3)
    return {
        "u": u.tolist(),
        "coordinates": [((branch - center) @ vectors).tolist() for branch in branches],
        "probabilities": [model(branch).softmax(1)[:, 1].tolist() for branch in branches],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/nd_spiral_wall"))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default=",".join(CONFIGURATIONS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task = TASK_BUILDERS["nd_spiral_high_rank"](args.seed)
    rows = []
    probes = []
    for configuration in args.models.split(","):
        print(f"START {configuration}", flush=True)
        model, row = train(configuration, task, args.width, args.seed, args.steps,
                           args.batch, args.lr, args.eval_every)
        probes.append({"configuration": configuration, **visual_probe(model, task, args.seed)})
        rows.append(row)
        (args.out / "results.partial.json").write_text(json.dumps({"runs": rows}, indent=2))
        (args.out / "probes.partial.json").write_text(json.dumps({"probes": probes}))
        print(json.dumps({key: row[key] for key in (
            "configuration", "validation_accuracy", "tail_accuracy",
            "parameters", "seconds", "antipodal_error")}), flush=True)
    (args.out / "results.json").write_text(json.dumps({
        "configuration": {**vars(args), "out": str(args.out)}, "runs": rows,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({"probes": probes}))
    print(json.dumps({"complete": True, "models": len(rows), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
