#!/usr/bin/env python3
"""Retrain selected models and export compact geometry probes for the report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch

from ML_experiment.metrics import predict
from ML_experiment.run_benchmark import train_variant
from ML_experiment.tasks import TASK_BUILDERS


def physical(task, value):
    return value if task.target_mean is None else value * task.target_std + task.target_mean


def field_probe(task, model, size):
    xmin, xmax, ymin, ymax = task.visual_limits; xs = torch.linspace(xmin, xmax, size); ys = torch.linspace(ymin, ymax, size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij"); points = torch.stack((xx.flatten(), yy.flatten()), 1)
    output = predict(model, points)
    if task.kind == "classification":
        predicted = output.argmax(1).float(); truth = task.truth(points).float() if task.truth else predicted.clone()
    else:
        predicted = physical(task, output).squeeze(1); truth = task.truth(points).squeeze(1)
    return {"type": "field", "size": size, "limits": list(task.visual_limits),
            "prediction": predicted.tolist(), "truth": truth.tolist()}


def curve_probe(task, model):
    x = task.x_test
    target = task.y_test
    if task.output_dim == 3:
        # The spatial plot must show what was observed and what is genuinely
        # continuation.  Join a deterministic observed subset to the ordered
        # extrapolation curve and retain the exact boundary as metadata.
        observed = torch.argsort(task.x_val.squeeze(1))
        x = torch.cat((task.x_val[observed], task.x_test))
        target = torch.cat((task.y_val[observed], task.y_test))
    output = physical(task, predict(model, x)); truth = physical(task, target)
    train_limits = [float(task.x_train.min()), float(task.x_train.max())]
    if output.shape[1] == 3:
        return {"type": "curve3d", "input": x.squeeze(1).tolist(),
                "prediction": output.tolist(), "truth": truth.tolist(),
                "train_limits": train_limits}
    return {"type": "curve", "x": x.squeeze(1).tolist(), "prediction": output.squeeze(1).tolist(),
            "truth": truth.squeeze(1).tolist(), "train_limits": train_limits}


def scatter_probe(task, model, maximum=1200):
    x = task.x_test[:maximum]; output = predict(model, x)
    if task.kind == "regression":
        truth = physical(task, task.y_test[:maximum]).flatten(); prediction = physical(task, output).flatten()
        return {"type": "parity", "truth": truth.tolist(), "prediction": prediction.tolist()}
    basis_source = torch.cat((task.x_train[:maximum], x)); centered = basis_source - basis_source.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False); xy = centered[-len(x):] @ vh[:2].T
    return {"type": "scatter", "xy": xy.tolist(), "truth": task.y_test[:maximum].tolist(),
            "prediction": output.argmax(1).tolist()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=Path("/tmp/ml_experiment_probes.json"))
    parser.add_argument("--tasks", default="spiral,checkerboard,nd_spiral_low_rank,nd_spiral_high_rank,radial_stripes,swiss_cheese,periodic_wells,ripple,multiscale_1d,chirp_1d,localized_steps_1d,fourier_mix_1d")
    parser.add_argument("--variants", default="ordinary_mlp,self_context,self_context_hard,self_context_uncertainty")
    parser.add_argument("--width", type=int, default=36); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600); parser.add_argument("--grid", type=int, default=61)
    args = parser.parse_args(); rows = []
    for task_name in args.tasks.split(","):
        task = TASK_BUILDERS[task_name](args.seed)
        for variant in args.variants.split(","):
            model, history, seconds, best_step = train_variant(variant, task, args.width, args.seed, args.steps, 256, 3e-3, 50)
            if task.input_dim == 1: probe = curve_probe(task, model)
            elif task.input_dim == 2 and task.truth is not None: probe = field_probe(task, model, args.grid)
            else: probe = scatter_probe(task, model)
            rows.append({"task": task_name, "variant": variant, "seconds": seconds, "best_step": best_step, **probe})
            print(json.dumps({"task": task_name, "variant": variant, "type": probe["type"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps({"configuration": {**vars(args), "out": str(args.out)}, "probes": rows}))


if __name__ == "__main__": main()
