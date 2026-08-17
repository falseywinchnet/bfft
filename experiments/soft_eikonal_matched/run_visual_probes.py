#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from metrics import predict
from models import make_mlp_pair, parameter_count
from run_benchmark import train
from tasks import TASK_BUILDERS


torch.set_num_threads(8)


def train_pair(task_name, seed, width, steps):
    task = TASK_BUILDERS[task_name](seed)
    torch.manual_seed(10000 + seed)
    mlp, soft = make_mlp_pair(task.input_dim, task.output_dim, width)
    for model in (mlp, soft):
        train(model, task, seed, steps, 256, 3e-3, 25)
    return task, mlp, soft


def one_dimensional(task, mlp, soft):
    x = torch.linspace(-5, 5, 801)[:, None]
    truth = task.truth(x).reshape(len(x), -1) if task.truth else None
    mlp_y, soft_y = predict(mlp, x), predict(soft, x)
    if task.target_mean is not None:
        mlp_y = mlp_y * task.target_std + task.target_mean
        soft_y = soft_y * task.target_std + task.target_mean
    return {"x": x[:, 0].tolist(), "truth": truth[:, 0].tolist(),
            "mlp": mlp_y[:, 0].tolist(), "soft": soft_y[:, 0].tolist(),
            "training_interval": [-3, 3]}


def two_dimensional(task, mlp, soft, resolution=101):
    xmin, xmax, ymin, ymax = task.visual_limits
    xs = torch.linspace(xmin, xmax, resolution); ys = torch.linspace(ymin, ymax, resolution)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy"); points = torch.stack((xx.flatten(), yy.flatten()), 1)
    if task.kind == "classification":
        mlp_value = torch.softmax(predict(mlp, points), 1)[:, 1]
        soft_value = torch.softmax(predict(soft, points), 1)[:, 1]
        truth = task.truth(points).float() if task.truth else torch.zeros(len(points))
    else:
        mlp_value, soft_value = predict(mlp, points)[:, 0], predict(soft, points)[:, 0]
        truth = task.truth(points).reshape(-1)
        if task.target_mean is not None:
            mlp_value = mlp_value * task.target_std[0, 0] + task.target_mean[0, 0]
            soft_value = soft_value * task.target_std[0, 0] + task.target_mean[0, 0]
    return {"limits": [xmin, xmax, ymin, ymax], "resolution": resolution,
            "truth": truth.tolist(), "mlp": mlp_value.tolist(), "soft": soft_value.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/soft_eikonal_visual_probes.json"))
    parser.add_argument("--width", type=int, default=36); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args(); result = {"configuration": vars(args) | {"out": str(args.out)}, "tasks": {}}
    names = ["spiral", "checkerboard", "periodic_wells", "multiscale_1d", "chirp_1d", "localized_steps_1d", "fourier_mix_1d"]
    for name in names:
        task, mlp, soft = train_pair(name, args.seed, args.width, args.steps)
        result["tasks"][name] = {"parameters": parameter_count(mlp), "expansion": mlp.expansion,
            "budget_remainder": mlp.extra.numel(), "kind": task.kind,
            "plot": one_dimensional(task, mlp, soft) if task.input_dim == 1 else two_dimensional(task, mlp, soft)}
        print(json.dumps({"task": name, "parameters": parameter_count(mlp), "expansion": mlp.expansion,
                          "budget_remainder": mlp.extra.numel()}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
