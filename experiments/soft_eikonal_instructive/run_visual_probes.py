#!/usr/bin/env python3
"""Train selected variants and export compact prediction fields for plotting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.soft_eikonal_instructive.models import PairZeroEvaluation
from experiments.soft_eikonal_instructive.run_screen import train_variant
from experiments.soft_eikonal_matched.metrics import predict
from experiments.soft_eikonal_matched.tasks import TASK_BUILDERS


def physical(task, value):
    if task.target_mean is None:
        return value
    return value * task.target_std + task.target_mean


def one_dimensional_probe(task, evaluator):
    x = torch.linspace(-5.0, 5.0, 801)[:, None]
    prediction = physical(task, predict(evaluator, x)).squeeze(1)
    truth = task.truth(x).squeeze(1)
    return {"type": "curve", "x": x.squeeze(1).tolist(),
            "prediction": prediction.tolist(), "truth": truth.tolist(), "train_limit": 3.0}


def two_dimensional_probe(task, evaluator, size):
    xmin, xmax, ymin, ymax = task.visual_limits
    xs = torch.linspace(xmin, xmax, size); ys = torch.linspace(ymin, ymax, size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx.flatten(), yy.flatten()), 1); output = predict(evaluator, points)
    if task.kind == "classification":
        field = torch.softmax(output, 1)[:, 1]; truth = task.truth(points).float()
    else:
        field = physical(task, output).squeeze(1); truth = task.truth(points).squeeze(1)
    return {"type": "field", "size": size, "limits": list(task.visual_limits),
            "field": field.tolist(), "truth": truth.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/soft_eikonal_instructive_probes.json"))
    parser.add_argument("--tasks", default="checkerboard,ripple,multiscale_1d,localized_steps_1d")
    parser.add_argument("--variants", default="soft_eikonal,self_context,secant_relational,temperature_hard")
    parser.add_argument("--width", type=int, default=36); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800); parser.add_argument("--grid", type=int, default=91)
    args = parser.parse_args()
    payload = {"configuration": {**vars(args), "out": str(args.out)}, "probes": []}
    for task_name in args.tasks.split(","):
        task = TASK_BUILDERS[task_name](args.seed)
        for variant in args.variants.split(","):
            model, evaluator, history, seconds, best_step = train_variant(
                variant, task, args.width, args.seed, args.steps, 256, 3e-3, 50)
            if variant == "paired_zero" and not isinstance(evaluator, PairZeroEvaluation):
                evaluator = PairZeroEvaluation(model)
            probe = (one_dimensional_probe(task, evaluator) if task.input_dim == 1
                     else two_dimensional_probe(task, evaluator, args.grid))
            payload["probes"].append({"task": task_name, "variant": variant,
                                      "seconds": seconds, "best_step": best_step,
                                      "history": history, **probe})
            print(json.dumps({"task": task_name, "variant": variant, "seconds": seconds}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
