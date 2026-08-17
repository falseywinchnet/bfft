#!/usr/bin/env python3
"""Train one radial-only dogfood round and export topology-first fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ML_experiment.metrics import predict
from ML_experiment.run_probes import field_probe
from ML_experiment.run_transport_study import train
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import RADIAL_LAB_VARIANTS


@torch.no_grad()
def radial_profile(model, maximum=1.5, radii_count=241, angles_count=128):
    radii = torch.linspace(0, maximum, radii_count)
    angles = torch.arange(angles_count) * (2 * torch.pi / angles_count)
    points = torch.stack((
        radii[:, None] * torch.cos(angles)[None],
        radii[:, None] * torch.sin(angles)[None],
    ), -1)
    probability = torch.softmax(predict(model, points.flatten(0, 1)), 1)[:, 1]
    probability = probability.view(radii_count, angles_count)
    mean_probability = probability.mean(1)
    radial_class = mean_probability >= .5
    truth = torch.floor(4.2 * radii).long().remainder(2).bool()
    transitions = int((radial_class[1:] != radial_class[:-1]).sum())
    truth_transitions = int((truth[1:] != truth[:-1]).sum())
    angular_consistency = torch.maximum(
        (probability >= .5).float().mean(1),
        (probability < .5).float().mean(1),
    )
    return {
        "radius": [round(float(value), 5) for value in radii],
        "class1_probability": [round(float(value), 5) for value in mean_probability],
        "truth": truth.long().tolist(),
        "radial_transitions": transitions,
        "truth_transitions": truth_transitions,
        "profile_accuracy": float((radial_class == truth).float().mean()),
        "angular_consistency": float(angular_consistency.mean()),
        "central_class0_probability": float(1 - mean_probability[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/radial_dogfood.json"))
    parser.add_argument("--variants", default=",".join(RADIAL_LAB_VARIANTS))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--grid", type=int, default=81)
    args = parser.parse_args()
    task = TASK_BUILDERS["radial_stripes"](args.seed)
    rows = []
    for variant in args.variants.split(","):
        model, history, state, seconds, best_step = train(
            variant, task, args.width, args.seed, args.steps, 256, 3e-3, 25
        )
        field = field_probe(task, model, args.grid)
        profile = radial_profile(model)
        row = {
            "variant": variant,
            "seconds": seconds,
            "best_step": best_step,
            "validation_score": max(point["score"] for point in history),
            "history": history,
            "profile": profile,
            "field": field,
            **state,
        }
        rows.append(row)
        print(json.dumps({
            "variant": variant,
            "score": row["validation_score"],
            **{key: value for key, value in profile.items() if not isinstance(value, list)},
        }), flush=True)
    args.out.write_text(json.dumps({
        "configuration": {**vars(args), "out": str(args.out)},
        "runs": rows,
    }))


if __name__ == "__main__":
    main()
