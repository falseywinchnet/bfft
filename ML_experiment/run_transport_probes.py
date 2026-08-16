#!/usr/bin/env python3
"""Generate focal probes for the self-context transport study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ML_experiment.run_probes import curve_probe, field_probe
from ML_experiment.run_transport_study import train
from ML_experiment.tasks import TASK_BUILDERS


VARIANTS = (
    "self_context",
    "self_context_transport_self_ray_odd",
    "self_context_jet_curvature_context",
    "self_context_jet_curvature_bounded",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/transport_probes.json"))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--grid", type=int, default=61)
    args = parser.parse_args()

    rows = []
    for task_name in ("radial_stripes", "multiscale_1d"):
        task = TASK_BUILDERS[task_name](args.seed)
        for variant in VARIANTS:
            model, history, state, seconds, best_step = train(
                variant, task, args.width, args.seed, args.steps, 256, 3e-3, 25
            )
            probe = curve_probe(task, model) if task.input_dim == 1 else field_probe(task, model, args.grid)
            rows.append({
                "task": task_name,
                "variant": variant,
                "best_step": best_step,
                "seconds": seconds,
                **state,
                **probe,
            })
            print(json.dumps({"task": task_name, "variant": variant, "type": probe["type"]}), flush=True)
    args.out.write_text(json.dumps({
        "configuration": {**vars(args), "out": str(args.out)},
        "probes": rows,
    }))


if __name__ == "__main__":
    main()
