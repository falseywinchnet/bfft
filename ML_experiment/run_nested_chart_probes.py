#!/usr/bin/env python3
"""Generate matched radial and multiscale probes for the nested-chart check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ML_experiment.run_nested_chart_check import CONFIGURATIONS, train
from ML_experiment.run_probes import curve_probe, field_probe
from ML_experiment.tasks import TASK_BUILDERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/nested_chart_probes.json"))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--switch-step", type=int, default=150)
    parser.add_argument("--grid", type=int, default=61)
    args = parser.parse_args()

    rows = []
    for task_name in ("radial_stripes", "multiscale_1d"):
        task = TASK_BUILDERS[task_name](args.seed)
        for configuration in CONFIGURATIONS:
            result, model = train(
                configuration, task, args.width, args.seed, args.steps,
                args.switch_step, 256, 3e-3, 25, return_model=True,
            )
            probe = curve_probe(task, model) if task.input_dim == 1 else field_probe(task, model, args.grid)
            rows.append({
                "task": task_name,
                "configuration": configuration,
                "best_step": result["best_step"],
                **probe,
            })
            print(json.dumps({"task": task_name, "configuration": configuration, "type": probe["type"]}), flush=True)

    args.out.write_text(json.dumps({
        "configuration": {**vars(args), "out": str(args.out)},
        "probes": rows,
    }))


if __name__ == "__main__":
    main()
