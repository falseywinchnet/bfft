#!/usr/bin/env python3
"""Measure whether fitted allocation, rather than the dense base, carries gains."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.soft_eikonal_instructive.run_screen import train_variant
from experiments.soft_eikonal_matched.metrics import soft_diagnostics
from experiments.soft_eikonal_matched.tasks import TASK_BUILDERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/soft_eikonal_instructive_diagnostics.json"))
    parser.add_argument("--tasks", default="radial_stripes,ripple,multiscale_1d")
    parser.add_argument("--variants", default="soft_eikonal,self_context")
    parser.add_argument("--seeds", type=int, default=3); parser.add_argument("--width", type=int, default=36)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args(); rows = []
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for variant in args.variants.split(","):
                _, evaluator, _, seconds, best_step = train_variant(
                    variant, task, args.width, seed, args.steps, 256, 3e-3, 50)
                row = {"task": task_name, "seed": seed, "variant": variant,
                       "seconds": seconds, "best_step": best_step,
                       **soft_diagnostics(evaluator, task)}
                rows.append(row); print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"configuration": {**vars(args), "out": str(args.out)}, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
