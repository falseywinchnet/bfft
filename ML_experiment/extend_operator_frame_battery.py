#!/usr/bin/env python3
"""Append newly measured tasks to the preserved operator-frame battery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path,
        default=Path("ML_experiment/results_operator_frame_battery_merged"),
    )
    parser.add_argument(
        "--extension", type=Path,
        default=Path("ML_experiment/results_sparse_sine_full_battery"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("ML_experiment/results_operator_frame_battery_24"),
    )
    args = parser.parse_args()

    base_results = json.loads((args.base / "results.json").read_text())
    base_probes = json.loads((args.base / "probes.json").read_text())
    extra_results = json.loads((args.extension / "results.json").read_text())
    extra_probes = json.loads((args.extension / "probes.json").read_text())

    variants = list(base_results["configuration"]["variants"])
    measured_variants = extra_results["configuration"]["variants"].split(",")
    if variants != measured_variants:
        raise ValueError(
            f"variant mismatch: base={variants}, extension={measured_variants}"
        )
    for key in ("width", "seeds", "steps", "batch", "lr", "eval_every", "grid"):
        left = base_results["configuration"][key]
        right = extra_results["configuration"][key]
        if left != right:
            raise ValueError(f"configuration mismatch for {key}: {left} != {right}")

    base_tasks = base_results["configuration"]["tasks"].split(",")
    extra_tasks = extra_results["configuration"]["tasks"].split(",")
    overlap = set(base_tasks) & set(extra_tasks)
    if overlap:
        raise ValueError(f"tasks already present in base battery: {sorted(overlap)}")
    tasks = base_tasks + extra_tasks
    runs = base_results["runs"] + extra_results["runs"]
    probes = base_probes["probes"] + extra_probes["probes"]
    expected = len(tasks) * len(variants) * base_results["configuration"]["seeds"]
    if len(runs) != expected or len(probes) != len(tasks) * len(variants):
        raise RuntimeError(
            f"incomplete battery: runs={len(runs)}, probes={len(probes)}, "
            f"expected runs={expected}, probes={len(tasks) * len(variants)}"
        )

    configuration = {
        **base_results["configuration"],
        "out": str(args.out),
        "tasks": ",".join(tasks),
        "sources": [str(args.base), str(args.extension)],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": runs,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({
        "tasks": len(tasks), "variants": len(variants),
        "runs": len(runs), "probes": len(probes), "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
