#!/usr/bin/env python3
"""Join preserved self/cone runs to the matched operator-frame battery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SELF = "self_context"
CONE = "self_contextual_full_learned_cone"


def selected(payload, variants):
    return [row for row in payload if row["variant"] in variants]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-results", type=Path,
        default=Path("ML_experiment/results_odd_context_battery_full"),
    )
    parser.add_argument(
        "--cone-results", type=Path,
        default=Path("ML_experiment/results_odd_context_learned_cone_battery"),
    )
    parser.add_argument(
        "--operator-results", type=Path,
        default=Path("ML_experiment/results_operator_frame_full_battery"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("ML_experiment/results_operator_frame_battery_merged"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        json.loads((args.self_results / "results.json").read_text()),
        json.loads((args.cone_results / "results.json").read_text()),
        json.loads((args.operator_results / "results.json").read_text()),
    ]
    configurations = [source["configuration"] for source in sources]
    for key in ("tasks", "width", "seeds", "steps", "batch", "lr", "eval_every", "grid"):
        values = [configuration[key] for configuration in configurations]
        if len(set(values)) != 1:
            raise ValueError(f"configuration mismatch for {key}: {values}")

    operator_variants = configurations[2]["variants"].split(",")
    variants = [
        "ordinary_mlp_self_budget",
        "ordinary_mlp_cone_budget",
        SELF,
        "cff",
        CONE,
        "self_contextual_operator_sphere_global_r2",
        "self_contextual_nested_operator_r2",
    ]
    runs = (
        selected(sources[2]["runs"], operator_variants)
        + selected(sources[0]["runs"], {SELF})
        + selected(sources[1]["runs"], {CONE})
    )
    order = {name: index for index, name in enumerate(variants)}
    task_order = {
        name: index for index, name in enumerate(configurations[0]["tasks"].split(","))
    }
    runs.sort(key=lambda row: (task_order[row["task"]], order[row["variant"]], row["seed"]))

    probe_sources = [
        json.loads((args.self_results / "probes.json").read_text())["probes"],
        json.loads((args.cone_results / "probes.json").read_text())["probes"],
        json.loads((args.operator_results / "probes.json").read_text())["probes"],
    ]
    probes = (
        selected(probe_sources[2], set(operator_variants))
        + selected(probe_sources[0], {SELF})
        + selected(probe_sources[1], {CONE})
    )
    probes.sort(key=lambda row: (task_order[row["task"]], order[row["variant"]]))
    expected = len(task_order) * len(variants)
    if len(runs) != expected or len(probes) != expected:
        raise RuntimeError(
            f"incomplete merged battery: runs={len(runs)}, probes={len(probes)}, expected={expected}"
        )
    configuration = {
        **configurations[0],
        "out": str(args.out),
        "variants": variants,
        "sources": [str(args.self_results), str(args.cone_results), str(args.operator_results)],
    }
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": runs,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({"runs": len(runs), "probes": len(probes), "variants": variants}))


if __name__ == "__main__":
    main()
