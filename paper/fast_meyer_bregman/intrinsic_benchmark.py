#!/usr/bin/env python3
"""Finite-flow Meyer jump against authored truth and the fused 64-pass state.

Every method is applied unchanged to every scene.  The authored components
measure absolute allocation quality; the optimized fused 64-pass result is the
algorithmic reference whose finite trajectory the jump compresses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft
from experiments.meyer_first_pass_conditioning import checker_support_scene
from experiments.meyer_capacity_route_ablation import score, truth
from experiments.meyer_deep_jump_causal_audit import (
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


def git_revision() -> str:
    override = os.environ.get("BFFT_GIT_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable-in-mirror"


def timed(function, warmups: int, repeats: int):
    result = None
    for _ in range(warmups):
        result = function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append(1000.0 * (time.perf_counter() - started))
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return result, {
        "median_ms": median,
        "mad_ms": mad,
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "warmups": warmups,
        "repeats": repeats,
        "samples_ms": samples,
    }


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "intrinsic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = (
        pure_edge_scene(args.size),
        pure_carrier_scene(args.size),
        symmetric_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        checker_support_scene(args.size),
        junction_texture_scene(args.size),
    )
    report = {
        "quality_basis": (
            "analytically authored cartoon/texture truth plus the optimized "
            "fused 64-pass trajectory reference"
        ),
        "scene_roles": {
            "causal_controls": ["pure_edge", "pure_carrier"],
            "material_support": [
                "symmetric_support", "checker_support"
            ],
            "crossing_material": [
                "multiscale_crossing", "junction_texture"
            ],
        },
        "interpretation": (
            "The old scalar hard jump, two fixed finite-flow schedules, and "
            "ordinary fused controls are applied without scene-dependent "
            "selection. Pure edge and pure carrier isolate halo and retained-"
            "carrier defects; four compound scenes test their superposition."
        ),
        "method_order": [
            "hard_jump",
            "flow_fast",
            "fused_pass_19",
            "flow_quality",
            "fused_pass_29",
            "fused_pass_64",
        ],
        "scenes": {},
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_revision": git_revision(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threads": args.threads,
            "lambda": 0.05,
            "mu": 40.0,
            "flow_fast_schedule": {
                "prefix_passes": 4, "horizon": 18,
                "settle_passes": 2, "jump_count": 3,
                "operator_pass_cost": 19,
            },
            "flow_quality_schedule": {
                "prefix_passes": 4, "horizon": 10,
                "settle_passes": 2, "jump_count": 5,
                "operator_pass_cost": 29,
            },
            "content_dependent_rules": "none",
        },
    }
    arrays: dict[str, np.ndarray] = {}

    for scene in scenes:
        name = scene["name"]
        source = np.ascontiguousarray(scene["source"], dtype=np.float64)
        one = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=args.threads
        )
        fused_plans = {
            passes: bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0,
                passes=passes, threads=args.threads
            )
            for passes in (19, 29, 64)
        }

        def fused_effective(passes: int):
            _, texture = fused_plans[passes].split_legacy(source)
            return source - texture, texture

        calls = {
            "hard_jump": (lambda: one.split_jump_measure(source), 2, 30),
            "flow_fast": (
                lambda: one.split_flow_jump(
                    source, prefix_passes=4, horizon=18,
                    settle_passes=2, jump_count=3
                ),
                2, 30,
            ),
            "fused_pass_19": (lambda: fused_effective(19), 2, 20),
            "flow_quality": (lambda: one.split_flow_jump(source), 2, 20),
            "fused_pass_29": (lambda: fused_effective(29), 2, 20),
            "fused_pass_64": (lambda: fused_effective(64), 2, 10),
        }
        rows = {}
        outputs = {}
        for method, (call, warmups, repeats) in calls.items():
            split, timing = timed(call, warmups, repeats)
            outputs[method] = split
            metrics = score(*split, scene)
            if "cartoon" in scene and "texture_interior" in scene:
                metrics.update(score_split(*split, scene))
            rows[method] = {"timing": timing, **metrics}
        reference_texture = outputs["fused_pass_64"][1]
        reference_scale = max(rms(reference_texture), 1e-12)
        for method, (_, texture) in outputs.items():
            rows[method]["texture_rms_error_to_fused64"] = rms(
                texture - reference_texture
            )
            rows[method]["texture_relative_error_to_fused64"] = (
                rows[method]["texture_rms_error_to_fused64"]
                / reference_scale
            )
        report["scenes"][name] = rows

        arrays[f"{name}_source"] = source.astype(np.float32)
        truth_cartoon, truth_texture = truth(scene)
        arrays[f"{name}_truth_cartoon"] = np.asarray(
            truth_cartoon, dtype=np.float32
        )
        arrays[f"{name}_truth_texture"] = np.asarray(
            truth_texture, dtype=np.float32
        )
        for method, (cartoon, texture) in outputs.items():
            arrays[f"{name}_{method}_cartoon"] = np.asarray(
                cartoon, dtype=np.float32
            )
            arrays[f"{name}_{method}_texture"] = np.asarray(
                texture, dtype=np.float32
            )

        print(name, flush=True)
        for method in report["method_order"]:
            row = rows[method]
            allocation = row.get("texture_over_contour_allocation_auc")
            suffix = "" if allocation is None else f"  AUC {allocation:.4f}"
            print(
                f"  {method:27s} {row['timing']['median_ms']:8.3f} ms  "
                f"tex {row['texture_relative_rms_error']:.4f}  "
                f"cart {row['cartoon_relative_rms_error']:.4f}{suffix}",
                flush=True,
            )

    (args.out / "intrinsic_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(args.out / "intrinsic_arrays.npz", **arrays)
    print(args.out / "intrinsic_benchmark.json")


if __name__ == "__main__":
    main()
