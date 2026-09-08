#!/usr/bin/env python3
"""Current finite-flow methods on the archived Gilles comparison sources.

The expensive cold nested Gilles run is immutable input.  Its texture is
retained, but its model survivor is folded into the effective cartoon
``f-v`` so every displayed method is an exact two-product decomposition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402


def timed(call, warmups: int, repeats: int):
    result = None
    for _ in range(warmups):
        result = call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    median = statistics.median(samples)
    return result, {
        "median_ms": median,
        "mad_ms": statistics.median(abs(x - median) for x in samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "warmups": warmups,
        "repeats": repeats,
        "samples_ms": samples,
    }


def effective_fused(plan: bfft.MeyerPlan, source: np.ndarray):
    _, texture = plan.split_legacy(source)
    return source - texture, texture


def metrics(source, candidate, reference):
    cu, cv = candidate
    ru, rv = reference
    source_norm = max(float(np.linalg.norm(source)), 1e-30)
    texture_norm = max(float(np.linalg.norm(rv)), 1e-30)
    return {
        "cartoon_relative_l2": float(np.linalg.norm(cu - ru) / source_norm),
        "texture_relative_l2": float(np.linalg.norm(cv - rv) / texture_norm),
        "texture_correlation": float(np.corrcoef(
            cv.ravel(), rv.ravel()
        )[0, 1]),
        "recomposition_linf": float(np.max(np.abs(source - cu - cv))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--archive", type=Path, default=HERE / "results" / "arrays.npz"
    )
    parser.add_argument(
        "--historical", type=Path,
        default=HERE / "results" / "benchmark.json"
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "flow_natural"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    old = np.load(args.archive)
    historical = json.loads(args.historical.read_text())
    names = sorted(
        key[:-7] for key in old.files if key.endswith("_source")
    )
    arrays: dict[str, np.ndarray] = {}
    report = {
        "comparison_contract": (
            "All outputs are exact two-product pairs (f-v, v). The archived "
            "cold nested Gilles texture is unchanged; its survivor f-u-v is "
            "folded into effective cartoon before comparison."
        ),
        "reference": "optimized fused 64-pass effective decomposition",
        "methods": {
            "hard_jump": {"role": "negative control"},
            "flow_fast": {
                "schedule": [4, 18, 2, 3], "operator_pass_cost": 19
            },
            "fused_19": {"operator_pass_cost": 19},
            "flow_quality": {
                "schedule": [4, 10, 2, 5], "operator_pass_cost": 29
            },
            "fused_29": {"operator_pass_cost": 29},
            "fused_64": {"operator_pass_cost": 64},
            "gilles_nested": {
                "role": "historical cold nested reference",
                "timing_source": str(args.historical),
            },
        },
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threads": args.threads,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "lambda": 0.05,
            "mu": 40.0,
            "content_dependent_rules": "none",
        },
        "scenes": {},
    }

    for name in names:
        source = np.ascontiguousarray(old[f"{name}_source"], dtype=np.float64)
        gilles_v = np.asarray(old[f"{name}_gilles_v"], dtype=np.float64)
        gilles = (source - gilles_v, gilles_v)
        one = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1,
            threads=args.threads
        )
        fused = {
            p: bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=p,
                threads=args.threads
            )
            for p in (19, 29, 64)
        }
        calls = {
            "hard_jump": lambda: one.split_jump_measure(source),
            "flow_fast": lambda: one.split_flow_jump(
                source, prefix_passes=4, horizon=18,
                settle_passes=2, jump_count=3
            ),
            "fused_19": lambda: effective_fused(fused[19], source),
            "flow_quality": lambda: one.split_flow_jump(source),
            "fused_29": lambda: effective_fused(fused[29], source),
            "fused_64": lambda: effective_fused(fused[64], source),
        }
        outputs = {"gilles_nested": gilles}
        rows = {
            "gilles_nested": {
                "timing": historical[name]["gilles_nested"],
            }
        }
        for method, call in calls.items():
            outputs[method], timing = timed(
                call, args.warmups, args.repeats
            )
            rows[method] = {"timing": timing}

        fused64 = outputs["fused_64"]
        for method, output in outputs.items():
            rows[method]["to_fused64"] = metrics(
                source, output, fused64
            )
            rows[method]["to_gilles_nested"] = metrics(
                source, output, gilles
            )
            arrays[f"{name}_{method}_cartoon"] = np.asarray(
                output[0], dtype=np.float32
            )
            arrays[f"{name}_{method}_texture"] = np.asarray(
                output[1], dtype=np.float32
            )
        arrays[f"{name}_source"] = source.astype(np.float32)
        report["scenes"][name] = rows
        print(name, flush=True)
        for method in calls:
            row = rows[method]
            print(
                f"  {method:14s} {row['timing']['median_ms']:9.3f} ms  "
                f"error64={row['to_fused64']['texture_relative_l2']:.5f}",
                flush=True,
            )

    np.savez_compressed(args.out / "arrays.npz", **arrays)
    (args.out / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
