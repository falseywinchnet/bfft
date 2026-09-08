#!/usr/bin/env python3
"""Native timing/quality gate for the coupled finite-flow Meyer jump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402


def benchmark_scene(size: int) -> np.ndarray:
    """Dependency-free multiscale crossing used by the synthetic audit."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    xn, yn = x / size, y / size
    source = 88.0 + 13.0 * xn - 7.0 * yn
    circle = (xn - 0.28) ** 2 + (yn - 0.31) ** 2 < 0.145 ** 2
    rectangle = (
        (xn > 0.56) & (xn < 0.87) & (yn > 0.55) & (yn < 0.82)
    )
    source = source + 76.0 * circle - 49.0 * rectangle
    coarse_radius = np.sqrt(
        ((xn - 0.43) / 0.36) ** 2 + ((yn - 0.58) / 0.31) ** 2
    )
    coarse_mask = np.clip((1.0 - coarse_radius) / 0.12, 0.0, 1.0)
    fine_distance = np.minimum.reduce((
        xn - 0.37, 0.94 - xn, yn - 0.12, 0.66 - yn,
    ))
    fine_mask = np.clip(fine_distance / 0.055, 0.0, 1.0)
    source += 19.0 * coarse_mask * np.cos(
        2.0 * np.pi * (x + 0.42 * y) / 17.0 + 0.31
    )
    source += 11.0 * fine_mask * (
        np.cos(2.0 * np.pi * (x - 1.35 * y) / 7.0 - 0.47)
        + 0.35 * np.cos(2.0 * np.pi * (x + 0.75 * y) / 5.0 + 0.19)
    )
    return source


def timing(operation, warmups: int, repeats: int) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def run_size(size: int, threads: int, warmups: int, repeats: int) -> dict:
    source = benchmark_scene(size)
    plans = {
        "hard_jump": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=threads
        ),
        "flow_fast": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=threads
        ),
        "flow_quality": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=threads
        ),
        "fused_19": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=19, threads=threads
        ),
        "fused_29": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=29, threads=threads
        ),
        "fused_64": bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=threads
        ),
    }
    operations = {
        "hard_jump": lambda: plans["hard_jump"].split_jump_measure(source),
        "flow_fast": lambda: plans["flow_fast"].split_flow_jump(
            source, prefix_passes=4, horizon=18,
            settle_passes=2, jump_count=3),
        "flow_quality": lambda: plans["flow_quality"].split_flow_jump(source),
        "fused_19": lambda: plans["fused_19"].split_legacy(source),
        "fused_29": lambda: plans["fused_29"].split_legacy(source),
        "fused_64": lambda: plans["fused_64"].split_legacy(source),
    }
    outputs = {name: operation() for name, operation in operations.items()}
    reference = outputs["fused_64"][1]
    measurements = {
        name: {
            **timing(operation, warmups, repeats),
            "texture_error_rms_to_fused64": rms(outputs[name][1] - reference),
            "recomposition_linf": float(np.max(np.abs(
                source - outputs[name][0] - outputs[name][1]
            ))),
        }
        for name, operation in operations.items()
    }
    for name in ("flow_fast", "flow_quality"):
        measurements[name]["speedup_vs_fused64"] = (
        measurements["fused_64"]["median_ms"]
        / measurements[name]["median_ms"]
        )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "experiments" / "out" / "meyer_flow_jump_benchmark.json",
    )
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    report = {
        "schedules": {
            "flow_fast": {
                "prefix_passes": 4, "arnoldi_depth": 2,
                "horizon": 18, "settle_passes": 2, "jump_count": 3,
                "exact_pass_equivalent_cost": 19,
            },
            "flow_quality": {
                "prefix_passes": 4, "arnoldi_depth": 2,
                "horizon": 10, "settle_passes": 2, "jump_count": 5,
                "exact_pass_equivalent_cost": 29,
            },
        },
        "threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "sizes": {},
    }
    for size in sizes:
        report["sizes"][str(size)] = run_size(
            size, args.threads, args.warmups, args.repeats
        )
        row = report["sizes"][str(size)]
        print(
            f"{size:4d}: hard={row['hard_jump']['median_ms']:.3f} ms "
            f"fast={row['flow_fast']['median_ms']:.3f} ms "
            f"quality={row['flow_quality']['median_ms']:.3f} ms "
            f"fused19={row['fused_19']['median_ms']:.3f} ms "
            f"fused29={row['fused_29']['median_ms']:.3f} ms "
            f"fused64={row['fused_64']['median_ms']:.3f} ms "
            f"speedup={row['flow_fast']['speedup_vs_fused64']:.2f}x/"
            f"{row['flow_quality']['speedup_vs_fused64']:.2f}x"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
