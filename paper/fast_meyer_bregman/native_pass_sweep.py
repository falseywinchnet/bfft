#!/usr/bin/env python3
"""Measure the pass/accuracy frontier of the fused native Bregman solver."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from skimage import metrics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402


def median_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def main() -> None:
    source_dir = HERE / "results"
    out_dir = HERE / "results" / "pass_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(source_dir / "arrays.npz")
    cases = ["barbara_512", "camera_256", "synthetic_256"]
    passes = [1, 2, 4, 8, 16, 32, 64, 96, 120]
    records = {}
    arrays = {}
    for name in cases:
        source = archive[f"{name}_source"].astype(np.float64)
        reference_u = archive[f"{name}_gilles_u"].astype(np.float64)
        reference_v = archive[f"{name}_gilles_v"].astype(np.float64)
        source_norm = max(float(np.linalg.norm(source)), 1e-30)
        texture_norm = max(float(np.linalg.norm(reference_v)), 1e-30)
        rows = []
        for count in passes:
            plan = bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=count, threads=8
            )
            plan.split_legacy(source)
            samples = []
            output = None
            for _ in range(7):
                start = time.perf_counter()
                output = plan.split_legacy(source)
                samples.append(1000.0 * (time.perf_counter() - start))
            median, mad = median_mad(samples)
            u, v = output
            rows.append(
                {
                    "passes": count,
                    "median_ms": median,
                    "mad_ms": mad,
                    "cartoon_relative_l2_to_gilles": float(
                        np.linalg.norm(u - reference_u) / source_norm
                    ),
                    "texture_relative_l2_to_gilles": float(
                        np.linalg.norm(v - reference_v) / texture_norm
                    ),
                    "cartoon_ssim_to_gilles": float(
                        metrics.structural_similarity(u, reference_u, data_range=255.0)
                    ),
                    "texture_correlation_to_gilles": float(
                        np.corrcoef(v.ravel(), reference_v.ravel())[0, 1]
                    ),
                    "source_recomposition_relative_l2": float(
                        np.linalg.norm(source - u - v) / source_norm
                    ),
                }
            )
            arrays[f"{name}_u_{count}"] = u.astype(np.float32)
            arrays[f"{name}_v_{count}"] = v.astype(np.float32)
        records[name] = rows
    (out_dir / "pass_sweep.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(out_dir / "pass_sweep_arrays.npz", **arrays)
    print(out_dir / "pass_sweep.json")


if __name__ == "__main__":
    main()
