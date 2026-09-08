#!/usr/bin/env python3
"""Reproducible Gilles-vs-native Meyer decomposition benchmark.

The benchmark keeps three algorithms separate:

1. ``gilles_nested`` is Algorithm 3 of Gilles--Osher with cold, converged
   Split Bregman inner solves (the repository's direct NumPy/SciPy oracle).
2. ``fused_interleaved_64`` is the July 2026 warm, one-sweep-per-block C++
   realization run for 64 outer passes.
3. ``native_jump_measure`` is the current fixed-cost C++ operator.  It is a
   compiled transport construction, not a claim to be an iterate-identical
   solver for the nested algorithm.

Images and complete numeric arrays are written beside the JSON record so the
paper figures can be regenerated without rerunning the expensive reference.
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
from PIL import Image
from scipy import ndimage
from skimage import data, metrics, transform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft
from experiments.meyer_bregman import SWEEPS, a2bc_cold


ASSETS = HERE / "assets"


def _git_revision() -> str:
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


def _median_mad(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return median, mad


def _time_call(function, warmups: int, repeats: int):
    result = None
    for _ in range(warmups):
        result = function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(1000.0 * (time.perf_counter() - start))
    median, mad = _median_mad(samples)
    return result, {
        "median_ms": median,
        "mad_ms": mad,
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "repeats": repeats,
        "warmups": warmups,
        "samples_ms": samples,
    }


def _gray(image: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 3:
        value = 0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]
    if value.max() <= 1.0:
        value *= 255.0
    if value.shape != (size, size):
        value = transform.resize(
            value, (size, size), order=3, anti_aliasing=True, preserve_range=True
        )
    return np.ascontiguousarray(value, dtype=np.float64)


def _synthetic(size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    x, y = (xx + 0.5) / size, (yy + 0.5) / size
    cartoon = 38.0 + 94.0 * x + 31.0 * (y > 0.65 - 0.28 * x)
    cartoon += 58.0 * ((x - 0.70) ** 2 + (y - 0.32) ** 2 < 0.155**2)
    window_a = np.exp(-((x - 0.34) ** 2 / 0.070 + (y - 0.58) ** 2 / 0.11) ** 4)
    window_b = np.exp(-((x - 0.70) ** 2 / 0.090 + (y - 0.70) ** 2 / 0.08) ** 4)
    texture = 19.0 * np.sin(2.0 * np.pi * (18.0 * x + 6.0 * y)) * window_a
    texture += 15.0 * np.sin(2.0 * np.pi * (4.0 * x + 21.0 * y)) * window_b
    return np.ascontiguousarray(cartoon + texture)


def _cases(quick: bool) -> dict[str, np.ndarray]:
    barbara = np.asarray(Image.open(ASSETS / "barbara_512.tif"), dtype=np.float64)
    if quick:
        barbara = _gray(barbara, 256)
        return {"barbara_256": barbara, "synthetic_256": _synthetic(256)}
    return {
        "barbara_512": barbara,
        "camera_256": _gray(data.camera(), 256),
        "synthetic_256": _synthetic(256),
    }


def _quality(source: np.ndarray, candidate, reference) -> dict[str, float]:
    cu, cv = candidate
    ru, rv = reference
    source_norm = max(float(np.linalg.norm(source)), 1e-30)
    texture_norm = max(float(np.linalg.norm(rv)), 1e-30)
    return {
        "cartoon_relative_l2_to_gilles": float(np.linalg.norm(cu - ru) / source_norm),
        "texture_relative_l2_to_gilles": float(np.linalg.norm(cv - rv) / texture_norm),
        "cartoon_ssim_to_gilles": float(metrics.structural_similarity(cu, ru, data_range=255.0)),
        "texture_correlation_to_gilles": float(np.corrcoef(cv.ravel(), rv.ravel())[0, 1]),
        "source_recomposition_relative_l2": float(
            np.linalg.norm(source - cu - cv) / source_norm
        ),
        "texture_rms": float(np.sqrt(np.mean(cv * cv))),
    }


def _reference(source: np.ndarray, outer_tol: float, inner_tol: float):
    SWEEPS["n"] = 0
    start = time.perf_counter()
    u, v, outer_iterations, _ = a2bc_cold(
        source,
        0.05,
        40.0,
        outer_tol=outer_tol,
        inner_tol=inner_tol,
        max_outer=120,
    )
    elapsed = time.perf_counter() - start
    return (u, v), {
        "elapsed_ms": 1000.0 * elapsed,
        "outer_iterations": outer_iterations,
        "inner_sweeps": int(SWEEPS["n"]),
        "outer_tolerance": outer_tol,
        "inner_tolerance": inner_tol,
        "implementation": "NumPy/SciPy rfft2, cold nested Split Bregman",
    }


def _native(source: np.ndarray, threads: int):
    creation_start = time.perf_counter()
    plan = bfft.MeyerPlan(source.shape, lam=0.05, mu=40.0, passes=64, threads=threads)
    creation_ms = 1000.0 * (time.perf_counter() - creation_start)
    fixed, fixed_timing = _time_call(lambda: plan.split(source), 3, 15)
    legacy, legacy_timing = _time_call(lambda: plan.split_legacy(source), 2, 7)
    fixed_timing["plan_creation_ms"] = creation_ms
    legacy_timing["plan_creation_ms"] = creation_ms
    return fixed, fixed_timing, legacy, legacy_timing


def _timing_scaling(threads: int) -> dict[str, dict[str, float]]:
    records = {}
    for size in (128, 256, 512, 1024):
        source = _synthetic(size)
        plan = bfft.MeyerPlan(source.shape, lam=0.05, mu=40.0, passes=64, threads=threads)
        _, fixed = _time_call(
            lambda: plan.split(source), 3, 15 if size < 1024 else 7
        )
        _, legacy = _time_call(
            lambda: plan.split_legacy(source), 1, 5 if size < 1024 else 3
        )
        records[str(size)] = {
            "pixels": size * size,
            "native_jump_measure_median_ms": fixed["median_ms"],
            "native_jump_measure_mad_ms": fixed["mad_ms"],
            "fused_interleaved_64_median_ms": legacy["median_ms"],
            "fused_interleaved_64_mad_ms": legacy["mad_ms"],
        }
    return records


def _save_figures(out: Path, cases: dict[str, dict], arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in cases if not name.startswith("_")]
    fig, axes = plt.subplots(len(names), 7, figsize=(14.0, 2.45 * len(names)), squeeze=False)
    column_titles = (
        "source",
        "Gilles cartoon",
        "current C++ cartoon",
        "|cartoon difference| x8",
        "Gilles texture",
        "current C++ texture",
        "July fused cartoon",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=9)
    for row, name in enumerate(names):
        source = arrays[f"{name}_source"]
        gu, gv = arrays[f"{name}_gilles_u"], arrays[f"{name}_gilles_v"]
        fu, fv = arrays[f"{name}_fixed_u"], arrays[f"{name}_fixed_v"]
        lu = arrays[f"{name}_legacy_u"]
        texture_scale = max(float(np.percentile(np.abs(np.r_[gv.ravel(), fv.ravel()]), 99.5)), 1.0)
        panels = (
            (source, "gray", 0.0, 255.0),
            (gu, "gray", 0.0, 255.0),
            (fu, "gray", 0.0, 255.0),
            (8.0 * np.abs(fu - gu), "magma", 0.0, 80.0),
            (gv, "gray", -texture_scale, texture_scale),
            (fv, "gray", -texture_scale, texture_scale),
            (lu, "gray", 0.0, 255.0),
        )
        for col, (value, cmap, lo, hi) in enumerate(panels):
            axes[row, col].imshow(value, cmap=cmap, vmin=lo, vmax=hi)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name.replace("_", "\n"), fontsize=9)
    fig.tight_layout(pad=0.55)
    fig.savefig(out / "comparison_grid.png", dpi=220)
    plt.close(fig)

    barbara_name = next(name for name in names if name.startswith("barbara"))
    source = arrays[f"{barbara_name}_source"]
    gu, gv = arrays[f"{barbara_name}_gilles_u"], arrays[f"{barbara_name}_gilles_v"]
    fu, fv = arrays[f"{barbara_name}_fixed_u"], arrays[f"{barbara_name}_fixed_v"]
    h, w = source.shape
    crop = (slice(int(0.48 * h), int(0.98 * h)), slice(0, int(0.55 * w)))
    texture_scale = max(float(np.percentile(np.abs(np.r_[gv[crop].ravel(), fv[crop].ravel()]), 99.5)), 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8))
    for axis, title, value, cmap, lo, hi in (
        (axes[0, 0], "source textile crop", source[crop], "gray", 0.0, 255.0),
        (axes[0, 1], "Gilles cartoon", gu[crop], "gray", 0.0, 255.0),
        (axes[0, 2], "current C++ cartoon", fu[crop], "gray", 0.0, 255.0),
        (axes[1, 0], "|cartoon difference| x8", 8.0 * np.abs(fu[crop] - gu[crop]), "magma", 0.0, 80.0),
        (axes[1, 1], "Gilles texture", gv[crop], "gray", -texture_scale, texture_scale),
        (axes[1, 2], "current C++ texture", fv[crop], "gray", -texture_scale, texture_scale),
    ):
        axis.imshow(value, cmap=cmap, vmin=lo, vmax=hi)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout(pad=0.7)
    fig.savefig(out / "barbara_textile_crop.png", dpi=240)
    plt.close(fig)

    scaling = cases["_scaling"]
    sizes = np.array([int(value) for value in scaling])
    fixed = np.array([scaling[str(value)]["native_jump_measure_median_ms"] for value in sizes])
    legacy = np.array([scaling[str(value)]["fused_interleaved_64_median_ms"] for value in sizes])
    fig, axis = plt.subplots(figsize=(6.3, 3.8))
    axis.loglog(sizes * sizes, fixed, "o-", label="current fixed-cost C++")
    axis.loglog(sizes * sizes, legacy, "s-", label="July fused, 64 passes")
    axis.set_xlabel("pixels")
    axis.set_ylabel("median wall time (ms)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "native_scaling.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--outer-tol", type=float, default=1e-5)
    parser.add_argument("--inner-tol", type=float, default=1e-5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sources = _cases(args.quick)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for name, source in sources.items():
        print(f"[{name}] Gilles nested reference", flush=True)
        reference, reference_timing = _reference(source, args.outer_tol, args.inner_tol)
        print(f"[{name}] native methods", flush=True)
        fixed, fixed_timing, legacy, legacy_timing = _native(source, args.threads)
        records[name] = {
            "shape": list(source.shape),
            "gilles_nested": reference_timing,
            "native_jump_measure": fixed_timing,
            "fused_interleaved_64": legacy_timing,
            "native_jump_measure_quality": _quality(source, fixed, reference),
            "fused_interleaved_64_quality": _quality(source, legacy, reference),
            "speedup_native_vs_gilles": reference_timing["elapsed_ms"] / fixed_timing["median_ms"],
            "speedup_native_vs_fused_64": legacy_timing["median_ms"] / fixed_timing["median_ms"],
        }
        for label, value in (
            ("source", source),
            ("gilles_u", reference[0]),
            ("gilles_v", reference[1]),
            ("fixed_u", fixed[0]),
            ("fixed_v", fixed[1]),
            ("legacy_u", legacy[0]),
            ("legacy_v", legacy[1]),
        ):
            arrays[f"{name}_{label}"] = np.asarray(value, dtype=np.float32)

    print("[scaling] native C++", flush=True)
    scaling = _timing_scaling(args.threads)
    records["_scaling"] = scaling
    records["_metadata"] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads_native": args.threads,
        "barbara_sha256": subprocess.check_output(
            ["shasum", "-a", "256", str(ASSETS / "barbara_512.tif")], text=True
        ).split()[0],
        "gilles_parameters": {"lambda": 0.05, "mu": 40.0},
        "native_parameters": {"lambda": 0.05, "mu": 40.0, "virtual_depth": 8},
        "note": "Gilles timing is a cold nested reference; native timings exclude plan construction and report it separately.",
    }
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    (args.out / "benchmark.json").write_text(json.dumps(records, indent=2) + "\n")
    _save_figures(args.out, records, arrays)
    print(args.out / "benchmark.json")


if __name__ == "__main__":
    main()
