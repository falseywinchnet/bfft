#!/usr/bin/env python3
"""Measure halo, oscillation, and displaced residue in known-path recovery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from denoiser.run_2d_denoiser_battery import sources, ssim

from .decomposition import two_stage_deblur_known
from .kernels import (
    curved_path_kernel,
    gaussian_kernel,
    line_kernel,
    path_kernel,
)
from .synthetic import degrade, random_camera_path


def _luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return value
    return value @ np.asarray((0.2126, 0.7152, 0.0722))


def _local_shift_spread(
    image: np.ndarray,
    truth: np.ndarray,
    *,
    cells: int = 4,
) -> dict[str, float]:
    first = _luminance(image)
    second = _luminance(truth)
    height, width = first.shape
    shifts = []
    weights = []
    for row in range(cells):
        for column in range(cells):
            y0, y1 = row * height // cells, (row + 1) * height // cells
            x0, x1 = column * width // cells, (column + 1) * width // cells
            a = first[y0:y1, x0:x1]
            b = second[y0:y1, x0:x1]
            window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
            af = np.fft.fft2((a - np.mean(a)) * window)
            bf = np.fft.fft2((b - np.mean(b)) * window)
            cross = af * np.conj(bf)
            cross /= np.maximum(np.abs(cross), 1e-12)
            correlation = np.abs(np.fft.ifft2(cross))
            py, px = np.unravel_index(int(np.argmax(correlation)), a.shape)
            dy = float(py if py <= a.shape[0] // 2 else py - a.shape[0])
            dx = float(px if px <= a.shape[1] // 2 else px - a.shape[1])
            shifts.append((dx, dy))
            weights.append(float(np.mean(ndimage.sobel(b, axis=0) ** 2
                                         + ndimage.sobel(b, axis=1) ** 2)))
    shift = np.asarray(shifts, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64) + 1e-12
    weight /= np.sum(weight)
    center = np.sum(weight[:, None] * shift, axis=0)
    spread = np.sqrt(np.sum(weight * np.sum((shift - center) ** 2, axis=1)))
    return {
        "weighted_mean_shift_x": float(center[0]),
        "weighted_mean_shift_y": float(center[1]),
        "weighted_local_shift_spread": float(spread),
    }


def _metrics(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(image, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    error = candidate - reference
    axes_sigma = (3.0, 3.0) if error.ndim == 2 else (3.0, 3.0, 0.0)
    fine_sigma = (1.0, 1.0) if error.ndim == 2 else (1.0, 1.0, 0.0)
    low = ndimage.gaussian_filter(error, sigma=axes_sigma, mode="reflect")
    fine = error - ndimage.gaussian_filter(error, sigma=fine_sigma, mode="reflect")
    mse = float(np.mean(error ** 2))
    if reference.ndim == 2:
        structural = float(ssim(reference, candidate))
    else:
        structural = float(np.mean([
            ssim(reference[..., channel], candidate[..., channel])
            for channel in range(reference.shape[2])
        ]))
    return {
        "psnr": float(-10.0 * math.log10(max(mse, np.finfo(float).tiny))),
        "ssim": structural,
        "broad_halo_error_rms": float(np.sqrt(np.mean(low ** 2))),
        "oscillatory_error_rms": float(np.sqrt(np.mean(fine ** 2))),
        **_local_shift_spread(candidate, reference),
    }


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _save_image(path: Path, image: np.ndarray) -> None:
    value = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    Image.fromarray(np.round(255.0 * value).astype(np.uint8)).save(path)


def _benchmark_sources(size: int) -> dict[str, np.ndarray]:
    result = dict(sources(size))
    step = np.zeros((size, size), dtype=np.float64)
    step[:, size // 2:] = 1.0
    result["canonical vertical step"] = step
    beads = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]
    for row, y in enumerate(np.linspace(12, size - 13, 4)):
        for column, x in enumerate(np.linspace(12, size - 13, 4)):
            amplitude = 0.55 + 0.45 * ((row + column) % 3) / 2.0
            beads += amplitude * np.exp(-0.5 * (
                ((xx - x) / 1.25) ** 2 + ((yy - y) / 1.25) ** 2))
    result["canonical sparse beads"] = np.clip(beads, 0.0, 1.0)
    return result


def run(
    size: int = 96,
    passes: int = 64,
    coverage_floor: float = 5e-4,
    local_constancy_floor: float = 0.004,
    basin_uncertainty_weight: float = 0.0,
    save_dir: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    kernels = {
        "gaussian": gaussian_kernel(2.0),
        "line": line_kernel(11.0, 30.0),
        "curve": curved_path_kernel(11.0, 30.0, 4.0),
        "random": path_kernel(
            random_camera_path(extent=7.0, samples=31, seed=17),
            name="random_path_extent_7_seed_17",
        ),
    }
    rows: list[dict[str, object]] = []
    source_images = _benchmark_sources(size)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for source_index, (source_name, truth) in enumerate(source_images.items()):
        for kernel_index, (kind, kernel) in enumerate(kernels.items()):
            observation = degrade(
                truth,
                kernel,
                gaussian_sigma=0.002,
                seed=17000 + 101 * source_index + kernel_index,
                boundary="reflect",
            )
            positive = two_stage_deblur_known(
                observation,
                kernel,
                passes=passes,
                coverage_floor=coverage_floor,
                local_constancy_floor=local_constancy_floor,
                path_authority_scale=0.0,
            )
            unified = two_stage_deblur_known(
                observation,
                kernel,
                passes=passes,
                coverage_floor=coverage_floor,
                local_constancy_floor=local_constancy_floor,
                path_basin_uncertainty_weight=basin_uncertainty_weight,
            )
            if save_dir is not None and source_name.startswith("canonical"):
                stem = source_name.removeprefix("canonical ").replace(" ", "_")
                _save_image(save_dir / f"{stem}_{kind}_truth.png", truth)
                for method, image in (
                    ("observation", observation),
                    ("positive_basin", positive.image),
                    ("unified", unified.image),
                ):
                    _save_image(save_dir / f"{stem}_{kind}_{method}.png", image)
                    residual = 0.5 + 4.0 * (np.asarray(image) - truth)
                    _save_image(save_dir / f"{stem}_{kind}_{method}_error_x4.png", residual)
            for method, image in (
                ("observation", observation),
                ("positive_basin", positive.image),
                ("unified_path_transport", unified.image),
            ):
                rows.append({
                    "source": source_name,
                    "kind": kind,
                    "method": method,
                    **_metrics(image, truth),
                })
            rows[-1]["path_diagnostics"] = unified.diagnostics[
                "characteristic_transport"]
            rows[-1]["support_diagnostics"] = unified.diagnostics[
                "support_gate"]
    summary = {}
    fields = (
        "psnr", "ssim", "broad_halo_error_rms", "oscillatory_error_rms",
        "weighted_local_shift_spread",
    )
    for kind in kernels:
        summary[kind] = {}
        for method in (
            "observation", "positive_basin", "unified_path_transport",
        ):
            selected = [
                row for row in rows
                if row["kind"] == kind and row["method"] == method
            ]
            summary[kind][method] = {
                field: _mean(selected, field) for field in fields
            }
    return {
        "experiment": "known_path_ringing_displacement_benchmark_v1",
        "size": int(size),
        "passes": int(passes),
        "coverage_floor": float(coverage_floor),
        "local_constancy_floor": float(local_constancy_floor),
        "basin_uncertainty_weight": float(basin_uncertainty_weight),
        "source_count": len(source_images),
        "summary": summary,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--coverage-floor", type=float, default=5e-4)
    parser.add_argument("--local-constancy-floor", type=float, default=0.004)
    parser.add_argument("--basin-uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        size=args.size,
        passes=args.passes,
        coverage_floor=args.coverage_floor,
        local_constancy_floor=args.local_constancy_floor,
        basin_uncertainty_weight=args.basin_uncertainty_weight,
        save_dir=args.save_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "summary": report["summary"],
        "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
