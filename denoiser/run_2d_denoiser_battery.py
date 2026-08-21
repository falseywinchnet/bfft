"""Diverse image battery for the empirical checkpoint and theoretical seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .cross_predictive_transport_2d import denoise_cross_predictive_transport_2d
from .continual_eikonal_noise_transport_2d import (
    denoise_continual_eikonal_noise_transport_2d,
)
from .continual_fabada_eikonal_2d import (
    denoise_complete_moment_residual_posterior_2d,
    denoise_continual_fabada_eikonal_2d,
    denoise_continual_residual_posterior_2d,
)
from .fmmt_certified import denoise_fmmt
from .reflection_consistent_posterior_2d import (
    denoise_reflection_consistent_posterior_2d,
)
from .probes import edge_retention, hair_edge_scene
from .sample_series import corrupt


CONDITIONS = (
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("Gaussian 0.10", "Gaussian additive", 0.10, 0.25),
    ("Laplace 0.08", "Laplace additive", 0.08, 0.25),
    ("multiplicative 0.12", "multiplicative", 0.12, 0.25),
    ("replacement 0.10", "random-value replacement", 0.10, 0.10),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("salt-pepper 0.10", "salt and pepper", 0.10, 0.10),
    ("mixed 0.10", "mixed replacement + uniform", 0.10, 0.10),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    pixels = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
    return np.asarray(
        Image.fromarray(pixels).resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0


def _cameraman(size: int) -> np.ndarray:
    montage = Path(__file__).parent / "truth_cameraman_compare.png"
    if not montage.exists():
        raise FileNotFoundError("truth_cameraman_compare.png is required")
    image = Image.open(montage).convert("L")
    clean = image.crop((27, 61, 535, 569))
    return np.asarray(
        clean.resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0


def _geometric(size: int) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    field = 0.18 + 0.18 * (xx + 1.0) / 2.0
    field += 0.25 * (((xx + 0.35) ** 2 + (yy + 0.25) ** 2) < 0.22 ** 2)
    field += 0.35 * ((xx - 0.25 > 0.55 * yy) & (yy > -0.45))
    ring = np.abs(np.hypot(xx - 0.25, yy + 0.30) - 0.28) < 0.025
    field[ring] = 0.92
    return ndimage.gaussian_filter(np.clip(field, 0.0, 1.0), 0.35, mode="reflect")


def _woven(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    phase_x = 2.0 * np.pi * (7.0 * xx + 13.0 * xx * xx)
    phase_y = 2.0 * np.pi * (9.0 * yy + 8.0 * yy * yy)
    envelope = 0.45 + 0.35 * np.exp(-((xx - 0.55) ** 2 + (yy - 0.5) ** 2) / 0.16)
    field = 0.48 + envelope * (
        0.14 * np.sin(phase_x) + 0.11 * np.sin(phase_y))
    field += 0.18 * (xx + yy > 1.35)
    return np.clip(field, 0.0, 1.0)


def _line_drawing(size: int) -> np.ndarray:
    scale = 4
    canvas = Image.new("L", (size * scale, size * scale), 238)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((18, 24, size * scale - 22, size * scale - 28), outline=45, width=5)
    for offset in range(34, size * scale - 35, 28):
        draw.line((28, offset, size * scale - 30, offset + 11), fill=75, width=3)
    draw.ellipse((size, size // 2, 3 * size, 5 * size // 2), outline=20, width=7)
    draw.line((25, 3 * size, 3 * size, size), fill=25, width=6)
    return np.asarray(
        canvas.resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0


def _multiscale_blobs(size: int) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    field = 0.2 + 0.06 * xx - 0.04 * yy
    for cx, cy, width, amplitude in (
        (-0.55, -0.45, 0.34, 0.42),
        (0.35, -0.35, 0.17, 0.50),
        (-0.12, 0.42, 0.09, 0.58),
        (0.58, 0.48, 0.045, 0.62),
    ):
        field += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / width ** 2)
    checker = ((np.floor((xx + 1) * 12) + np.floor((yy + 1) * 12)) % 2) * 0.12
    field += checker * (xx > 0.05) * (yy > 0.0)
    return np.clip(field, 0.0, 1.0)


def sources(size: int) -> dict[str, np.ndarray]:
    hair = hair_edge_scene(size=size, seed=719)[0]
    return {
        "cameraman": _cameraman(size),
        "tapered hair": hair,
        "geometric interfaces": _geometric(size),
        "woven chirps": _woven(size),
        "line drawing": _line_drawing(size),
        "multiscale blobs": _multiscale_blobs(size),
    }


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Standard Gaussian-window structural similarity without skimage."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    ux = ndimage.gaussian_filter(x, 1.5, mode="reflect", truncate=3.5)
    uy = ndimage.gaussian_filter(y, 1.5, mode="reflect", truncate=3.5)
    uxx = ndimage.gaussian_filter(x * x, 1.5, mode="reflect", truncate=3.5)
    uyy = ndimage.gaussian_filter(y * y, 1.5, mode="reflect", truncate=3.5)
    uxy = ndimage.gaussian_filter(x * y, 1.5, mode="reflect", truncate=3.5)
    vx = np.maximum(uxx - ux * ux, 0.0)
    vy = np.maximum(uyy - uy * uy, 0.0)
    covariance = uxy - ux * uy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    value = ((2.0 * ux * uy + c1) * (2.0 * covariance + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(np.mean(value))


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(estimate) - np.asarray(truth)
    truth_variance = float(np.var(truth))
    truth_range = float(np.quantile(truth, 0.95) - np.quantile(truth, 0.05))
    return {
        "mse": float(np.mean(error * error)),
        "ssim": ssim(truth, estimate),
        "variance_ratio": float(
            np.var(estimate) / max(truth_variance, np.finfo(float).tiny)),
        "central_range_ratio": float(
            (np.quantile(estimate, 0.95) - np.quantile(estimate, 0.05))
            / max(truth_range, np.finfo(float).tiny)),
        "edge_retention": edge_retention(estimate, truth),
        "mean_bias": float(np.mean(estimate) - np.mean(truth)),
    }


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = tuple(metrics(np.zeros((8, 8)), np.zeros((8, 8))))
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run(size: int, seeds: int) -> dict:
    clean_sources = sources(size)

    def wiener(value: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = signal.wiener(value, (3, 3))
        return np.clip(np.nan_to_num(result, nan=float(np.mean(value))), 0.0, 1.0)

    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "observation": lambda value: value.copy(),
        "Gaussian sigma=1 control": lambda value: ndimage.gaussian_filter(
            value, 1.0, mode="reflect"),
        "median 3x3 control": lambda value: ndimage.median_filter(
            value, 3, mode="reflect"),
        "Wiener 3x3 control": wiener,
        "integrated FMMT checkpoint": lambda value: denoise_fmmt(value)[0],
        "continual phase-eikonal checkpoint": lambda value: (
            denoise_continual_eikonal_noise_transport_2d(value)[0]),
        "FABADA-order eikonal averaging": lambda value: (
            denoise_continual_fabada_eikonal_2d(value)[0]),
        "transported residual posterior": lambda value: (
            denoise_continual_residual_posterior_2d(value)[0]),
        "complete-moment metric posterior": lambda value: (
            denoise_complete_moment_residual_posterior_2d(value)[0]),
        "reflection-consistent posterior": lambda value: (
            denoise_reflection_consistent_posterior_2d(value)[0]),
        "four-direction theoretical seed": lambda value: (
            denoise_cross_predictive_transport_2d(value)[0]),
    }
    clean_rows = []
    noisy_rows = []
    continuation_records = []
    for source, truth in clean_sources.items():
        for method, estimator in methods.items():
            clean_rows.append({
                "source": source,
                "method": method,
                **metrics(estimator(truth), truth),
            })
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth, kind, amount=amount, density=density,
                    seed=9100 + seed)
                for method, estimator in methods.items():
                    noisy_rows.append({
                        "source": source,
                        "condition": condition,
                        "seed": seed,
                        "method": method,
                        **metrics(estimator(observation), truth),
                    })
                _estimate, diagnostic = denoise_cross_predictive_transport_2d(
                    observation)
                continuation_records.append({
                    "source": source,
                    "condition": condition,
                    "seed": seed,
                    "accepted_continuations": diagnostic["accepted_continuations"],
                    "continuation_ceiling_hit": diagnostic["continuation_ceiling_hit"],
                })

    clean_summary = {}
    noisy_summary = {}
    by_condition = {}
    by_source = {}
    for method in methods:
        clean_summary[method] = _mean_metrics([
            row for row in clean_rows if row["method"] == method])
        noisy_summary[method] = _mean_metrics([
            row for row in noisy_rows if row["method"] == method])
    for condition, *_rest in CONDITIONS:
        by_condition[condition] = {
            method: _mean_metrics([
                row for row in noisy_rows
                if row["condition"] == condition and row["method"] == method])
            for method in methods
        }
    for source in clean_sources:
        by_source[source] = {
            method: _mean_metrics([
                row for row in noisy_rows
                if row["source"] == source and row["method"] == method])
            for method in methods
        }

    case_keys = sorted({
        (row["source"], row["condition"], row["seed"])
        for row in noisy_rows})
    mse_wins = {method: 0 for method in methods}
    ssim_wins = {method: 0 for method in methods}
    for key in case_keys:
        case = [row for row in noisy_rows if (
            row["source"], row["condition"], row["seed"]) == key]
        mse_wins[min(case, key=lambda row: row["mse"])["method"]] += 1
        ssim_wins[max(case, key=lambda row: row["ssim"])["method"]] += 1

    candidate = "four-direction theoretical seed"
    phase_candidate = "continual phase-eikonal checkpoint"
    empirical = "integrated FMMT checkpoint"
    gates = {
        "candidate_reaches_equilibrium": not any(
            row["continuation_ceiling_hit"] for row in continuation_records),
        "candidate_reduces_observation_mse": (
            noisy_summary[candidate]["mse"]
            < noisy_summary["observation"]["mse"]),
        "candidate_retains_half_variance": (
            noisy_summary[candidate]["variance_ratio"] > 0.5),
        "candidate_retains_half_central_range": (
            noisy_summary[candidate]["central_range_ratio"] > 0.5),
        "candidate_matches_empirical_mse": (
            noisy_summary[candidate]["mse"] <= noisy_summary[empirical]["mse"]),
        "candidate_matches_empirical_ssim": (
            noisy_summary[candidate]["ssim"] >= noisy_summary[empirical]["ssim"]),
        "phase_candidate_reduces_observation_mse": (
            noisy_summary[phase_candidate]["mse"]
            < noisy_summary["observation"]["mse"]),
        "phase_candidate_retains_half_variance": (
            noisy_summary[phase_candidate]["variance_ratio"] > 0.5),
        "phase_candidate_retains_half_central_range": (
            noisy_summary[phase_candidate]["central_range_ratio"] > 0.5),
    }
    return {
        "status": "2-D gate; theoretical seed is promoted only if every gate passes",
        "size": int(size),
        "seeds": int(seeds),
        "sources": list(clean_sources),
        "conditions": [condition for condition, *_rest in CONDITIONS],
        "clean_summary": clean_summary,
        "noisy_summary": noisy_summary,
        "by_condition": by_condition,
        "by_source": by_source,
        "mse_case_wins": mse_wins,
        "ssim_case_wins": ssim_wins,
        "gates": gates,
        "continuation_summary": {
            "mean_accepted": float(np.mean([
                row["accepted_continuations"] for row in continuation_records])),
            "maximum_accepted": int(max(
                row["accepted_continuations"] for row in continuation_records)),
            "ceiling_hits": int(sum(
                row["continuation_ceiling_hit"] for row in continuation_records)),
        },
        "clean_rows": clean_rows,
        "noisy_rows": noisy_rows,
        "continuation_records": continuation_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    report = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "candidate": report["noisy_summary"]["four-direction theoretical seed"],
        "empirical": report["noisy_summary"]["integrated FMMT checkpoint"],
        "wins_mse": report["mse_case_wins"],
        "wins_ssim": report["ssim_case_wins"],
        "gates": report["gates"],
        "continuation": report["continuation_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
