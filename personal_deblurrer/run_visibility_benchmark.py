#!/usr/bin/env python3
"""Measure positive latent ownership under moving-layer disocclusion and folds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import shift

from denoiser.run_2d_denoiser_battery import metrics, sources

from .dense_estimation import deblur_dense_pair_consensus


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _layered_pair(
    background: np.ndarray,
    displacement: float,
    *,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = background.shape
    yy, xx = np.mgrid[:height, :width]
    mask = (
        ((xx - 0.5 * width) / (0.22 * width)) ** 2
        + ((yy - 0.5 * height) / (0.28 * height)) ** 2
        < 1.0
    ).astype(np.float64)
    foreground = np.clip(0.1 + 0.8 * (1.0 - background), 0.0, 1.0)
    truth = mask * foreground + (1.0 - mask) * background
    observations = []
    for capture, offset in enumerate((-displacement, displacement)):
        moved_mask = shift(
            mask, (0.0, offset), order=1,
            mode="constant", cval=0.0, prefilter=False)
        moved_foreground = shift(
            foreground, (0.0, offset), order=1,
            mode="reflect", prefilter=False)
        observation = (
            moved_mask * moved_foreground
            + (1.0 - moved_mask) * background)
        rng = np.random.default_rng(seed + capture)
        observations.append(np.clip(
            observation + rng.normal(0.0, noise_sigma, observation.shape),
            0.0,
            1.0,
        ))
    return truth, observations


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {
        "moderate_disocclusion": 2.0,
        "folded_disocclusion": 3.0,
    }
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    direct_gains = []
    all_gains = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, displacement in cases.items():
            truth, observations = _layered_pair(
                background,
                displacement,
                noise_sigma=0.002,
                seed=21000 + 41 * source_index,
            )
            result = deblur_dense_pair_consensus(
                observations[0],
                observations[1],
                duty_cycle=0.0,
                passes=passes,
            )
            methods = {
                "best_capture": min(
                    observations,
                    key=lambda item: float(np.mean((item - truth) ** 2)),
                ),
                "unregistered_average": np.mean(observations, axis=0),
                "positive_ownership_consensus": result.image,
            }
            scored = {}
            for method, image in methods.items():
                score = _score(image, truth)
                scored[method] = score
                rows.append({
                    "source": source,
                    "case": case,
                    "method": method,
                    **score,
                })
            gain = (
                scored["positive_ownership_consensus"]["psnr"]
                - scored["unregistered_average"]["psnr"])
            all_gains.append(gain)
            if result.diagnostics["execution_chart"] == "direct_joint_operator":
                direct_gains.append(gain)
            audits[f"{source}/{case}"] = {
                "displacement_each_side_pixels": displacement,
                "psnr_gain_over_average": gain,
                "flow_fold_fractions": result.diagnostics["fold_fractions"],
                "execution_chart": result.diagnostics["execution_chart"],
                "coverage_decision": result.diagnostics["coverage_decision"],
                "unsupported_visibility_fraction": result.diagnostics[
                    "unsupported_visibility_fraction"],
                "ownership_entropy_mean": result.diagnostics[
                    "ownership_entropy_mean"],
                "ownership_entropy_min": result.diagnostics[
                    "ownership_entropy_min"],
                "joint_coverage_min": result.diagnostics["joint_coverage_min"],
                "uncertainty_rms": result.diagnostics["uncertainty_rms"],
            }

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary = {}
    for case in cases:
        summary[case] = {}
        for method in (
            "best_capture", "unregistered_average",
            "positive_ownership_consensus",
        ):
            summary[case][method] = _mean([
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == case and row["method"] == method
            ])
    result = {
        "experiment": "positive_latent_ownership_visibility_transport_v1",
        "size": int(size),
        "maximum_passes": int(passes),
        "noise_sigma": 0.002,
        "cases": cases,
        "sources": list(sources(size)),
        "summary": summary,
        "direct_joint_operator_trials": len(direct_gains),
        "minimum_direct_joint_gain_db": (
            float(np.min(direct_gains)) if direct_gains else None),
        "mean_direct_joint_gain_db": (
            float(np.mean(direct_gains)) if direct_gains else None),
        "minimum_all_gain_db": float(np.min(all_gains)),
        "mean_all_gain_db": float(np.mean(all_gains)),
        "audits": audits,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/visibility_results"))
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(f"  {method:29s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        f"direct folds: {result['direct_joint_operator_trials']}; "
        f"gain mean/min {result['mean_direct_joint_gain_db']:.3f}/"
        f"{result['minimum_direct_joint_gain_db']:.3f} dB")
    print(
        f"all gain mean/min {result['mean_all_gain_db']:.3f}/"
        f"{result['minimum_all_gain_db']:.3f} dB")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
