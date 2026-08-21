#!/usr/bin/env python3
"""Measure collapse, ambiguity, and calibration of blur uncertainty transport."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .kernels import gaussian_kernel, line_kernel
from .solver import fuse_transport_observations
from .synthetic import degrade
from .uncertainty import (
    deblur_pair_posterior,
    estimate_noise_discrepancy,
    estimate_pair_posterior,
)
from .workbench import workbench_catalog


def _score(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(estimate, truth)
    record["psnr"] = float(-10.0 * math.log10(max(
        float(record["mse"]), np.finfo(float).tiny)))
    return record


def run(size: int, seeds: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    catalog = workbench_catalog()
    cases = {
        "complementary_motion": (
            line_kernel(11.0, 0.0), line_kernel(11.0, 90.0)),
        "identical_gaussian_gauge": (
            gaussian_kernel(2.5), gaussian_kernel(2.5)),
    }
    noise_levels = (0.0, 0.002, 0.01)
    rows: list[dict[str, object]] = []
    for source_index, (source_name, truth) in enumerate(sources(size).items()):
        for case_name, kernels in cases.items():
            for noise in noise_levels:
                for seed in range(max(int(seeds), 1)):
                    observations = [
                        degrade(
                            truth, kernel, gaussian_sigma=noise,
                            seed=10000 * source_index + 100 * seed + index)
                        for index, kernel in enumerate(kernels)
                    ]
                    posterior = estimate_pair_posterior(
                        observations[0], observations[1], catalog,
                        noise_sigma=noise)
                    exact = bool(
                        posterior.best.first.name == kernels[0].name
                        and posterior.best.second.name == kernels[1].name)
                    if posterior.common_blur_unidentifiable:
                        policy = np.mean(observations, axis=0)
                        retained = 0.0
                        uncertainty_mean = 0.0
                        decision = "abstain_common_blur_gauge"
                    else:
                        uncertain = deblur_pair_posterior(
                            observations[0], observations[1], posterior,
                            noise_sigma=noise, credibility=0.95,
                            maximum_branches=6, passes=16)
                        policy = uncertain.image
                        retained = uncertain.retained_probability
                        uncertainty_mean = float(np.mean(
                            uncertain.standard_deviation))
                        decision = "transport_posterior"
                    oracle = fuse_transport_observations(
                        observations, list(kernels), tv_weight=0.0012,
                        flux_penalty=0.035, passes=16)
                    noise_records = [
                        estimate_noise_discrepancy(
                            observation, degrade(truth, kernel, clip=False))
                        for observation, kernel in zip(observations, kernels)
                    ]
                    true_probability = float(sum(
                        hypothesis.probability
                        for hypothesis in posterior.hypotheses
                        if hypothesis.first.name == kernels[0].name
                        and hypothesis.second.name == kernels[1].name
                    ))
                    rows.append({
                        "source": source_name,
                        "case": case_name,
                        "noise_sigma": noise,
                        "seed": seed,
                        "truth": [kernel.name for kernel in kernels],
                        "best": [
                            posterior.best.first.name,
                            posterior.best.second.name],
                        "best_probability": posterior.best.probability,
                        "true_probability": true_probability,
                        "exact": exact,
                        "entropy": posterior.entropy,
                        "effective_hypotheses": posterior.effective_hypotheses,
                        "consistent_hypotheses": int(sum(
                            item.consistent for item in posterior.hypotheses)),
                        "common_blur_unidentifiable": (
                            posterior.common_blur_unidentifiable),
                        "decision": decision,
                        "retained_probability": retained,
                        "mean_image_uncertainty": uncertainty_mean,
                        "estimated_read_sigma": float(np.mean([
                            item.read_sigma for item in noise_records])),
                        "structured_residual": float(np.mean([
                            item.structured_rms for item in noise_records])),
                        "policy": _score(policy, truth),
                        "oracle": _score(oracle.image, truth),
                        "capture_mean": _score(np.mean(observations, axis=0), truth),
                    })
    complementary = [
        row for row in rows if row["case"] == "complementary_motion"]
    by_noise = {}
    for noise in noise_levels:
        subset = [row for row in complementary if row["noise_sigma"] == noise]
        by_noise[str(noise)] = {
            "exact_fraction": float(np.mean([row["exact"] for row in subset])),
            "mean_true_probability": float(np.mean([
                row["true_probability"] for row in subset])),
            "mean_effective_hypotheses": float(np.mean([
                row["effective_hypotheses"] for row in subset])),
            "mean_policy_psnr": float(np.mean([
                row["policy"]["psnr"] for row in subset])),
            "mean_oracle_psnr": float(np.mean([
                row["oracle"]["psnr"] for row in subset])),
            "mean_estimated_read_sigma": float(np.mean([
                row["estimated_read_sigma"] for row in subset])),
        }
    result = {
        "experiment": "personal_deblurrer_uncertainty_transport_v0",
        "size": int(size),
        "seeds": max(int(seeds), 1),
        "candidate_count": len(catalog),
        "noise_summary": by_noise,
        "identical_gauge_detection_fraction": float(np.mean([
            row["common_blur_unidentifiable"] for row in rows
            if row["case"] == "identical_gaussian_gauge"])),
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "uncertainty_results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/uncertainty_results"))
    args = parser.parse_args()
    result = run(args.size, args.seeds, args.out)
    for noise, summary in result["noise_summary"].items():
        print(
            f"noise {noise:>5s}: exact {summary['exact_fraction']:.3f}; "
            f"true p {summary['mean_true_probability']:.3f}; effective "
            f"{summary['mean_effective_hypotheses']:.2f}; policy "
            f"{summary['mean_policy_psnr']:.3f} dB; oracle "
            f"{summary['mean_oracle_psnr']:.3f} dB")
    print("identical gauge detection:",
          f"{result['identical_gauge_detection_fraction']:.3f}")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
