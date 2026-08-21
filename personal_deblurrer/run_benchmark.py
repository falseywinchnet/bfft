#!/usr/bin/env python3
"""Run the first exposure-transport deblurring battery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from denoiser.run_2d_denoiser_battery import metrics, sources

from .estimation import estimate_kernel_pair
from .kernels import (
    TransportKernel,
    disk_kernel,
    gaussian_kernel,
    identity_kernel,
    line_kernel,
)
from .solver import fuse_transport_observations, multi_wiener
from .synthetic import degrade


def candidate_catalog() -> list[TransportKernel]:
    result = [identity_kernel()]
    result.extend(gaussian_kernel(value) for value in (1.0, 2.0, 3.0))
    result.extend(disk_kernel(value) for value in (2.0, 3.0, 4.0))
    result.extend(
        line_kernel(length, angle)
        for length in (7.0, 11.0, 15.0)
        for angle in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    )
    return result


def cases() -> dict[str, tuple[list[TransportKernel], float]]:
    return {
        "single_gaussian": ([gaussian_kernel(2.0)], 0.002),
        "orthogonal_motion": (
            [line_kernel(11.0, 0.0), line_kernel(11.0, 90.0)], 0.002),
        "oblique_motion": (
            [line_kernel(11.0, 30.0), line_kernel(11.0, 120.0)], 0.002),
        "motion_plus_defocus": (
            [line_kernel(11.0, 0.0), disk_kernel(3.0)], 0.002),
        "identical_defocus_control": (
            [disk_kernel(3.0), disk_kernel(3.0)], 0.002),
        "three_angle_tomography": (
            [
                line_kernel(11.0, 0.0),
                line_kernel(11.0, 60.0),
                line_kernel(11.0, 120.0),
            ],
            0.0025,
        ),
    }


def _score(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(estimate, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    record["psnr"] = float(-10.0 * math.log10(mse))
    return record


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _coverage_json(diagnostics: dict[str, object]) -> dict[str, float]:
    coverage = diagnostics["coverage"]
    return {
        "dead_fraction": float(coverage["dead_fraction"]),
        "minimum": float(coverage["minimum"]),
        "median": float(coverage["median"]),
    }


def _save(path: Path, image: np.ndarray) -> None:
    value = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(value, mode="L").save(path)


def run(size: int, seeds: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    catalog = candidate_catalog()
    clean_sources = sources(size)
    all_rows: list[dict[str, object]] = []
    estimation_rows: list[dict[str, object]] = []
    case_reports: dict[str, object] = {}
    for case_name, (kernels, noise) in cases().items():
        case_rows: list[dict[str, object]] = []
        coverage_record: dict[str, float] | None = None
        for source_index, (source_name, truth) in enumerate(clean_sources.items()):
            for seed in range(max(int(seeds), 1)):
                observations = [
                    degrade(
                        truth,
                        kernel,
                        gaussian_sigma=noise,
                        seed=10000 * source_index + 100 * seed + index,
                    )
                    for index, kernel in enumerate(kernels)
                ]
                methods: dict[str, np.ndarray] = {
                    "capture_0": observations[0],
                    "pixel_average": np.mean(observations, axis=0),
                }
                for index, observation in enumerate(observations[1:], 1):
                    methods[f"capture_{index}"] = observation
                # Use the same conservative fixed regularization as the
                # coverage-gated fallback.  This keeps the closed-form inverse
                # baseline honest in weakly covered cases instead of tuning it
                # to the proposed transport solve.
                wiener = multi_wiener(
                    observations, kernels, regularization=1.0e-3)
                transport = fuse_transport_observations(
                    observations,
                    kernels,
                    tv_weight=0.0012,
                    flux_penalty=0.035,
                    passes=20,
                )
                methods["oracle_multi_wiener"] = wiener.image
                methods["oracle_exposure_transport"] = transport.image
                coverage_record = _coverage_json(transport.diagnostics)

                if len(observations) == 2:
                    estimate = estimate_kernel_pair(
                        observations[0], observations[1], catalog)
                    if estimate.common_blur_unidentifiable:
                        methods["estimated_policy"] = np.mean(observations, axis=0)
                        estimate_decision = "abstain_common_blur_gauge"
                    else:
                        estimated = fuse_transport_observations(
                            observations,
                            [estimate.first, estimate.second],
                            tv_weight=0.0012,
                            flux_penalty=0.035,
                            passes=20,
                        )
                        methods["estimated_policy"] = estimated.image
                        estimate_decision = str(estimated.diagnostics["method"])
                    estimation_rows.append({
                        "case": case_name,
                        "source": source_name,
                        "seed": seed,
                        "truth": [kernel.name for kernel in kernels],
                        "estimate": [estimate.first.name, estimate.second.name],
                        "exact": bool(
                            estimate.first.name == kernels[0].name
                            and estimate.second.name == kernels[1].name),
                        "score": estimate.score,
                        "runner_up_score": estimate.runner_up_score,
                        "ambiguity_ratio": estimate.ambiguity_ratio,
                        "relative_transport_strength": (
                            estimate.relative_transport_strength),
                        "common_blur_unidentifiable": (
                            estimate.common_blur_unidentifiable),
                        "decision": estimate_decision,
                        "ranked": estimate.ranked,
                    })

                for method, image in methods.items():
                    row = {
                        "case": case_name,
                        "source": source_name,
                        "seed": seed,
                        "method": method,
                        **_score(image, truth),
                    }
                    case_rows.append(row)
                    all_rows.append(row)

                if source_index == 0 and seed == 0:
                    case_dir = output / case_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    _save(case_dir / "truth.png", truth)
                    for method, image in methods.items():
                        _save(case_dir / f"{method}.png", image)

        method_names = sorted({str(row["method"]) for row in case_rows})
        summary = {
            method: _mean([
                {key: float(value) for key, value in row.items()
                 if key in ("mse", "ssim", "variance_ratio",
                            "central_range_ratio", "edge_retention",
                            "mean_bias", "psnr")}
                for row in case_rows if row["method"] == method
            ])
            for method in method_names
        }
        case_reports[case_name] = {
            "kernels": [kernel.name for kernel in kernels],
            "noise_sigma": noise,
            "coverage": coverage_record,
            "summary": summary,
        }

    pair_estimates = [row for row in estimation_rows]
    result = {
        "experiment": "personal_positive_exposure_transport_deblur_v0",
        "size": int(size),
        "seeds": max(int(seeds), 1),
        "sources": list(clean_sources),
        "candidate_count": len(catalog),
        "cases": case_reports,
        "estimation": {
            "rows": pair_estimates,
            "exact_fraction_all": float(np.mean([
                row["exact"] for row in pair_estimates])) if pair_estimates else 0.0,
            "exact_fraction_complementary": float(np.mean([
                row["exact"] for row in pair_estimates
                if row["case"] != "identical_defocus_control"
            ])),
        },
        "rows": all_rows,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.seeds, args.out)
    for case, record in result["cases"].items():
        print(case)
        for method, values in record["summary"].items():
            print(f"  {method:30s} {values['psnr']:7.3f} dB  "
                  f"SSIM {values['ssim']:.4f}")
    print("exact complementary kernel pairs:",
          f"{result['estimation']['exact_fraction_complementary']:.3f}")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
