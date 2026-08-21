#!/usr/bin/env python3
"""Measure shift-first and centered-mixing recovery on reflect-boundary blur."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .decomposition import two_stage_deblur_blind, two_stage_deblur_known
from .kernels import (
    curved_path_kernel,
    disk_kernel,
    gaussian_kernel,
    line_kernel,
    translated_kernel,
)
from .synthetic import degrade


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(size: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {
        "gaussian_center_mix": gaussian_kernel(2.0),
        "disk_center_mix": disk_kernel(3.0),
        "line_center_mix": line_kernel(11.0, 0.0),
        "curve_center_mix": curved_path_kernel(11.0, 0.0, 4.0),
        "shift_then_line_mix": translated_kernel(
            line_kernel(9.0, 30.0), (3.0, -2.0)),
    }
    case_records: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    for case_name, kernel in cases.items():
        method_rows: dict[str, list[dict[str, float]]] = {
            "observation": [], "known_two_stage": [], "blind_two_stage": []}
        audit: dict[str, object] | None = None
        for source_name, truth in sources(size).items():
            observation = degrade(
                truth,
                kernel,
                gaussian_sigma=0.002,
                seed=len(all_rows) + 17,
                boundary="reflect",
            )
            known = two_stage_deblur_known(
                observation, kernel, passes=64, reference=truth)
            blind = two_stage_deblur_blind(observation, passes=64)
            images = {
                "observation": observation,
                "known_two_stage": known.image,
                "blind_two_stage": blind.image,
            }
            for method, image in images.items():
                score = _score(image, truth)
                method_rows[method].append(score)
                all_rows.append({
                    "case": case_name,
                    "source": source_name,
                    "method": method,
                    **score,
                })
            audit = {
                "true_kernel": kernel.name,
                "deterministic_shift_xy": list(
                    known.factorization.deterministic_shift_xy),
                "observation_unchanged": known.diagnostics[
                    "observation_unchanged"],
                "boundary": known.diagnostics["boundary"],
                "support_gate": known.diagnostics["support_gate"],
                "characteristic_transport": known.diagnostics[
                    "characteristic_transport"],
                "blind_estimated_kernel": blind.diagnostics[
                    "estimated_centered_kernel"],
                "blind_estimation_branch": blind.diagnostics[
                    "estimation_branch"],
                "blind_shift_observability": blind.diagnostics[
                    "shift_observability"],
            }
        summary = {method: _mean(rows) for method, rows in method_rows.items()}
        case_records[case_name] = {**(audit or {}), "summary": summary}
    result = {
        "experiment": "shift_first_continuous_positive_exposure_transport_v4",
        "size": int(size),
        "known_maximum_passes": 64,
        "blind_maximum_passes": 8,
        "sources": list(sources(size)),
        "noise_sigma": 0.002,
        "cases": case_records,
        "rows": all_rows,
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/shift_mix_results"))
    args = parser.parse_args()
    result = run(args.size, args.out)
    for case, record in result["cases"].items():
        print(case)
        for method, score in record["summary"].items():
            print(f"  {method:20s} {score['psnr']:7.3f} dB  "
                  f"SSIM {score['ssim']:.4f}")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
