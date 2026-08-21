#!/usr/bin/env python3
"""Measure the unified transport inverse on Wronski binomial operators."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .analytic_support import analyze_transport_support
from .decomposition import two_stage_deblur_known
from .kernels import (
    translated_kernel,
    wronski_binomial_kernel,
    wronski_separable_kernel,
)
from .synthetic import degrade


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    angle = math.radians(31.0)
    oblique = (math.cos(angle), math.sin(angle))
    cases = {
        "binomial_single_axis": wronski_binomial_kernel(),
        "binomial_repeated_axis": wronski_binomial_kernel(stages=2),
        "binomial_repeated_oblique": wronski_binomial_kernel(
            oblique, stages=2),
        "binomial_separable": wronski_separable_kernel(),
        "shift_then_repeated_binomial": translated_kernel(
            wronski_binomial_kernel(oblique, stages=2), (3.0, -2.0)),
    }
    records: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for case_name, kernel in cases.items():
        scores = {"observation": [], "transport_inverse": []}
        terminal: dict[str, object] | None = None
        for source_index, (source_name, truth) in enumerate(sources(size).items()):
            observation = degrade(
                truth,
                kernel,
                gaussian_sigma=0.002,
                seed=1900 + source_index,
                boundary="reflect",
            )
            result = two_stage_deblur_known(
                observation, kernel, passes=passes, reference=truth)
            for method, image in (
                ("observation", observation),
                ("transport_inverse", result.image),
            ):
                score = _score(image, truth)
                scores[method].append(score)
                rows.append({
                    "case": case_name,
                    "source": source_name,
                    "method": method,
                    **score,
                })
            terminal = {
                "operator": kernel.name,
                "analytic_support": analyze_transport_support(
                    result.factorization.centered_mixing).diagnostics,
                "deterministic_shift_xy": list(
                    result.factorization.deterministic_shift_xy),
                "support_gate": result.diagnostics["support_gate"],
                "blur_family_selected": result.diagnostics[
                    "blur_family_selected"],
            }
        summary = {method: _mean(values) for method, values in scores.items()}
        records[case_name] = {**(terminal or {}), "summary": summary}
    payload = {
        "experiment": "wronski_positive_measure_fourier_eikonal_v1",
        "size": int(size),
        "passes": int(passes),
        "noise_sigma": 0.002,
        "operator_decision": "analytic_positive_measure_no_family_selection",
        "cases": records,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/personal_deblurrer_wronski_operator.json"),
    )
    args = parser.parse_args()
    payload = run(args.size, args.passes, args.out)
    for case, record in payload["cases"].items():
        before = record["summary"]["observation"]
        after = record["summary"]["transport_inverse"]
        print(
            f"{case:31s} {before['psnr']:7.3f} -> {after['psnr']:7.3f} dB  "
            f"SSIM {before['ssim']:.4f} -> {after['ssim']:.4f}"
        )
    print(f"wall: {payload['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
