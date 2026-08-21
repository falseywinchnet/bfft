#!/usr/bin/env python3
"""Measure blind continuous flow-atlas ownership on moving-layer controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .flow_fiber_estimation import deblur_flow_fiber_consensus
from .run_visibility_benchmark import _layered_pair


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, object]], case: str, method: str) -> dict[str, float]:
    chosen = [
        row for row in rows
        if row["case"] == case and row["method"] == method
    ]
    keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    return {
        key: float(np.mean([float(row[key]) for row in chosen]))
        for key in keys
    }


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cases = {"moderate_disocclusion": 2.0, "folded_disocclusion": 3.0}
    methods = (
        "best_capture", "unregistered_average", "single_flow_consensus",
        "raw_fourier_circle_atlas", "blind_positive_flow_atlas",
    )
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    fiber_gains = []
    fiber_minus_single = []
    raw_minus_single = []
    injective_changes = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, displacement in cases.items():
            truth, observations = _layered_pair(
                background,
                displacement,
                noise_sigma=0.002,
                seed=21000 + 41 * source_index,
            )
            fiber = deblur_flow_fiber_consensus(
                observations[0], observations[1], passes=passes)
            single = fiber.common_image
            images = {
                "best_capture": min(
                    observations,
                    key=lambda item: float(np.mean((item - truth) ** 2)),
                ),
                "unregistered_average": np.mean(observations, axis=0),
                "single_flow_consensus": single,
                "raw_fourier_circle_atlas": (
                    fiber.fiber_solution.image
                    if fiber.fiber_solution is not None else single),
                "blind_positive_flow_atlas": fiber.image,
            }
            scored = {}
            for method, image in images.items():
                score = _score(image, truth)
                scored[method] = score
                rows.append({
                    "source": source, "case": case, "method": method, **score,
                })
            gain = (
                scored["blind_positive_flow_atlas"]["psnr"]
                - scored["unregistered_average"]["psnr"])
            delta = (
                scored["blind_positive_flow_atlas"]["psnr"]
                - scored["single_flow_consensus"]["psnr"])
            fiber_gains.append(gain)
            fiber_minus_single.append(delta)
            raw_minus_single.append(
                scored["raw_fourier_circle_atlas"]["psnr"]
                - scored["single_flow_consensus"]["psnr"])
            fold_fractions = [
                field.diagnostics()["fold_fraction"]
                for field in fiber.estimate.fields
            ]
            if max(fold_fractions) == 0.0:
                injective_changes.append(float(np.max(np.abs(
                    fiber.image - single))))
            audits[f"{source}/{case}"] = {
                "gain_over_average_db": gain,
                "delta_from_single_flow_db": delta,
                "raw_delta_from_single_flow_db": (
                    scored["raw_fourier_circle_atlas"]["psnr"]
                    - scored["single_flow_consensus"]["psnr"]),
                "flow_support": fiber.diagnostics.get("flow_support", []),
                "flow_measure_entropy_mean": fiber.diagnostics.get(
                    "latent_measure_entropy_mean", 0.0),
                "closure_temperature": fiber.diagnostics.get(
                    "closure_temperature", 0.0),
                "fold_fractions": fold_fractions,
                "fold_defect_mean": fiber.diagnostics.get(
                    "jacobian_pressure_mean", 0.0),
                "correction_authority_mean": fiber.diagnostics.get(
                    "correction_authority_mean", 0.0),
                "correction_authority_max": fiber.diagnostics.get(
                    "correction_authority_max", 0.0),
                "executed_fiber_passes": fiber.diagnostics.get(
                    "fiber_passes", 0),
                "fourier_circle_translation_xy": fiber.diagnostics.get(
                    "fourier_circle_translation_xy"),
                "dense_circle_disagreement": fiber.diagnostics.get(
                    "dense_circle_disagreement", 0.0),
                "normalized_circle_dispersion": fiber.diagnostics.get(
                    "normalized_circle_dispersion", 0.0),
                "coherent_disagreement_authority": fiber.diagnostics.get(
                    "coherent_disagreement_authority", 0.0),
                "estimation_decision": fiber.diagnostics[
                    "estimation_decision"],
            }

    summary = {
        case: {method: _mean(rows, case, method) for method in methods}
        for case in cases
    }
    result = {
        "experiment": "blind_fourier_circle_positive_flow_atlas_v3",
        "scope": (
            "dense and global/local Fourier-circle connections, tensor "
            "quadrature over scale and atlas coordinate, distinct appearance "
            "per support point, forward/reverse cross-predictive ownership, "
            "continuous coherence-disagreement authority"),
        "size": int(size),
        "maximum_passes": int(passes),
        "support_count": 5,
        "maximum_fiber_passes": 1,
        "noise_sigma": 0.002,
        "cases": cases,
        "sources": list(sources(size)),
        "summary": summary,
        "mean_gain_over_average_db": float(np.mean(fiber_gains)),
        "minimum_gain_over_average_db": float(np.min(fiber_gains)),
        "mean_delta_from_single_flow_db": float(np.mean(fiber_minus_single)),
        "minimum_delta_from_single_flow_db": float(np.min(fiber_minus_single)),
        "positive_delta_from_single_flow_trials": int(np.sum(
            np.asarray(fiber_minus_single) > 0.0)),
        "mean_raw_delta_from_single_flow_db": float(np.mean(raw_minus_single)),
        "minimum_raw_delta_from_single_flow_db": float(np.min(raw_minus_single)),
        "maximum_injective_change": (
            float(np.max(injective_changes)) if injective_changes else None),
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
        default=Path("personal_deblurrer/flow_fiber_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, methods in result["summary"].items():
        print(case)
        for method, score in methods.items():
            print(f"  {method:31s} {score['psnr']:7.3f} dB  SSIM {score['ssim']:.4f}")
    print(
        f"gain over average mean/min {result['mean_gain_over_average_db']:.3f}/"
        f"{result['minimum_gain_over_average_db']:.3f} dB")
    print(
        f"delta from single mean/min {result['mean_delta_from_single_flow_db']:.3f}/"
        f"{result['minimum_delta_from_single_flow_db']:.3f} dB; positive "
        f"{result['positive_delta_from_single_flow_trials']}/12")
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
