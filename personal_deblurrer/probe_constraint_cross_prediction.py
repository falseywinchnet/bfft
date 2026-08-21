#!/usr/bin/env python3
"""Probe independent brightness/derivative evidence for non-fold multi-motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .dense_estimation import (
    _luminance,
    _one_way_dense_flow,
    _sample,
    estimate_dense_pair_exposure,
)
from .run_visibility_benchmark import _layered_pair


def _gradient_residual(
    reference: np.ndarray,
    moving: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    first_y, first_x = np.gradient(_luminance(reference))
    second_y, second_x = np.gradient(_luminance(moving))
    return np.sqrt(
        (_sample(second_x, flow) - first_x) ** 2
        + (_sample(second_y, flow) - first_y) ** 2)


def run(size: int, output: Path) -> dict[str, object]:
    rows = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, displacement in (
            ("moderate_disocclusion", 2.0),
            ("folded_disocclusion", 3.0),
        ):
            _, observations = _layered_pair(
                background,
                displacement,
                noise_sigma=0.002,
                seed=21000 + 41 * source_index,
            )
            estimate = estimate_dense_pair_exposure(
                observations[0], observations[1], duty_cycle=0.0, atoms=1)
            brightness_flow, brightness_record = _one_way_dense_flow(
                observations[0],
                observations[1],
                pyramid_levels=3,
                warp_iterations=5,
                cg_iterations=60,
                smoothness=0.12,
                finest_gradient_weight=0.0,
            )
            primary = estimate.forward_sampling_flow_xy
            disagreement = np.sqrt(np.sum(
                (primary - brightness_flow) ** 2, axis=2))
            flow_magnitude = np.sqrt(np.sum(primary * primary, axis=2))
            primary_gradient = _gradient_residual(
                observations[0], observations[1], primary)
            brightness_gradient = _gradient_residual(
                observations[0], observations[1], brightness_flow)
            primary_photo = np.abs(
                _sample(_luminance(observations[1]), primary)
                - _luminance(observations[0]))
            brightness_photo = np.abs(
                _sample(_luminance(observations[1]), brightness_flow)
                - _luminance(observations[0]))
            rows.append({
                "source": source,
                "case": case,
                "disagreement_rms_pixels": float(np.sqrt(np.mean(
                    disagreement ** 2))),
                "disagreement_q90_pixels": float(np.quantile(
                    disagreement, 0.90)),
                "relative_disagreement": float(np.mean(
                    disagreement / (0.25 + flow_magnitude))),
                "primary_gradient_rms": float(np.sqrt(np.mean(
                    primary_gradient ** 2))),
                "brightness_gradient_rms": float(np.sqrt(np.mean(
                    brightness_gradient ** 2))),
                "derivative_cross_predictive_gain": float(np.mean(
                    brightness_gradient - primary_gradient)),
                "primary_photo_rms": float(np.sqrt(np.mean(
                    primary_photo ** 2))),
                "brightness_photo_rms": float(np.sqrt(np.mean(
                    brightness_photo ** 2))),
                "brightness_flow_rms": brightness_record["flow_rms"],
                "primary_flow_rms": estimate.diagnostics["flow_rms"],
                "primary_fold_fraction": float(max(
                    field.diagnostics()["fold_fraction"]
                    for field in estimate.fields)),
            })
    result = {
        "experiment": "brightness_derivative_cross_predictive_flow_probe_v1",
        "size": int(size),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/constraint_cross_prediction.json"),
    )
    args = parser.parse_args()
    result = run(args.size, args.out)
    for row in result["rows"]:
        print(
            f"{row['source']:22s} {row['case']:23s} "
            f"disagree {row['disagreement_rms_pixels']:.3f} px  "
            f"relative {row['relative_disagreement']:.3f}  "
            f"d-cross {row['derivative_cross_predictive_gain']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
