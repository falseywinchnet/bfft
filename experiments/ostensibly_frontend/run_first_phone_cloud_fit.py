#!/usr/bin/env python3
"""Build and register identically processed BDL and Dave/Simon /R/ clouds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .crazy_climber_geometry import (
    CrazyClimberConfig,
    HessianRidgeConfig,
    crazy_climber_occupation,
    multiscale_hessian_ridge_surface,
)
from .depthmap_geometry import DepthmapGeometryConfig, multiscale_morphological_residual
from .occupation_point_cloud import (
    CloudFitConfig,
    PointCloudConfig,
    fit_phone_cloud,
    occupation_to_point_cloud,
    transform_phone_cloud,
)


def _occupation_from_trace(
    trace: np.ndarray,
    morphology_config: DepthmapGeometryConfig,
    ridge_config: HessianRidgeConfig,
    climber_config: CrazyClimberConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual, _ = multiscale_morphological_residual(trace, morphology_config)
    ridge = multiscale_hessian_ridge_surface(residual, ridge_config)
    occupation = crazy_climber_occupation(ridge.saliency, climber_config)
    return residual, ridge.saliency, occupation.weighted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument(
        "--rows",
        type=int,
        default=128,
        help="low-frequency step-3 rows retained by the accepted visualization",
    )
    args = parser.parse_args()

    with np.load(args.pair) as saved:
        source_trace = np.asarray(saved["source_trace"][: args.rows], dtype=np.float64)
        target_trace = np.asarray(saved["target_trace"][: args.rows], dtype=np.float64)

    morphology_config = DepthmapGeometryConfig()
    ridge_config = HessianRidgeConfig()
    climber_config = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    fit_config = CloudFitConfig()
    source_residual, source_ridge, source_occupation = _occupation_from_trace(
        source_trace, morphology_config, ridge_config, climber_config
    )
    target_residual, target_ridge, target_occupation = _occupation_from_trace(
        target_trace, morphology_config, ridge_config, climber_config
    )
    source_cloud = occupation_to_point_cloud(source_occupation, cloud_config)
    target_cloud = occupation_to_point_cloud(target_occupation, cloud_config)
    fit = fit_phone_cloud(target_cloud.points, source_cloud.points, fit_config)
    target_prefit = transform_phone_cloud(
        target_cloud.points,
        np.zeros(5, dtype=np.float64),
        fit.moving_center,
        fit.reference_center,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out / "first_phone_cloud_fit.npz"
    np.savez_compressed(
        arrays_path,
        source_trace=source_trace,
        target_trace=target_trace,
        source_residual=source_residual,
        target_residual=target_residual,
        source_ridge=source_ridge,
        target_ridge=target_ridge,
        source_occupation=source_occupation,
        target_occupation=target_occupation,
        source_high_resolution=source_cloud.high_resolution_field,
        target_high_resolution=target_cloud.high_resolution_field,
        source_points=source_cloud.points,
        target_points=target_cloud.points,
        target_prefit_points=target_prefit,
        target_fitted_points=fit.transformed_points,
        target_center=fit.moving_center,
        source_center=fit.reference_center,
    )
    improvement = 1.0 - fit.postfit_distance / max(fit.prefit_distance, 1e-30)
    metadata = {
        "source": {
            "role": "reference",
            "speaker": "CMU ARCTIC BDL",
            "phone": "R",
            "trace_shape": list(source_trace.shape),
            "point_count": int(source_cloud.points.shape[0]),
        },
        "target": {
            "role": "candidate",
            "speaker": "Dave/Simon first utterance",
            "phone_hypothesis": "R",
            "trace_shape": list(target_trace.shape),
            "point_count": int(target_cloud.points.shape[0]),
        },
        "pipeline": {
            "morphology": asdict(morphology_config),
            "ridge": asdict(ridge_config),
            "climbers": asdict(climber_config),
            "point_cloud": asdict(cloud_config),
        },
        "fit": {
            "config": asdict(fit_config),
            "degrees_of_freedom": 5,
            "row_scale": fit.row_scale,
            "frame_scale": fit.frame_scale,
            "row_shift": fit.row_shift,
            "frame_shift": fit.frame_shift,
            "pitch_drift_rows_per_frame": fit.pitch_drift,
            "target_center": fit.moving_center.tolist(),
            "reference_center": fit.reference_center.tolist(),
            "effective_center_row_displacement": float(
                fit.reference_center[0] + fit.row_shift - fit.moving_center[0]
            ),
            "effective_center_frame_displacement": float(
                fit.reference_center[1] + fit.frame_shift - fit.moving_center[1]
            ),
            "prefit_distance": fit.prefit_distance,
            "postfit_distance": fit.postfit_distance,
            "relative_improvement": improvement,
            "optimizer_success": fit.optimizer_success,
            "optimizer_evaluations": fit.optimizer_evaluations,
        },
    }
    metadata_path = args.out / "first_phone_cloud_fit.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"arrays": str(arrays_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
