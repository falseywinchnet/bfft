#!/usr/bin/env python3
"""Profile generated covariance transport against materialized nine atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .multicapture_transport import _spatial_positive_sigma_measure
from .spatial_transport import (
    CovarianceReflectedExposureOperator,
    SpatialExposureField,
    SpatialReflectedExposureOperator,
)


def run(size: int, repeats: int, output: Path) -> dict[str, object]:
    extent = max(int(size), 16)
    yy, xx = np.mgrid[:extent, :extent]
    angle = 0.9 * np.pi * xx / max(extent - 1, 1)
    major = 0.4 + 4.5 * yy / max(extent - 1, 1)
    minor = 0.15 + 0.7 * xx / max(extent - 1, 1)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    covariance = np.empty((extent, extent, 2, 2), dtype=np.float64)
    covariance[..., 0, 0] = major * cosine ** 2 + minor * sine ** 2
    covariance[..., 0, 1] = (major - minor) * cosine * sine
    covariance[..., 1, 0] = covariance[..., 0, 1]
    covariance[..., 1, 1] = major * sine ** 2 + minor * cosine ** 2
    points, weights = _spatial_positive_sigma_measure(covariance)
    field = SpatialExposureField.from_barycentric_paths(
        "materialized_nine_atom_covariance",
        np.zeros((extent, extent, 2), dtype=np.float64),
        points,
        weights,
    )
    materialized = SpatialReflectedExposureOperator(field)
    generated = CovarianceReflectedExposureOperator(covariance)
    rng = np.random.default_rng(7441)
    image = rng.random((extent, extent, 3))
    dual = rng.random(image.shape)
    forward_error = float(np.max(np.abs(
        materialized.forward(image) - generated.forward(image))))
    adjoint_error = float(np.max(np.abs(
        materialized.adjoint(dual) - generated.adjoint(dual))))
    closure_error = float(abs(
        np.vdot(generated.forward(image), dual)
        - np.vdot(image, generated.adjoint(dual))))
    for _ in range(2):
        materialized.forward(image)
        materialized.adjoint(dual)
        generated.forward(image)
        generated.adjoint(dual)
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        materialized.forward(image)
        materialized.adjoint(dual)
    materialized_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        generated.forward(image)
        generated.adjoint(dual)
    generated_seconds = time.perf_counter() - started
    field_bytes = int(field.displacements_xy.nbytes + field.weights.nbytes)
    plan_bytes = int(
        materialized._source_indices.nbytes + materialized._coefficients.nbytes)
    generated_axis_bytes = int(generated._axes.nbytes)
    generated_shape_bytes = int(generated._side_weights.nbytes)
    generated_bytes = generated_axis_bytes + generated_shape_bytes
    pixels = extent * extent
    full_pixels = 800 * 800
    plans = 12
    report = {
        "experiment": "generated_positive_covariance_native_v1",
        "size": extent,
        "repeats": int(repeats),
        "materialized_backend": materialized.backend,
        "generated_backend": generated.backend,
        "maximum_forward_parity_error": forward_error,
        "maximum_adjoint_parity_error": adjoint_error,
        "generated_adjoint_closure_error": closure_error,
        "materialized_seconds": materialized_seconds,
        "generated_seconds": generated_seconds,
        "generated_speedup": (
            materialized_seconds / max(generated_seconds, 1e-12)),
        "materialized_field_bytes": field_bytes,
        "materialized_plan_bytes": plan_bytes,
        "generated_covariance_axis_bytes": generated_axis_bytes,
        "generated_axis_shape_bytes": generated_shape_bytes,
        "operator_storage_reduction": (
            (field_bytes + plan_bytes) / max(generated_bytes, 1)),
        "projected_800x800_twelve_capture_materialized_bytes": int(
            (field_bytes + plan_bytes) / pixels * full_pixels * plans),
        "projected_800x800_twelve_capture_generated_bytes": int(
            generated_axis_bytes / pixels * full_pixels * plans
            + generated_shape_bytes * plans),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/covariance_native_profile.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.size, args.repeats, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
