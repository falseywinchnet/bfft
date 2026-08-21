#!/usr/bin/env python3
"""Profile the generated FFT exposure against materialized-plan storage."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .spatial_transport import (
    CompactGlobalExposureField,
    CompactGlobalReflectedExposureOperator,
)


def profile(size: int = 800, atom_count: int = 153) -> dict[str, object]:
    if atom_count < 3:
        raise ValueError("profile needs at least three atoms")
    random = np.random.default_rng(4021)
    pair_count = atom_count // 2
    half = random.normal(0.0, 3.0, size=(pair_count, 2))
    points = np.concatenate((half, -half), axis=0)
    if atom_count % 2:
        points = np.concatenate((points, np.zeros((1, 2))), axis=0)
    weights = random.uniform(0.1, 1.0, size=len(points))
    # Antipodal weights make the measure centered before constructor
    # normalization, matching the full-quartic positive dictionary.
    weights[:pair_count] = weights[pair_count:2 * pair_count]
    field = CompactGlobalExposureField(
        "profiled_full_quartic_measure", (size, size), points, weights)
    construction_start = time.perf_counter()
    operator = CompactGlobalReflectedExposureOperator(field)
    construction_seconds = time.perf_counter() - construction_start
    image = random.random((size, size))
    cotangent = random.normal(size=(size, size))
    forward_start = time.perf_counter()
    prediction = operator.forward(image)
    forward_seconds = time.perf_counter() - forward_start
    adjoint_start = time.perf_counter()
    gradient = operator.adjoint(cotangent)
    adjoint_seconds = time.perf_counter() - adjoint_start
    materialized_bytes = 7 * len(points) * size * size * 8
    return {
        "experiment": "compact_generated_global_transport_profile_v1",
        "shape": [size, size],
        "atom_count": len(points),
        "construction_seconds": construction_seconds,
        "forward_seconds": forward_seconds,
        "adjoint_seconds": adjoint_seconds,
        "compact_storage_bytes": operator.storage_bytes,
        "reference_materialized_storage_bytes": materialized_bytes,
        "storage_reduction_factor": materialized_bytes / operator.storage_bytes,
        "adjoint_inner_product_error": abs(
            float(np.vdot(prediction, cotangent))
            - float(np.vdot(image, gradient))),
        "unit_mass_error": float(np.max(np.abs(
            operator.forward(np.ones_like(image)) - 1.0))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--atoms", type=int, default=153)
    args = parser.parse_args()
    print(json.dumps(profile(args.size, args.atoms), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
