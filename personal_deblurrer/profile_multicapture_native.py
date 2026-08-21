#!/usr/bin/env python3
"""Profile compact native global exposure plans used by multicapture transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .relative_mixing_transport import _positive_sigma_measure
from .spatial_transport import (
    SpatialExposureField,
    SpatialReflectedExposureOperator,
)


def run(size: int, repeats: int, plans: int, output: Path) -> dict[str, object]:
    shape = (int(size), int(size))
    operators = []
    for index in range(max(int(plans), 3)):
        angle = np.pi * index / max(int(plans), 3)
        direction = np.asarray((np.cos(angle), np.sin(angle)))
        normal = np.asarray((-direction[1], direction[0]))
        covariance = (
            (1.0 + 0.3 * index) * np.outer(direction, direction)
            + 0.15 * np.outer(normal, normal)
        )
        points, weights = _positive_sigma_measure(covariance)
        flow = np.broadcast_to(
            np.asarray((0.2 * index, -0.1 * index))[None, None, :],
            (*shape, 2),
        )
        operators.append(SpatialReflectedExposureOperator(
            SpatialExposureField.from_barycentric_paths(
                name=f"multicapture_native_profile_{index}",
                barycentric_flow_xy=flow,
                residual_displacements_xy=points,
                weights=weights,
                compact_global=True,
            )))
    rng = np.random.default_rng(9217)
    images = rng.random((len(operators), *shape, 3))

    def native_roundtrip() -> list[np.ndarray]:
        return [operator.adjoint(operator.forward(image))
                for operator, image in zip(operators, images)]

    def numpy_roundtrip() -> list[np.ndarray]:
        return [operator._adjoint_numpy(operator._forward_numpy(image))
                for operator, image in zip(operators, images)]

    expected = numpy_roundtrip()
    actual = native_roundtrip()
    maximum_error = float(max(
        np.max(np.abs(first - second))
        for first, second in zip(expected, actual)))
    for _ in range(3):
        native_roundtrip()
        numpy_roundtrip()
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        numpy_roundtrip()
    numpy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        native_roundtrip()
    native_seconds = time.perf_counter() - started
    pixel_count = shape[0] * shape[1]
    spatial_coefficient_bytes_avoided = int(sum(
        operator._source_indices.shape[0] * pixel_count * 8
        for operator in operators))
    report = {
        "experiment": "multicapture_compact_native_global_exposure_v1",
        "size": int(size),
        "plans": len(operators),
        "repeats": int(repeats),
        "backends": sorted({operator.backend for operator in operators}),
        "all_compact_global": all(
            operator._scalar_coefficients is not None for operator in operators),
        "maximum_parity_error": maximum_error,
        "numpy_seconds": numpy_seconds,
        "native_seconds": native_seconds,
        "native_speedup": numpy_seconds / max(native_seconds, 1e-12),
        "spatial_coefficient_bytes_avoided": spatial_coefficient_bytes_avoided,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--plans", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/multicapture_native_profile.json"))
    args = parser.parse_args()
    report = run(args.size, args.repeats, args.plans, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
