#!/usr/bin/env python3
"""Profile ABI-v3 batched spatial transport against per-sheet native calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .spatial_transport import (
    SpatialExposureField,
    SpatialExposureOperatorBatch,
    SpatialReflectedExposureOperator,
)


def run(
    size: int,
    repeats: int,
    output: Path,
    plans: int = 5,
) -> dict[str, object]:
    yy, xx = np.mgrid[:size, :size]
    flow = np.empty((size, size, 2), dtype=np.float64)
    flow[..., 0] = 3.0 + 0.4 * np.sin(2.0 * np.pi * yy / size)
    flow[..., 1] = -1.5 + 0.3 * np.sin(2.0 * np.pi * xx / size)
    times = np.linspace(-0.25, 0.25, 7)[:, None, None, None]
    plan_count = max(int(plans), 2)
    operators = tuple(
        SpatialReflectedExposureOperator(
            SpatialExposureField.from_barycentric_paths(
                name=f"batch_profile_{value}",
                barycentric_flow_xy=float(value) * flow,
                residual_displacements_xy=times * float(value) * flow[None, ...],
                weights=np.ones((7, size, size), dtype=np.float64),
            )
        )
        for value in np.linspace(0.0, 1.0, plan_count)
    )
    batch = SpatialExposureOperatorBatch(operators)
    rng = np.random.default_rng(881)
    images = rng.random((plan_count, size, size))

    def loop_apply(value: np.ndarray) -> np.ndarray:
        forward = np.stack([
            operator.forward(image)
            for operator, image in zip(operators, value)
        ], axis=0)
        return np.stack([
            operator.adjoint(image)
            for operator, image in zip(operators, forward)
        ], axis=0)

    def batch_apply(value: np.ndarray) -> np.ndarray:
        return batch.adjoint(batch.forward(value))

    expected = loop_apply(images)
    actual = batch_apply(images)
    maximum_error = float(np.max(np.abs(expected - actual)))
    for _ in range(5):
        loop_apply(images)
        batch_apply(images)
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        loop_apply(images)
    loop_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        batch_apply(images)
    batch_seconds = time.perf_counter() - started
    result = {
        "experiment": "native_spatial_batch_profile_v2",
        "size": int(size),
        "plan_count": plan_count,
        "repeats": int(repeats),
        "individual_backend": operators[0].backend,
        "batch_backend": batch.backend,
        "maximum_parity_error": maximum_error,
        "loop_seconds": loop_seconds,
        "batch_seconds": batch_seconds,
        "batch_speedup": loop_seconds / max(batch_seconds, 1e-12),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--plans", type=int, default=5)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/spatial_batch_profile.json"),
    )
    args = parser.parse_args()
    result = run(args.size, args.repeats, args.out, plans=args.plans)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
