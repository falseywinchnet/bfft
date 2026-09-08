"""Native CPU timing packet for the finalized fast two-order CONV*."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator

from experiments.conv_distilled_core import distilled_conv_resize
from standalone_conv_resize_demo.backend import (
    backend_description,
    easu_resize,
    lanczos3_resize,
    linear_resize,
)


def _source(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(0.0, 1.0, shape[1]),
        indexing="ij",
    )
    return np.stack((
        0.5 + 0.34 * np.sin(2.0 * math.pi * (2.1 * x + 0.7 * y)),
        0.5 + 0.42 * np.tanh(((x - 0.5) * 0.8 + (y - 0.5) * 0.6) / 0.035),
        0.1 + 0.8 * np.exp(-((x - 0.47) ** 2 + (y - 0.53) ** 2) / 0.08),
    ), axis=-1).astype(np.float32)


def _pchip(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    tx = np.linspace(0.0, source.shape[1] - 1.0, target[1])
    ty = np.linspace(0.0, source.shape[0] - 1.0, target[0])
    along_x = PchipInterpolator(
        np.arange(source.shape[1]), source, axis=1
    )(tx)
    return np.asarray(PchipInterpolator(
        np.arange(source.shape[0]), along_x, axis=0
    )(ty), dtype=np.float32)


METHODS = {
    "CONV*": distilled_conv_resize,
    "Lanczos-3": lanczos3_resize,
    "bilinear": linear_resize,
    "EASU": easu_resize,
    "PCHIP": _pchip,
}


def _measure(operation, repeats: int) -> tuple[float, float, float]:
    for _ in range(3):
        output = operation()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        output = operation()
        samples.append(1000.0 * (perf_counter() - started))
    if not np.all(np.isfinite(output)):
        raise AssertionError("nonfinite timing output")
    values = np.asarray(samples)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, float(np.min(values))


def run(repeats: int) -> dict[str, object]:
    backend = backend_description()
    records: list[dict[str, object]] = []
    for source_shape, target_shape in (
        ((64, 64), (512, 512)),
        ((256, 256), (512, 512)),
    ):
        source = _source(source_shape)
        for name, method in METHODS.items():
            median, mad, minimum = _measure(
                lambda method=method: method(source, target_shape), repeats
            )
            records.append({
                "mode": "synthesis",
                "method": name,
                "source_shape": list(source_shape),
                "target_shape": list(target_shape),
                "median_ms": median,
                "mad_ms": mad,
                "minimum_ms": minimum,
                "nanoseconds_per_output_channel": (
                    median * 1.0e6 / (target_shape[0] * target_shape[1] * 3)
                ),
                "repeats": repeats,
            })

    return {
        "qualification": (
            "single process; common three-channel float32 fields; three warmups; "
            "median of repeated wall times; native methods share the same worker pool"
        ),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "backend": backend,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["records"]:
        shape = row.get("coarse_shape", row["source_shape"])
        print(
            f"{row['mode']:13s} {row['method']:10s} {shape} "
            f"{row['median_ms']:9.3f} ms (MAD {row['mad_ms']:.3f})"
        )


if __name__ == "__main__":
    main()
