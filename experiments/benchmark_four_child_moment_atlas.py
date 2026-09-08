#!/usr/bin/env python3
"""Paired throughput and conformance screen for the fused moment atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from standalone_conv_resize_demo.backend import backend_description
from standalone_conv_resize_demo.conservative import (
    restrict_2x2,
    zero_detail_synthesis,
    zero_detail_synthesis_reference,
)


def _elapsed(operation, source: np.ndarray, repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        operation(source)
    return (time.perf_counter() - started) / repetitions


def benchmark(
    sizes: tuple[int, ...], channels: tuple[int, ...], repeats: int
) -> dict[str, object]:
    rng = np.random.default_rng(20260901)
    records: list[dict[str, object]] = []
    for side in sizes:
        for channel_count in channels:
            shape = (side, side) if channel_count == 1 else (
                side, side, channel_count
            )
            source = rng.random(shape, dtype=np.float32)
            fused = zero_detail_synthesis(source)
            reference = zero_detail_synthesis_reference(source)
            fused_times: list[float] = []
            reference_times: list[float] = []
            inner = max(1, min(32, round(100000 / source.size)))
            for iteration in range(repeats):
                if iteration % 2:
                    reference_times.append(_elapsed(
                        zero_detail_synthesis_reference, source, inner
                    ))
                    fused_times.append(_elapsed(
                        zero_detail_synthesis, source, inner
                    ))
                else:
                    fused_times.append(_elapsed(
                        zero_detail_synthesis, source, inner
                    ))
                    reference_times.append(_elapsed(
                        zero_detail_synthesis_reference, source, inner
                    ))
            fused_median = float(np.median(fused_times))
            reference_median = float(np.median(reference_times))
            returned = restrict_2x2(fused)
            records.append({
                "source_shape": list(shape),
                "fused_median_ms": 1000.0 * fused_median,
                "decomposed_median_ms": 1000.0 * reference_median,
                "speedup": reference_median / fused_median,
                "maximum_fused_reference_difference": float(
                    np.max(np.abs(fused - reference), initial=0.0)
                ),
                "maximum_parent_mean_residual": float(
                    np.max(np.abs(returned - source), initial=0.0)
                ),
            })
    return {"backend": backend_description(), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="25,64,128,256")
    parser.add_argument("--channels", default="1,3")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = benchmark(
        tuple(int(value) for value in args.sizes.split(",")),
        tuple(int(value) for value in args.channels.split(",")),
        max(1, int(args.repeats)),
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out)


if __name__ == "__main__":
    main()
