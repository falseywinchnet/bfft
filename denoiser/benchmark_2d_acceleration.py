"""Repeatable M4 timing gate for the active continuous-support 2-D FMMT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .transport_support import (
    TransportResolution,
    denoise_2d_fmmt,
    support_density,
)


def _timed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def run(sizes: tuple[int, ...], repeats: int) -> dict:
    resolution = TransportResolution()
    records = []
    for size in sizes:
        truth = sources(size)["cameraman"]
        observation = corrupt(
            truth,
            "mixed replacement + uniform",
            amount=0.10,
            density=0.25,
            seed=11107,
        )
        direct_seconds = []
        duplicate_gui_seconds = []
        shared_gui_seconds = []
        stage_rows = []
        direct_output = None
        shared_output = None
        duplicate_output = None
        diagnostic = None
        for _ in range(repeats):
            (direct_output, diagnostic), elapsed = _timed(
                lambda: denoise_2d_fmmt(observation, resolution=resolution))
            direct_seconds.append(elapsed)
            stage_rows.append(diagnostic["stage_seconds"])

            def duplicate_gui_path():
                support_density(observation, resolution)
                return denoise_2d_fmmt(observation, resolution=resolution)

            (duplicate_output, _), elapsed = _timed(duplicate_gui_path)
            duplicate_gui_seconds.append(elapsed)

            def shared_gui_path():
                support = support_density(observation, resolution)
                return denoise_2d_fmmt(
                    observation,
                    resolution=resolution,
                    precomputed_support=support,
                )

            (shared_output, _), elapsed = _timed(shared_gui_path)
            shared_gui_seconds.append(elapsed)

        assert direct_output is not None
        assert duplicate_output is not None
        assert shared_output is not None
        assert diagnostic is not None
        stage_names = tuple(stage_rows[0])
        records.append({
            "size": int(size),
            "pixels": int(size * size),
            "front_batch": int(diagnostic["front_batch"]),
            "direct_seconds": _summary(direct_seconds),
            "gui_duplicate_support_seconds": _summary(duplicate_gui_seconds),
            "gui_shared_support_seconds": _summary(shared_gui_seconds),
            "gui_median_speedup": float(
                np.median(duplicate_gui_seconds)
                / np.median(shared_gui_seconds)
            ),
            "stage_seconds": {
                name: _summary([float(row[name]) for row in stage_rows])
                for name in stage_names
            },
            "maximum_direct_shared_difference": float(np.max(np.abs(
                direct_output - shared_output))),
            "maximum_duplicate_shared_difference": float(np.max(np.abs(
                duplicate_output - shared_output))),
            "metrics": metrics(direct_output, truth),
        })
    return {
        "purpose": (
            "measure representation-only acceleration of the active 2-D "
            "continuous-support FMMT on the M4 CPU"
        ),
        "input": "Cameraman, mixed replacement + uniform 0.25, seed 11107",
        "repeats": int(repeats),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="128,256")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        tuple(int(value) for value in args.sizes.split(",") if value),
        int(args.repeats),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
