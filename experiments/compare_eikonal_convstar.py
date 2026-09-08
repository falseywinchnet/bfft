"""Measure fast CONV* against the paper's formal Eikonal factor blend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.conv_distilled_core import distilled_conv_synthesis
from experiments.conv_paper_operator import eikonal_factor_blend_synthesis
from standalone_conv_resize_demo.backend import conv_basin_average


def _cases() -> list[dict[str, float | str]]:
    return [
        {"kind": "edge", "angle": angle, "phase": 0.0, "offset": offset}
        for angle in (15.0, 37.5, 60.0, 82.5)
        for offset in (-0.004, 0.004)
    ] + [
        {"kind": "carrier", "angle": angle, "phase": phase, "offset": 0.0}
        for angle in (22.5, 52.5)
        for phase in (0.0, math.pi / 2.0)
    ] + [
        {"kind": "curved", "angle": 0.0, "phase": 0.0, "offset": offset}
        for offset in (-0.004, 0.004)
    ] + [
        {"kind": "crossing", "angle": 0.0, "phase": phase, "offset": 0.0}
        for phase in (0.0, math.pi / 2.0)
    ]


def _scene(
    kind: str,
    side: int,
    *,
    angle: float = 0.0,
    phase: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, side)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    theta = math.radians(angle)
    along = x * math.cos(theta) + y * math.sin(theta)
    normal = (x - 0.5) * math.cos(theta) + (y - 0.5) * math.sin(theta)
    if kind == "edge":
        return 0.5 + 0.43 * np.tanh((normal - offset) / 0.035)
    if kind == "carrier":
        envelope = np.sin(math.pi * x) ** 2 * np.sin(math.pi * y) ** 2
        return 0.5 + 0.34 * envelope * np.sin(
            2.0 * math.pi * 3.25 * along + phase
        )
    if kind == "curved":
        radius = np.hypot(x - 0.47, y - 0.53)
        return 0.5 + 0.42 * np.tanh((radius - 0.27 - offset) / 0.032)
    if kind == "crossing":
        first = np.sin(2.0 * math.pi * (2.75 * x + 0.9 * y) + phase)
        second = np.sin(2.0 * math.pi * (-0.9 * x + 2.75 * y) - phase)
        return 0.5 + 0.19 * (first + second)
    raise KeyError(kind)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    residual = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(residual * residual))


def _geomean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-300)))))


def run(source_side: int, scale: int) -> dict[str, object]:
    target_side = scale * (source_side - 1) + 1
    rows: list[dict[str, object]] = []
    for case in _cases():
        truth = _scene(
            str(case["kind"]), target_side,
            angle=float(case["angle"]), phase=float(case["phase"]),
            offset=float(case["offset"]),
        ).astype(np.float32)
        coarse = conv_basin_average(
            truth, (source_side, source_side)
        ).astype(np.float32)
        started = perf_counter()
        eikonal = eikonal_factor_blend_synthesis(coarse, truth.shape)
        eikonal_seconds = perf_counter() - started
        started = perf_counter()
        fast = distilled_conv_synthesis(coarse, truth.shape)
        fast_seconds = perf_counter() - started
        rows.append({
            "case": case,
            "eikonal_truth_mse": _mse(eikonal, truth),
            "fast_truth_mse": _mse(fast, truth),
            "fast_eikonal_mse": _mse(fast, eikonal),
            "fast_eikonal_linf": float(np.max(np.abs(
                np.asarray(fast, dtype=np.float64)
                - np.asarray(eikonal, dtype=np.float64)
            ))),
            "eikonal_seconds": eikonal_seconds,
            "fast_seconds": fast_seconds,
        })
    return {
        "source_side": source_side,
        "target_side": target_side,
        "scale": scale,
        "case_count": len(rows),
        "aggregate": {
            "eikonal_truth_geomean_mse": _geomean([
                float(row["eikonal_truth_mse"]) for row in rows
            ]),
            "fast_truth_geomean_mse": _geomean([
                float(row["fast_truth_mse"]) for row in rows
            ]),
            "fast_eikonal_geomean_mse": _geomean([
                float(row["fast_eikonal_mse"]) for row in rows
            ]),
            "fast_eikonal_maximum_linf": max(
                float(row["fast_eikonal_linf"]) for row in rows
            ),
            "fast_eikonal_rms": float(np.sqrt(np.mean([
                float(row["fast_eikonal_mse"]) for row in rows
            ]))),
            "fast_eikonal_maximum_mse": max(
                float(row["fast_eikonal_mse"]) for row in rows
            ),
            "fast_eikonal_median_linf": float(np.median([
                float(row["fast_eikonal_linf"]) for row in rows
            ])),
            "eikonal_median_seconds": float(np.median([
                float(row["eikonal_seconds"]) for row in rows
            ])),
            "fast_median_seconds": float(np.median([
                float(row["fast_seconds"]) for row in rows
            ])),
            "fast_truth_wins": sum(
                float(row["fast_truth_mse"]) < float(row["eikonal_truth_mse"])
                for row in rows
            ),
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-side", type=int, default=9)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/support_geometry/eikonal_convstar_conformance.json"),
    )
    args = parser.parse_args()
    result = run(args.source_side, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
