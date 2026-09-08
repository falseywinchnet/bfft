"""Universal factor-order contrast sweep for distilled CONV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_distilled_core import (
    nodal_current_geometry,
    reverse_conv_resize,
)


def _cases() -> list[dict[str, float | str]]:
    return [
        {"kind": "edge", "angle": angle, "phase": 0.0, "offset": offset}
        for angle in (15.0, 37.5, 60.0, 82.5)
        for offset in (-0.004, 0.004)
    ] + [
        {"kind": "carrier", "angle": angle, "phase": phase, "offset": 0.0}
        for angle in (22.5, 52.5) for phase in (0.0, math.pi / 2.0)
    ] + [
        {"kind": "curved", "angle": 0.0, "phase": 0.0, "offset": offset}
        for offset in (-0.004, 0.004)
    ] + [
        {"kind": "crossing", "angle": 0.0, "phase": phase, "offset": 0.0}
        for phase in (0.0, math.pi / 2.0)
    ]


def _scene(kind: str, side: int, *, angle: float, phase: float, offset: float) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, side)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    theta = math.radians(angle)
    along = x * math.cos(theta) + y * math.sin(theta)
    normal = (x - 0.5) * math.cos(theta) + (y - 0.5) * math.sin(theta)
    if kind == "edge":
        return 0.5 + 0.43 * np.tanh((normal - offset) / 0.035)
    if kind == "carrier":
        envelope = np.sin(math.pi * x) ** 2 * np.sin(math.pi * y) ** 2
        return 0.5 + 0.34 * envelope * np.sin(2.0 * math.pi * 3.25 * along + phase)
    if kind == "curved":
        radius = np.hypot(x - 0.47, y - 0.53)
        return 0.5 + 0.42 * np.tanh((radius - 0.27 - offset) / 0.032)
    first = np.sin(2.0 * math.pi * (2.75 * x + 0.9 * y) + phase)
    second = np.sin(2.0 * math.pi * (-0.9 * x + 2.75 * y) - phase)
    return 0.5 + 0.19 * (first + second)


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    residual = np.asarray(first, float) - np.asarray(second, float)
    return float(np.mean(residual * residual))
from standalone_conv_resize_demo.backend import (
    conv_basin_average,
    conv_resize,
    q1_order_blend,
)


def order_coordinate(gxx: np.ndarray, gyy: np.ndarray, q: float) -> np.ndarray:
    """Symmetric powered energy ratio, evaluated without overflow."""

    if q <= 0.0:
        raise ValueError("order exponent must be positive")
    trace = gxx + gyy
    if math.isinf(q):
        beta = np.where(gyy > gxx, 1.0, np.where(gyy < gxx, 0.0, 0.5))
    else:
        ratio = np.divide(
            gyy, trace, out=np.full_like(trace, 0.5), where=trace > 0.0
        )
        left = np.power(ratio, q)
        right = np.power(1.0 - ratio, q)
        beta = np.divide(
            left, left + right,
            out=np.full_like(left, 0.5), where=(left + right) > 0.0,
        )
    return np.where(trace > 0.0, beta, 0.5).astype(np.float32)


def geomean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-300)))))


def run(source_side: int, scale: int, exponents: tuple[float, ...]) -> dict[str, object]:
    target_side = scale * (source_side - 1) + 1
    records: list[dict[str, object]] = []
    for case in _cases():
        truth = _scene(
            str(case["kind"]), target_side,
            angle=float(case["angle"]), phase=float(case["phase"]),
            offset=float(case["offset"]),
        ).astype(np.float32)
        coarse = conv_basin_average(truth, (source_side, source_side))
        forward = conv_resize(coarse, truth.shape)
        reverse = reverse_conv_resize(coarse, truth.shape)
        gxx, _, gyy, _ = nodal_current_geometry(coarse)
        low, high = float(np.min(coarse)), float(np.max(coarse))
        for q in exponents:
            beta = order_coordinate(gxx, gyy, q)
            output = q1_order_blend(beta, forward, reverse)
            records.append({
                "case": case,
                "q": "infinity" if math.isinf(q) else q,
                "truth_mse": _mse(output, truth),
                "range_excursion": float(max(
                    0.0, np.max(output) - high, low - np.min(output)
                )),
                "factor_order_rms": float(np.sqrt(_mse(forward, reverse))),
            })
    summary: dict[str, object] = {}
    for q in exponents:
        key = "infinity" if math.isinf(q) else str(q)
        rows = [row for row in records if row["q"] == ("infinity" if math.isinf(q) else q)]
        by_family = {}
        for family in sorted({str(row["case"]["kind"]) for row in rows}):
            selected = [row for row in rows if row["case"]["kind"] == family]
            by_family[family] = geomean([float(row["truth_mse"]) for row in selected])
        summary[key] = {
            "truth_geomean_mse": geomean([float(row["truth_mse"]) for row in rows]),
            "maximum_range_excursion": max(float(row["range_excursion"]) for row in rows),
            "family_geomean_mse": by_family,
            "wins": sum(
                float(row["truth_mse"]) == min(
                    float(other["truth_mse"]) for other in records
                    if other["case"] == row["case"]
                )
                for row in rows
            ),
        }
    return {
        "definition": {"source_side": source_side, "target_side": target_side},
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-side", type=int, default=17)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--exponents", default="0.5,1,2,4,inf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exponents = tuple(float(value) for value in args.exponents.split(","))
    result = run(args.source_side, args.scale, exponents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
