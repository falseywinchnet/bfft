#!/usr/bin/env python3
"""Constant-metric control for continuous local transport direction.

The graph walk restricts a characteristic to stencil edges.  This experiment
uses the same Cartesian samples but performs the Hopf--Lax update over each
triangle of a metric-reduced superbase.  The interpolation coordinate inside
the opposing edge is continuous, so path direction is not enumerated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from port_needed.metric_reduced_stencil import (  # noqa: E402
    metric_reduced_superbase,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


def ordered_signed_superbase(superbase: np.ndarray) -> np.ndarray:
    vectors = np.concatenate((superbase, -superbase), axis=0)
    angle = np.arctan2(vectors[:, 1], vectors[:, 0])
    return np.ascontiguousarray(vectors[np.argsort(angle)], dtype=np.int32)


@_compile
def _norm(dx, dy, mxx, mxy, myy):
    return math.sqrt(max(
        mxx * dx * dx + 2.0 * mxy * dx * dy + myy * dy * dy,
        1e-30,
    ))


@_compile
def _triangle_update(
    first_value,
    second_value,
    first_x,
    first_y,
    second_x,
    second_y,
    mxx,
    mxy,
    myy,
):
    """Minimize interpolated arrival plus metric distance on one edge."""
    delta_value = second_value - first_value
    delta_x = second_x - first_x
    delta_y = second_y - first_y

    def derivative(t):
        rx = first_x + t * delta_x
        ry = first_y + t * delta_y
        length = _norm(rx, ry, mxx, mxy, myy)
        directional = (
            delta_x * (mxx * rx + mxy * ry)
            + delta_y * (mxy * rx + myy * ry)
        )
        return delta_value + directional / length

    left_derivative = derivative(0.0)
    if left_derivative >= 0.0:
        return first_value + _norm(
            first_x, first_y, mxx, mxy, myy)
    right_derivative = derivative(1.0)
    if right_derivative <= 0.0:
        return second_value + _norm(
            second_x, second_y, mxx, mxy, myy)

    low, high = 0.0, 1.0
    for _ in range(24):
        middle = 0.5 * (low + high)
        if derivative(middle) < 0.0:
            low = middle
        else:
            high = middle
    t = 0.5 * (low + high)
    rx = first_x + t * delta_x
    ry = first_y + t * delta_y
    return (
        first_value
        + t * delta_value
        + _norm(rx, ry, mxx, mxy, myy)
    )


@_compile
def _sweep_distance(
    size: int,
    mxx: float,
    mxy: float,
    myy: float,
    directions: np.ndarray,
    cycles: int,
) -> np.ndarray:
    distance = np.full((size, size), 1e300)
    source = size // 2
    distance[source, source] = 0.0
    for _ in range(cycles):
        for sweep in range(4):
            if sweep == 0 or sweep == 1:
                y_start, y_stop, y_step = 0, size, 1
            else:
                y_start, y_stop, y_step = size - 1, -1, -1
            if sweep == 0 or sweep == 2:
                x_start, x_stop, x_step = 0, size, 1
            else:
                x_start, x_stop, x_step = size - 1, -1, -1
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    if y == source and x == source:
                        continue
                    value = distance[y, x]
                    count = len(directions)
                    for index in range(count):
                        ux = directions[index, 0]
                        uy = directions[index, 1]
                        vx = directions[(index + 1) % count, 0]
                        vy = directions[(index + 1) % count, 1]
                        first_x, first_y = x + ux, y + uy
                        second_x, second_y = x + vx, y + vy
                        first_inside = (
                            0 <= first_x < size and 0 <= first_y < size)
                        second_inside = (
                            0 <= second_x < size and 0 <= second_y < size)
                        if first_inside:
                            candidate = (
                                distance[first_y, first_x]
                                + _norm(ux, uy, mxx, mxy, myy)
                            )
                            if candidate < value:
                                value = candidate
                        if second_inside:
                            candidate = (
                                distance[second_y, second_x]
                                + _norm(vx, vy, mxx, mxy, myy)
                            )
                            if candidate < value:
                                value = candidate
                        if first_inside and second_inside:
                            first_value = distance[first_y, first_x]
                            second_value = distance[second_y, second_x]
                            if first_value < 1e299 and second_value < 1e299:
                                candidate = _triangle_update(
                                    first_value,
                                    second_value,
                                    ux,
                                    uy,
                                    vx,
                                    vy,
                                    mxx,
                                    mxy,
                                    myy,
                                )
                                if candidate < value:
                                    value = candidate
                    distance[y, x] = value
    return distance


def metric(angle_degrees: float, condition: float):
    angle = math.radians(angle_degrees)
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    matrix = (
        np.outer(tangent, tangent)
        + float(condition) * np.outer(normal, normal)
    )
    return matrix


def run(size: int, condition: float, cycles: int):
    records = []
    fields = []
    for angle in np.linspace(0.0, 45.0, 10):
        matrix = metric(float(angle), condition)
        superbase = metric_reduced_superbase(
            np.array([[matrix[0, 0]]]),
            np.array([[matrix[0, 1]]]),
            np.array([[matrix[1, 1]]]),
        )[0, 0]
        directions = ordered_signed_superbase(superbase)
        measured = _sweep_distance(
            size,
            matrix[0, 0],
            matrix[0, 1],
            matrix[1, 1],
            directions,
            cycles,
        )
        yy, xx = np.mgrid[:size, :size]
        center = size // 2
        dx = xx - center
        dy = yy - center
        exact = np.sqrt(np.maximum(
            matrix[0, 0] * dx * dx
            + 2.0 * matrix[0, 1] * dx * dy
            + matrix[1, 1] * dy * dy,
            0.0,
        ))
        radius = np.hypot(dx, dy)
        mask = (radius >= 0.30 * size) & (radius <= 0.42 * size)
        relative = np.abs(measured[mask] - exact[mask]) / np.maximum(
            exact[mask], 1e-12)
        records.append({
            "angle": float(angle),
            "mean_relative_error": float(np.mean(relative)),
            "p95_relative_error": float(np.percentile(relative, 95.0)),
            "maximum_reduced_reach": int(np.max(np.abs(superbase))),
            "superbase": superbase.tolist(),
        })
        fields.append((angle, measured, exact))
    return records, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=129)
    parser.add_argument("--condition", type=float, default=64.0)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/hopf_lax_rotation.png",
    )
    args = parser.parse_args()
    records, fields = run(args.size, args.condition, args.cycles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2))

    angles = [item["angle"] for item in records]
    mean = [item["mean_relative_error"] for item in records]
    p95 = [item["p95_relative_error"] for item in records]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(angles, mean, marker="o", label="mean")
    axes[0].plot(angles, p95, marker="o", label="p95")
    axes[0].set_xlabel("metric tangent angle")
    axes[0].set_ylabel("relative distance error")
    axes[0].legend()
    for axis, field_index in zip(axes[1:], (0, len(fields) // 2)):
        angle, measured, exact = fields[field_index]
        error = (measured - exact) / np.maximum(exact, 1e-12)
        image = axis.imshow(error, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        axis.set_title(f"{angle:.1f}° signed relative error")
        axis.axis("off")
    figure.colorbar(image, ax=axes[1:], shrink=0.8)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    print(json.dumps({
        "mean_error_range": [min(mean), max(mean)],
        "p95_error_range": [min(p95), max(p95)],
        "output": str(args.output),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
