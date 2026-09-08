"""Reference comparison of density-screened Green and local CONV admission.

The Green arm uses the exact free-space constant-metric kernel
K0(sqrt(rho) r).  It is a destination reference for the admission law, not a
replacement for the variable-metric PDE discretization described in
GREEN_KERNEL_ADMISSION_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import k0

from experiments.compare_eikonal_convstar import _cases, _scene
from experiments.conv_distilled_core import (
    nodal_current_geometry,
    reverse_conv_resize,
)
from standalone_conv_resize_demo.backend import (
    conv_basin_average,
    conv_resize,
    linear_resize,
)


Array = np.ndarray


def _green_beta(eta: Array, target: tuple[int, int]) -> Array:
    """Normalized free-space Green coordinate with density-selected screening."""

    height, width = eta.shape
    source_y, source_x = np.indices((height, width), dtype=np.float64)
    source_x = source_x.reshape(-1)
    source_y = source_y.reshape(-1)
    labels = np.asarray(eta, dtype=np.float64).reshape(-1)
    target_y = np.linspace(0.0, height - 1.0, target[0])
    target_x = np.linspace(0.0, width - 1.0, target[1])
    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    points_x = xx.reshape(-1)
    points_y = yy.reshape(-1)

    # det(M)=1 and unit source spacing give one source per metric unit area.
    screening = 1.0
    numerator = np.empty(points_x.size, dtype=np.float64)
    denominator = np.empty(points_x.size, dtype=np.float64)
    block = 512
    for start in range(0, points_x.size, block):
        stop = min(start + block, points_x.size)
        dx = points_x[start:stop, None] - source_x[None, :]
        dy = points_y[start:stop, None] - source_y[None, :]
        distance = np.hypot(dx, dy)
        exact = distance == 0.0
        response = k0(np.maximum(screening * distance, np.finfo(float).tiny))
        response[exact] = 0.0
        numerator[start:stop] = response @ labels
        denominator[start:stop] = np.sum(response, axis=1)
        if np.any(exact):
            row, column = np.nonzero(exact)
            numerator[start + row] = labels[column]
            denominator[start + row] = 1.0
    return (numerator / denominator).reshape(target)


def _mse(first: Array, second: Array) -> float:
    residual = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.mean(residual * residual))


def _geomean(values: list[float]) -> float:
    data = np.maximum(np.asarray(values, dtype=np.float64), 1.0e-300)
    return float(np.exp(np.mean(np.log(data))))


def run(source_sides: tuple[int, ...]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for source_side in source_sides:
        target_side = 2 * (source_side - 1) + 1
        for case in _cases():
            truth = _scene(
                str(case["kind"]),
                target_side,
                angle=float(case["angle"]),
                phase=float(case["phase"]),
                offset=float(case["offset"]),
            ).astype(np.float32)
            coarse = conv_basin_average(
                truth, (source_side, source_side)
            ).astype(np.float32)
            forward = np.asarray(
                conv_resize(coarse, truth.shape), dtype=np.float64
            )
            reverse = np.asarray(
                reverse_conv_resize(coarse, truth.shape), dtype=np.float64
            )
            eta = nodal_current_geometry(coarse)[3]
            local_beta = np.asarray(
                linear_resize(eta.astype(np.float32), truth.shape),
                dtype=np.float64,
            )
            green_beta = _green_beta(eta, truth.shape)
            local = (1.0 - local_beta) * forward + local_beta * reverse
            green = (1.0 - green_beta) * forward + green_beta * reverse
            records.append({
                "source_side": source_side,
                "target_side": target_side,
                "case": case,
                "local_truth_mse": _mse(local, truth),
                "green_truth_mse": _mse(green, truth),
                "green_local_beta_rms": float(np.sqrt(_mse(
                    green_beta, local_beta
                ))),
                "green_local_beta_linf": float(np.max(np.abs(
                    green_beta - local_beta
                ))),
                "green_local_output_rms": float(np.sqrt(_mse(green, local))),
                "green_local_output_linf": float(np.max(np.abs(green - local))),
            })

    summary: dict[str, object] = {}
    for source_side in source_sides:
        group = [row for row in records if row["source_side"] == source_side]
        local_mse = [float(row["local_truth_mse"]) for row in group]
        green_mse = [float(row["green_truth_mse"]) for row in group]
        summary[str(source_side)] = {
            "case_count": len(group),
            "local_truth_geomean_mse": _geomean(local_mse),
            "green_truth_geomean_mse": _geomean(green_mse),
            "green_over_local_truth_mse_ratio": _geomean([
                green / max(local, 1.0e-300)
                for green, local in zip(green_mse, local_mse)
            ]),
            "green_truth_wins": sum(
                green < local for green, local in zip(green_mse, local_mse)
            ),
            "green_local_beta_rms": float(np.sqrt(np.mean([
                float(row["green_local_beta_rms"]) ** 2 for row in group
            ]))),
            "green_local_beta_maximum_linf": max(
                float(row["green_local_beta_linf"]) for row in group
            ),
            "green_local_output_rms": float(np.sqrt(np.mean([
                float(row["green_local_output_rms"]) ** 2 for row in group
            ]))),
            "green_local_output_maximum_linf": max(
                float(row["green_local_output_linf"]) for row in group
            ),
        }
    return {
        "definition": {
            "kernel": "K0(sqrt(rho) r)/(2 pi); normalization cancels 1/(2 pi)",
            "rho": 1.0,
            "rho_rule": "one source per determinant-one metric unit area",
            "geometry": "free-space constant metric reference",
            "cycle": "CONV basin analysis followed by the stated admission synthesis",
        },
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sides", default="9,17,33")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/support_geometry/green_admission_reference.json"),
    )
    args = parser.parse_args()
    source_sides = tuple(int(value) for value in args.source_sides.split(","))
    result = run(source_sides)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
