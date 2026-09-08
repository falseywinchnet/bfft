"""Two-dimensional universality check for CONV zeta/alpha candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_order_exponent_sweep import _cases, _mse, _scene
from experiments.conv_distilled_core import distilled_conv_synthesis, nodal_current_geometry
from experiments.conv_zeta_alpha_sweep import (
    gram, proposal, project_weighted, synthesize,
)
from experiments.convstar import _ordered_sign_ledger_reference
from standalone_conv_resize_demo.backend import conv_basin_average, q1_order_blend


def resize_axis(values: np.ndarray, target: int, axis: int, z0: float, z1: float, weight: np.ndarray) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, float), axis, 0)
    shape = moved.shape
    lines = moved.reshape(shape[0], -1)
    raw, delta = proposal(lines, z0, z1)
    signs = _ordered_sign_ledger_reference(raw, delta)
    current = project_weighted(raw, signs, delta, weight)
    output = synthesize(lines, current, target).reshape((target,) + shape[1:])
    return np.moveaxis(output, 0, axis)


def synthesis(values: np.ndarray, target: tuple[int, int], z0: float, z1: float, weight: np.ndarray, reverse: bool) -> np.ndarray:
    if reverse:
        return resize_axis(resize_axis(values, target[0], 0, z0, z1, weight), target[1], 1, z0, z1, weight)
    return resize_axis(resize_axis(values, target[1], 1, z0, z1, weight), target[0], 0, z0, z1, weight)


def run(
    zeta0_values: tuple[float, ...], zeta1_values: tuple[float, ...],
    alphas: tuple[float, ...],
) -> dict[str, object]:
    cases = []
    for case in _cases():
        truth = _scene(str(case["kind"]), 65, angle=float(case["angle"]), phase=float(case["phase"]), offset=float(case["offset"])).astype(np.float32)
        coarse = conv_basin_average(truth, (17, 17))
        baseline = distilled_conv_synthesis(coarse, truth.shape)
        beta = nodal_current_geometry(coarse)[3].astype(np.float32)
        cases.append((case, truth, coarse, baseline, beta))
    records = []
    for alpha in alphas:
        weight = gram(alpha)
        for z0 in zeta0_values:
            for z1 in zeta1_values:
                errors = []
                ratios = []
                wins = 0
                for _, truth, coarse, baseline, beta in cases:
                    forward = synthesis(coarse, truth.shape, z0, z1, weight, False)
                    reverse = synthesis(coarse, truth.shape, z0, z1, weight, True)
                    output = q1_order_blend(beta, forward.astype(np.float32), reverse.astype(np.float32))
                    error = _mse(output, truth)
                    base = _mse(baseline, truth)
                    errors.append(error)
                    ratios.append(error / base)
                    wins += error < base
                records.append({
                    "zeta0": z0, "zeta1": z1,
                    "alpha": "infinity" if math.isinf(alpha) else alpha,
                    "truth_geomean_mse": float(np.exp(np.mean(np.log(errors)))),
                    "geomean_ratio_to_current": float(np.exp(np.mean(np.log(ratios)))),
                    "maximum_ratio_to_current": max(ratios),
                    "wins": wins,
                })
    ordered = sorted(records, key=lambda row: (
        row["maximum_ratio_to_current"], row["geomean_ratio_to_current"]
    ))
    return {"best": ordered[0], "top": ordered[:20], "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zetas", default="-.02,-.01,-.005,0,.005,.01,.02")
    parser.add_argument("--zeta0")
    parser.add_argument("--zeta1")
    parser.add_argument("--alphas", default="0,.25,1,inf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        tuple(float(x) for x in (args.zeta0 or args.zetas).split(",")),
        tuple(float(x) for x in (args.zeta1 or args.zetas).split(",")),
        tuple(float(x) for x in args.alphas.split(",")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
