"""Exact constrained sweep of CONV proposal zeta and projection alpha."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_projection_metric_certificate import matrices
from experiments.convstar import (
    _ordered_sign_ledger_reference,
    raw_current_fused_bank,
)


Array = np.ndarray
FIFTH_DIFFERENCE = np.array((-1, 5, -10, 10, -5, 1), dtype=float)


def gram(alpha: float) -> Array:
    w0, w1 = matrices()
    first = np.array([[float(value) for value in row] for row in w0])
    second = np.array([[float(value) for value in row] for row in w1])
    if math.isinf(alpha):
        return second
    return first + float(alpha) * second


def project_weighted(raw: Array, signs: Array, delta: Array, weight: Array) -> Array:
    """Enumerate the 31 signed faces of each five-current fibre exactly."""

    cells, order, components = raw.shape
    values = raw.transpose(0, 2, 1).reshape(-1, order)
    orientation = signs.transpose(0, 2, 1).reshape(-1, order)
    totals = delta.reshape(-1)
    best = np.zeros_like(values)
    best_cost = np.full(values.shape[0], np.inf)
    wa = values @ weight.T
    tolerance = 2.0e-11
    for mask in range(1, 1 << order):
        active = np.array([index for index in range(order) if mask & (1 << index)])
        sub = weight[np.ix_(active, active)]
        inverse = np.linalg.inv(sub)
        mapping = inverse @ weight[active]
        base = wa[:, active] @ inverse.T
        ones = inverse @ np.ones(active.size)
        denominator = float(np.sum(ones))
        multiplier = (np.sum(base, axis=1) - totals) / denominator
        candidate_active = base - multiplier[:, None] * ones[None, :]
        feasible = np.all(
            orientation[:, active] * candidate_active >= -tolerance, axis=1
        )
        candidate = np.zeros_like(values)
        candidate[:, active] = candidate_active
        residual = candidate - values
        cost = np.einsum("ni,ij,nj->n", residual, weight, residual, optimize=True)
        update = feasible & (cost < best_cost)
        best[update] = candidate[update]
        best_cost[update] = cost[update]
    missing = ~np.isfinite(best_cost)
    if np.any(missing & (np.abs(totals) > tolerance)):
        raise RuntimeError("weighted signed-fibre projection found no feasible face")
    return best.reshape(cells, components, order).transpose(0, 2, 1)


def proposal(lines: Array, zeta0: float, zeta1: float) -> tuple[Array, Array]:
    raw, delta = raw_current_fused_bank(lines)
    if lines.shape[0] >= 6:
        windows = np.stack(
            [lines[offset:offset + lines.shape[0] - 5] for offset in range(6)],
            axis=1,
        )
        fifth = np.einsum("k,ikc->ic", FIFTH_DIFFERENCE, windows)
        zeta = np.array((
            zeta0, zeta1, -2.0 * (zeta0 + zeta1), zeta1, zeta0
        ))
        raw[2:lines.shape[0] - 3] += fifth[:, None, :] * zeta[None, :, None]
    return raw, delta


def tail_weights(u: Array) -> Array:
    u = np.asarray(u, float)
    v = 1.0 - u
    b = np.stack((
        v**5, 5*u*v**4, 10*u**2*v**3, 10*u**3*v**2,
        5*u**4*v, u**5,
    ), axis=1)
    return np.flip(np.cumsum(np.flip(b[:, 1:], axis=1), axis=1), axis=1)


def synthesize(lines: Array, current: Array, target: int) -> Array:
    sites = np.linspace(0.0, lines.shape[0] - 1.0, target)
    cell = np.floor(sites).astype(np.intp)
    cell = np.minimum(cell, lines.shape[0] - 2)
    u = sites - cell
    tail = tail_weights(u)
    return lines[cell] + np.einsum("mk,mkc->mc", tail, current[cell], optimize=True)


def battery(source_count: int, scale: int) -> tuple[Array, Array, list[dict[str, object]]]:
    target_count = scale * (source_count - 1) + 1
    xs = np.linspace(0.0, 1.0, source_count)
    xt = np.linspace(0.0, 1.0, target_count)
    source: list[Array] = []
    truth: list[Array] = []
    labels: list[dict[str, object]] = []
    nyquist = 0.5 * (source_count - 1)
    for fraction in (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75):
        cycles = fraction * nyquist
        for phase_index in range(8):
            phase = phase_index * math.pi / 4.0
            source.append(0.5 + 0.45 * np.sin(2 * math.pi * cycles * xs + phase))
            truth.append(0.5 + 0.45 * np.sin(2 * math.pi * cycles * xt + phase))
            labels.append({"family": "sine", "fraction_nyquist": fraction, "phase": phase_index})
    for phase_index in range(8):
        phase = phase_index * math.pi / 4.0
        # Instantaneous frequency rises from 1/16 to 3/4 source Nyquist.
        f0 = nyquist / 16.0
        f1 = 0.75 * nyquist
        source.append(0.5 + 0.45 * np.sin(2 * math.pi * (f0*xs + 0.5*(f1-f0)*xs**2) + phase))
        truth.append(0.5 + 0.45 * np.sin(2 * math.pi * (f0*xt + 0.5*(f1-f0)*xt**2) + phase))
        labels.append({"family": "chirp", "phase": phase_index})
    return np.stack(source, axis=1), np.stack(truth, axis=1), labels


def evaluate(
    source: Array, truth: Array, zeta0: float, zeta1: float, alpha: float
) -> dict[str, float]:
    raw, delta = proposal(source, zeta0, zeta1)
    signs = _ordered_sign_ledger_reference(raw, delta)
    current = project_weighted(raw, signs, delta, gram(alpha))
    output = synthesize(source, current, truth.shape[0])
    residual = output - truth
    variance = np.mean((truth - np.mean(truth, axis=0)) ** 2, axis=0)
    normalized = np.mean(residual * residual, axis=0) / np.maximum(variance, 1e-15)
    return {
        "maximum_normalized_mse": float(np.max(normalized)),
        "mean_normalized_mse": float(np.mean(normalized)),
        "geomean_normalized_mse": float(np.exp(np.mean(np.log(np.maximum(normalized, 1e-300))))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
    }


def run(
    source_counts: tuple[int, ...], scale: int,
    zeta0_values: tuple[float, ...], zeta1_values: tuple[float, ...],
    alphas: tuple[float, ...],
) -> dict[str, object]:
    batteries = [battery(count, scale) for count in source_counts]
    records = []
    for alpha in alphas:
        for zeta0 in zeta0_values:
            for zeta1 in zeta1_values:
                metrics = [
                    evaluate(source, truth, zeta0, zeta1, alpha)
                    for source, truth, _ in batteries
                ]
                records.append({
                    "zeta0": zeta0, "zeta1": zeta1, "alpha": alpha,
                    "maximum_normalized_mse": max(item["maximum_normalized_mse"] for item in metrics),
                    "mean_normalized_mse": float(np.mean([item["mean_normalized_mse"] for item in metrics])),
                    "geomean_normalized_mse": float(np.exp(np.mean(np.log([
                        item["geomean_normalized_mse"] for item in metrics
                    ])))),
                    "maximum_absolute_error": max(item["maximum_absolute_error"] for item in metrics),
                })
    ordered = sorted(records, key=lambda row: (
        row["maximum_normalized_mse"], row["mean_normalized_mse"]
    ))
    return {
        "definition": {
            "source_counts": source_counts, "scale": scale,
            "selection": "lexicographic minimum of maximum then mean normalized MSE",
        },
        "best": ordered[0],
        "top": ordered[:20],
        "records": records,
    }


def parse_values(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-counts", default="17,33")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--zetas", default="-0.02,-0.01,-0.005,0,0.005,0.01,0.02")
    parser.add_argument("--zeta0")
    parser.add_argument("--zeta1")
    parser.add_argument("--alphas", default="0,0.015625,0.0625,0.25,1,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        tuple(int(value) for value in args.source_counts.split(",")),
        args.scale,
        parse_values(args.zeta0 or args.zetas),
        parse_values(args.zeta1 or args.zetas),
        parse_values(args.alphas),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
