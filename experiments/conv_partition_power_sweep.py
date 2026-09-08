"""Measure conservative powered partitions for CONV basin analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.duchon1979_conv_response import _criteria, _fit_response
from standalone_conv_resize_demo.backend import (
    conv_basin_average,
    conv_evaluate_profile,
    conv_resize,
)


Array = np.ndarray


def powered_partition_average(
    values: Array, target: int, power: float, quadrature_order: int = 16
) -> Array:
    """Integrate a CONV profile against normalized powered affine ownership."""

    source = np.asarray(values, dtype=np.float32).reshape(-1)
    if math.isinf(power):
        return conv_basin_average(source, target)
    if power <= 0.0:
        raise ValueError("partition power must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    interval = np.arange(source.size - 1, dtype=float)[:, None]
    sites = (interval + 0.5 * (nodes[None, :] + 1.0)).reshape(-1)
    measure = np.broadcast_to(0.5 * weights, interval.shape[:-1] + (quadrature_order,)).reshape(-1)
    profile = np.asarray(conv_evaluate_profile(source, sites), dtype=float)

    coordinate = sites * (target - 1) / (source.size - 1)
    left = np.floor(coordinate).astype(np.intp)
    left = np.minimum(left, target - 2)
    fraction = coordinate - left
    first = np.power(1.0 - fraction, power)
    second = np.power(fraction, power)
    denominator = first + second
    first /= denominator
    second /= denominator
    mass = (
        np.bincount(left, weights=measure * first, minlength=target)
        + np.bincount(left + 1, weights=measure * second, minlength=target)
    )
    moment = (
        np.bincount(left, weights=measure * first * profile, minlength=target)
        + np.bincount(left + 1, weights=measure * second * profile, minlength=target)
    )
    return (moment / mass).astype(np.float32)


def measure_case(
    cutoff: float,
    power: float,
    intervals: int,
    frequencies: Array,
    phases: Array,
) -> dict[str, object]:
    target = int(round(2.0 * cutoff * intervals)) + 1
    index = np.arange(intervals + 1, dtype=float)
    crop = slice(intervals // 4, 3 * intervals // 4 + 1)
    lower = np.full(frequencies.shape, np.inf)
    upper = np.full(frequencies.shape, -np.inf)
    generated = np.zeros(frequencies.shape)
    for fi, frequency in enumerate(frequencies):
        for phase in phases:
            source = 0.5 + 0.45 * np.sin(
                2.0 * np.pi * float(frequency) * index + float(phase)
            )
            coarse = powered_partition_average(source, target, power)
            output = conv_resize(coarse, intervals + 1)
            fit = _fit_response(source, output, float(frequency), float(phase), crop)
            if fit is None:
                continue
            response, distortion = fit
            lower[fi] = min(lower[fi], response)
            upper[fi] = max(upper[fi], response)
            generated[fi] = max(generated[fi], distortion)
    lower[0] = upper[0] = 1.0

    step_excursion = 0.0
    conservation_error = 0.0
    for offset in (0.125, 0.375, 0.625, 0.875):
        source = (index >= intervals / 2.0 + offset).astype(np.float32)
        coarse = powered_partition_average(source, target, power)
        coarse[np.abs(coarse) <= 2.0e-6] = 0.0
        coarse[np.abs(coarse - 1.0) <= 2.0e-6] = 1.0
        output = conv_resize(coarse, intervals + 1)
        step_excursion = max(
            step_excursion,
            float(max(0.0, np.max(output) - 1.0, -np.min(output))),
        )
        # Weighted masses partition the continuous domain exactly.  Equal
        # interior masses make this equivalent to the usual coarse sum away
        # from the two half-mass endpoints.
        expected = float(np.mean(source))
        reconstructed_mean = float(np.trapezoid(output) / intervals)
        conservation_error = max(conservation_error, abs(reconstructed_mean - expected))
    criteria = _criteria(frequencies, lower, upper, cutoff)
    return {
        "cutoff": cutoff,
        "power": "infinity" if math.isinf(power) else power,
        "criteria": criteria,
        "maximum_generated_energy_ratio": float(np.max(generated)),
        "unit_step_excursion": step_excursion,
        "cycle_mean_error_on_step": conservation_error,
    }


def run(intervals: int, frequency_count: int, phase_count: int, powers: tuple[float, ...]) -> dict[str, object]:
    frequency = np.linspace(0.0, 0.5, frequency_count, endpoint=False)
    phases = np.arange(phase_count) * (2.0 * np.pi / phase_count)
    records = [
        measure_case(cutoff, power, intervals, frequency, phases)
        for cutoff in (0.15, 0.20, 0.30)
        for power in powers
    ]
    return {
        "definition": {
            "intervals": intervals,
            "frequency_count": frequency_count,
            "phase_count": phase_count,
            "quadrature_order": 16,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=510)
    parser.add_argument("--frequency-count", type=int, default=201)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--powers", default="1,2,4,8,16,inf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    powers = tuple(float(value) for value in args.powers.split(","))
    result = run(args.intervals, args.frequency_count, args.phase_count, powers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for record in result["records"]:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
