"""Reproduce Duchon (1979) response criteria for Lanczos and CONV.

Duchon's filter is linear.  CONV is not, so its reported response is the
worst phase envelope of sinusoidal probes through a matched reduction and
reconstruction cycle.  The interior half of each record is fitted to the
input frequency; generated non-fundamental energy is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from standalone_conv_resize_demo.backend import conv_basin_average, conv_resize


Array = np.ndarray


def duchon_weights(cutoff: float, n: int) -> tuple[Array, Array]:
    k = np.arange(-n, n + 1, dtype=float)
    ideal = np.empty_like(k)
    ideal[k == 0] = 2.0 * cutoff
    nz = k != 0
    ideal[nz] = np.sin(2.0 * np.pi * cutoff * k[nz]) / (np.pi * k[nz])
    sigma = np.sinc(k / n)
    return k, ideal * sigma


def duchon_response(cutoff: float, n: int, frequency: Array) -> Array:
    k, weights = duchon_weights(cutoff, n)
    return np.cos(2.0 * np.pi * frequency[:, None] * k) @ weights


def _fit_response(
    source: Array, signal: Array, frequency: float, phase: float, sl: slice
) -> tuple[float, float] | None:
    index = np.arange(signal.size, dtype=float)[sl]
    observed = np.asarray(signal, float)[sl] - 0.5
    theta = 2.0 * np.pi * frequency * index + phase
    design = np.column_stack((np.sin(theta), np.cos(theta), np.ones_like(theta)))
    if np.linalg.cond(design) > 1.0e8:
        return None
    source_centered = np.asarray(source, float)[sl] - np.mean(np.asarray(source, float)[sl])
    source_rms = float(np.sqrt(np.mean(source_centered * source_centered)))
    if source_rms < 0.1:
        return None
    coefficient, *_ = np.linalg.lstsq(design, observed, rcond=None)
    fundamental = design[:, :2] @ coefficient[:2]
    residual = observed - design @ coefficient
    signed_gain = float(coefficient[0] / 0.45)
    amplitude_gain = float(np.hypot(coefficient[0], coefficient[1]) / 0.45)
    return signed_gain, float(np.sqrt(np.mean(residual * residual)) / source_rms)


def conv_phase_envelope(
    cutoff: float, frequencies: Array, phases: Array, intervals: int
) -> tuple[Array, Array, Array, float]:
    target_intervals = int(round(2.0 * cutoff * intervals))
    if target_intervals < 4:
        raise ValueError("matched coarse lattice is too small")
    index = np.arange(intervals + 1, dtype=float)
    crop = slice(intervals // 4, 3 * intervals // 4 + 1)
    lower = np.full(frequencies.shape, np.inf)
    upper = np.full(frequencies.shape, -np.inf)
    maximum_thd = np.zeros(frequencies.shape)
    range_excursion = 0.0
    for fi, frequency in enumerate(frequencies):
        for phase in phases:
            source = 0.5 + 0.45 * np.sin(
                2.0 * np.pi * float(frequency) * index + float(phase)
            )
            coarse = conv_basin_average(source.astype(np.float32), target_intervals + 1)
            output = conv_resize(coarse, intervals + 1)
            fit = _fit_response(source, output, float(frequency), float(phase), crop)
            if fit is None:
                continue
            signed, thd = fit
            lower[fi] = min(lower[fi], signed)
            upper[fi] = max(upper[fi], signed)
            maximum_thd[fi] = max(maximum_thd[fi], thd)
            range_excursion = max(
                range_excursion,
                float(max(0.0, np.max(output) - np.max(source), np.min(source) - np.min(output))),
            )
    lower[0] = upper[0] = 1.0
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("no nondegenerate phase existed at a sampled frequency")
    return lower, upper, maximum_thd, range_excursion


def _first_crossing(frequency: Array, value: Array, level: float) -> float:
    hit = np.flatnonzero(value <= level)
    if not hit.size:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(frequency[0])
    x0, x1 = frequency[i - 1:i + 1]
    y0, y1 = value[i - 1:i + 1]
    if y0 == y1:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _criteria(frequency: Array, lower: Array, upper: Array, cutoff: float) -> dict[str, float]:
    passband = frequency <= cutoff
    stopband = frequency >= cutoff
    f90 = _first_crossing(frequency, lower, 0.9)
    f10 = _first_crossing(frequency, upper, 0.1)
    half_index = int(np.argmin(np.abs(frequency - 0.5 * cutoff)))
    return {
        "G_plus": float(max(0.0, np.max(upper[passband] - 1.0))),
        "G_minus": float(max(0.0, np.max(-lower[stopband]))),
        "maximum_absolute_stop_response": float(np.max(np.maximum(
            np.abs(lower[stopband]), np.abs(upper[stopband])
        ))),
        "response_at_half_cutoff_lower": float(lower[half_index]),
        "response_at_half_cutoff_upper": float(upper[half_index]),
        "f_90_percent": f90,
        "f_10_percent": f10,
        "f_90_to_f_10_width": f10 - f90,
    }


def _duchon_step_excursion(cutoff: float, n: int) -> float:
    _, weights = duchon_weights(cutoff, n)
    # Interior response of a unit step is the cumulative FIR weight.
    response = np.concatenate(([0.0], np.cumsum(weights), [np.sum(weights)]))
    return float(max(0.0, np.max(response) - 1.0, -np.min(response)))


def _conv_step_excursion(cutoff: float, intervals: int, phase_count: int) -> float:
    target_intervals = int(round(2.0 * cutoff * intervals))
    index = np.arange(intervals + 1, dtype=float)
    maximum = 0.0
    for phase in (np.arange(phase_count) + 0.5) / phase_count:
        source = (index >= intervals / 2.0 + phase).astype(np.float32)
        coarse = conv_basin_average(source, target_intervals + 1)
        # The analytic basin means on either constant half-line are exactly
        # zero or one.  Canonicalize their float32 quadrature representatives;
        # otherwise alternating 1 and 1-2^-24 constitute artificial observed
        # sign reversals, a different input problem.
        coarse[np.abs(coarse) <= 2.0e-6] = 0.0
        coarse[np.abs(coarse - 1.0) <= 2.0e-6] = 1.0
        output = conv_resize(coarse, intervals + 1)
        maximum = max(maximum, float(max(0.0, np.max(output) - 1.0, -np.min(output))))
    return maximum


def run(intervals: int, frequency_count: int, phase_count: int) -> dict[str, object]:
    # Nyquist has only one real phase degree of freedom and makes the usual
    # sine/cosine response fit rank deficient.  Approach it from below.
    frequencies = np.linspace(0.0, 0.5, frequency_count, endpoint=False)
    phases = np.arange(phase_count, dtype=float) * (2.0 * np.pi / phase_count)
    # The first three reproduce Duchon's one-dimensional examples.  The last
    # is the y factor of his 21-by-21 two-dimensional example (x uses 0.20).
    cases = ((0.20, 10), (0.15, 15), (0.15, 25), (0.30, 10))
    records = []
    for cutoff, n in cases:
        lanczos = duchon_response(cutoff, n, frequencies)
        conv_lower, conv_upper, conv_thd, excursion = conv_phase_envelope(
            cutoff, frequencies, phases, intervals
        )
        records.append({
            "cutoff": cutoff,
            "weights": 2 * n + 1,
            "duchon_lanczos": _criteria(frequencies, lanczos, lanczos, cutoff),
            "conv_worst_phase": _criteria(frequencies, conv_lower, conv_upper, cutoff),
            "conv_maximum_generated_energy_ratio": float(np.max(conv_thd)),
            "conv_range_excursion": excursion,
            "unit_step_excursion": {
                "duchon_lanczos": _duchon_step_excursion(cutoff, n),
                "conv": _conv_step_excursion(cutoff, intervals, phase_count),
            },
            "samples": {
                "frequency": frequencies.tolist(),
                "duchon_response": lanczos.tolist(),
                "conv_lower_response": conv_lower.tolist(),
                "conv_upper_response": conv_upper.tolist(),
                "conv_generated_energy_ratio": conv_thd.tolist(),
            },
        })
    return {
        "definition": {
            "source_intervals": intervals,
            "phase_count": phase_count,
            "frequency_count": frequency_count,
            "matched_coarse_intervals": "round(2*f_c*source_intervals)",
            "fit_region": "central one-half of endpoint-aligned record",
            "amplitude": 0.45,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=1020)
    parser.add_argument("--frequency-count", type=int, default=201)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.intervals, args.frequency_count, args.phase_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["records"]:
        print(json.dumps({key: value for key, value in row.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
