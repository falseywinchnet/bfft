"""Benchmark and audit the bin-driven CONV* FIR design prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.convstar_filter_design import (
    design_from_bins,
    design_minimax_from_bins,
    frequency_response,
    linear_phase_target,
)


def run() -> dict[str, object]:
    records = []
    for length in (16, 32, 64, 128, 256, 512):
        bin_count = max(1025, 8 * length + 1)
        frequencies = np.linspace(0.0, 0.5, bin_count)
        cutoff = 0.18
        transition = 0.04
        amplitude = np.where(
            frequencies <= cutoff,
            1.0,
            np.where(
                frequencies >= cutoff + transition,
                0.0,
                0.5 + 0.5 * np.cos(
                    np.pi * (frequencies - cutoff) / transition
                ),
            ),
        )
        desired = linear_phase_target(amplitude, frequencies, length)
        design = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            symmetry="even",
            exact_frequencies=[0.0],
            exact_values=[1.0],
            regularization=1e-14,
        )
        dense = np.linspace(0.0, 0.5, 16 * bin_count + 1)
        dense_response = frequency_response(design.taps, dense)
        records.append({
            "length": length,
            "bins": bin_count,
            "solve_seconds": design.certificate.solve_seconds,
            "weighted_rms_error": design.certificate.weighted_rms_error,
            "maximum_weighted_error_on_bins": (
                design.certificate.maximum_weighted_error
            ),
            "stationarity_residual": design.certificate.stationarity_residual,
            "exact_constraint_residual": (
                design.certificate.exact_constraint_residual
            ),
            "dense_peak_gain": float(np.max(np.abs(dense_response))),
        })

    rng = np.random.default_rng(20260831)
    arbitrary_records = []
    for length in (16, 32, 64, 128, 256, 512):
        bin_count = max(1025, 8 * length + 1)
        frequencies = np.sort(rng.uniform(0.0, 0.5, bin_count))
        desired = (
            rng.normal(size=bin_count) + 1j * rng.normal(size=bin_count)
        )
        weights = np.exp(rng.uniform(-3.0, 3.0, size=bin_count))
        result = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            weights=weights,
            regularization=1.0e-10,
        )
        arbitrary_records.append({
            "length": length,
            "bins": bin_count,
            "solve_seconds": result.certificate.solve_seconds,
            "weighted_rms_error": result.certificate.weighted_rms_error,
            "stationarity_residual": result.certificate.stationarity_residual,
        })

    passband = np.linspace(0.0, 0.18, 600)
    stopband = np.linspace(0.24, 0.5, 800)
    frequencies = np.concatenate((passband, stopband))
    amplitude = np.concatenate((np.ones(passband.size), np.zeros(stopband.size)))
    minimax_records = []
    for length in (31, 63, 127):
        result = design_minimax_from_bins(
            amplitude,
            length=length,
            frequencies=frequencies,
            symmetry="even",
            exact_frequencies=[0.0],
            exact_amplitudes=[1.0],
        )
        minimax_records.append({
            "length": length,
            "bins": frequencies.size,
            "solve_seconds": result.certificate.solve_seconds,
            "maximum_weighted_error": (
                result.certificate.maximum_weighted_error
            ),
            "lp_residual": result.certificate.stationarity_residual,
        })

    # Classical Type-I low-pass comparison.  Both methods use the same two
    # declared bands; the dense audit detects peaks between the LP bins.
    from scipy.signal import freqz, remez

    comparison_length = 63
    comparison_bins = np.concatenate((
        np.linspace(0.0, 0.18, 1200),
        np.linspace(0.24, 0.5, 1600),
    ))
    comparison_amplitude = np.concatenate((
        np.ones(1200), np.zeros(1600),
    ))
    projected = design_minimax_from_bins(
        comparison_amplitude,
        length=comparison_length,
        frequencies=comparison_bins,
        symmetry="even",
    )
    remez_started = perf_counter()
    exchanged_taps = remez(
        comparison_length,
        (0.0, 0.18, 0.24, 0.5),
        (1.0, 0.0),
        fs=1.0,
        grid_density=128,
        maxiter=100,
    )
    remez_seconds = perf_counter() - remez_started
    dense_frequency = np.concatenate((
        np.linspace(0.0, 0.18, 100_001),
        np.linspace(0.24, 0.5, 100_001),
    ))
    dense_target = np.concatenate((
        np.ones(100_001), np.zeros(100_001),
    ))
    projected_response = frequency_response(projected.taps, dense_frequency)
    _, exchanged_response = freqz(
        exchanged_taps, worN=2.0 * np.pi * dense_frequency
    )
    projected_amplitude = (
        np.exp(
            2j * np.pi * dense_frequency * ((comparison_length - 1) / 2.0)
        ) * projected_response
    ).real
    exchanged_amplitude = (
        np.exp(
            2j * np.pi * dense_frequency * ((comparison_length - 1) / 2.0)
        ) * exchanged_response
    ).real
    comparison = {
        "length": comparison_length,
        "convstar_lp_seconds": projected.certificate.solve_seconds,
        "convstar_discrete_peak": projected.certificate.maximum_weighted_error,
        "convstar_dense_peak": float(
            np.max(np.abs(projected_amplitude - dense_target))
        ),
        "remez_dense_peak": float(
            np.max(np.abs(exchanged_amplitude - dense_target))
        ),
        "remez_seconds": remez_seconds,
        "maximum_tap_difference": float(
            np.max(np.abs(projected.taps - exchanged_taps))
        ),
    }
    return {
        "weighted_l2": records,
        "arbitrary_complex_weighted_l2": arbitrary_records,
        "discrete_minimax": minimax_records,
        "remez_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
