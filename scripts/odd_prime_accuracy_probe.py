#!/usr/bin/env python3
"""Odd/composite generalized-Bruun accuracy probe.

This is an intentionally slow metrology script for scratch_genbruun_exact.py.  It
compares the generalized Bruun prototype and NumPy rfft against an mpmath DFT
reference evaluated at high precision.  The final reductions use math.fsum so
that the reported RMS values are not dominated by Python summation order.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scratch_genbruun_exact import rfft_gen_exact  # noqa: E402


def factor_string(n: int) -> str:
    factors: list[str] = []
    d = 2
    value = n
    while d * d <= value:
        count = 0
        while value % d == 0:
            value //= d
            count += 1
        if count == 1:
            factors.append(str(d))
        if count > 1:
            factors.append(f"{d}^{count}")
        d += 1
    if value > 1:
        factors.append(str(value))
    return " * ".join(factors)


def mpmath_rfft(x: np.ndarray, dps: int) -> np.ndarray:
    mp.mp.dps = dps
    n = int(x.shape[0])
    result: list[complex] = []
    for k in range(n // 2 + 1):
        real_part = mp.mpf("0")
        imag_part = mp.mpf("0")
        for sample_index, sample_value in enumerate(x):
            angle = -2 * mp.pi * k * sample_index / n
            exact_sample = mp.mpf(str(float(sample_value)))
            real_part += exact_sample * mp.cos(angle)
            imag_part += exact_sample * mp.sin(angle)
        result.append(complex(float(real_part), float(imag_part)))
    return np.asarray(result, dtype=np.complex128)


def row_for_size(n: int, dps: int) -> tuple[str, ...]:
    x = np.random.default_rng(n).standard_normal(n)
    reference = mpmath_rfft(x, dps)
    generalized = rfft_gen_exact(x)
    numpy_rfft = np.fft.rfft(x)

    scale = max(float(np.max(np.abs(reference))), 1.0)
    generalized_error = np.abs(generalized - reference) / scale
    numpy_error = np.abs(numpy_rfft - reference) / scale

    generalized_rms = math.sqrt(
        math.fsum(float(v * v) for v in generalized_error) / len(generalized_error)
    )
    numpy_rms = math.sqrt(
        math.fsum(float(v * v) for v in numpy_error) / len(numpy_error)
    )

    if float(np.max(numpy_error)) == 0.0:
        ratio = math.inf
    else:
        ratio = float(np.max(generalized_error)) / float(np.max(numpy_error))

    return (
        str(n),
        factor_string(n),
        f"{float(np.max(generalized_error)):.3e}",
        f"{float(np.max(numpy_error)):.3e}",
        f"{generalized_rms:.3e}",
        f"{numpy_rms:.3e}",
        f"{ratio:.2f}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=int, default=90)
    parser.add_argument(
        "sizes",
        nargs="*",
        type=int,
        default=[3, 5, 7, 9, 15, 21, 25, 27, 45, 75, 81, 125, 225, 243, 405, 729],
    )
    args = parser.parse_args()

    header = (
        "N",
        "factors",
        "max_bruun",
        "max_numpy",
        "rms_bruun",
        "rms_numpy",
        "max_ratio",
    )
    print("\t".join(header))
    for n in args.sizes:
        print("\t".join(row_for_size(n, args.dps)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
