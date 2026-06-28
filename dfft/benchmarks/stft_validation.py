"""Validate and time BFFT's native STFTPlan against the reference Numba STFT.

Run from a checkout after building/installing the package::

    python benchmarks/stft_validation.py

The comparison imports the repository-root ``stft.py`` reference
implementation. BFFT's public ``STFTPlan`` stores its inverse overlap buffer
internally, so validation resets it before each timed fresh inverse stream.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import bfft
from stft import STFT as ReferenceSTFT


def _bench(label, fn, repeats=5):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    print(f"{label:24s} {best * 1e3:9.3f} ms")
    return result


def _signal_from_istft_result(result, label):
    if result is None:
        raise RuntimeError(f"{label} returned None; expected an array or (array, buffer)")
    if isinstance(result, tuple):
        if len(result) == 0:
            raise RuntimeError(f"{label} returned an empty tuple")
        return result[0]
    if isinstance(result, list):
        if len(result) == 0:
            raise RuntimeError(f"{label} returned an empty list")
        return result[0]
    return result


def _fresh_native_istft(plan, Zx):
    plan.reset_buffer()
    return plan.istft(Zx)


def validate(transform="rfft", n=24576, n_fft=512, hop=128):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n)

    ref = ReferenceSTFT(n=n, n_fft=n_fft, hop_length=hop, transform=transform)
    native = bfft.STFTPlan(n=n, n_fft=n_fft, hop_length=hop, transform=transform)

    Z_ref = _bench(f"reference {transform} stft", lambda: ref.stft(x))
    Z_native = _bench(f"native {transform} stft", lambda: native.stft(x))

    xr_native = _bench(
        f"native {transform} istft",
        lambda: _fresh_native_istft(native, Z_native),
    )
    xr_ref_result = _bench(f"reference {transform} istft", lambda: ref.istft(Z_ref))
    xr_ref = _signal_from_istft_result(xr_ref_result, "reference istft")

    z_err = np.max(np.abs(Z_native - Z_ref))
    x_err = np.max(np.abs(xr_native - xr_ref))
    print(f"{transform} max |Z_native - Z_ref| = {z_err:.3e}")
    print(f"{transform} max |x_native - x_ref| = {x_err:.3e}")
    if z_err > 1e-10 or x_err > 1e-10:
        raise SystemExit(f"{transform} validation failed")


if __name__ == "__main__":
    validate("rfft")
    validate("odft")
