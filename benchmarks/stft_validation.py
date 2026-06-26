"""Validate and time BFFT's native STFTPlan against the reference Numba STFT.

Run from a checkout after building/installing the package::

    python benchmarks/stft_validation.py

The comparison uses the repository's reference ``stft.py`` implementation, which
keeps the original fully-nopython algorithm. BFFT's public ``STFTPlan`` stores
its inverse overlap buffer internally, so validation resets it before fresh
inverse streams.
"""

from __future__ import annotations

import time
import sys
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


def validate(transform="rfft", n=24576, n_fft=512, hop=128):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n)

    ref = ReferenceSTFT(n=n, n_fft=n_fft, hop_length=hop, transform=transform).compile()
    native = bfft.STFTPlan(n=n, n_fft=n_fft, hop_length=hop, transform=transform)

    Z_ref = _bench(f"reference {transform} stft", lambda: ref.stft(x))
    Z_native = _bench(f"native {transform} stft", lambda: native.stft(x))

    native.reset_buffer()
    xr_native = _bench(f"native {transform} istft", lambda: native.istft(Z_native))
    xr_ref, _ = _bench(f"reference {transform} istft", lambda: ref.istft(Z_ref))

    z_err = np.max(np.abs(Z_native - Z_ref))
    x_err = np.max(np.abs(xr_native - xr_ref))
    print(f"{transform} max |Z_native - Z_ref| = {z_err:.3e}")
    print(f"{transform} max |x_native - x_ref| = {x_err:.3e}")
    if z_err > 1e-10 or x_err > 1e-10:
        raise SystemExit(f"{transform} validation failed")


if __name__ == "__main__":
    validate("rfft")
    validate("odft")
