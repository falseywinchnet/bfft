#!/usr/bin/env python3
"""HD crossover and native-sideport benchmark for the sigma pipeline.

This is intentionally an isolated study: it imports the optimized experiment
kernels but does not change the viewer or library.  The FFT path reproduces
``scipy.ndimage.gaussian_filter(..., mode="reflect", truncate=4)`` by using
the identical sampled Gaussian kernels and an explicit half-sample-symmetric
halo.

Run:

    PYTHONPATH=.:viewer:experiments .venv/bin/python \
        experiments/sigma_opt/bench_fft_cpp_opportunities.py
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
import time

import numpy as np
from scipy import ndimage as ndi
from scipy import signal
from scipy.ndimage._filters import _gaussian_kernel1d

from opt_normal_assembly import (
    _accumulate,
    _render,
    build_pattern,
)
from opt_ridge_scan import _scan


LAB_WEIGHTS = np.array([1.0, 1.5, 1.5], dtype=np.float64)


def timed(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    value = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), float(np.median(samples)), value


def direct_triplet(plane, sigma):
    return np.stack(
        [
            ndi.gaussian_filter(plane, sigma, mode="reflect"),
            ndi.gaussian_filter(
                plane, sigma, order=(0, 1), mode="reflect"
            ),
            ndi.gaussian_filter(
                plane, sigma, order=(1, 0), mode="reflect"
            ),
        ]
    )


def fft_triplet(plane, sigma):
    """One input FFT, three Gaussian/derivative spectra, exact sampled kernel."""
    radius = int(4.0 * sigma + 0.5)
    smooth = _gaussian_kernel1d(sigma, 0, radius)
    deriv = _gaussian_kernel1d(sigma, 1, radius)
    kernels = np.stack(
        [
            np.outer(smooth, smooth),
            np.outer(smooth, deriv),
            np.outer(deriv, smooth),
        ]
    )
    # ndimage "reflect" is half-sample symmetric; numpy "symmetric" supplies
    # that same halo.  "valid" removes it after the linear convolution.
    padded = np.pad(
        plane, ((radius, radius), (radius, radius)), mode="symmetric"
    )
    return signal.fftconvolve(
        padded[None, ...], kernels, mode="valid", axes=(-2, -1)
    )


def run_fft():
    rng = np.random.default_rng(20260726)
    cases = [
        (512, 512),
        (720, 1280),
        (1080, 1920),
    ]
    sigmas = (0.8, 2.0, 4.0, 8.0, 16.0, 32.0)
    print("\nFFT crossover: one plane -> mean, dx, dy")
    print(
        "shape       sigma radius  direct-ms    fft-ms  speedup  max-abs-gap"
    )
    for h, w in cases:
        plane = rng.standard_normal((h, w))
        for sigma in sigmas:
            direct_ms, _, direct = timed(
                lambda: direct_triplet(plane, sigma),
                repeats=2,
                warmup=1,
            )
            fft_ms, _, transformed = timed(
                lambda: fft_triplet(plane, sigma),
                repeats=2,
                warmup=1,
            )
            gap = float(np.max(np.abs(direct - transformed)))
            radius = int(4.0 * sigma + 0.5)
            print(
                f"{w:4d}x{h:<4d} {sigma:5.1f} {radius:6d} "
                f"{direct_ms:10.2f} {fft_ms:9.2f} "
                f"{direct_ms / fft_ms:8.2f} {gap:12.3e}"
            )
            del direct, transformed


def ownership_grid(h, w, n):
    """Deterministic approximately-square cells plus a right-hand runner."""
    rows = max(1, int(round(math.sqrt(n * h / w))))
    cols = max(1, int(math.ceil(n / rows)))
    yy, xx = np.mgrid[:h, :w]
    cy = np.minimum(yy * rows // h, rows - 1)
    cx = np.minimum(xx * cols // w, cols - 1)
    owner = np.minimum(cy * cols + cx, n - 1).astype(np.int64).ravel()
    other = np.minimum(owner + 1, n - 1)
    valid = other != owner
    seed_x = (cx + 0.5) * w / cols
    seed_y = (cy + 0.5) * h / rows
    dx = (xx - seed_x).ravel().astype(np.float64)
    dy = (yy - seed_y).ravel().astype(np.float64)
    return owner, other, valid, dx, dy


def run_native_candidates():
    rng = np.random.default_rng(20260726)
    cases = [
        (512, 512, 2400),
        (720, 1280, 2400),
        (1080, 1920, 2400),
    ]
    angles = 16
    bins = 41
    theta = np.linspace(0.0, np.pi, angles, endpoint=False)
    cosines = np.cos(theta)
    sines = np.sin(theta)
    print("\nCurrent optimized JIT kernels (native-sideport candidates)")
    print(
        "shape/cells          ridge-ms pattern-ms accum-ms render-ms "
        "pattern-MB"
    )
    for h, w, n in cases:
        owner, other, valid, dx, dy = ownership_grid(h, w, n)
        npix = h * w
        weight = np.full(npix, 0.7)
        residual = rng.standard_normal((npix, 3))
        spacing = math.sqrt(npix / n)
        # Warm the same specializations on a tiny prefix through the actual
        # full calls; numba caches machine code between cases/processes.
        ridge_ms, _, _ = timed(
            lambda: _scan(
                owner,
                weight,
                residual,
                dx,
                dy,
                cosines,
                sines,
                spacing,
                n,
                angles,
                bins,
                2.5,
                LAB_WEIGHTS,
            ),
            repeats=2,
            warmup=1,
        )

        pattern_ms, _, pattern = timed(
            lambda: build_pattern(owner, other, valid, n),
            repeats=2,
            warmup=1,
        )
        first = np.column_stack(
            [np.ones(npix), dx / spacing, dy / spacing]
        )
        second = first.copy()
        w1 = weight
        w2 = 1.0 - weight

        def accumulate():
            blocks = np.zeros((pattern["blocks"], 3, 3), dtype=np.float64)
            rhs = np.zeros((n, 3, 3), dtype=np.float64)
            _accumulate(
                owner,
                other,
                valid,
                w1,
                w2,
                first,
                second,
                residual,
                pattern["diag_of"],
                pattern["slot_forward"],
                pattern["slot_reverse"],
                blocks,
                rhs,
            )
            return blocks, rhs

        accum_ms, _, _ = timed(accumulate, repeats=3, warmup=1)
        coeff = rng.standard_normal((n, 3, 3))

        def render():
            pred_first = np.empty((npix, 3))
            pred_second = np.empty((npix, 3))
            field = np.empty((npix, 3))
            _render(
                coeff,
                owner,
                other,
                valid,
                w1,
                w2,
                first,
                second,
                pred_first,
                pred_second,
                field,
            )
            return field

        render_ms, _, _ = timed(render, repeats=3, warmup=1)
        pattern_bytes = sum(
            value.nbytes
            for value in pattern.values()
            if isinstance(value, np.ndarray)
        )
        print(
            f"{w:4d}x{h:<4d}/{n:<5d} "
            f"{ridge_ms:9.2f} {pattern_ms:10.2f} "
            f"{accum_ms:8.2f} {render_ms:9.2f} "
            f"{pattern_bytes / 2**20:10.1f}"
        )
        del owner, other, valid, dx, dy, residual, first, second, pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", choices=("all", "fft", "native"), default="all"
    )
    args = parser.parse_args()
    print(
        f"{platform.platform()} | numpy {np.__version__} | "
        f"{platform.machine()}"
    )
    if args.part in ("all", "fft"):
        run_fft()
    if args.part in ("all", "native"):
        run_native_candidates()


if __name__ == "__main__":
    main()
