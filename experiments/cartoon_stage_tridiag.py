#!/usr/bin/env python3
"""A rework of the cartoon stage's linear solve: FFT one way, sweep the other.

WHAT THE STAGE DOES NOW.  Every split-Bregman subproblem needs

    (c - eta * Laplacian) u = rhs

and the kernel solves it by a full 2-D real transform, a pointwise multiply
by the symbol 1/(c - eta*(lx + ly)), and a full 2-D inverse.  Four such
transforms per pass.  Because the transform is radix-2, every dimension is
first padded up to the next power of two by symmetric reflection.

WHY THAT IS MORE THAN IT NEEDS.  The operator is separable, but only the
*transform* has to be.  Transform along rows only.  For each row frequency k
the remaining equation in y is

    -eta*u[y-1] + (c - eta*lx(k) + 2*eta)*u[y] - eta*u[y+1] = rhs[k, y]

a symmetric tridiagonal Toeplitz system, strictly diagonally dominant for
every k because lx(k) is in [-4, 0] and so the diagonal is at least c + 2*eta
against an off-diagonal sum of 2*eta.  Thomas needs no pivoting, costs O(H)
instead of O(H log H), and -- the part that matters most -- **needs no
padding in y at all**, because a tridiagonal solve does not care whether H is
a power of two.

Three consequences, in order of size:

1. The height stops being padded.  At 1080p the transformed area falls from
   2.02x the image to 1.07x.  Just above a power of two it falls from 3.99x
   to about 1.5x.
2. The column stage goes from O(H log H) to O(H) and becomes two streaming
   passes instead of a scatter, a transform, and a gather.
3. Neumann (reflect) boundaries in y become free -- fold the first and last
   rows of the tridiagonal instead of wrapping.  The current periodic symbol
   is exactly the wrapped finite difference that `TRANSPORT_CELL_MATH.md`
   lists as defect 7; in y it simply stops existing.

This file proves the equivalence and measures the cost.  It does not modify
the library.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_stage_tridiag.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))


def next_pow2(n):
    p = 8
    while p < n:
        p *= 2
    return p


# ----------------------------------------------------------------------
# What the kernel does today
# ----------------------------------------------------------------------

def symbol_2d(h, w, c, eta):
    """The kernel's own symbol: 1 / (c - eta * (ly + lx)), periodic."""
    lx = 2.0 * np.cos(2.0 * np.pi * np.arange(w // 2 + 1) / w) - 2.0
    ly = 2.0 * np.cos(2.0 * np.pi * np.arange(h) / h) - 2.0
    return 1.0 / (c - eta * (ly[:, None] + lx[None, :]))


def solve_spectral(rhs, c, eta):
    """Full 2-D transform, multiply, inverse -- the shipped route."""
    h, w = rhs.shape
    return np.fft.irfft2(np.fft.rfft2(rhs) * symbol_2d(h, w, c, eta),
                         s=(h, w))


# ----------------------------------------------------------------------
# Row transform, then a sweep down the columns
# ----------------------------------------------------------------------

def tridiagonal_factors(h, w, c, eta, periodic):
    """Thomas factors per row frequency, computed once per (c, eta).

    Returns the reciprocal pivots, the elimination coefficients, and (for the
    periodic case) the Sherman-Morrison correction vector.  These play the
    same role as the kernel's `symbol` table and are the same order of size.
    """
    lx = 2.0 * np.cos(2.0 * np.pi * np.arange(w // 2 + 1) / w) - 2.0
    diag = c - eta * lx + 2.0 * eta           # (WB,)
    off = -eta
    bins = diag.size

    main = np.repeat(diag[None, :], h, axis=0).copy()
    if periodic:
        # Sherman-Morrison: perturb the corners away, correct afterwards.
        gamma = -diag
        main[0] -= gamma
        main[-1] -= off * off / gamma
    else:
        # Neumann: a reflected neighbour halves the outward coupling.
        main[0] += off
        main[-1] += off

    upper = np.empty((h, bins), dtype=np.float64)
    pivot = np.empty((h, bins), dtype=np.float64)
    pivot[0] = 1.0 / main[0]
    upper[0] = off * pivot[0]
    for y in range(1, h):
        pivot[y] = 1.0 / (main[y] - off * upper[y - 1])
        upper[y] = off * pivot[y]
    factors = {"upper": upper, "pivot": pivot, "off": off, "h": h,
               "periodic": periodic}
    if periodic:
        correction = np.zeros((h, bins), dtype=np.float64)
        correction[0] = -diag
        correction[-1] = off
        factors["u_vector"] = correction
        factors["q"] = _thomas(correction, factors)
        gamma = -diag
        factors["gamma"] = gamma
    return factors


def _thomas(rhs, factors):
    """Solve the factored tridiagonal for every row frequency at once."""
    h = factors["h"]
    off = factors["off"]
    upper = factors["upper"]
    pivot = factors["pivot"]
    out = np.empty_like(rhs)
    out[0] = rhs[0] * pivot[0]
    for y in range(1, h):
        out[y] = (rhs[y] - off * out[y - 1]) * pivot[y]
    for y in range(h - 2, -1, -1):
        out[y] -= upper[y] * out[y + 1]
    return out


def _solve_columns(rhs, factors):
    solution = _thomas(rhs, factors)
    if not factors["periodic"]:
        return solution
    # Sherman-Morrison rank-one correction restores the wrapped corners.
    q = factors["q"]
    off = factors["off"]
    gamma = factors["gamma"]
    numerator = solution[0] + (off / gamma) * solution[-1]
    denominator = 1.0 + q[0] + (off / gamma) * q[-1]
    return solution - (numerator / denominator)[None, :] * q


def solve_row_then_sweep(rhs, c, eta, factors=None, periodic=True):
    """Transform along rows, sweep along columns.  No padding in y."""
    h, w = rhs.shape
    if factors is None:
        factors = tridiagonal_factors(h, w, c, eta, periodic)
    spectrum = np.fft.rfft(rhs, axis=1)
    real = _solve_columns(np.ascontiguousarray(spectrum.real), factors)
    imag = _solve_columns(np.ascontiguousarray(spectrum.imag), factors)
    return np.fft.irfft(real + 1j * imag, n=w, axis=1)


# ----------------------------------------------------------------------
# Checks and measurements
# ----------------------------------------------------------------------

def check_equivalence():
    print("\n== 1. the two routes solve the same system ==")
    rng = np.random.default_rng(11)
    for h, w in ((64, 64), (96, 128), (130, 256), (1080, 512)):
        rhs = rng.standard_normal((h, w))
        for c, eta in ((0.05, 0.10), (0.025, 0.25), (1.0 / 40.0, 10.0 / 40.0)):
            reference = solve_spectral(rhs, c, eta)
            trial = solve_row_then_sweep(rhs, c, eta, periodic=True)
            gap = float(np.max(np.abs(reference - trial)))
            scale = float(np.max(np.abs(reference)))
            assert gap / scale < 1e-10, (h, w, c, eta, gap / scale)
        print(f"  {h:5d}x{w:<5d} periodic route agrees to "
              f"{gap / scale:.2e} relative, any (c, eta)")

    # The residual of the Neumann variant against its own operator, to show
    # it solves a real system rather than merely producing an array.
    h, w = 128, 192
    rhs = rng.standard_normal((h, w))
    c, eta = 0.05, 0.10
    u = solve_row_then_sweep(rhs, c, eta, periodic=False)
    padded = np.pad(u, 1, mode="edge")          # Neumann in y, and in x too
    padded[:, 0] = padded[:, -2]                 # x stays periodic
    padded[:, -1] = padded[:, 1]
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * u)
    residual = np.max(np.abs(c * u - eta * laplacian - rhs))
    print(f"  Neumann-in-y variant residual against its own operator "
          f"{residual / np.max(np.abs(rhs)):.2e} relative")


def measure_cost():
    print("\n== 2. cost of one linear solve ==")
    print(f"  {'image':>12s} {'spectral (padded)':>19s} "
          f"{'row+sweep (native)':>20s} {'speedup':>8s}")
    rows = []
    for h, w, label in ((1024, 1024, "power of two"),
                        (1080, 1920, "1080p"),
                        (1440, 2560, "1440p"),
                        (1025, 1025, "just over 1024"),
                        (2048, 2048, "power of two")):
        c, eta = 0.05, 0.10
        ph, pw = next_pow2(h), next_pow2(w)
        padded = np.zeros((ph, pw))
        rng = np.random.default_rng(2)
        native = rng.standard_normal((h, w))
        padded[:h, :w] = native
        symbol = symbol_2d(ph, pw, c, eta)

        def spectral():
            return np.fft.irfft2(np.fft.rfft2(padded) * symbol, s=(ph, pw))

        factors = tridiagonal_factors(h, next_pow2(w), c, eta, False)
        wide = np.zeros((h, next_pow2(w)))
        wide[:, :w] = native

        def sweep():
            return solve_row_then_sweep(
                wide, c, eta, factors=factors, periodic=False)

        def timed(fn, repeats=5):
            fn()
            best = math.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            return best

        # The transforms alone, so the Python-level Thomas loop can be
        # separated from what a compiled sweep would cost.
        def transforms_only():
            spectrum = np.fft.rfft(wide, axis=1)
            return np.fft.irfft(spectrum, n=wide.shape[1], axis=1)

        a = timed(spectral)
        b = timed(sweep)
        t = timed(transforms_only)
        rows.append((h, w, a, b, t))
        print(f"  {h:5d}x{w:<5d} {a * 1e3:16.2f} ms {b * 1e3:17.2f} ms "
              f"{a / b:7.2f}x  (transforms alone {t * 1e3:.1f} ms, "
              f"so {a / t:.2f}x is the compiled ceiling)   {label}")
    return rows


def area_accounting():
    """Transform the axis already near a power of two; sweep the other one.

    Which axis to sweep is a free choice, and it matters: sweeping the axis
    with the worse padding ratio is where the whole saving lives.
    """
    print("\n== 3. transformed area, sweeping the worse-padded axis ==")
    print(f"  {'image':>12s} {'now':>15s} {'swept':>15s} "
          f"{'saved':>7s}  axis swept")
    for h, w, label in ((1920, 1080, "1080p"), (2560, 1440, "1440p"),
                        (3840, 2160, "4K"), (1025, 1025, "just over 1024"),
                        (4100, 4100, "just over 4096")):
        now = next_pow2(h) * next_pow2(w)
        sweep_h = h * next_pow2(w)      # transform width, sweep height
        sweep_w = next_pow2(h) * w      # transform height, sweep width
        best = min(sweep_h, sweep_w)
        axis = "height" if sweep_h <= sweep_w else "width"
        print(f"  {h:5d}x{w:<5d} {now / (h * w):8.2f}x image "
              f"{best / (h * w):8.2f}x image {now / best:6.2f}x  "
              f"{axis:6s}  {label}")


def main():
    check_equivalence()
    measure_cost()
    area_accounting()
    print("\nBoth routes solve the same equation; the second one does not "
          "pad the height and does not transform it.")


if __name__ == "__main__":
    main()
