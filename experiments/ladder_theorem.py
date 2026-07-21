#!/usr/bin/env python3
"""NEXT-PROJECT task 10: the ladder theorem, empirically grounded.

Two claims measured:

(i) CLOSED FORM: the pin strength of a packet pair under window N is the
    normalized spectrogram CROSS-OVERLAP of the two components.  Linearizing
    the gauge rotation b -> e^{i phi} b: per T-F cell, |A(a + e^{i phi} b)|
    differs from |A(a+b)| by ~ the interference term, so

        residual^2 ~ sum_cells (|A a| |A b|)^2 / sum_cells |A z|^4-ish;

    empirically we test the simplest predictor
    O_N = sqrt(sum |A a|^2 |A b|^2) / sum |A z|^2 against the measured
    gauge residual over a (dt, df) grid x window ladder.  If it holds, the
    coverage function is computable analytically from single-component
    spectrograms -- no probe needed -- and coverage = transversality
    becomes a formula.

(ii) r*(w): for geometric ladders with ratio r, the interior worst-case
    strength along the dt ridge (df=0) relative to the on-rung peak, vs r,
    for packet widths w.  The ladder theorem candidate: no interior hole
    iff r <= r*(w); r* shrinks as packets narrow relative to rung spacing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))
OUT = Path(__file__).resolve().parent / "out"

from two_lattice import MagnitudeFamily  # noqa: E402

L = 8192


def packet(t0, f, width, phase=0.0):
    env = np.zeros(L)
    env[t0:t0 + width] = np.hanning(width)
    t = np.arange(L, dtype=np.float64)
    return env * np.exp(2j * np.pi * f * t + 1j * phase)


def measured_and_predicted(a, b, n, h):
    z = a + b
    fam = MagnitudeFamily(z, n, h)
    zr = z + (np.exp(1j * np.pi / 2) - 1.0) * b
    measured = fam.relative_error(zr)
    Sa = np.abs(np.fft.fft(a[fam.indices] * fam.window[None, :], axis=1))
    Sb = np.abs(np.fft.fft(b[fam.indices] * fam.window[None, :], axis=1))
    # Rotating b moves |A z| only in cells BOTH components occupy, by ~ the
    # smaller of the two: residual ~ sqrt(2 sum min^2 / sum |A z|^2).
    # (A product form (Sa Sb)^2 measures the SQUARE of this: slope 1/2.)
    overlap = np.sqrt(2.0 * np.sum(np.minimum(Sa, Sb) ** 2) /
                      (np.sum(fam.target ** 2) + 1e-30))
    return measured, overlap


def part_i():
    print("=== (i) pin strength == spectrogram cross-overlap? ===")
    rng = np.random.default_rng(2)
    meas, pred = [], []
    w = 200
    for _ in range(120):
        dt = int(rng.integers(60, 2200))
        df = rng.uniform(0, 12.0 / 1024)
        n = int(rng.choice([256, 512, 1024, 2048]))
        a = packet(2800, 0.07, w)
        b = packet(2800 + dt, 0.07 + df, w)
        m, p = measured_and_predicted(a, b, n, n // 8)
        if m > 1e-4:                      # above the numerical gauge floor
            meas.append(m)
            pred.append(p)
    meas, pred = np.array(meas), np.array(pred)
    lm, lp = np.log10(meas), np.log10(pred)
    r = np.corrcoef(lm, lp)[0, 1]
    slope = np.polyfit(lp, lm, 1)[0]
    ratio = meas / pred
    print(f"  {len(meas)} pinned cases: log-log Pearson r = {r:.4f}, "
          f"slope = {slope:.3f}")
    print(f"  measured/predicted ratio: median {np.median(ratio):.3f}, "
          f"IQR {np.percentile(ratio,25):.3f}..{np.percentile(ratio,75):.3f}")
    return meas, pred


def ridge_strength(w, rungs, dts):
    """Ladder strength along the df=0 ridge (max over rungs per dt)."""
    out = np.zeros(len(dts))
    for j, dt in enumerate(dts):
        a = packet(2600, 0.07, w)
        b = packet(2600 + int(dt), 0.07, w)
        best = 0.0
        for n in rungs:
            m, _ = measured_and_predicted(a, b, n, max(32, n // 8))
            best = max(best, m)
        out[j] = best
    return out


def part_ii():
    """Pin strength decays intrinsically with dt even at the optimal rung
    (fewer shared cells, magnitude spread over more bins), so a raw
    dip-to-peak ratio conflates that decay with ladder holes.  The hole
    metric is the ladder's strength relative to the DENSE (r=2) envelope
    at the same dt: hole(r) = min_dt s_r(dt)/s_2(dt)."""
    print("\n=== (ii) interior hole vs geometric ratio r (df=0 ridge) ===")
    print(f"{'w':>5s} {'r':>4s} {'rungs':>26s} "
          f"{'min_dt s_r/s_dense':>19s}")
    for w in (100, 200, 400):
        n0 = 256
        dts = np.unique(np.linspace(0.8 * n0, 0.8 * 4096, 24).astype(int))
        curves = {}
        for r in (2, 4, 8):
            rungs = [n0]
            while rungs[-1] * r <= 4096:
                rungs.append(rungs[-1] * r)
            curves[r] = (rungs, ridge_strength(w, rungs, dts))
        dense = curves[2][1] + 1e-30
        for r in (2, 4, 8):
            rungs, s = curves[r]
            hole = float(np.min(s / dense))
            print(f"{w:5d} {r:4d} {str(rungs):>26s} {hole:19.3f}")
    print("\n  hole ~1 means the sparse ladder matches the dense envelope "
          "everywhere; the r where it collapses is r*(w).")


if __name__ == "__main__":
    m, p = part_i()
    part_ii()