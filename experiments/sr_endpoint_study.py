#!/usr/bin/env python3
"""Locate finite-chirp endpoint loss in the SR pipeline.

Separates latent reconstruction from reassigned readout on deterministic
complex chirps with abrupt starts/ends.  This is deliberately headless and
uses the same native kernels/geometries as the waterfall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))
import iqwaterfall as iqw  # noqa: E402
from two_lattice import MagnitudeFamily  # noqa: E402


def fixture(length=16384, seed=41):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    z = 0.006 * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    events = [
        # start, stop, normalized start/end frequency, amplitude
        (1800, 7900, -0.31, -0.08, 0.35),
        (7000, 14500, 0.05, 0.34, 0.22),
    ]
    for start, stop, f0, f1, amp in events:
        u = np.arange(stop - start)
        rate = (f1 - f0) / max(1, stop - start - 1)
        phase = 2 * np.pi * (f0 * u + 0.5 * rate * u * u)
        # Abrupt support is intentional: endpoint survival is the measurement.
        z[start:stop] += amp * np.exp(1j * phase)
    return z, events


def reassigned(z, n=4096, hop=1024):
    rows = (len(z) - n) // hop + 1
    iq = np.empty(2 * len(z), np.float32)
    iq[0::2] = z.real
    iq[1::2] = z.imag
    engine = iqw.Reassign(n)
    engine.set_bilinear(True)
    db = engine.render_mem(iq, n // 2, hop, rows)
    engine.close()
    return np.power(10.0, db / 10.0), n // 2 + np.arange(rows) * hop


def guided_reassigned(energy_signal, phase_guide, n=4096, hop=1024):
    """Unified energy scattered using measured/raw phase coordinates."""
    rows = (len(energy_signal) - n) // hop + 1
    centers = n // 2 + np.arange(rows) * hop
    g = np.hanning(n); dg = np.gradient(g)
    tg = (np.arange(n) - (n - 1) / 2) * g
    P = np.zeros((rows, n), np.float64)
    for r, c in enumerate(centers):
        es = energy_signal[c-n//2:c+n//2]
        gs = phase_guide[c-n//2:c+n//2]
        Ye = np.fft.fft(g * es)
        Y = np.fft.fft(g * gs); Yt = np.fft.fft(tg * gs); Yd = np.fft.fft(dg * gs)
        E = np.abs(Ye) ** 2; G = np.abs(Y) ** 2
        good = E > 1e-8 * (E.max() + 1e-30)
        that = r + np.real(Yt[good] * np.conj(Y[good])) / (G[good] + 1e-30) / hop
        khat = (np.arange(n)[good] - np.imag(Yd[good] * np.conj(Y[good])) /
                (G[good] + 1e-30) * n / (2*np.pi))
        for tc, fc, e in zip(that, khat, E[good]):
            c0 = int(np.floor(tc)); ft = tc-c0
            k0 = int(np.floor(fc)); ff = fc-k0
            for cc, wt in ((c0,1-ft),(c0+1,ft)):
                cc = min(rows-1,max(0,cc))
                P[cc,k0 % n] += e*wt*(1-ff)
                P[cc,(k0+1) % n] += e*wt*ff
    return np.fft.fftshift(P, axes=1), centers


def endpoint_metric(power, centers, events, n=4096):
    rows = []
    for start, stop, f0, f1, _amp in events:
        # Last analysis center still inside the event.
        valid = np.flatnonzero((centers >= start) & (centers < stop))
        r = int(valid[-1])
        frac = (centers[r] - start) / (stop - start - 1)
        freq = f0 + frac * (f1 - f0)
        k = (int(round(freq * n)) + n // 2) % n
        idx = (k + np.arange(-8, 9)) % n
        band = float(np.sum(power[r, idx]))
        total = float(np.sum(power[r])) + 1e-30
        rows.append({"row": r, "center": int(centers[r]),
                     "band_fraction": band / total, "band_power": band})
    return rows


def tail_frequency_power(power, events, n=4096, tail=0.20):
    out = []
    for _start, _stop, f0, f1, _amp in events:
        fa = f0 + (1.0 - tail) * (f1 - f0)
        lo, hi = sorted((fa, f1))
        freqs = (np.arange(n) - n // 2) / n
        mask = (freqs >= lo) & (freqs <= hi)
        out.append(float(np.sum(power[:, mask])))
    return out


def long_endpoint_magnitude(z, events, n=4096, hop=1024):
    centers = n // 2 + np.arange((len(z) - n) // hop + 1) * hop
    win = np.hanning(n)
    vals = []
    for start, stop, f0, f1, _amp in events:
        valid = np.flatnonzero((centers >= start) & (centers < stop))
        r = int(valid[-1]); c = int(centers[r])
        frac = (c - start) / (stop - start - 1)
        k = int(round((f0 + frac * (f1 - f0)) * n)) % n
        Y = np.fft.fft(z[c - n // 2:c + n // 2] * win)
        vals.append(float(np.sum(np.abs(Y[(k + np.arange(-8, 9)) % n]) ** 2)))
    return vals


def main():
    z, events = fixture()
    shared, _ = iqw.dip_run_complex(
        z, n_steps=1, fuse_seed=False, nb=4096, ns=1024)
    unified, _ = iqw.dip_unified(
        z, nb=4096, ns=1024, h_short=512, unified_steps=1,
        final_long_relax=0.0)
    long_family = MagnitudeFamily(z, 4096, 512)
    short_family = MagnitudeFamily(z, 1024, 512)
    long_exact = long_family.project(unified, unified)
    variants = {"raw": z, "shared": shared, "unified": unified}
    for relax in (0.25, 0.5, 0.75, 1.0):
        variants[f"unified+long{relax:g}"] = (
            unified + relax * (long_exact - unified))
    raw_mag = long_endpoint_magnitude(z, events)
    raw_tail = None
    for name, signal in variants.items():
        P, centers = reassigned(signal)
        metrics = endpoint_metric(P, centers, events)
        mags = long_endpoint_magnitude(signal, events)
        ratios = [m / (r + 1e-30) for m, r in zip(mags, raw_mag)]
        tails = tail_frequency_power(P, events)
        if raw_tail is None:
            raw_tail = tails
        tail_ratios = [v / (r + 1e-30) for v, r in zip(tails, raw_tail)]
        print(name, "endpoint_reassignment", metrics,
              "pre_RA_long_power_vs_raw", ratios,
              "tail_raster_power_vs_raw", tail_ratios,
              "family_error", (long_family.relative_error(signal),
                               short_family.relative_error(signal)))
    guided, centers = guided_reassigned(unified, z)
    print("unified-energy/raw-phase-guide",
          "endpoint_reassignment", endpoint_metric(guided, centers, events),
          "tail_raster_power_vs_raw",
          [v/(r+1e-30) for v, r in zip(tail_frequency_power(guided, events),
                                       raw_tail)])


if __name__ == "__main__":
    main()
