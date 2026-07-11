#!/usr/bin/env python3
"""NEXT-PROJECT tasks 1-3: coverage maps, transversality, dead-zone recovery.

1. Coverage map.  For packet pairs on a (dt, df) grid, each window family's
   gauge residual (rotate packet B by 90 deg, measure |STFT| change) maps
   what that rung can pin.  A ladder's coverage is the max over its rungs.
   Verifies the visibility law  dt + w <~ N,  df - 1/w <~ 1/N  and the
   ratio-4 no-hole claim, and shows what a 2048 rung adds to {1024, 256}.

2. Transversality vs ladder.  Near the true signal, fit the asymptotic AP
   contraction rate with 2 vs 3 rungs.  Separates "a rung raises the
   manifold angle" (faster) from "a rung only extends coverage" (same rate,
   more pinned directions).

3. Dead-zone recovery.  A slow pair (dt = 1600 > NB) starts ON the gauge
   orbit (packet B rotated by a random angle).  If the ladder cannot pin
   the pair, iterations retain the rotation; if it can, they restore it.
   Recovered relative-phase error, multi-seed, 2-rung vs 3-rung: the audio
   analog of the third-blur-angle image measurement.
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
W = 200                       # packet width
T0 = 2800
RUNGS = {4096: 512, 2048: 256, 1024: 128, 512: 64, 256: 32}


def packet(t0, f, phase=0.0, width=W):
    env = np.zeros(L)
    env[t0:t0 + width] = np.hanning(width)
    t = np.arange(L, dtype=np.float64)
    return env * np.exp(2j * np.pi * f * t + 1j * phase)


def pair_signal(dt, df, phase_b=0.0):
    return packet(T0, 0.07) + packet(T0 + dt, 0.07 + df, phase_b)


def gauge_residual(z, b, n, h):
    fam = MagnitudeFamily(z, n, h)
    zr = z + (np.exp(1j * np.pi / 2) - 1.0) * b
    return fam.relative_error(zr)


def task1_coverage():
    print("=== task 1: coverage maps over (dt, df) ===")
    dts = np.linspace(100, 2400, 24)
    dfs = np.linspace(0.0, 14.0 / 1024, 15)
    t1 = {n: RUNGS[n] for n in (2048, 1024, 512, 256)}
    res = {n: np.zeros((len(dfs), len(dts))) for n in t1}
    for j, dt in enumerate(dts):
        for i, df in enumerate(dfs):
            b = packet(T0 + int(dt), 0.07 + df)
            z = packet(T0, 0.07) + b
            for n, h in t1.items():
                res[n][i, j] = gauge_residual(z, b, n, h)
    thresh = 0.02
    pinnable = np.maximum.reduce(list(res.values())) > thresh
    lad2 = np.maximum(res[1024], res[256]) > thresh
    lad3u = np.maximum.reduce([res[2048], res[1024], res[256]]) > thresh
    lad3m = np.maximum.reduce([res[1024], res[512], res[256]]) > thresh
    print(f"pinnable fraction of grid (any rung 256..2048): "
          f"{pinnable.mean():.3f}")
    for name, m in (("{1024,256}", lad2), ("{2048,1024,256}", lad3u),
                    ("{1024,512,256}", lad3m)):
        cov = (m & pinnable).sum() / max(pinnable.sum(), 1)
        print(f"  ladder {name:16s}: covers {cov:.3f} of the pinnable set")
    hole = pinnable & ~lad2
    if hole.any():
        jj, ii = np.where(hole)
        print(f"  {hole.sum()} uncovered pinnable cells for the deployed "
              f"pair; dt range "
              f"{dts[ii].min():.0f}..{dts[ii].max():.0f}, df*NB range "
              f"{1024*dfs[jj].min():.1f}..{1024*dfs[jj].max():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), sharex=True,
                             sharey=True)
    panels = [(f"rung {n}", np.log10(res[n] + 1e-4)) for n in res]
    panels += [("ladder {1024,256}",
                np.log10(np.maximum(res[1024], res[256]) + 1e-4)),
               ("ladder {2048,1024,256}",
                np.log10(np.maximum.reduce(
                    [res[2048], res[1024], res[256]]) + 1e-4))]
    for ax, (name, img) in zip(axes.ravel(), panels):
        pc = ax.pcolormesh(dts, 1024 * dfs, img, cmap="inferno",
                           vmin=-4, vmax=0, shading="auto")
        ax.set_title(name, fontsize=10)
        # visibility-law boundary of the largest rung in the panel
        fig.colorbar(pc, ax=ax, shrink=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("dt (samples)")
    for ax in axes[:, 0]:
        ax.set_ylabel("df (long bins)")
    fig.suptitle("gauge residual log10 (bright = pinned)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ladder_coverage.png", dpi=130)
    print(f"  wrote {OUT/'ladder_coverage.png'}")
    return res


def run_ap(z, fams, u0, iters, beta=0.88):
    latent, prev = u0.copy(), u0.copy()
    hist = []
    for _ in range(iters):
        trial = latent + beta * (latent - prev)
        prev = latent
        out = trial
        for f in fams:
            out = f.project(out, out)
        latent = out
        hist.append(latent.copy())
    return latent, hist


def task2_transversality():
    """Distance-to-z floors at the unpinned gauge subspace the perturbation
    excites (measured: rate 1.0000), so the right observable is CONSTRAINT
    VIOLATION decay -- the contraction of the composed projector on the
    reachable subspace."""
    print("\n=== task 2: violation contraction near z, 2 vs 3 rungs ===")
    rng = np.random.default_rng(4)
    z = pair_signal(400, 1.0 / 1024) + 0.3 * packet(1200, 0.18)
    rms = np.sqrt(np.mean(np.abs(z) ** 2))
    for name, rungs in (("{1024,256}", (1024, 256)),
                        ("{2048,1024,256}", (2048, 1024, 256))):
        fams = [MagnitudeFamily(z, n, RUNGS[n]) for n in rungs]
        u0 = z + 0.02 * rms * (rng.standard_normal(L) +
                               1j * rng.standard_normal(L))
        _, hist = run_ap(z, fams, u0, 48)
        viol = np.array([max(f.relative_error(u) for f in fams)
                         for u in hist])
        tail = viol[24:]
        rate = np.median(tail[1:] / np.maximum(tail[:-1], 1e-30))
        print(f"  {name:16s}: violation it1={viol[0]:.2e} "
              f"it48={viol[-1]:.2e}  tail rate/iter={rate:.4f}")


def task3_recovery():
    """Recovery speed is set by pin STRENGTH, not bare coverage: the 2048
    rung pins dt=1600 only at residual 0.021 (pair at its cell edge), so
    contraction is ~O(strength^2)-slow.  A 4096 rung holds the pair deep in
    its cell.  Track the recovered relative phase vs iterations."""
    print("\n=== task 3: dead-zone recovery (slow pair dt=1600) ===")
    a = packet(T0, 0.07)
    b = packet(T0 + 1600, 0.07)
    z = a + b
    ladders = {"{1024,256}": (1024, 256),
               "{2048,1024,256}": (2048, 1024, 256),
               "{4096,1024,256}": (4096, 1024, 256)}
    for name, rungs in ladders.items():
        pins = [gauge_residual(z, b, n, RUNGS[n]) for n in rungs]
        fams = [MagnitudeFamily(z, n, RUNGS[n]) for n in rungs]
        rng = np.random.default_rng(11)
        finals, traj = [], None
        for s in range(4):
            theta = rng.uniform(0.4, np.pi)
            u0 = a + np.exp(1j * theta) * b
            u, hist = run_ap(z, fams, u0, 300)
            phis = []
            for uu in (hist[0], hist[19], hist[74], hist[299]):
                ca = np.vdot(a, uu)
                uu = uu * np.conj(ca) / abs(ca)
                phis.append(abs(np.degrees(np.angle(np.vdot(b, uu)))))
            finals.append(phis[-1] / np.degrees(theta))
            if s == 0:
                traj = (np.degrees(theta), phis)
        t0, ph = traj
        print(f"  {name:16s} pin strengths {['%.3f' % p for p in pins]}  "
              f"start {t0:5.1f} deg -> it1 {ph[0]:5.1f}, it20 {ph[1]:5.1f}, "
              f"it75 {ph[2]:5.1f}, it300 {ph[3]:5.1f} | mean retained "
              f"{100*np.mean(finals):.0f}%")


if __name__ == "__main__":
    task1_coverage()
    task2_transversality()
    task3_recovery()
