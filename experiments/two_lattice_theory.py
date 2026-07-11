#!/usr/bin/env python3
"""Formal/empirical study of the two-lattice magnitude-projection solver.

The viewer's super-resolution recovers one latent complex waveform from TWO
symmetric-Hann STFT magnitude families (long aperture: fine frequency; short
aperture: fine time) by alternating waveform-domain projections

    P_i(u) = OLA_i( |STFT_i target| * STFT_i(u)/|STFT_i(u)| ),

with heavy-ball momentum, i.e. accelerated nonconvex alternating projection
onto M_long \\cap M_short.  This script measures the quantities the theory
says govern that iteration:

  A. convergence: family residuals vs iteration, asymptotic linear rate,
     momentum effect (beta 0 vs 0.88) -- local rate of nonconvex AP is set by
     the angle between constraint manifolds at the intersection;
  B. identifiability/gauge: does M_long \\cap M_short pin the waveform up to
     global phase, and how much of that is due to the SECOND family
     (long-only / short-only / alternating, multi-seed scatter);
  C. SNR: iterations project a noisy observation toward joint magnitude
     consistency -- output SNR vs iteration count and noise level
     ("multiple iterations push SNR", quantified);
  D. why two lattices: region-resolved fidelity -- long-only fails at clicks
     (timing lives in discarded phase), short-only fails on the close tone
     pair (below its resolution), alternating recovers both.

Fixture (normalized units): tone pair split by 1.4 bins of the long aperture
(irresolvable at N/4), a 60-sample Hann click train, one crossing chirp, and
a -40 dB noise floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))

from two_lattice import MagnitudeFamily  # noqa: E402

L = 8192
NB, NS = 1024, 256
HB, HS = NB // 8, NB // 8   # viewer geometry: h_short = ns/2 = nb/8


def fixture(L=L, noise=0.01, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(L, dtype=np.float64)
    z = 0.40 * np.exp(2j * np.pi * 0.0710 * t)
    z += 0.40 * np.exp(2j * np.pi * (0.0710 + 1.4 / NB) * t + 1.1j)
    click = np.hanning(60)
    click_pos = list(range(700, L - 64, 1024))
    for c in click_pos:
        z[c:c + 60] += 1.6 * click * np.exp(1j * rng.uniform(0, 2 * np.pi))
    z += 0.25 * np.exp(2j * np.pi * (-0.2 * t + 0.4 * t * t / (2 * L)))
    z += noise * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    return z, click_pos


def align(u, ref):
    c = np.vdot(u, ref)
    return u * (c / abs(c)) if abs(c) > 1e-30 else u


def corr(u, ref):
    return abs(np.vdot(u, ref)) / (np.linalg.norm(u) * np.linalg.norm(ref)
                                   + 1e-30)


def run_iters(z, iters, beta=0.88, families=("long", "short"), seed=0,
              initial=None, targets_from=None):
    """The literal two_lattice loop, instrumented.  `targets_from` lets the
    magnitude targets come from a different (e.g. noisy) record than z."""
    obs = z if targets_from is None else targets_from
    fam = {"long": MagnitudeFamily(obs, NB, HB),
           "short": MagnitudeFamily(obs, NS, HS)}
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    if initial is None:
        latent = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) \
            * (0.01 * rms / np.sqrt(2))
    else:
        latent = np.array(initial, np.complex128)
    prev = latent.copy()
    hist = []
    for k in range(iters):
        trial = latent + beta * (latent - prev)
        if not np.all(np.isfinite(trial)) or \
                np.max(np.abs(trial)) > 50 * rms:
            trial = latent.copy()
        prev = latent
        out = trial
        for name in families:
            out = fam[name].project(out, out)
        latent = out
        hist.append((fam["long"].relative_error(latent),
                     fam["short"].relative_error(latent),
                     corr(latent, z)))
    return latent, np.array(hist)


def degraded_seed(z, jitter=1.0, seed=1):
    """Magnitude-true, phase-degraded start: what a PGHI-style seed looks
    like.  Long-frame phases are jittered by jitter*U(-pi,pi), then OLA."""
    fam = MagnitudeFamily(z, NB, HB)
    frames = z[fam.indices] * fam.window[None, :]
    S = np.fft.fft(frames, axis=1)
    rng = np.random.default_rng(seed)
    S = np.abs(S) * np.exp(1j * (np.angle(S) +
                                 jitter * rng.uniform(-np.pi, np.pi, S.shape)))
    fitted = np.fft.ifft(S, axis=1)
    acc = np.zeros(L, np.complex128)
    np.add.at(acc, fam.indices.ravel(),
              (fitted * fam.window[None, :]).ravel())
    out = acc / np.maximum(fam.den, 1e-8)
    return np.where(fam.support, out, z * 0)


def exp_a_convergence():
    print("=== A. convergence & momentum (alternating, clean fixture) ===")
    z, _ = fixture()
    for beta in (0.0, 0.88):
        _, h = run_iters(z, 64, beta=beta)
        tail = h[40:, 0]
        rate = np.median(tail[1:] / np.maximum(tail[:-1], 1e-15))
        print(f"beta={beta:4.2f}: err(long) it1={h[0,0]:.3f} it8={h[7,0]:.4f} "
              f"it24={h[23,0]:.4f} it64={h[-1,0]:.4f}  "
              f"asymptotic rate/iter={rate:.4f}  corr@64={h[-1,2]:.5f}")
    # seed sensitivity: start AT the observation (what the viewer's DIP seed
    # approximates) vs cold random
    z, _ = fixture()
    _, h_obs = run_iters(z, 8, initial=z)
    _, h_rng = run_iters(z, 8)
    print(f"seeded at observation: err(long) it1={h_obs[0,0]:.5f} "
          f"it8={h_obs[7,0]:.5f} corr it1={h_obs[0,2]:.6f}")
    print(f"cold random seed:      err(long) it1={h_rng[0,0]:.5f} "
          f"it8={h_rng[7,0]:.5f} corr it1={h_rng[0,2]:.6f}")


def exp_b_gauge():
    """Which phase directions do the families actually pin?

    Rotate the phase of ONE time-frequency component of the fixture and
    measure how far the rotated signal is from both constraint sets.  A ~zero
    residual direction is a genuine gauge freedom of the intersection; a
    large one is pinned.  This is the honest identifiability statement --
    global AP runs from random seeds only measure stagnation.
    """
    print("\n=== B. gauge probe: component phase rotations (residual vs eps) ===")
    t = np.arange(L, dtype=np.float64)
    tone1 = 0.40 * np.exp(2j * np.pi * 0.0710 * t)
    tone2 = 0.40 * np.exp(2j * np.pi * (0.0710 + 1.4 / NB) * t + 1.1j)
    chirp = 0.25 * np.exp(2j * np.pi * (-0.2 * t + 0.4 * t * t / (2 * L)))
    click = np.zeros(L, np.complex128)
    click[3000:3060] = 1.6 * np.hanning(60)
    z = tone1 + tone2 + chirp + click
    lf, sf = MagnitudeFamily(z, NB, HB), MagnitudeFamily(z, NS, HS)
    print(f"{'rotated component':>22s} {'long resid':>11s} {'short resid':>12s}")
    for name, comp in (("global (all)", z), ("chirp only", chirp),
                       ("click only", click), ("tone1 of close pair", tone1)):
        zr = z + (np.exp(1j * np.pi / 2) - 1.0) * comp   # rotate comp by 90 deg
        print(f"{name:>22s} {lf.relative_error(zr):11.5f} "
              f"{sf.relative_error(zr):12.5f}")


def exp_c_snr():
    print("\n=== C. what iterations actually push (phase-degraded seed) ===")
    clean, _ = fixture(noise=0.0)
    print("targets from CLEAN record, seed = long magnitudes + jittered phase:")
    for jit in (0.5, 1.0):
        u0 = degraded_seed(clean, jitter=jit)
        row = [f"  jitter={jit:.1f}: fid(seed)="
               f"{corr(u0, clean):.4f}"]
        for k in (1, 2, 4, 8, 24):
            u, _ = run_iters(clean, k, initial=u0)
            row.append(f"it{k}={corr(align(u, clean), clean):.4f}")
        print(" ".join(row))
    print("targets from NOISY record (sigma=0.2, input SNR 7.4 dB), same seeds:")
    rng = np.random.default_rng(77)
    noisy = clean + 0.2 * (rng.standard_normal(L) +
                           1j * rng.standard_normal(L))
    u0 = degraded_seed(noisy, jitter=1.0)
    row = [f"  fid-to-CLEAN: seed={corr(u0, clean):.4f}"]
    for k in (1, 2, 4, 8, 24):
        u, _ = run_iters(noisy, k, initial=u0, targets_from=noisy)
        row.append(f"it{k}={corr(align(u, clean), clean):.4f}")
    row.append(f" | obs itself={corr(noisy, clean):.4f} (the cap)")
    print(" ".join(row))


def exp_d_why_two():
    print("\n=== D. why two lattices (phase-scrambled start, 32 iterations) ===")
    z, clicks = fixture(noise=0.0)
    click_mask = np.zeros(L, bool)
    for c in clicks:
        click_mask[max(0, c - 64):c + 128] = True
    interior = np.zeros(L, bool)
    interior[NB:L - NB] = True          # avoid window-support edge effects
    u0 = degraded_seed(z, jitter=1.0)
    for fams in (("long",), ("short",), ("long", "short")):
        u, _ = run_iters(z, 32, families=fams, initial=u0)
        u = align(u, z)
        m1 = click_mask & interior
        m2 = (~click_mask) & interior
        # A lone transient's absolute phase is a gauge direction (exp B), so
        # complex error there is dominated by invisible gauge: report the
        # ENVELOPE error, which is what any magnitude display shows.
        env = lambda v, m: np.linalg.norm(np.abs(u[m]) - np.abs(z[m])) / \
            np.linalg.norm(np.abs(z[m]))
        name = "+".join(fams)
        print(f"{name:12s}: envelope err near clicks {env(u, m1):.4f}   "
              f"steady (tone pair+chirp) {env(u, m2):.4f}   "
              f"overall corr {corr(u, z):.4f}")


def exp_e_ladder():
    """The waterfall dead zone and the third aperture.

    A window of length N pins the relative phase of a packet pair only if
    the pair shares an analysis cell: dt <~ N and df <~ 1/N.  Pairs with
    dt*df <~ 1 that fall BETWEEN the two deployed scales are invisible to
    both — the audio analog of the jointly-dead blur band.  Rotate packet B
    by 90 degrees and measure each family's residual: ~0 = that family
    cannot see the pair's relative phase.
    """
    print("\n=== E. aperture-ladder gauge probe: packet pair at (dt, df) ===")
    t = np.arange(L, dtype=np.float64)

    def packet(t0, f, width=200):
        env = np.zeros(L)
        env[t0:t0 + width] = np.hanning(width)
        return env * np.exp(2j * np.pi * f * t)

    # Visibility model: some window N must see both packets in one cell,
    # dt + w <~ N and df - 1/w <~ 1/N (w = packet width 200).  The deployed
    # ladder is {1024, 256}; dead-for-deployed pairs live OUTSIDE it.
    cases = [("covered: inside long cell (400, 1/NB)", 400, 1.0 / NB),
             ("DEAD slow pair (dt=1600, df=1/NB)", 1600, 1.0 / NB),
             ("DEAD fast pair (dt=40, df=20/NB)", 40, 20.0 / NB),
             ("interior gap, thin at ratio 4 (400, 7/NB)", 400, 7.0 / NB),
             ("unpinnable by ANY window (1600, 12/NB)", 1600, 12.0 / NB)]
    wins = (("2048", 2048, 256), ("1024", 1024, 128), ("512", 512, 64),
            ("256", 256, 128), ("128", 128, 32))
    hdr = " ".join(f"{n:>7s}" for n, _, _ in wins)
    print(f"{'pair':>44s} {hdr}")
    for name, dt, df in cases:
        a = packet(2000, 0.07)
        b = packet(2000 + dt, 0.07 + df)
        z = a + b + 0.001 * np.random.default_rng(1).standard_normal(L)
        zr = z + (np.exp(1j * np.pi / 2) - 1.0) * b
        res = [MagnitudeFamily(z, n, h).relative_error(zr)
               for _, n, h in wins]
        print(f"{name:>44s} " + " ".join(f"{r:7.4f}" for r in res))


if __name__ == "__main__":
    exp_a_convergence()
    exp_b_gauge()
    exp_c_snr()
    exp_d_why_two()
    exp_e_ladder()
