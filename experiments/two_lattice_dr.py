#!/usr/bin/env python3
"""Douglas-Rachford / RAAR vs alternating projections on the cross-hatch.

Setting: the strong-complementarity fusion of two_lattice_blur_study part
A2 (orthogonal 11-px motion blurs, max-consensus magnitude targets +
positivity).  AP leaves a cross-hatch residual originating from the band
where BOTH blur OTFs are small: there the consensus magnitude target is
noise-level, phase has no anchor, and plain AP parks in a local trap.

Douglas-Rachford iterates through reflections,

    x <- x + P_C(2 P_M(x) - x) - P_M(x),

(M = consensus-magnitude set, C = positivity), which is the standard
phase-retrieval escape from AP stagnation; RAAR relaxes it toward P_M by
(1-beta).  Readout u = clip(P_M(x), 0).

Scored globally (PSNR) and in the spectral bands that define the problem:
  dead band:  |H_h| < tau  AND |H_v| < tau   (neither capture measured it)
  live band:  the complement.
Band error = relative truth-spectrum-weighted error energy in the band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from two_lattice_mff import PatchFamily  # noqa: E402
from two_lattice_blur_study import ground_truth, psnr  # noqa: E402

TAU = 0.3


def conv2(img, k):
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    kk = np.zeros_like(pad)
    kk[:k.shape[0], :k.shape[1]] = k
    kk = np.roll(kk, (-ph, -pw), axis=(0, 1))
    out = np.fft.irfft2(np.fft.rfft2(pad) * np.fft.rfft2(kk), s=pad.shape)
    return out[ph:ph + img.shape[0], pw:pw + img.shape[1]]


def otf_mag(k, shape):
    kk = np.zeros(shape)
    kk[:k.shape[0], :k.shape[1]] = k
    kk = np.roll(kk, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1))
    return np.abs(np.fft.fft2(kk))


def make_problem():
    gt = ground_truth()
    rng = np.random.default_rng(9)
    kh = np.ones((1, 11)) / 11.0
    kv = np.ones((11, 1)) / 11.0
    caps = [conv2(gt, kh) + 0.002 * rng.standard_normal(gt.shape),
            conv2(gt, kv) + 0.002 * rng.standard_normal(gt.shape)]
    fams = [PatchFamily(c) for c in caps]
    consensus = PatchFamily(caps[0])
    consensus.target = np.maximum(fams[0].target, fams[1].target)
    dead = (otf_mag(kh, gt.shape) < TAU) & (otf_mag(kv, gt.shape) < TAU)
    return gt, caps, consensus, dead


def p_m(fam, u):
    S = fam.analyze(u)
    radial = fam.target * S / (np.abs(S) + 1e-12)
    return fam.synthesize(radial, u)


def band_errors(u, gt, dead):
    E = np.abs(np.fft.fft2(u - gt)) ** 2
    G = np.abs(np.fft.fft2(gt)) ** 2
    e_dead = float(E[dead].sum() / (G[dead].sum() + 1e-30))
    e_live = float(E[~dead].sum() / (G[~dead].sum() + 1e-30))
    return e_dead, e_live


def run_ap(fam, seed, iters=40, beta=0.7):
    latent, prev = seed.copy(), seed.copy()
    for _ in range(iters):
        trial = latent + beta * (latent - prev)
        prev = latent
        latent = np.clip(p_m(fam, trial), 0.0, None)
    return latent


def absolute_gate(fam, kappa=2.0):
    """Enforcement gate from the data: coefficients whose consensus target
    sits at the noise floor should not be enforced at all.  The floor is a
    robust (median) estimate over the outer-frequency shell of the patch
    plane; gate = t^2/(t^2 + (kappa*floor)^2).  This completes the weighted
    theory: softmax weights are RELATIVE (who to trust), this is ABSOLUTE
    (whether to trust anyone)."""
    p = fam.patch
    f = np.fft.fftfreq(p) * p
    rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    outer = rr > 0.75 * rr.max()
    floor = np.median(fam.target[:, outer])
    t2 = fam.target ** 2
    return t2 / (t2 + (kappa * floor) ** 2)


def run_ap_gated(fam, seed, iters=40, beta=0.7, kappa=2.0):
    gate = absolute_gate(fam, kappa)
    latent, prev = seed.copy(), seed.copy()
    for _ in range(iters):
        trial = latent + beta * (latent - prev)
        prev = latent
        S = fam.analyze(trial)
        radial = fam.target * S / (np.abs(S) + 1e-12)
        out = fam.synthesize((1.0 - gate) * S + gate * radial, trial)
        latent = np.clip(out, 0.0, None)
    return latent


def run_dr(fam, seed, gt, iters=100, raar_beta=None, track=None):
    """DR (raar_beta=None) or RAAR; returns final readout and best readout."""
    x = seed.copy()
    best, best_ps = None, -np.inf
    for k in range(iters):
        pm = p_m(fam, x)
        y = 2.0 * pm - x
        pc = np.clip(y, 0.0, None)
        if raar_beta is None:
            x = x + pc - pm
        else:
            b = raar_beta
            x = 0.5 * b * (2.0 * pc - y + x) + (1.0 - b) * pm
        readout = np.clip(pm, 0.0, None)
        ps = psnr(readout, gt)
        if track is not None:
            track.append(ps)
        if ps > best_ps:
            best_ps, best = ps, readout
    return readout, best


def main():
    gt, caps, fam, dead = make_problem()
    seed = sum(caps) / 2
    print(f"dead band fraction of plane: {dead.mean():.3f} (tau={TAU})")
    rows = []

    ap = run_ap(fam, seed, iters=40)
    rows.append(("AP + positivity (40 it, baseline)", ap))

    dr_final, dr_best = run_dr(fam, seed, gt, iters=100)
    rows.append(("DR final (100 it)", dr_final))
    rows.append(("DR best iterate", dr_best))

    for b in (0.9, 0.75):
        rf, rb = run_dr(fam, seed, gt, iters=100, raar_beta=b)
        rows.append((f"RAAR beta={b} final", rf))
        rows.append((f"RAAR beta={b} best", rb))

    # standard practice: DR exploration, then a short AP landing
    dr_ap = run_ap(fam, dr_best, iters=8)
    rows.append(("DR best -> 8 AP polish", dr_ap))

    for kappa in (1.0, 2.0, 4.0):
        rows.append((f"AP + absolute gate k={kappa}",
                     run_ap_gated(fam, seed, kappa=kappa)))

    print(f"{'config':>34s} {'PSNR':>7s} {'dead-band err':>14s} "
          f"{'live-band err':>14s}")
    for name, u in rows:
        ed, el = band_errors(u, gt, dead)
        print(f"{name:>34s} {psnr(u, gt):7.2f} {ed:14.4f} {el:14.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = 96, 208, 64, 176
    gated = run_ap_gated(fam, seed, kappa=2.0)
    show = [("ground truth", gt), ("AP baseline", ap),
            ("DR best", dr_best), ("AP + absolute gate", gated)]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for c, (name, im) in enumerate(show):
        axes[0, c].imshow(im[y0:y1, x0:x1], cmap="gray",
                          vmin=gt.min(), vmax=gt.max())
        t = name if c == 0 else f"{name} ({psnr(im, gt):.2f} dB)"
        axes[0, c].set_title(t, fontsize=10)
        err = np.abs(im - gt)
        axes[1, c].imshow(err[y0:y1, x0:x1], cmap="magma",
                          vmin=0, vmax=0.35)
        axes[1, c].set_title("|error|", fontsize=9)
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_dr.png", dpi=130)
    print(OUT / "two_lattice_dr.png")


if __name__ == "__main__":
    main()
