#!/usr/bin/env python3
"""NEXT-PROJECT task 7: the tomographic angle curve.

K captures of one scene through 11-px line blurs at angles i*180/K.  Each
blur destroys frequencies along its motion direction and preserves the
perpendicular line; K angles sample the frequency plane like tomographic
projections.  Measured: fused PSNR and joint dead-band fraction vs K, under
two noise accountings:

  fixed per-capture noise  (more captures = more total exposure), and
  fixed total budget       (per-capture sigma grows as sqrt(K)).

Fusion = max-consensus magnitude target + AP + positivity (the operator
family already shown to be at the information limit, notes S4e), so the
curve isolates pure coverage value.
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
from two_lattice_dr import conv2, otf_mag, run_ap, TAU  # noqa: E402


def line_kernel(angle, length=11, size=15, ss=8):
    k = np.zeros((size, size))
    c = (size - 1) / 2
    for t_ in np.linspace(-(length - 1) / 2, (length - 1) / 2,
                          length * ss):
        y = c + t_ * np.sin(angle)
        x = c + t_ * np.cos(angle)
        i0, j0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - i0, x - j0
        k[i0, j0] += (1 - fy) * (1 - fx)
        k[i0 + 1, j0] += fy * (1 - fx)
        k[i0, j0 + 1] += (1 - fy) * fx
        k[i0 + 1, j0 + 1] += fy * fx
    return k / k.sum()


def consensus_family(caps):
    fams = [PatchFamily(c) for c in caps]
    fam = PatchFamily(caps[0])
    fam.target = np.maximum.reduce([f.target for f in fams])
    return fam


def main():
    gt = ground_truth()
    ks = range(1, 7)
    results = {"fixed per-capture": [], "fixed total budget": []}
    dead_fracs = []
    for K in ks:
        kernels = [line_kernel(i * np.pi / K) for i in range(K)]
        dead = np.ones(gt.shape, bool)
        for k in kernels:
            dead &= otf_mag(k, gt.shape) < TAU
        dead_fracs.append(dead.mean())
        for model, sigma in (("fixed per-capture", 0.002),
                             ("fixed total budget", 0.002 * np.sqrt(K))):
            rng = np.random.default_rng(9)
            caps = [conv2(gt, k) + sigma * rng.standard_normal(gt.shape)
                    for k in kernels]
            fused = run_ap(consensus_family(caps), sum(caps) / K, iters=24)
            results[model].append(
                (psnr(fused, gt), max(psnr(c, gt) for c in caps)))
    print(f"{'K':>2s} {'dead frac':>9s} "
          f"{'fused (per-cap)':>15s} {'best cap':>9s} "
          f"{'fused (budget)':>14s} {'best cap':>9s}")
    for i, K in enumerate(ks):
        f1, b1 = results["fixed per-capture"][i]
        f2, b2 = results["fixed total budget"][i]
        print(f"{K:2d} {dead_fracs[i]:9.3f} {f1:15.2f} {b1:9.2f} "
              f"{f2:14.2f} {b2:9.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, marker in (("fixed per-capture", "o"),
                          ("fixed total budget", "s")):
        ax1.plot(list(ks), [r[0] for r in results[model]], marker=marker,
                 label=f"fused, {model}")
    ax1.plot(list(ks), [r[1] for r in results["fixed per-capture"]],
             marker="x", ls="--", label="best single capture")
    ax1.set_xlabel("number of blur angles K")
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(list(ks), dead_fracs, marker="o", color="crimson")
    ax2.set_xlabel("number of blur angles K")
    ax2.set_ylabel(f"joint dead-band fraction (tau={TAU})")
    ax2.grid(alpha=0.3)
    fig.suptitle("tomographic coverage: fusion gain vs angular sampling")
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_tomo.png", dpi=130)
    print(OUT / "two_lattice_tomo.png")


if __name__ == "__main__":
    main()
