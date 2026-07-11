#!/usr/bin/env python3
"""2D prototype: dual-window Gabor-magnitude fusion of one latent image.

The two-lattice audio solver generalizes verbatim: a latent 2D scene, two
windowed-DFT magnitude families measured with different patch sizes (large
patch = fine spatial frequency, small patch = fine localization), and
alternating radial-projection + windowed overlap-add.  This is the algebraic
skeleton of "image fusion from two captures with different properties": each
capture contributes one constraint family on a shared latent scene.

Scene: a close-pair grating (two nearby spatial frequencies -- only the large
window can resolve the pair), point sources and a thin line (only the small
window localizes them), and a smooth background.  Targets are measured from
the truth; recovery starts from a phase-scrambled seed.  Compare large-only,
small-only, and alternating recoveries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"


def make_scene(n=192, seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    img = 0.15 + 0.05 * np.sin(2 * np.pi * (x + 2 * y) / n)
    # close-pair grating: spatial frequencies 0.11 and 0.11+1.3/48 cyc/px
    band = (y > 20) & (y < 90)
    img += band * 0.35 * (np.sin(2 * np.pi * 0.11 * x) +
                          np.sin(2 * np.pi * (0.11 + 1.3 / 48) * x + 0.7))
    # point sources and a thin diagonal line (localization content)
    for _ in range(12):
        py, px = rng.integers(105, n - 8, 2)
        img[py:py + 2, px:px + 2] += 1.4
    for k in range(60):
        img[100 + k, 20 + k] += 1.0
    img += 0.01 * rng.standard_normal((n, n))
    return img


class Family2D:
    """One 2D separable-Hann windowed-DFT magnitude family."""

    def __init__(self, observed, patch, hop):
        n = observed.shape[0]
        self.patch, self.hop = patch, hop
        w1 = np.hanning(patch)
        self.win = np.outer(w1, w1)
        offs = np.arange(0, n - patch + 1, hop)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        self.oy, self.ox = oy.ravel(), ox.ravel()
        iy = self.oy[:, None, None] + np.arange(patch)[None, :, None]
        ix = self.ox[:, None, None] + np.arange(patch)[None, None, :]
        self.iy, self.ix = iy, ix
        patches = observed[iy, ix] * self.win[None]
        self.target = np.abs(np.fft.fft2(patches, axes=(1, 2)))
        self.den = np.zeros((n, n))
        np.add.at(self.den, (iy, ix), np.broadcast_to(self.win ** 2,
                                                      patches.shape))
        med = np.median(self.den[self.den > 0])
        self.support = self.den > 1e-2 * med

    def project(self, latent):
        patches = latent[self.iy, self.ix] * self.win[None]
        S = np.fft.fft2(patches, axes=(1, 2))
        S *= self.target / (np.abs(S) + 1e-12)
        fitted = np.fft.ifft2(S, axes=(1, 2)).real
        acc = np.zeros_like(latent)
        np.add.at(acc, (self.iy, self.ix), fitted * self.win[None])
        out = acc / np.maximum(self.den, 1e-8)
        return np.where(self.support, out, latent)

    def relative_error(self, latent):
        S = np.abs(np.fft.fft2(latent[self.iy, self.ix] * self.win[None],
                               axes=(1, 2)))
        return float(np.sqrt(np.sum((S - self.target) ** 2) /
                             (np.sum(self.target ** 2) + 1e-30)))


def degraded_seed(truth, fam, seed=1):
    rng = np.random.default_rng(seed)
    patches = truth[fam.iy, fam.ix] * fam.win[None]
    S = np.fft.fft2(patches, axes=(1, 2))
    S = np.abs(S) * np.exp(1j * rng.uniform(-np.pi, np.pi, S.shape))
    fitted = np.fft.ifft2(S, axes=(1, 2)).real
    acc = np.zeros_like(truth)
    np.add.at(acc, (fam.iy, fam.ix), fitted * fam.win[None])
    return np.where(fam.support, acc / np.maximum(fam.den, 1e-8), 0.0)


def psnr(u, ref):
    rng_ = ref.max() - ref.min()
    return 10 * np.log10(rng_ ** 2 / np.mean((u - ref) ** 2))


def recover(truth, families, iterations=80, beta=0.88, seed_img=None,
            positivity=False):
    latent = seed_img.copy()
    prev = latent.copy()
    for _ in range(iterations):
        trial = latent + beta * (latent - prev)
        prev = latent
        out = trial
        for f in families:
            out = f.project(out)
        # Scenes are nonnegative: positivity is a third convex constraint
        # that kills the local sign gauge of real-image magnitude retrieval
        # (the polarity flips and patch-sign domains seen without it).
        latent = np.maximum(out, 0.0) if positivity else out
    return latent


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = make_scene()
    big = Family2D(truth, 48, 12)
    small = Family2D(truth, 12, 4)
    seed_img = degraded_seed(truth, big, seed=11)

    results = {"seed": seed_img}
    results["large only"] = recover(truth, [big], seed_img=seed_img,
                                    positivity=True)
    results["small only"] = recover(truth, [small], seed_img=seed_img,
                                    positivity=True)
    results["alternating"] = recover(truth, [big, small], seed_img=seed_img,
                                     positivity=True)

    print(f"{'recovery':>12s} {'PSNR dB':>8s} {'big resid':>10s} "
          f"{'small resid':>12s}")
    for name, u in results.items():
        print(f"{name:>12s} {psnr(u, truth):8.2f} "
              f"{big.relative_error(u):10.4f} "
              f"{small.relative_error(u):12.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    panels = [("truth", truth)] + list(results.items())
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=truth.min(), vmax=truth.max())
        title = name if name == "truth" else \
            f"{name} ({psnr(img, truth):.1f} dB)"
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    png = OUT / "two_lattice_image.png"
    fig.savefig(png, dpi=130)
    print(png)


if __name__ == "__main__":
    main()
