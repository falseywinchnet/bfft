#!/usr/bin/env python3
"""Multi-focus image fusion via relaxed/weighted magnitude projections.

The captures of a focus stack are INCONSISTENT constraint families: each is
a correct measurement of the latent scene only where it is in focus.  This
is the setting the audio two-lattice method never faces (its two lattices
are exact measurements of one record), and it is where the relaxed
projection enters:

    P_i^w(u):  S = A_i u;   S <- (1 - w_i) * S + w_i * m_i * S/|S|;   u = A_i^+ S

a per-coefficient convex combination between keeping the current analysis
coefficient and radially projecting it onto capture i's magnitude.  w_i = 1
is the hard constraint of the consistent theory; w_i = 0 ignores the
family.  Equivalently u <- u + A_i^+ ( w_i * radial-residual ): the exact
operator analog of unbalanced-OT marginal relaxation.  Fixed points settle
at w-weighted compromises wherever families disagree.

Reliability is measured, not assumed: defocus is a local low-pass, so the
in-focus capture has the largest high-band patch energy.  Weights are a
softmax of per-patch high-frequency energy across captures.

Families here are {captures} x {one patch lattice}; positivity closes the
sign gauge as in two_lattice_image.py.  RGB channels share the luminance
weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

PATCH, HOP = 32, 8
GAMMA = 2.0            # softmax sharpness of the focus evidence
BETA = 0.7             # momentum
ITERS = 16
LOW_R = 2.0            # |f| <= LOW_R bins counts as low band (shared by all)


class PatchFamily:
    """One capture's patched 2D Hann windowed-DFT magnitude family."""

    def __init__(self, img, patch=PATCH, hop=HOP):
        h, w = img.shape
        self.patch, self.hop = patch, hop
        w1 = np.hanning(patch)
        self.win = np.outer(w1, w1)
        oy = np.arange(0, h - patch + 1, hop)
        ox = np.arange(0, w - patch + 1, hop)
        gy, gx = np.meshgrid(oy, ox, indexing="ij")
        self.gy, self.gx = gy, gx
        iy = gy.ravel()[:, None, None] + np.arange(patch)[None, :, None]
        ix = gx.ravel()[:, None, None] + np.arange(patch)[None, None, :]
        self.iy, self.ix = iy, ix
        self.den = np.zeros((h, w))
        np.add.at(self.den, (iy, ix),
                  np.broadcast_to(self.win ** 2,
                                  (iy.shape[0], patch, patch)))
        med = np.median(self.den[self.den > 0])
        self.support = self.den > 1e-2 * med
        self.target = np.abs(self.analyze(img))

    def analyze(self, img):
        return np.fft.fft2(img[self.iy, self.ix] * self.win[None],
                           axes=(1, 2))

    def synthesize(self, S, fallback):
        fitted = np.fft.ifft2(S, axes=(1, 2)).real
        acc = np.zeros_like(fallback)
        np.add.at(acc, (self.iy, self.ix), fitted * self.win[None])
        out = acc / np.maximum(self.den, 1e-8)
        return np.where(self.support, out, fallback)

    def project_weighted(self, u, w_patch):
        """Relaxed radial projection; w_patch is [n_patches] in [0,1]."""
        S = self.analyze(u)
        radial = self.target * S / (np.abs(S) + 1e-12)
        w = w_patch[:, None, None]
        return self.synthesize((1.0 - w) * S + w * radial, u)

    def high_energy(self):
        """Per-patch high-band target energy (the focus evidence)."""
        p = self.patch
        f = np.fft.fftfreq(p) * p
        rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
        high = rr > LOW_R
        return np.sum((self.target ** 2) * high[None], axis=(1, 2))


def focus_weights(fams):
    e = np.stack([f.high_energy() for f in fams])          # [n_cap, n_patch]
    e = e ** GAMMA
    return e / (e.sum(axis=0, keepdims=True) + 1e-30)


def weight_map(fam, w_patch):
    """Upsample per-patch weights to a per-pixel map (win^2 OLA)."""
    acc = np.zeros_like(fam.den)
    np.add.at(acc, (fam.iy, fam.ix),
              w_patch[:, None, None] * fam.win[None] ** 2)
    return acc / np.maximum(fam.den, 1e-8)


def fuse(channels, lum_list):
    """channels: list over captures of [h,w,3]; lum_list: luminances."""
    fams_lum = [PatchFamily(lum) for lum in lum_list]
    w = focus_weights(fams_lum)                            # [n_cap, n_patch]
    maps = [weight_map(fams_lum[i], w[i]) for i in range(len(lum_list))]

    fused = np.zeros_like(channels[0])
    for c in range(3):
        fams = [PatchFamily(img[:, :, c]) for img in channels]
        # seed: reliability-weighted pixel average (the classical soft fuse)
        seed = sum(m * img[:, :, c] for m, img in zip(maps, channels))
        latent, prev = seed.copy(), seed.copy()
        for _ in range(ITERS):
            trial = latent + BETA * (latent - prev)
            prev = latent
            out = trial
            for i, fam in enumerate(fams):
                out = fam.project_weighted(out, w[i])
            latent = np.clip(out, 0.0, None)
        fused[:, :, c] = latent
    return np.clip(fused, 0, 1), maps


def sharpness(img):
    g = img.mean(axis=2) if img.ndim == 3 else img
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(np.var(lap))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
        "7ec9c357-2c5f-45c7-83cc-25520967d9ba/scratchpad")
    imgs = []
    for i in (1, 2, 3):
        a = mpimg.imread(str(base / f"mff_image{i}.png")).astype(np.float64)
        imgs.append(a[:, :, :3])
    lums = [im.mean(axis=2) for im in imgs]

    fused, maps = fuse(imgs, lums)
    average = sum(imgs) / len(imgs)

    # classic hard baseline: per-pixel argmax of smoothed Laplacian energy
    def lap_energy(g):
        lap = np.zeros_like(g)
        lap[1:-1, 1:-1] = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                           - g[1:-1, :-2] - g[1:-1, 2:])
        e = lap ** 2
        k = 15
        box = np.ones(k) / k
        e = np.apply_along_axis(lambda r: np.convolve(r, box, "same"), 1, e)
        e = np.apply_along_axis(lambda r: np.convolve(r, box, "same"), 0, e)
        return e
    energies = np.stack([lap_energy(l) for l in lums])
    pick = np.argmax(energies, axis=0)
    hard = np.take_along_axis(
        np.stack(imgs), pick[None, :, :, None], axis=0)[0]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [("source 1", imgs[0]), ("source 2", imgs[1]),
            ("source 3", imgs[2]), ("average", average),
            ("argmax-Laplacian", hard), ("weighted projections", fused)]
    print(f"{'image':>22s} {'laplacian variance':>19s}")
    for name, im in rows:
        print(f"{name:>22s} {sharpness(im):19.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 7.6))
    for ax, (name, im) in zip(axes.ravel(), rows):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(f"{name} (lapvar {sharpness(im):.4f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_mff.png", dpi=120)
    plt.close(fig)

    # zoom crops: shell center / shell rim / sand
    crops = [("shell center", (100, 260, 180, 340)),
             ("shell rim", (250, 410, 400, 560)),
             ("sand", (20, 140, 480, 620))]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    sel = [("source 1", imgs[0]), ("source 2", imgs[1]),
           ("average", average), ("weighted projections", fused)]
    for r, (cname, (y0, y1, x0, x1)) in enumerate(crops):
        for c, (name, im) in enumerate(sel):
            axes[r, c].imshow(np.clip(im[y0:y1, x0:x1], 0, 1))
            if r == 0:
                axes[r, c].set_title(name)
            if c == 0:
                axes[r, c].set_ylabel(cname)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_mff_zoom.png", dpi=120)
    print(OUT / "two_lattice_mff.png")
    print(OUT / "two_lattice_mff_zoom.png")


if __name__ == "__main__":
    main()
