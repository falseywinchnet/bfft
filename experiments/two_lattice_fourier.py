#!/usr/bin/env python3
"""NEXT-PROJECT task 9: Fourier-domain families (domain freedom, made useful).

Because the latent is the scene itself, a constraint family can act in any
linearly reachable domain.  The demo where this MATTERS: capture B is sharp
but MISREGISTERED (unknown shift).  Translation only changes Fourier phase,
so B's global Fourier MODULUS is a valid, shift-invariant constraint family
— usable without registering B at all.  Capture A is registered but blurred:
its spatial patched family is reliable only at low frequencies (defocus
preserves lows), and supplies the position anchor.

Families:
  P_F (Fourier modulus of B): u <- ifft2(|F(B)| . F(u)/|F(u)|).real  —
      the FFT is orthogonal, so this radial step IS the exact metric
      projection onto the modulus set;
  P_A (spatial patches of A, low-band only): relaxed projection with a
      per-coefficient low-pass reliability mask (blur keeps lows honest);
  positivity.

Baselines: A alone, B used naively as registered (ghost-shifted), pixel
average.  PSNR against ground truth in A's frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from two_lattice_mff import PatchFamily  # noqa: E402
from two_lattice_blur_study import psnr, disk_kernel  # noqa: E402
from two_lattice_dr import conv2  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
               "7ec9c357-2c5f-45c7-83cc-25520967d9ba/scratchpad")
N = 256
SHIFT = (7, -12)


def scene_pair():
    """Two misregistration models.

    'roll': periodic shift — modulus invariance is exact; the clean
    demonstration of the principle.
    'crop': the physically real case — the shifted aperture admits new
    boundary content, and the unwindowed FFT modulus is dominated by the
    wrap-around discontinuity, so invariance breaks at the boundary.
    Measured below as the honest caveat.
    """
    import matplotlib.image as mpimg
    big = mpimg.imread(str(SCRATCH / "mff_image1.png")) \
        .astype(np.float64)[:, :, :3].mean(axis=2)
    y0, x0 = 60, 100
    gt = big[y0:y0 + N, x0:x0 + N]
    crop = big[y0 + SHIFT[0]:y0 + SHIFT[0] + N,
               x0 + SHIFT[1]:x0 + SHIFT[1] + N]
    rng = np.random.default_rng(9)
    noise = lambda: 0.002 * rng.standard_normal(gt.shape)
    cap_a = conv2(gt, disk_kernel(3.0)) + noise()
    cap_b_roll = np.roll(gt, SHIFT, axis=(0, 1)) + noise()
    cap_b_crop = crop + noise()
    return gt, cap_a, cap_b_roll, cap_b_crop


def lowband_mask(fam, radius=4.0):
    p = fam.patch
    f = np.fft.fftfreq(p) * p
    rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return (rr <= radius).astype(np.float64)


def fuse(cap_a, cap_b, iters=60, beta=0.7):
    fam_a = PatchFamily(cap_a)
    w_low = lowband_mask(fam_a)[None]          # A reliable at low bands only
    mod_b = np.abs(np.fft.fft2(cap_b))         # shift-invariant evidence
    latent, prev = cap_a.copy(), cap_a.copy()
    for _ in range(iters):
        trial = latent + beta * (latent - prev)
        prev = latent
        # Fourier-modulus family of B (exact projection; FFT orthogonal)
        U = np.fft.fft2(trial)
        out = np.fft.ifft2(mod_b * U / (np.abs(U) + 1e-12)).real
        # spatial family of A, relaxed to its reliable low band
        S = fam_a.analyze(out)
        radial = fam_a.target * S / (np.abs(S) + 1e-12)
        out = fam_a.synthesize((1.0 - w_low) * S + w_low * radial, out)
        latent = np.clip(out, 0.0, None)
    return latent


def main():
    gt, cap_a, cap_b_roll, cap_b_crop = scene_pair()
    fused_roll = fuse(cap_a, cap_b_roll)
    fused_crop = fuse(cap_a, cap_b_crop)
    rows = [("capture A (blurred, registered)", cap_a),
            ("capture B (sharp, shifted)", cap_b_roll),
            ("pixel average", 0.5 * (cap_a + cap_b_roll)),
            ("fusion, periodic shift", fused_roll),
            ("fusion, real crop (caveat)", fused_crop)]
    print(f"{'image':>34s} {'PSNR vs GT (dB)':>16s}")
    for name, im in rows:
        print(f"{name:>34s} {psnr(im, gt):16.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.6))
    for ax, (name, im) in zip(axes, [("ground truth", gt)] + rows):
        ax.imshow(im, cmap="gray", vmin=gt.min(), vmax=gt.max())
        t = name if name == "ground truth" else \
            f"{name}\n({psnr(im, gt):.1f} dB)"
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_fourier.png", dpi=125)
    print(OUT / "two_lattice_fourier.png")


if __name__ == "__main__":
    main()
