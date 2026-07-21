#!/usr/bin/env python3
"""Directional denoise on each axis, then BLUE-fuse (not geomean).

Formal backing:
  1D Wiener along axis u is the MMSE linear estimator for that marginal.
  Inverse-variance (BLUE / information-form) fusion of unbiased estimates
  is minimum-variance and has error covariance <= each input on EVERY axis.
  Geometric mean is neither unbiased nor covariance-aware, so it cannot
  reach the per-axis minima -- this is the strawman we replace.

Estimators:
  est_rows = 1D Wiener along axis 1 (kills row-axis noise; blurs vertical
             edges = column-varying content).
  est_cols = 1D Wiener along axis 0 (dual).
Fusions:
  geomean  = sqrt(est_rows * est_cols)              (biased strawman)
  arithmean= 0.5(est_rows + est_cols)               (variance-blind)
  blue_oracle = inverse (GT) local-MSE weights      (upper bound of fusion)
  blue_practical = inverse local residual-variance weights (GT-free; this
             is our relaxed weighted projection with identity aperture,
             omega ~ Sigma^{-1}).
Reference:
  wiener2d = full 2D Wiener (joint MMSE linear) -- the number the fusion of
             two 1D estimators aims to approach, NOT to beat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from two_lattice_blur_study import ground_truth, psnr  # noqa: E402

SIGMA = 0.08


def box(img, k=9):
    c = np.cumsum(np.pad(img, ((k, k), (0, 0)), mode="reflect"), axis=0)
    img = (c[2 * k:, :] - c[:-2 * k, :]) / (2 * k)
    c = np.cumsum(np.pad(img, ((0, 0), (k, k)), mode="reflect"), axis=1)
    return (c[:, 2 * k:] - c[:, :-2 * k]) / (2 * k)


def marginal_wiener(noisy, sigma, axis):
    n = noisy.shape[axis]
    Y = np.fft.fft(noisy, axis=axis)
    other = 1 - axis
    Py = (np.abs(Y) ** 2).mean(axis=other, keepdims=True)   # avg periodogram
    noise_power = sigma ** 2 * n                            # unnormalized FFT
    Ps = np.maximum(Py - noise_power, 0.0)
    W = Ps / (Ps + noise_power)
    return np.real(np.fft.ifft(Y * W, axis=axis))


def wiener2d(noisy, sigma):
    Y = np.fft.fft2(noisy)
    P = np.abs(Y) ** 2
    noise_power = sigma ** 2 * noisy.size
    Ps = np.maximum(P - noise_power, 0.0)
    W = Ps / (Ps + noise_power)
    return np.real(np.fft.ifft2(Y * W))


def blue_fuse(e1, e2, v1, v2):
    w1 = (1.0 / (v1 + 1e-12))
    w2 = (1.0 / (v2 + 1e-12))
    return (w1 * e1 + w2 * e2) / (w1 + w2)


def oriented_scene(n=256):
    """Spatially varying orientation: left half = horizontal edges (content
    varies vertically), right half = vertical edges.  The two axes fail in
    DIFFERENT places, so inverse-variance weighting must decisively beat any
    fixed blend."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    left = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * y / 12))     # horiz edges
    right = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * x / 12))    # vert edges
    img = np.where(x < n / 2, left, right)
    return np.clip(img, 0, 1)


def run_scene(gt, sigma, seed, tag):
    rng = np.random.default_rng(seed)
    noisy = gt + sigma * rng.standard_normal(gt.shape)
    est_rows = marginal_wiener(noisy, sigma, axis=1)
    est_cols = marginal_wiener(noisy, sigma, axis=0)
    geomean = np.sqrt(np.clip(est_rows, 0, None) * np.clip(est_cols, 0, None))
    v1 = box((noisy - est_rows) ** 2)
    v2 = box((noisy - est_cols) ** 2)
    blue = blue_fuse(est_rows, est_cols, v1, v2)
    print(f"[{tag}] input {psnr(noisy, gt):.2f} | rows {psnr(est_rows, gt):.2f}"
          f" cols {psnr(est_cols, gt):.2f} | geomean {psnr(geomean, gt):.2f} |"
          f" BLUE {psnr(blue, gt):.2f} | gain over geomean "
          f"{psnr(blue, gt) - psnr(geomean, gt):+.2f} dB")


def main():
    print("== decisive case: spatially-varying orientation ==")
    run_scene(oriented_scene(), 0.12, 1, "oriented")
    print("\n== honest baseline: near-isotropic natural texture ==")
    gt = ground_truth()                       # sharp natural crop, ~256^2
    rng = np.random.default_rng(0)
    noisy = gt + SIGMA * rng.standard_normal(gt.shape)

    est_rows = marginal_wiener(noisy, SIGMA, axis=1)
    est_cols = marginal_wiener(noisy, SIGMA, axis=0)

    geomean = np.sqrt(np.clip(est_rows, 0, None) * np.clip(est_cols, 0, None))
    arith = 0.5 * (est_rows + est_cols)

    # oracle inverse-MSE (upper bound of the fusion idea)
    v1_o = box((est_rows - gt) ** 2)
    v2_o = box((est_cols - gt) ** 2)
    blue_oracle = blue_fuse(est_rows, est_cols, v1_o, v2_o)

    # practical, GT-free: residual variance spikes where a filter blurred a
    # feature along its direction (structured residual), so it self-reports
    # unreliability there.
    v1_p = box((noisy - est_rows) ** 2)
    v2_p = box((noisy - est_cols) ** 2)
    blue_practical = blue_fuse(est_rows, est_cols, v1_p, v2_p)

    ref2d = wiener2d(noisy, SIGMA)

    rows = [("noisy input", noisy),
            ("est axis=rows (1D Wiener)", est_rows),
            ("est axis=cols (1D Wiener)", est_cols),
            ("geometric mean (strawman)", geomean),
            ("arithmetic mean", arith),
            ("BLUE fuse (practical omega)", blue_practical),
            ("BLUE fuse (oracle omega)", blue_oracle),
            ("2D Wiener (joint ref)", ref2d)]
    print(f"sigma={SIGMA}, input PSNR={psnr(noisy, gt):.2f} dB")
    print(f"{'method':>30s} {'PSNR (dB)':>10s}")
    for name, im in rows:
        print(f"{name:>30s} {psnr(im, gt):10.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, (name, im) in zip(axes.ravel(), rows):
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\n{psnr(im, gt):.2f} dB", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "directional_fusion.png", dpi=120)
    print(OUT / "directional_fusion.png")


if __name__ == "__main__":
    main()
