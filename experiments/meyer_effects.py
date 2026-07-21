#!/usr/bin/env python3
"""Recomposition effects over the Meyer split: gains, shading, colour.

Three experiments, each writing a figure to experiments/out:

  gains    the cartoon/texture gain plane on a grey image, plus the layers
  shade    the shading layer (cartoon - ROF(cartoon)) amplified, measured
           against unsharp masking at matched boost for halo
  colour   per-channel decomposition in the four supported spaces, and the
           gains that only OKLab luma/chroma can express

Run:  .venv/bin/python experiments/meyer_effects.py [gains|shade|colour|all]
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bfft                          # noqa: E402
from bfft.effects import (meyer_channels, recompose, recompose_channels,
                          shade)     # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def grey_image():
    """Barbara if the test set is present, else a synthetic ramp+weave."""
    from meyer_bregman import load_barbara
    try:
        return load_barbara()
    except Exception:
        n = 256
        y, x = np.mgrid[0:n, 0:n].astype(np.float64)
        f = 60 + 120 * x / n                       # smooth illumination
        f += 30 * (y > n / 2)                      # a jump
        f += 22 * np.cos(2 * np.pi * (x + y) / 7)  # weave
        return np.clip(f, 0, 255)


def colour_image():
    from skimage import data
    from skimage.transform import resize
    a = data.astronaut().astype(np.float64) / 255.0
    return resize(a, (256, 256), anti_aliasing=True)


def _show(ax, a, title, vmin=None, vmax=None, cmap="gray"):
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# --- 1. the gain plane ---------------------------------------------------

def exp_gains():
    f = grey_image()
    t0 = time.perf_counter()
    u, v = bfft.meyer_split(f)
    t_split = time.perf_counter() - t0
    print(f"split {f.shape}: {t_split * 1e3:.0f} ms")

    combos = [(1.0, 2.5, "texture x2.5"),
              (1.0, -1.0, "texture x-1 (phase inverted)"),
              (0.6, 1.8, "cartoon x0.6, texture x1.8"),
              (1.0, 0.0, "texture removed (gt = 0)")]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    _show(axes[0, 0], f, "input", 0, 255)
    _show(axes[0, 1], u, "cartoon u", 0, 255)
    _show(axes[0, 2], v + 128, "texture v (+128)", 0, 255)
    _show(axes[0, 3], f - u - v + 128, "residual (+128)", 0, 255)
    for ax, (gc, gt, name) in zip(axes[1], combos):
        _show(ax, recompose(f, u, v, gc, gt, clip=(0, 255)), name, 0, 255)
    fig.suptitle("Meyer split recomposition: independent layer gains",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / "meyer_effect_gains.png"
    fig.savefig(p, dpi=110)
    print("wrote", p)


# --- 2. the shading layer vs unsharp masking -----------------------------

def _unsharp(f, sigma, amount):
    """Classic unsharp mask, for the halo comparison."""
    from scipy.ndimage import gaussian_filter
    return f + amount * (f - gaussian_filter(f, sigma))


def _halo_metric(before, after, thr=25.0):
    """Overshoot introduced next to strong edges.

    Finds pixels adjacent to a strong gradient, and reports how far the
    result runs past the local input range there -- the signature of a
    halo.  Reported as a fraction of the input range."""
    from scipy.ndimage import maximum_filter, minimum_filter
    gx = np.diff(before, axis=1, append=before[:, :1])
    gy = np.diff(before, axis=0, append=before[:1, :])
    mag = np.hypot(gx, gy)
    near = maximum_filter(mag, size=5) > thr
    hi = maximum_filter(before, size=5)
    lo = minimum_filter(before, size=5)
    over = np.maximum(after - hi, 0.0) + np.maximum(lo - after, 0.0)
    rng = before.max() - before.min()
    return float(over[near].mean() / rng), float(over[near].max() / rng)


def exp_shade():
    f = grey_image()
    u, v = bfft.meyer_split(f)
    t0 = time.perf_counter()
    s = shade(u, c=0.02)
    print(f"shade solve: {(time.perf_counter() - t0) * 1e3:.0f} ms, "
          f"range [{s.min():.1f}, {s.max():.1f}], "
          f"energy {np.sqrt((s ** 2).mean()):.2f}")

    boosted = recompose(f, u, v, 1.0, 1.0, gain_shade=1.5, shade_c=0.02)
    # Match the unsharp amount to the same added energy, so the halo
    # comparison is at equal strength rather than equal parameter.
    add = boosted - f
    for amount in np.linspace(0.05, 3.0, 60):
        cand = _unsharp(f, 3.0, amount) - f
        if np.sqrt((cand ** 2).mean()) >= np.sqrt((add ** 2).mean()):
            break
    un = _unsharp(f, 3.0, amount)
    print(f"matched unsharp amount {amount:.2f}; added rms "
          f"shade {np.sqrt((add ** 2).mean()):.2f} vs "
          f"unsharp {np.sqrt(((un - f) ** 2).mean()):.2f}")

    h_s = _halo_metric(f, boosted)
    h_u = _halo_metric(f, un)
    print(f"edge overshoot (mean, max as fraction of range): "
          f"shade {h_s[0]:.2e} {h_s[1]:.2e} | unsharp {h_u[0]:.2e} "
          f"{h_u[1]:.2e}")

    flat = u - s
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    _show(axes[0, 0], u, "cartoon u", 0, 255)
    _show(axes[0, 1], flat, "ROF(u): the flat cartoon", 0, 255)
    _show(axes[0, 2], s, "shading u - ROF(u)")
    _show(axes[1, 0], f, "input", 0, 255)
    _show(axes[1, 1], np.clip(boosted, 0, 255),
          f"shading x2.5 (overshoot {h_s[0]:.1e})", 0, 255)
    _show(axes[1, 2], np.clip(un, 0, 255),
          f"unsharp, matched energy (overshoot {h_u[0]:.1e})", 0, 255)
    fig.suptitle("The shading layer: illumination the flat cartoon discards",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / "meyer_effect_shade.png"
    fig.savefig(p, dpi=110)
    print("wrote", p)
    return h_s, h_u


# --- 3. colour -----------------------------------------------------------

def exp_colour():
    img = colour_image()
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for col, space in enumerate(("rgb", "oklab", "oklab_lc", "gray")):
        t0 = time.perf_counter()
        sp = meyer_channels(img, space=space)
        dt = time.perf_counter() - t0
        k = sp.planes.shape[2]
        print(f"{space:9s} {k} channel(s) {sp.names} in {dt:.2f}s")
        cart = recompose_channels(sp, 1.0, 0.0)
        tex = recompose_channels(sp, 1.0, 2.5)
        _show(axes[0, col], cart, f"{space}: cartoon only", cmap=None)
        _show(axes[1, col], tex, f"{space}: texture x2.5", cmap=None)
        ident = recompose_channels(sp, clip=False)
        _show(axes[2, col], np.clip(np.abs(ident - img) * 1e12, 0, 1),
              f"{space}: identity err x1e12 "
              f"(max {np.abs(ident - img).max():.1e})", cmap=None)

    fig.suptitle("Per-channel decomposition and recomposition", fontsize=11)
    fig.tight_layout()
    p = OUT / "meyer_effect_colour.png"
    fig.savefig(p, dpi=105)
    print("wrote", p)

    # what only a luma/chroma split can express
    sp = meyer_channels(img, space="oklab_lc")
    variants = [((1, 1), (1, 1), (0, 0), "identity"),
                ((1, 1), (2.5, 1.0), (0, 0), "luma detail x2.5, chroma held"),
                ((1, 1), (1.0, 3.0), (0, 0), "chroma detail x3, luma held"),
                ((1, 1), (1, 1), (2.0, 0.0), "luma shading x3"),
                ((1, 0.4), (1, 1), (0, 0), "chroma cartoon x0.4 (desaturate)"),
                ((1, 1), (0.0, 1.0), (0, 0), "luma detail removed")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    for ax, (gc, gt, gs, name) in zip(axes.ravel(), variants):
        _show(ax, recompose_channels(sp, gc, gt, gs), name, cmap=None)
    fig.suptitle("OKLab luma/chroma: gains that RGB cannot express",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / "meyer_effect_oklab.png"
    fig.savefig(p, dpi=110)
    print("wrote", p)


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    if which in ("gains", "all"):
        exp_gains()
    if which in ("shade", "all"):
        exp_shade()
    if which in ("colour", "color", "all"):
        exp_colour()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
