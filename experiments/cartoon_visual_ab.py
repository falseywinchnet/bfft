#!/usr/bin/env python3
"""Look at the two cartoons at full size, side by side.

The downstream objective cannot see grid bias, and anisotropic TV has one --
it prefers axis-aligned edges, so curved and diagonal contours are where it
would show. Pikachu is almost entirely curved and diagonal contours, which is
why it is the right image to look at.

The crop is chosen objectively: the window with the most diagonally-oriented
gradient energy **in the target**, so neither method is shown its best or
worst case by my choosing.

The orientation histogram is the quantitative half. Anisotropic TV biases
toward 0 and 90 degrees; if it is doing so visibly, the histogram says so
without anyone having to squint.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_visual_ab.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import srgb_to_lab  # noqa: E402

from cartoon_stage_isotropy_cost import aniso_meyer_split  # noqa: E402


def diagonal_crop(light, size=112):
    """Window with the most diagonal gradient energy in the target."""
    from scipy import ndimage as ndi
    gx = ndi.sobel(light, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(light, axis=0, mode="reflect") / 8.0
    # |gx*gy| is largest where the contour is at 45 degrees and vanishes on
    # axis-aligned edges, so it points exactly at the risky places.
    diagonal = ndi.gaussian_filter(np.abs(gx * gy), 6.0, mode="reflect")
    h, w = light.shape
    half = size // 2
    interior = diagonal[half:h - half, half:w - half]
    index = int(np.argmax(interior))
    cy, cx = np.unravel_index(index, interior.shape)
    cy += half
    cx += half
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def orientation_profile(plane, bins=90):
    from scipy import ndimage as ndi
    gx = ndi.sobel(plane, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(plane, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(gx, gy)
    keep = magnitude > np.percentile(magnitude, 90.0)
    angle = np.degrees(np.arctan2(gy[keep], gx[keep])) % 180.0
    weights = magnitude[keep]
    counts, edges = np.histogram(
        angle, bins=bins, range=(0.0, 180.0), weights=weights)
    return 0.5 * (edges[:-1] + edges[1:]), counts / max(counts.sum(), 1e-30)


def axis_share(plane, width=10.0):
    """Fraction of edge energy within `width` degrees of an axis."""
    centres, density = orientation_profile(plane)
    near = ((centres < width) | (centres > 180.0 - width) |
            (np.abs(centres - 90.0) < width))
    return float(density[near].sum())


def main():
    rgb = gallery.load("pikachu")
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64))
    light = np.ascontiguousarray(lab[..., 0] * 255.0)
    print(f"pikachu at {light.shape[0]}x{light.shape[1]}, full size")

    t0 = time.perf_counter()
    iso_cartoon, iso_texture = bfft.meyer_split(
        light, lam=0.05, mu=40.0, passes=24, threads=4)
    iso_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ani_cartoon, ani_texture = aniso_meyer_split(
        light, lam=0.05, mu=40.0, passes=8)
    ani_s = time.perf_counter() - t0
    print(f"  isotropic, 24 passes  {iso_s * 1e3:7.0f} ms")
    print(f"  anisotropic, 8 passes {ani_s * 1e3:7.0f} ms")

    scale = float(np.linalg.norm(iso_cartoon))
    print(f"  relative distance between the two cartoons: "
          f"{float(np.linalg.norm(ani_cartoon - iso_cartoon)) / scale:.4f}")
    for label, plane in (("target", light), ("isotropic", iso_cartoon),
                         ("anisotropic", ani_cartoon)):
        print(f"  edge energy within 10 deg of an axis, {label:12s} "
              f"{axis_share(plane) * 100:5.1f}%")

    rows, cols = diagonal_crop(light)
    print(f"  diagonal crop at rows {rows.start}:{rows.stop}, "
          f"cols {cols.start}:{cols.stop}")

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.8))
    low, high = float(light.min()), float(light.max())
    panels = (("target", light), ("isotropic cartoon — shipped", iso_cartoon),
              ("anisotropic cartoon — taut string", ani_cartoon))
    for column, (title, plane) in enumerate(panels):
        axes[0, column].imshow(plane, cmap="gray", vmin=low, vmax=high)
        axes[0, column].set_title(title, fontsize=11)
        axes[0, column].add_patch(plt.Rectangle(
            (cols.start, rows.start), cols.stop - cols.start,
            rows.stop - rows.start, fill=False, edgecolor="#e8442e",
            linewidth=1.4))
        axes[1, column].imshow(plane[rows, cols], cmap="gray",
                               vmin=low, vmax=high, interpolation="nearest")
        axes[1, column].set_title(
            f"{title.split(' — ')[0]}, diagonal crop", fontsize=10)
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])

    difference = ani_cartoon - iso_cartoon
    span = float(np.percentile(np.abs(difference), 99.5))
    image = axes[2, 0].imshow(difference, cmap="RdBu_r",
                              vmin=-span, vmax=span)
    axes[2, 0].set_title("anisotropic − isotropic", fontsize=10)
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    fig.colorbar(image, ax=axes[2, 0], fraction=0.046)

    axes[2, 1].imshow(np.abs(difference)[rows, cols], cmap="magma",
                      interpolation="nearest")
    axes[2, 1].set_title("|difference|, same crop", fontsize=10)
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])

    for label, plane, colour in (("target", light, "#888888"),
                                 ("isotropic", iso_cartoon, "#2b6cb0"),
                                 ("anisotropic", ani_cartoon, "#e8442e")):
        centres, density = orientation_profile(plane)
        axes[2, 2].plot(centres, density, label=label, color=colour,
                        linewidth=1.5)
    for mark in (0, 90, 180):
        axes[2, 2].axvline(mark, color="#000000", linewidth=0.6, alpha=0.25)
    axes[2, 2].set_title("edge orientation density\n"
                         "(grid bias would pile up at 0 and 90)",
                         fontsize=10)
    axes[2, 2].set_xlabel("gradient angle, degrees")
    axes[2, 2].set_xlim(0, 180)
    axes[2, 2].legend(fontsize=8)

    fig.suptitle(
        f"Pikachu {light.shape[0]}x{light.shape[1]} cartoon layer: "
        f"shipped split Bregman (24 passes) against exact 1-D taut string "
        f"(8 passes)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = ROOT / "experiments" / "out" / "cartoon_visual_ab.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
