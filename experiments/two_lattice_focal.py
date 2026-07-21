#!/usr/bin/env python3
"""NEXT-PROJECT task 8: the focus stack as a focal ladder.

Reinterpret MFFW3/4 through the coverage law: each focus slice is a rung
whose "cell" is its depth-of-field band.  With no ground truth, the
OTF-proxy for coverage is relative high-band patch energy: slice i covers
patch p at strength E_i(p)/max_j E_j(p) (defocus only removes high-band
energy, so the per-patch max across ALL slices is the best local proxy for
the scene's content).

Measured questions:
  - dominance map: which slice owns which patches, and by what margin;
  - joint dead zone of each PAIR of slices (coverage < threshold relative
    to the triple) — the focal analog of the blur-angle dead band;
  - fusion test: fuse each pair vs the triple; sharpness measured INSIDE
    the region the left-out slice uniquely owns.  If the third slice is a
    real rung, pair-fusions must be soft exactly there and the triple must
    fix it.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from two_lattice_mff import PatchFamily, fuse, weight_map, sharpness  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
               "7ec9c357-2c5f-45c7-83cc-25520967d9ba/scratchpad")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    imgs = [mpimg.imread(str(SCRATCH / f"mff_image{i}.png"))
            .astype(np.float64)[:, :, :3] for i in (1, 2, 3)]
    lums = [im.mean(axis=2) for im in imgs]
    fams = [PatchFamily(lum) for lum in lums]
    E = np.stack([f.high_energy() for f in fams])          # [3, n_patch]
    best = E.max(axis=0) + 1e-30

    # --- dominance and pairwise dead zones (patch lattice) ---
    dom = np.argmax(E, axis=0)
    print("slice dominance (fraction of patches):",
          [f"slice{i+1}={np.mean(dom == i):.3f}" for i in range(3)])
    thresh = 0.5
    print(f"pairwise focal dead zones (pair coverage < {thresh} of triple):")
    dead = {}
    for pair in combinations(range(3), 2):
        cov = E[list(pair)].max(axis=0) / best
        dead[pair] = cov < thresh
        print(f"  slices {pair[0]+1}&{pair[1]+1}: dead fraction "
              f"{dead[pair].mean():.3f}")

    # --- fusion test: leave one slice out ---
    triple, _ = fuse(imgs, lums)
    print(f"\n{'config':>16s} {'lapvar overall':>14s} "
          f"{'lapvar in left-out slice territory':>35s}")
    shape = lums[0].shape
    for pair in combinations(range(3), 2):
        left_out = ({0, 1, 2} - set(pair)).pop()
        fused_pair, _ = fuse([imgs[i] for i in pair],
                             [lums[i] for i in pair])
        # territory = pixels where the left-out slice dominates by margin
        own = (dom == left_out) & (E[left_out] > 1.5 *
                                   E[list(pair)].max(axis=0))
        mask = weight_map(fams[0], own.astype(np.float64)) > 0.5
        def lv(im, m=None):
            g = im.mean(axis=2)
            lap = np.zeros_like(g)
            lap[1:-1, 1:-1] = (4*g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                               - g[1:-1, :-2] - g[1:-1, 2:])
            sel = lap if m is None else lap[m]
            return float(np.var(sel))
        name = f"{{{pair[0]+1},{pair[1]+1}}} (drop {left_out+1})"
        print(f"{name:>16s} {lv(fused_pair):14.6f} "
              f"{lv(fused_pair, mask):25.6f}  (triple there: "
              f"{lv(triple, mask):.6f}, territory {mask.mean()*100:.1f}%)")
    print(f"{'triple':>16s} {sharpness(triple):14.6f}")

    # --- figure: dominance map + a pair's dead territory ---
    dom_map = np.zeros(shape)
    for i in range(3):
        dom_map += (i + 1) * (weight_map(fams[0],
                                         (dom == i).astype(np.float64)) > 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].imshow(imgs[0])
    axes[0].set_title("source 1")
    im1 = axes[1].imshow(dom_map, cmap="viridis")
    axes[1].set_title("focal dominance map (1/2/3)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    pair_cov = E[[0, 1]].max(axis=0) / best
    axes[2].imshow(weight_map(fams[0], pair_cov),
                   cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("coverage of pair {1,2} (dark = focal dead zone)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "two_lattice_focal.png", dpi=120)
    print(OUT / "two_lattice_focal.png")


if __name__ == "__main__":
    main()
