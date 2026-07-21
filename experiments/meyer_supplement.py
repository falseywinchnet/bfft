#!/usr/bin/env python3
"""Visual supplement: TGFD decomposition + 3-rung ladder on 20 test images.

Set12 (12 classic images, 256^2/512^2) + first 8 of BSD68 (481x321).
Per image: TGFD split (lam=0.05, mu=40, budget scaled by pixel count),
then ladder bands mu = {40, 10, 2.5}.  One 2x3 panel PNG per image
(f / u / v ; coarse / mid / fine band) into experiments/out/supplement/,
plus a summary table printed for the LaTeX document.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "supplement"
IMGS = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
            "63c039dd-f490-43f0-aba0-e10153ab8f90/scratchpad/testimgs")

from meyer_bregman import a2bc_budget, texture_ladder, SWEEPS  # noqa: E402

NAMES = {
    "set12_01": "Cameraman", "set12_02": "House", "set12_03": "Peppers",
    "set12_04": "Starfish", "set12_05": "Monarch", "set12_06": "Airplane",
    "set12_07": "Parrot", "set12_08": "Lena", "set12_09": "Barbara",
    "set12_10": "Boat", "set12_11": "Man", "set12_12": "Couple",
    "bsd68_001": "BSD68-01", "bsd68_002": "BSD68-02",
    "bsd68_003": "BSD68-03", "bsd68_004": "BSD68-04",
    "bsd68_005": "BSD68-05", "bsd68_006": "BSD68-06",
    "bsd68_007": "BSD68-07", "bsd68_008": "BSD68-08",
}

LAM, MU = 0.05, 40.0
MUS = [40.0, 10.0, 2.5]


def load(path):
    import matplotlib.image as mpimg
    img = mpimg.imread(str(path)).astype(np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img * 255.0 if img.max() <= 1.5 else img


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, title in NAMES.items():
        f = load(IMGS / f"{stem}.png")
        # budget scaled to pixel count: 0.9 s at 512^2
        budget = 0.9 * f.size / 512 ** 2
        SWEEPS["n"] = 0
        t0 = time.perf_counter()
        u, v, npass = a2bc_budget(f, LAM, MU, budget_s=budget)
        t_split = time.perf_counter() - t0
        t0 = time.perf_counter()
        s0, bands = texture_ladder(v, MUS)
        t_ladder = time.perf_counter() - t0
        rows.append((title, f.shape, npass, t_split, t_ladder))
        print(f"{title:>10s} {f.shape[0]}x{f.shape[1]} passes {npass:4d} "
              f"split {t_split:5.2f}s ladder {t_ladder:5.2f}s")

        landscape = f.shape[1] >= f.shape[0]
        panels = [("$f$", f, 0, 255), ("$u$ (cartoon)", u + s0, 0, 255),
                  ("$v$ (texture)", v, -40, 40),
                  (r"band $\mu\,40\to10$", bands[0], -30, 30),
                  (r"band $\mu\,10\to2.5$", bands[1], -30, 30),
                  (r"band $\mu<2.5$", bands[2], -30, 30)]
        ar = f.shape[0] / f.shape[1]
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 2 * 4.2 * ar + 1.7))
        for ax, (name, im, lo, hi) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{title} ({f.shape[0]}x{f.shape[1]}) — "
                     f"TGFD {t_split:.2f}s, {npass} passes; "
                     f"ladder {t_ladder:.2f}s", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.2)
        fig.savefig(OUT / f"{stem}.png", dpi=110)
        plt.close(fig)

    print("\n% LaTeX table rows")
    for title, shape, npass, ts, tl in rows:
        print(f"{title} & ${shape[0]}\\times{shape[1]}$ & {npass} & "
              f"{ts:.2f} & {tl:.2f} \\\\")


if __name__ == "__main__":
    main()
