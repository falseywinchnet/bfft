#!/usr/bin/env python3
"""Visual supplement: TGFD vs nested Gilles Alg.3 + ladder on 20 images.

Set12 (12 classic images) + first 8 of BSD68.  Per image:
  - reference: Gilles Algorithm 3, nested inner solves to tol 1e-4
    (optimized implementation: warm inner states, eta=10/mu, threaded FFT);
  - TGFD split (lam=0.05, mu=40, budget 0.9 s x pixels/512^2);
  - ladder bands mu = {40, 10, 2.5} on the TGFD texture layer.
One 3x3 panel PNG per image into experiments/out/supplement/:
  row 1: f, Gilles u, Gilles v
  row 2: |u_TGFD - u_Gilles| x20, TGFD u, TGFD v
  row 3: coarse / mid / fine band.
Prints the LaTeX table rows (sizes, passes, times, agreement).
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

from meyer_bregman import (a2bc_budget, a2bc_nested_warm, texture_ladder,
                           SWEEPS)  # noqa: E402

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
        nf = np.linalg.norm(f)

        # reference: nested Gilles Alg.3 (optimized implementation)
        t0 = time.perf_counter()
        ug, vg, nout = a2bc_nested_warm(f, LAM, MU)
        t_gilles = time.perf_counter() - t0

        # TGFD
        budget = 0.9 * f.size / 512 ** 2
        t0 = time.perf_counter()
        u, v, npass = a2bc_budget(f, LAM, MU, budget_s=budget)
        t_split = time.perf_counter() - t0

        # ladder on the TGFD texture layer
        t0 = time.perf_counter()
        s0, bands = texture_ladder(v, MUS)
        t_ladder = time.perf_counter() - t0

        agree = np.linalg.norm(u - ug) / nf
        rows.append((title, f.shape, nout, t_gilles, npass, t_split,
                     t_ladder, agree))
        print(f"{title:>10s} {f.shape[0]}x{f.shape[1]} | gilles {nout:3d} "
              f"outer {t_gilles:5.1f}s | tgfd {npass:3d} passes "
              f"{t_split:4.2f}s | ladder {t_ladder:4.1f}s | "
              f"agree {agree:.2e}")

        panels = [
            ("$f$", f, 0, 255),
            (f"$u$ Gilles Alg.3 ({t_gilles:.1f} s)", ug, 0, 255),
            ("$v$ Gilles Alg.3", vg, -40, 40),
            (r"$|u_{\rm TGFD}-u_{\rm Gilles}|\times 20$",
             20 * np.abs(u - ug), 0, 255),
            (f"$u$ TGFD ({t_split:.2f} s)", u + s0, 0, 255),
            ("$v$ TGFD", v, -40, 40),
            (r"band $\mu\,40\to10$", bands[0], -30, 30),
            (r"band $\mu\,10\to2.5$", bands[1], -30, 30),
            (r"band $\mu<2.5$", bands[2], -30, 30),
        ]
        ar = f.shape[0] / f.shape[1]
        fig, axes = plt.subplots(3, 3, figsize=(12.6, 3 * 4.2 * ar + 2.2))
        for ax, (name, im, lo, hi) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"{title} ({f.shape[0]}x{f.shape[1]}) — Gilles Alg.3 "
            f"{t_gilles:.1f}s vs TGFD {t_split:.2f}s "
            f"({t_gilles / t_split:.0f}x); agreement "
            f"$\\|\\Delta u\\|/\\|f\\|$ = {agree:.1e}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0)
        fig.savefig(OUT / f"{stem}.png", dpi=110)
        plt.close(fig)

    print("\n% LaTeX table rows")
    for title, shape, nout, tg, npass, ts, tl, agree in rows:
        print(f"{title} & ${shape[0]}\\times{shape[1]}$ & {tg:.1f} & "
              f"{ts:.2f} & {tg/ts:.0f}$\\times$ & {tl:.1f} & "
              f"${agree*1e3:.1f}\\times10^{{-3}}$ \\\\")


if __name__ == "__main__":
    main()
