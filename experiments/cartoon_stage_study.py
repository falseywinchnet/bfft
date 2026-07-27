#!/usr/bin/env python3
"""Where the BFFT cartoon stage spends its time on large images.

The cartoon stage is a Gilles-Osher / split-Bregman TGFD alternation. Per
pass it does exactly four real 2-D transforms -- forward on each of the two
divergence fields, inverse on each of the two spectra -- plus two pointwise
shrinkage sweeps. The linear subproblem is solved exactly in the spectrum by
the symbol 1/(c - eta*(lx + ly)), which is the periodic DFT symbol of
(c - eta*Laplacian).

Three structural questions, none of them micro-optimization:

1. How much work is spent on padding?  `_meyer_padded` rounds each dimension
   up to the next power of two by symmetric reflection.  The penalty is
   worst just above a power of two and can approach 4x in area.
2. Is the fixed pass count needed?  `run_passes` has no convergence test --
   unlike `rof_from_spec`, which does.  If the alternation settles early,
   every remaining pass is four transforms of pure waste.
3. Does a coarse solve predict the fine one?  If it does, the fine level can
   be warm-started from a cheap coarse solve, which is the standard cascadic
   multigrid argument and the only lever that gets better as images grow.

Nothing here modifies the library.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_stage_study.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402


def next_pow2(n):
    p = 8
    while p < n:
        p *= 2
    return p


def synthetic(h, w, seed=3):
    """Cartoon-plus-texture test field, so the split has real work to do."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy / max(h - 1, 1)
    x = xx / max(w - 1, 1)
    image = np.zeros((h, w), dtype=np.float64)
    for cy, cx, r, v in ((0.30, 0.35, 0.22, 200.0), (0.68, 0.62, 0.18, 60.0),
                         (0.50, 0.80, 0.12, 150.0)):
        image[(y - cy) ** 2 + (x - cx) ** 2 < r * r] = v
    image += 40.0 * x
    image += 18.0 * np.sin(2.0 * np.pi * 26.0 * x) * np.cos(
        2.0 * np.pi * 21.0 * y)
    image += 6.0 * rng.standard_normal((h, w))
    return np.clip(image, 0.0, 255.0)


def timed(fn, repeats=3):
    fn()
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def study_padding():
    """The area actually transformed against the area supplied."""
    print("\n== 1. padding penalty ==")
    print(f"  {'image':>13s} {'transformed':>13s} {'waste':>7s}")
    rows = []
    for h, w, label in (
            (256, 256, "power of two"),
            (512, 512, "power of two"),
            (1920, 1080, "1080p"),
            (2048, 2048, "power of two"),
            (2560, 1440, "1440p"),
            (3840, 2160, "4K"),
            (1025, 1025, "just over 1024"),
            (4096, 4096, "power of two"),
            (4100, 4100, "just over 4096")):
        ph, pw = next_pow2(h), next_pow2(w)
        waste = (ph * pw) / (h * w)
        rows.append({"h": h, "w": w, "padded": [ph, pw],
                     "waste": waste, "label": label})
        print(f"  {h:5d}x{w:<5d} {ph:6d}x{pw:<6d} {waste:6.2f}x  {label}")
    return rows


def study_scaling(passes=24):
    """Cost per pass against transform size, and the per-pass slope."""
    print(f"\n== 2. cost scaling, passes={passes} ==")
    rows = []
    for side in (256, 512, 1024, 2048):
        image = synthetic(side, side)
        one = timed(lambda: bfft.meyer_split(
            image, passes=1, threads=4), repeats=3)
        many = timed(lambda: bfft.meyer_split(
            image, passes=passes, threads=4), repeats=2)
        per_pass = (many - one) / max(passes - 1, 1)
        pixels = side * side
        rows.append({"side": side, "one_pass_s": one, "full_s": many,
                     "per_pass_s": per_pass,
                     "ns_per_pixel_per_pass": per_pass / pixels * 1e9})
        print(f"  {side:5d}^2  first pass {one * 1e3:8.1f} ms   "
              f"{passes} passes {many * 1e3:8.1f} ms   "
              f"per pass {per_pass * 1e3:7.2f} ms   "
              f"{per_pass / pixels * 1e9:6.2f} ns/px")
    return rows


def study_convergence(side=1024, passes=48, lam=0.05, mu=40.0):
    """Relative movement of the cartoon layer per pass.

    `meyer_trace` returns every intermediate state from one native pass
    sequence, so this costs one run rather than one run per pass count.
    """
    print(f"\n== 3. convergence of the alternation, {side}^2, "
          f"passes={passes} ==")
    image = synthetic(side, side)
    cartoon, texture = bfft.meyer_trace(
        image, lam=lam, mu=mu, passes=passes, threads=4)
    final = cartoon[-1]
    scale = float(np.linalg.norm(final))
    rows = []
    print(f"  {'pass':>5s} {'step':>11s} {'distance to final':>18s}")
    for p in range(passes):
        step = (float(np.linalg.norm(cartoon[p] - cartoon[p - 1])) / scale
                if p else float("nan"))
        distance = float(np.linalg.norm(cartoon[p] - final)) / scale
        rows.append({"pass": p + 1, "step": step, "distance": distance})
        if p < 6 or (p + 1) % 6 == 0 or p == passes - 1:
            print(f"  {p + 1:5d} {step:11.3e} {distance:18.3e}")
    for target in (1e-2, 3e-3, 1e-3, 3e-4):
        reached = next((r["pass"] for r in rows if r["distance"] <= target),
                       None)
        print(f"    within {target:8.0e} of the {passes}-pass answer "
              f"after {reached} passes")
    return rows


def study_coarse_prediction(side=1024, passes=24, lam=0.05, mu=40.0):
    """Does a half-resolution solve predict the full-resolution one?

    The discrete ROF functional at pixel width h is
    sum|grad u| + (c*h/2) sum|u - f|^2, so halving the resolution doubles
    the effective fidelity constant.  That prediction is tested here rather
    than assumed: several rescalings are tried and the best is reported.
    """
    print(f"\n== 4. coarse solve as a predictor, {side}^2 ==")
    image = synthetic(side, side)
    t0 = time.perf_counter()
    fine_cartoon, _ = bfft.meyer_split(
        image, lam=lam, mu=mu, passes=passes, threads=4)
    fine_s = time.perf_counter() - t0
    scale = float(np.linalg.norm(fine_cartoon))

    half = 0.25 * (image[0::2, 0::2] + image[1::2, 0::2] +
                   image[0::2, 1::2] + image[1::2, 1::2])
    rows = []
    for factor, name in ((1.0, "lam unchanged"), (2.0, "lam x2 (h scaling)"),
                         (0.5, "lam /2")):
        t0 = time.perf_counter()
        coarse_cartoon, _ = bfft.meyer_split(
            half, lam=lam * factor, mu=mu, passes=passes, threads=4)
        coarse_s = time.perf_counter() - t0
        lifted = np.repeat(np.repeat(coarse_cartoon, 2, axis=0), 2, axis=1)
        lifted = lifted[:side, :side]
        distance = float(np.linalg.norm(lifted - fine_cartoon)) / scale
        rows.append({"factor": factor, "name": name, "distance": distance,
                     "coarse_s": coarse_s})
        print(f"  {name:22s} distance to fine answer {distance:.4f}   "
              f"coarse cost {coarse_s / fine_s * 100:5.1f}% of fine")

    # What a fine-level iterate costs to reach the same distance.
    trace_cartoon, _ = bfft.meyer_trace(
        image, lam=lam, mu=mu, passes=passes, threads=4)
    best = min(rows, key=lambda r: r["distance"])
    equivalent = next(
        (p + 1 for p in range(passes)
         if float(np.linalg.norm(trace_cartoon[p] - fine_cartoon)) / scale
         <= best["distance"]), None)
    print(f"  best coarse start ({best['name']}) sits where fine pass "
          f"{equivalent} of {passes} sits, for "
          f"{best['coarse_s'] / fine_s * 100:.1f}% of the cost")
    return rows


def main():
    record = {
        "padding": study_padding(),
        "scaling": study_scaling(),
        "convergence": study_convergence(),
        "coarse": study_coarse_prediction(),
    }
    out = ROOT / "experiments" / "out" / "cartoon_stage_study.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
