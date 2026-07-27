#!/usr/bin/env python3
"""What is actually slow in the cartoon stage's inner problem?

The alternation is still moving at pass 48. Before proposing a different
solver class, decide which part of the problem is responsible, because the
candidate classes attack different parts:

* if the **jump set** is still being discovered late, the difficulty is
  combinatorial and the level-set / min-cut family is the relevant answer;
* if the jump set settles early but the **values** crawl, the difficulty is
  that a first-order method cannot exploit an already-correct active set, and
  the semismooth-Newton family is the relevant answer;
* if the remaining error is **low frequency**, the difficulty is coarse-grid
  propagation and multigrid is the relevant answer.

These are distinguishable by measurement, so measure.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/cartoon_stage_difficulty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402


def gradient_magnitude(u):
    gx = np.roll(u, -1, axis=1) - u
    gy = np.roll(u, -1, axis=0) - u
    return np.sqrt(gx * gx + gy * gy)


def jaccard(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)) / max(union, 1)


def report(name, image, passes=64, lam=0.05, mu=40.0):
    print(f"\n=== {name} ===")
    cartoon, _ = bfft.meyer_trace(
        image, lam=lam, mu=mu, passes=passes, threads=4)
    final = cartoon[-1]
    scale = float(np.linalg.norm(final))

    # The jump set of the converged answer: where the cartoon actually has
    # edges. Threshold set from the answer itself so it is not a free knob.
    magnitude_final = gradient_magnitude(final)
    cut = float(np.percentile(magnitude_final, 95.0))
    active_final = magnitude_final > cut
    print(f"  jump set is {100 * active_final.mean():.1f}% of pixels "
          f"(|grad| > {cut:.3f})")

    # Frequency split of the remaining error, at the scale of a cell.
    h, w = final.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    low = radius <= 0.05

    print(f"  {'pass':>5s} {'rel err':>9s} {'jaccard':>8s} "
          f"{'err on jump':>12s} {'err off jump':>13s} {'low-freq':>9s}")
    rows = []
    for p in range(passes):
        u = cartoon[p]
        error = u - final
        relative = float(np.linalg.norm(error)) / scale
        active = gradient_magnitude(u) > cut
        overlap = jaccard(active, active_final)
        on = float(np.linalg.norm(error[active_final]))
        off = float(np.linalg.norm(error[~active_final]))
        total = max(np.hypot(on, off), 1e-30)
        spectrum = np.abs(np.fft.rfft2(error)) ** 2
        low_share = float(spectrum[low].sum() / max(spectrum.sum(), 1e-30))
        rows.append((p + 1, relative, overlap, on / total, off / total,
                     low_share))
        if p + 1 in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
            print(f"  {p + 1:5d} {relative:9.3e} {overlap:8.4f} "
                  f"{on / total:12.3f} {off / total:13.3f} "
                  f"{low_share:9.3f}")

    settled = next(
        (r[0] for r in rows if r[2] >= 0.98), None)
    print(f"  jump set within 2% of final after pass {settled}; "
          f"relative error there is "
          f"{next(r[1] for r in rows if r[0] == settled):.3e}")
    tail = [r for r in rows if r[0] >= (settled or 1)]
    if len(tail) > 1:
        first, last = tail[0], tail[-1]
        print(f"  from pass {first[0]} to {last[0]} the jump set moves "
              f"{first[2]:.4f} -> {last[2]:.4f} while the error falls "
              f"{first[1]:.2e} -> {last[1]:.2e}")
    return rows


def main():
    rng = np.random.default_rng(4)
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    y, x = yy / (h - 1), xx / (w - 1)
    synthetic = np.zeros((h, w))
    for cy, cx, r, v in ((0.30, 0.35, 0.22, 200.0), (0.68, 0.62, 0.18, 60.0)):
        synthetic[(y - cy) ** 2 + (x - cx) ** 2 < r * r] = v
    synthetic += 40.0 * x
    synthetic += 18.0 * np.sin(2 * np.pi * 26 * x) * np.cos(
        2 * np.pi * 21 * y)
    synthetic += 6.0 * rng.standard_normal((h, w))
    report("synthetic cartoon + texture", np.clip(synthetic, 0, 255))

    camera = np.asarray(gallery.load("camera"), dtype=np.float64)
    report("cameraman", camera)


if __name__ == "__main__":
    main()
