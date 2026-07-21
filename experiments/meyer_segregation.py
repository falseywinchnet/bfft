#!/usr/bin/env python3
"""Texture segregation from the Meyer split, scored against ground truth.

The gallery's segregation fields are built so that mean and standard
deviation are identical on both sides of every boundary: intensity carries
no information at all, and only the geometry of the oscillation -- its
scale or its orientation -- distinguishes the regions.  That is the case
worth scoring, because separating a bright region from a dark one requires
nothing of a decomposition.

For each field we compute region features three ways and cluster them:

  raw       local mean and standard deviation of the image (the baseline
            that these fields are constructed to defeat)
  texture   local energy of the texture layer v
  bands     local energy of each of the three ladder bands, which resolves
            the texture layer by scale

and score the clustering against the known labels under the best
permutation of cluster indices.

MEASURED (mean accuracy over the four fields): raw 0.555, texture layer
0.762, ladder bands 0.853.  Field by field the bands give 0.983 on the
curved brick/grass boundary, 0.894 on the four-way mosaic, 0.993 where
only the period changes -- and 0.542 on the orientation field, which is
chance.

That last one is a real limit and it is a limit of the FEATURE, not of the
decomposition: band energy is isotropic, so two regions carrying the same
scales at different orientations produce the same feature vector.  The
boundary itself is plainly visible in the texture layer and in the band
clustering as a ring; what fails is labelling the interiors.  Separating
orientation would need an oriented feature, which the scale ladder does
not provide.

Run:  .venv/bin/python experiments/meyer_segregation.py
"""

import itertools
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft                            # noqa: E402
import gallery                         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

WIN = 17          # feature window, odd


def boxmean(a, w=WIN):
    """Mean over a w x w window, periodic, by two cumulative passes."""
    k = w // 2
    p = np.pad(a, k, mode="wrap")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    tot = (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w])
    return tot / (w * w)


def energy(a):
    """Local RMS of a zero-mean-ish layer."""
    return np.sqrt(np.maximum(boxmean(a * a) - boxmean(a) ** 2, 0.0))


def kmeans(feats, k, seed=0, iters=60):
    """Plain k-means over standardized features, shape (N, D)."""
    x = feats - feats.mean(0)
    s = x.std(0)
    x = x / np.where(s > 1e-12, s, 1.0)
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), k, replace=False)]
    lab = np.zeros(len(x), dtype=np.int32)
    for _ in range(iters):
        d = ((x[:, None, :] - cent[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = x[m].mean(0)
    return lab


def accuracy(pred, truth, k):
    """Best-permutation agreement: clustering carries no label identity."""
    best = 0.0
    for perm in itertools.permutations(range(k)):
        mapped = np.array(perm)[pred]
        best = max(best, float((mapped == truth).mean()))
    return best


def boundary_f1(pred, truth, tol=WIN // 2):
    """F1 of the predicted region boundary against the true one.

    The tolerance is the feature window's half width: every window that
    straddles a boundary sees a mixture, so no feature computed this way
    can localize better than that, and a tighter tolerance would measure
    the window size rather than the separation."""
    from scipy.ndimage import binary_dilation

    def edges(lab):
        e = np.zeros(lab.shape, bool)
        e[:-1] |= lab[:-1] != lab[1:]
        e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        return e

    ep, et = edges(pred), edges(truth)
    if not ep.any() or not et.any():
        return 0.0
    st = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    prec = float((ep & binary_dilation(et, st)).sum() / ep.sum())
    rec = float((et & binary_dilation(ep, st)).sum() / et.sum())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run_one(key, sub=4):
    img, truth = gallery.load_truth(key)
    k = int(truth.max()) + 1

    t0 = time.perf_counter()
    cart, tex, b0, b1, b2 = bfft.meyer(img, rung_sweeps=200)
    t_full = time.perf_counter() - t0

    feats = {
        "raw": np.stack([boxmean(img), energy(img)], -1),
        "texture": energy(tex)[..., None],
        "bands": np.stack([energy(b0), energy(b1), energy(b2)], -1),
    }
    # Interior only: every window straddling a boundary is a mixture, and
    # scoring those measures window size rather than separation.
    res = {}
    labmaps = {}
    for name, f in feats.items():
        flat = f.reshape(-1, f.shape[-1])[::sub]
        lab_sub = kmeans(flat, k)
        # relabel the whole field by nearest centroid on the full grid
        full = f.reshape(-1, f.shape[-1])
        x = (full - flat.mean(0)) / np.where(flat.std(0) > 1e-12,
                                             flat.std(0), 1.0)
        xs = (flat - flat.mean(0)) / np.where(flat.std(0) > 1e-12,
                                              flat.std(0), 1.0)
        cent = np.stack([xs[lab_sub == j].mean(0) if (lab_sub == j).any()
                         else xs[0] for j in range(k)])
        lab = ((x[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        lab = lab.reshape(img.shape)
        labmaps[name] = lab
        res[name] = (accuracy(lab.ravel(), truth.ravel(), k),
                     boundary_f1(lab, truth))
    return img, truth, (cart, tex, b0, b1, b2), res, labmaps, t_full


def main():
    keys = list(gallery.TRUTH)
    rows = []
    fig, axes = plt.subplots(len(keys), 6, figsize=(16.5, 2.9 * len(keys)))
    print(f"{'field':14s} {'k':>2s} "
          f"{'raw acc':>8s} {'raw F1':>7s} "
          f"{'tex acc':>8s} {'tex F1':>7s} "
          f"{'band acc':>9s} {'band F1':>8s} {'time':>7s}")
    for r, key in enumerate(keys):
        img, truth, layers, res, labs, t = run_one(key)
        k = int(truth.max()) + 1
        rows.append((key, res))
        print(f"{key:14s} {k:2d} "
              f"{res['raw'][0]:8.3f} {res['raw'][1]:7.3f} "
              f"{res['texture'][0]:8.3f} {res['texture'][1]:7.3f} "
              f"{res['bands'][0]:9.3f} {res['bands'][1]:8.3f} "
              f"{t:6.2f}s")

        cart, tex = layers[0], layers[1]
        panels = [(img, "field", "gray"), (cart, "cartoon", "gray"),
                  (tex + 128, "texture", "gray"),
                  (truth, "truth", "viridis"),
                  (labs["raw"], f"raw {res['raw'][0]:.2f}", "viridis"),
                  (labs["bands"], f"bands {res['bands'][0]:.2f}", "viridis")]
        for c, (a, title, cm) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(a, cmap=cm)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            elif c >= 4:
                ax.set_title(title, fontsize=8)   # per-row accuracies
            if c == 0:
                ax.set_ylabel(key, fontsize=8)

    fig.suptitle("Texture segregation at matched first and second order "
                 "statistics: clustering local features", fontsize=11)
    fig.tight_layout()
    p = OUT / "meyer_segregation.png"
    fig.savefig(p, dpi=105)
    print("\nwrote", p)

    m = np.array([[res[n][0] for n in ("raw", "texture", "bands")]
                  for _, res in rows])
    print(f"mean accuracy: raw {m[:, 0].mean():.3f}  "
          f"texture {m[:, 1].mean():.3f}  bands {m[:, 2].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
