#!/usr/bin/env python3
"""Recomposition effects and the exported ROF solve.

  1. ROF: bfft.MeyerPlan.rof matches the python specification's Split
     Bregman sweeps, and the arbitrary-size shim matches a manual
     pad/solve/crop.
  2. LADDER UNCHANGED: the ladder rungs now run through the same extracted
     solver, so decompose must still match the python spec bit for bit
     (covered in meyer_test.py; here we only check the rung constants line
     up -- rof at (1/mu, 10/mu) reproduces the coarsest rung).
  3. IDENTITY: recompose and recompose_channels at unit gains reproduce
     the input, in every colour space, to roundoff.
  4. GAINS: layer gains act only on their own layer and are linear.
  5. COLOUR: sRGB <-> OKLab round trips to roundoff; hue is invariant
     under oklab_lc gains; alpha is carried untouched.
  6. SHADE: the shading layer is what a flat cartoon discards -- shade +
     ROF(cartoon) == cartoon exactly -- and amplifying it introduces less
     mean overshoot beside strong edges than unsharp masking at matched
     added energy.

Run from the repo root:  .venv/bin/python tests/meyer_effects_test.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import bfft  # noqa: E402
from bfft.effects import (SPACES, lab_to_srgb, meyer_channels, recompose,
                          recompose_channels, shade, srgb_to_lab)  # noqa: E402
from meyer_bregman import rof_sb  # noqa: E402


def scene(n=128, seed=5):
    """Ramp + jump + weave + noise: every layer is non-trivial."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    f = 55 + 130 * x / n
    f += 35 * (y > n / 2)
    f += 20 * np.cos(2 * np.pi * (x + y) / 7)
    f += 12 * np.cos(2 * np.pi * y / 5) * (x < n / 2)
    return np.clip(f + 2.0 * rng.standard_normal((n, n)), 0, 255)


def colour_scene(n=96, seed=11):
    rng = np.random.default_rng(seed)
    g = scene(n, seed) / 255.0
    rgb = np.stack([g, np.roll(g, 7, axis=1), np.roll(g, 5, axis=0)], -1)
    return np.clip(rgb * (0.55 + 0.4 * rng.random((n, n, 3))), 0.02, 0.98)


def main():
    ok = True
    f = scene()

    # 1. the exported ROF solve vs the python specification
    plan = bfft.MeyerPlan(f.shape)
    c = 0.05
    got = plan.rof(f, c, sweeps=60, tol=0.0)
    want, _ = rof_sb(f, c, eta=10.0 * c, sweeps=60)
    e = float(np.max(np.abs(got - want)) / f.max())
    print(f"1. rof vs python spec: {e:.3e}", "ok" if e < 1e-9 else "FAIL")
    ok &= e < 1e-9

    fa = f[3:3 + 71, 9:9 + 100].copy()
    shim = bfft.rof(fa, c=c, sweeps=60, tol=0.0)
    top, left = (128 - 71) // 2, (128 - 100) // 2
    padded = np.pad(fa, ((top, 128 - 71 - top), (left, 128 - 100 - left)),
                    mode="symmetric")
    manual = plan.rof(padded, c, sweeps=60,
                      tol=0.0)[top:top + 71, left:left + 100]
    e = float(np.max(np.abs(shim - manual)))
    print(f"   rof shim pad/crop: {e:.3e}", "ok" if e == 0.0 else "FAIL")
    ok &= e == 0.0

    # 2. the rung constants: rof at the coarse rung reproduces the ladder's
    #    coarsest survivor, i.e. decompose's cartoon minus u
    mu = 40.0
    pl = bfft.MeyerPlan(f.shape, mu=mu, rung_sweeps=80, rung_tol=0.0)
    cart, tex, _, _, _ = pl.decompose(f)
    u, v = pl.split(f)
    s0 = pl.rof(v, 1.0 / mu, eta=10.0 / mu, sweeps=80, tol=0.0)
    e = float(np.max(np.abs((cart - u) - s0)) / f.max())
    print(f"2. rof reproduces the coarse rung: {e:.3e}",
          "ok" if e < 1e-12 else "FAIL")
    ok &= e < 1e-12

    # 3. identity at unit gains
    u, v = bfft.meyer_split(f)
    e = float(np.max(np.abs(recompose(f, u, v) - f)))
    print(f"3. recompose identity: {e:.3e}", "ok" if e < 1e-9 else "FAIL")
    ok &= e < 1e-9

    img = colour_scene()
    for space in SPACES:
        sp = meyer_channels(img, space=space, passes=16)
        back = recompose_channels(sp, clip=False)
        e = float(np.max(np.abs(back - img)))
        print(f"   channels identity [{space:9s}]: {e:.3e}",
              "ok" if e < 1e-12 else "FAIL")
        ok &= e < 1e-12

    # 4. gain linearity: out(gc, gt) - out(1, 1) == (gc-1)*u + (gt-1)*v
    a = recompose(f, u, v, 1.7, 0.3)
    b = recompose(f, u, v, 1.0, 1.0)
    e = float(np.max(np.abs((a - b) - (0.7 * u - 0.7 * v))) / f.max())
    print(f"4. gain linearity: {e:.3e}", "ok" if e < 1e-12 else "FAIL")
    ok &= e < 1e-12
    # a texture gain must leave the cartoon layer of the result alone
    z = recompose(f, u, v, 1.0, 0.0)
    e = float(np.max(np.abs(z - (f - v))) / f.max())
    print(f"   gt=0 removes exactly the texture: {e:.3e}",
          "ok" if e < 1e-12 else "FAIL")
    ok &= e < 1e-12

    # 5. colour transform and hue invariance
    e = float(np.max(np.abs(lab_to_srgb(srgb_to_lab(img)) - img)))
    print(f"5. sRGB<->OKLab round trip: {e:.3e}",
          "ok" if e < 1e-12 else "FAIL")
    ok &= e < 1e-12

    sp = meyer_channels(img, space="oklab_lc", passes=16)
    out = recompose_channels(sp, gain_cartoon=(1.0, 0.5),
                             gain_texture=(2.0, 1.5), clip=False)
    lab0, lab1 = srgb_to_lab(img), srgb_to_lab(out)
    h0 = np.arctan2(lab0[..., 2], lab0[..., 1])
    h1 = np.arctan2(lab1[..., 2], lab1[..., 1])
    # Hue is only defined where there is chroma to carry it, and a gain may
    # drive chroma to the zero clamp (saturation cannot go negative), which
    # leaves the pixel grey with no hue to preserve.
    strong = (np.hypot(lab0[..., 1], lab0[..., 2]) > 0.02) & \
             (np.hypot(lab1[..., 1], lab1[..., 2]) > 1e-6)
    dh = np.abs(np.angle(np.exp(1j * (h1 - h0))))[strong]
    print(f"   hue defined on {100 * strong.mean():.0f}% of pixels "
          f"(chroma clamped on {100 * (np.hypot(lab1[..., 1], lab1[..., 2]) <= 1e-6).mean():.1f}%)")
    print(f"   hue invariant under oklab_lc gains: max {dh.max():.3e} rad",
          "ok" if dh.max() < 1e-9 else "FAIL")
    ok &= dh.max() < 1e-9

    rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 0.37)], -1)
    sp = meyer_channels(rgba, space="rgb", passes=16)
    out = recompose_channels(sp, gain_texture=3.0, clip=False)
    alpha_ok = out.shape[2] == 4 and np.array_equal(out[..., 3], rgba[..., 3])
    print("   alpha carried untouched:", "ok" if alpha_ok else "FAIL")
    ok &= alpha_ok

    # 6. the shading layer
    u, v = bfft.meyer_split(f)
    s = shade(u, c=0.02, tol=0.0, sweeps=80)
    flat = bfft.rof(u, c=0.02, tol=0.0, sweeps=80)
    e = float(np.max(np.abs(s + flat - u)) / f.max())
    print(f"6. shade + ROF(cartoon) == cartoon: {e:.3e}",
          "ok" if e < 1e-12 else "FAIL")
    ok &= e < 1e-12
    # the flat cartoon must be flatter: strictly less total variation
    tv = lambda a: float(np.abs(np.diff(a, axis=0)).sum()
                         + np.abs(np.diff(a, axis=1)).sum())
    flatter = tv(flat) < tv(u)
    print(f"   TV(ROF(u)) {tv(flat):.3e} < TV(u) {tv(u):.3e}",
          "ok" if flatter else "FAIL")
    ok &= flatter

    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    def overshoot(before, after):
        gx = np.diff(before, axis=1, append=before[:, :1])
        gy = np.diff(before, axis=0, append=before[:1, :])
        near = maximum_filter(np.hypot(gx, gy), size=5) > 25.0
        hi, lo = maximum_filter(before, 5), minimum_filter(before, 5)
        over = (np.maximum(after - hi, 0.0) + np.maximum(lo - after, 0.0))
        return float(over[near].mean() / (before.max() - before.min()))

    boosted = recompose(f, u, v, 1.0, 1.0, gain_shade=1.5, shade_c=0.02)
    rms = np.sqrt(((boosted - f) ** 2).mean())
    for amount in np.linspace(0.05, 4.0, 80):
        un = f + amount * (f - gaussian_filter(f, 3.0))
        if np.sqrt(((un - f) ** 2).mean()) >= rms:
            break
    o_s, o_u = overshoot(f, boosted), overshoot(f, un)
    print(f"   edge overshoot at matched energy: shade {o_s:.2e} vs "
          f"unsharp {o_u:.2e} ({o_u / o_s:.2f}x)",
          "ok" if o_s < o_u else "FAIL")
    ok &= o_s < o_u

    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
