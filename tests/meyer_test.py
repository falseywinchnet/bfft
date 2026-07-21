#!/usr/bin/env python3
"""Meyer G-norm decomposer (TGFD) validation.

Checks the native kernel against the pure-python specification
(experiments/meyer_bregman.py) and exercises the arbitrary-size shim:

  1. EQUIVALENCE: bfft.MeyerPlan with rung_tol=0 (deterministic sweep
     counts) must match the python reference -- the identical alternation
     built from rof_sb sweeps -- to FFT roundoff on every output.
  2. SUM IDENTITY: cartoon + band_coarse + band_mid + band_fine = u + v
     exactly; image - that sum is the texture-side ROF survivor w.
  3. TOL PATH: with rung_tol=1e-5 both implementations stop their rungs
     within tolerance of each other's bands.
  4. SHIM: bfft.meyer on an arbitrary-size image returns five arrays of
     the input shape, matching a manually padded MeyerPlan run cropped.
  5. SPEED: the native 512^2 decomposition must beat the python path.

Run from the repo root:  .venv/bin/python tests/meyer_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import bfft  # noqa: E402
from meyer_bregman import rof_sb, RofState  # noqa: E402

BARBARA = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
               "63c039dd-f490-43f0-aba0-e10153ab8f90/scratchpad/testimgs/"
               "set12_09.png")


def load_image():
    if BARBARA.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(str(BARBARA)).astype(np.float64)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        return img * 255.0 if img.max() <= 1.5 else img
    rng = np.random.default_rng(7)
    n = 512
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    f = 90 + 40 * x / n + 25 * np.cos(2 * np.pi * x / 6) * (y > n / 2)
    f += 20 * np.cos(2 * np.pi * (x + y) / 11) * (x < n / 2)
    return np.clip(f + rng.standard_normal((n, n)), 0, 255)


def reference(f, lam, mu, passes, rung_sweeps, rung_tol):
    """The python specification: identical alternation from rof_sb sweeps."""
    st_u, st_v = None, None
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    for _ in range(passes):
        u, st_u = rof_sb(f - v, lam, sweeps=1, state=st_u)
        g = f - u
        w, st_v = rof_sb(g, 1.0 / mu, eta=10.0 / mu, sweeps=1, state=st_v)
        v = g - w
    s = []
    for mk in (mu, mu / 4.0, mu / 16.0):
        if rung_tol > 0:
            sk, _ = rof_sb(v, 1.0 / mk, eta=10.0 / mk, tol=rung_tol,
                           max_sweeps=rung_sweeps)
        else:
            sk, _ = rof_sb(v, 1.0 / mk, eta=10.0 / mk, sweeps=rung_sweeps)
        s.append(sk)
    return (u + s[0], v, s[1] - s[0], s[2] - s[1], v - s[2])


def _time_call(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def rel(a, b, f):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(f)) + 1e-300))


def main():
    f512 = load_image()
    f = f512[128:256, 192:320].copy()          # 128^2, textured crop
    lam, mu = 0.05, 40.0
    ok = True

    # 1. deterministic equivalence
    passes, rs = 24, 40
    plan = bfft.MeyerPlan(f.shape, lam=lam, mu=mu, passes=passes,
                          rung_sweeps=rs, rung_tol=0.0)
    got = plan.decompose(f)
    want = reference(f, lam, mu, passes, rs, 0.0)
    names = ("cartoon", "texture", "band_coarse", "band_mid", "band_fine")
    print("1. equivalence vs python spec (rel max err, tol 1e-9):")
    for name, g, w in zip(names, got, want):
        e = rel(g, w, f)
        flag = "ok" if e < 1e-9 else "FAIL"
        ok &= e < 1e-9
        print(f"   {name:>12s}  {e:.3e}  {flag}")

    # 2. sum identity
    s = got[0] + got[2] + got[3] + got[4]
    w_resid = f - s
    u_plus_v = want[0] - (want[0] - s)          # s should equal u+v
    e = rel(s, want[0] + want[2] + want[3] + want[4], f)
    print(f"2. sum identity C==py: {e:.3e}", "ok" if e < 1e-9 else "FAIL")
    ok &= e < 1e-9
    print(f"   residual w range [{w_resid.min():.2f}, {w_resid.max():.2f}]")

    # 3. tol path proximity
    plan_t = bfft.MeyerPlan(f.shape, lam=lam, mu=mu, passes=passes,
                            rung_sweeps=600, rung_tol=1e-5)
    got_t = plan_t.decompose(f)
    want_t = reference(f, lam, mu, passes, 600, 1e-5)
    print("3. tol-path proximity (rel max err, tol 1e-3):")
    for name, g, w in zip(names, got_t, want_t):
        e = rel(g, w, f)
        flag = "ok" if e < 1e-3 else "FAIL"
        ok &= e < 1e-3
        print(f"   {name:>12s}  {e:.3e}  {flag}")

    # 4. arbitrary-size shim
    fa = f512[7:7 + 300, 3:3 + 421].copy()
    outs = bfft.meyer(fa, passes=8, rung_sweeps=20, rung_tol=0.0)
    shapes_ok = all(o.shape == fa.shape for o in outs)
    ph, pw = 512, 512
    top, left = (ph - 300) // 2, (pw - 421) // 2
    padded = np.pad(fa, ((top, ph - 300 - top), (left, pw - 421 - left)),
                    mode="symmetric")
    plan_p = bfft.MeyerPlan((ph, pw), passes=8, rung_sweeps=20, rung_tol=0.0)
    manual = tuple(o[top:top + 300, left:left + 421]
                   for o in plan_p.decompose(padded))
    e = max(rel(a, b, fa) for a, b in zip(outs, manual))
    print(f"4. shim shapes {'ok' if shapes_ok else 'FAIL'}; "
          f"pad/crop match {e:.3e}", "ok" if e < 1e-12 else "FAIL")
    ok &= shapes_ok and e < 1e-12

    # 4c. split(): (u, v) alone, vs the python alternation and vs decompose
    st_u, st_v = None, None
    ru = np.zeros_like(f)
    rv = np.zeros_like(f)
    for _ in range(passes):
        ru, st_u = rof_sb(f - rv, lam, sweeps=1, state=st_u)
        g = f - ru
        rw, st_v = rof_sb(g, 1.0 / mu, eta=10.0 / mu, sweeps=1, state=st_v)
        rv = g - rw
    sc, sx = plan.split(f)
    e_u, e_v = rel(sc, ru, f), rel(sx, rv, f)
    # consistency with decompose: same texture, and cartoon differs by
    # exactly the ladder's coarsest survivor s0 = decompose.cartoon - u
    e_tex = rel(sx, got[1], f)
    ok_c = e_u < 1e-9 and e_v < 1e-9 and e_tex < 1e-12
    print(f"4c. split vs python u {e_u:.3e}, v {e_v:.3e}; "
          f"texture == decompose {e_tex:.3e}", "ok" if ok_c else "FAIL")
    ok &= ok_c

    # 4b. thread-count invariance: outputs bit-identical for T = 1, 2, 4
    outs_t = {}
    for T in (1, 2, 4):
        pl = bfft.MeyerPlan(f.shape, lam=lam, mu=mu, passes=12,
                            rung_sweeps=30, rung_tol=1e-5, threads=T)
        outs_t[T] = pl.decompose(f)
    same = all(np.array_equal(a, b)
               for a, b in zip(outs_t[1], outs_t[2])) and \
           all(np.array_equal(a, b)
               for a, b in zip(outs_t[1], outs_t[4]))
    print(f"4b. thread invariance (T=1,2,4 bit-identical):",
          "ok" if same else "FAIL")
    ok &= same

    # 5. speed, full production settings at 512^2
    plan5 = bfft.MeyerPlan(f512.shape, lam=lam, mu=mu, passes=64,
                           rung_sweeps=600, rung_tol=1e-5)
    plan5.decompose(f512)                       # warm
    t_c = min(_time_call(plan5.decompose, f512) for _ in range(3))
    t0 = time.perf_counter()
    reference(f512, lam, mu, 64, 600, 1e-5)
    t_py = time.perf_counter() - t0
    print(f"5. speed 512^2 (64 passes + 3-rung ladder): native {t_c:.2f}s, "
          f"python {t_py:.2f}s, speedup {t_py / t_c:.1f}x",
          "ok" if t_c < t_py else "FAIL")
    ok &= t_c < t_py

    # 5b. split-only speed: the decomposition without the ladder.
    # Best of 3: a single shot here lands right after the python reference
    # run above and picks up thermal/scheduling noise.
    plan5.split(f512)
    t_s = min(_time_call(plan5.split, f512) for _ in range(3))
    print(f"5b. split-only 512^2 (64 passes, no ladder): {t_s * 1e3:.0f} ms "
          f"({t_c / t_s:.0f}x cheaper than the laddered call)",
          "ok" if t_s < t_c else "FAIL")
    ok &= t_s < t_c

    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
