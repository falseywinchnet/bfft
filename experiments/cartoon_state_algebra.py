#!/usr/bin/env python3
"""How much state does the cartoon stage's split-Bregman actually need?

The kernel stores, per subproblem, four planes: the Bregman dual `b` as
(bx, by) and the fused `d - b` as (dbx, dby).  This file shows that two of
those four are redundant -- not approximately, exactly -- and verifies the
claim by running both recursions side by side.

THE IDENTITY.  Write theta = 1/eta and let

    t_k = grad(u_k) + b_{k-1}                       (the pre-shrink field)

Soft shrinkage and ball projection partition their argument:

    shrink(t, theta) + proj(t, theta) = t     for every t,

because shrink(t) = t*(1 - theta/|t|)_+ and proj(t) = t*min(1, theta/|t|).
Therefore

    d_k = shrink(t_k),   b_k = t_k - d_k = proj(t_k),

and the quantity the linear solve actually consumes is

    p_k = d_k - b_k = t_k - 2*proj(t_k, theta),

which is a pointwise function of t_k alone -- it is the Douglas-Rachford
reflection (2P - I) of t_k about the ball, negated.  The whole recursion
closes on t:

    p_k   = t_k - 2*proj(t_k)                       (consumed, never stored)
    u_{k+1} = (c - eta*Lap)^-1 (c*g - eta*div p_k)
    t_{k+1} = grad(u_{k+1}) + proj(t_k)

So one R^2-valued field per split constraint suffices.  That is also the
floor: ADMM / Douglas-Rachford carries exactly one dual field per constraint,
and the constraint here is d = grad(u), a vector field.  The kernel carries
two.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_state_algebra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def grad(u):
    return (np.roll(u, -1, axis=1) - u, np.roll(u, -1, axis=0) - u)


def div(px, py):
    return ((px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0)))


def symbol(h, w, c, eta):
    lx = 2.0 * np.cos(2.0 * np.pi * np.arange(w // 2 + 1) / w) - 2.0
    ly = 2.0 * np.cos(2.0 * np.pi * np.arange(h) / h) - 2.0
    return 1.0 / (c - eta * (ly[:, None] + lx[None, :]))


def solve(rhs, sym, shape):
    return np.fft.irfft2(np.fft.rfft2(rhs) * sym, s=shape)


def shrink(tx, ty, theta):
    magnitude = np.sqrt(tx * tx + ty * ty)
    scale = np.where(magnitude > theta, 1.0 - theta / np.maximum(
        magnitude, 1e-300), 0.0)
    return tx * scale, ty * scale


def project(tx, ty, theta):
    magnitude = np.sqrt(tx * tx + ty * ty)
    scale = np.minimum(1.0, theta / np.maximum(magnitude, 1e-300))
    return tx * scale, ty * scale


def shipped_form(g, c, eta, passes):
    """Four planes per subproblem: b and the fused d - b."""
    h, w = g.shape
    sym = symbol(h, w, c, eta)
    g_hat = g
    bx = np.zeros_like(g)
    by = np.zeros_like(g)
    pbx = np.zeros_like(g)      # d - b, the stored fusion
    pby = np.zeros_like(g)
    trace = []
    for _ in range(passes):
        u = solve(c * g_hat - eta * div(pbx, pby), sym, (h, w))
        gx, gy = grad(u)
        tx, ty = gx + bx, gy + by
        dx, dy = shrink(tx, ty, 1.0 / eta)
        bx, by = tx - dx, ty - dy
        pbx, pby = dx - bx, dy - by
        trace.append(u.copy())
    return trace


def reflected_form(g, c, eta, passes):
    """Two planes per subproblem: the pre-shrink field alone."""
    h, w = g.shape
    sym = symbol(h, w, c, eta)
    tx = np.zeros_like(g)
    ty = np.zeros_like(g)
    trace = []
    for _ in range(passes):
        qx, qy = project(tx, ty, 1.0 / eta)
        # p = t - 2*proj(t), formed on the fly and never stored
        u = solve(c * g - eta * div(tx - 2.0 * qx, ty - 2.0 * qy),
                  sym, (h, w))
        gx, gy = grad(u)
        tx, ty = gx + qx, gy + qy
        trace.append(u.copy())
    return trace


def check():
    print("== the two recursions are the same recursion ==")
    rng = np.random.default_rng(5)
    for h, w in ((64, 64), (96, 128)):
        image = rng.standard_normal((h, w))
        image[h // 4:h // 2, w // 4:w // 2] += 6.0
        for c, eta in ((0.05, 0.10), (1.0 / 40.0, 10.0 / 40.0),
                       (0.5, 0.05)):
            a = shipped_form(image, c, eta, 24)
            b = reflected_form(image, c, eta, 24)
            gap = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
            scale = float(np.max(np.abs(a[-1])))
            assert gap / scale < 1e-13, (h, w, c, eta, gap / scale)
        print(f"  {h:4d}x{w:<4d} every one of 24 iterates agrees to "
              f"{gap / scale:.2e} relative, all (c, eta)")

    print("\n== the partition identity the reduction rests on ==")
    t = rng.standard_normal((4096, 2)) * np.array([3.0, 3.0])
    for theta in (0.1, 1.0, 10.0):
        sx, sy = shrink(t[:, 0], t[:, 1], theta)
        px, py = project(t[:, 0], t[:, 1], theta)
        residual = max(float(np.max(np.abs(sx + px - t[:, 0]))),
                       float(np.max(np.abs(sy + py - t[:, 1]))))
        print(f"  theta={theta:5.1f}  max |shrink(t) + proj(t) - t| "
              f"= {residual:.2e}")
        assert residual < 1e-14


def inventory():
    """Plane-equivalents allocated, live, and needed, for `meyer_split`."""
    print("\n== state inventory, in image-plane equivalents ==")
    allocated = {
        "u, w": 2.0,
        "bux, buy, dbux, dbuy (u-solver)": 4.0,
        "bvx, bvy, dbvx, dbvy (v-solver)": 4.0,
        "rbx, rby, rdbx, rdby (ROF path only)": 4.0,
        "xit, prev (ROF path only)": 2.0,
        "vplane": 1.0,
        "reT, imT (column stage)": 1.0,
        "f_spec, u_spec, w_spec, d_spec": 4.0,
        "v_spec (ladder path only)": 1.0,
        "s_u, s_v": 1.0,
        "s_r0, s_r1, s_r2, s_gen (ladder/ROF only)": 2.0,
    }
    needed = {
        "u, w": 2.0,
        "t_u = (tux, tuy)": 2.0,
        "t_v = (tvx, tvy)": 2.0,
        "vplane -> caller's output buffer": 0.0,
        "reT, imT": 1.0,
        "f_spec, u_spec, w_spec, d_spec": 4.0,
        "s_u, s_v": 1.0,
    }
    total_alloc = sum(allocated.values())
    total_need = sum(needed.values())
    for name, size in allocated.items():
        print(f"  {size:5.1f}  {name}")
    print(f"  {total_alloc:5.1f}  TOTAL ALLOCATED")
    print()
    for name, size in needed.items():
        print(f"  {size:5.1f}  {name}")
    print(f"  {total_need:5.1f}  TOTAL NEEDED BY meyer_split")
    print(f"\n  reduction {total_alloc / total_need:.2f}x")
    for side in (1024, 2048, 4096):
        plane = side * side * 8 / 2**20
        print(f"    {side}^2: {total_alloc * plane:8.0f} MB "
              f"-> {total_need * plane:7.0f} MB")


def traffic():
    """Plane traversals per pass, which is what a bandwidth-bound stage pays."""
    print("\n== plane traversals per pass ==")
    shipped = {
        "fwd2d_div reads dbx, dby": 2,
        "inv2d writes u": 1,
        "shrink reads u": 1,
        "shrink reads and writes bx, by": 4,
        "shrink writes dbx, dby": 2,
    }
    reduced = {
        "fused div reads tx, ty": 2,
        "inv2d writes u": 1,
        "shrink reads u": 1,
        "shrink reads and writes tx, ty": 4,
    }
    a = 2 * sum(shipped.values())
    b = 2 * sum(reduced.values())
    for name, count in shipped.items():
        print(f"  {count:3d}  {name}")
    print(f"  {a:3d}  TOTAL, both subproblems")
    print()
    for name, count in reduced.items():
        print(f"  {count:3d}  {name}")
    print(f"  {b:3d}  TOTAL, both subproblems      ({a / b:.2f}x less)")


if __name__ == "__main__":
    check()
    inventory()
    traffic()
