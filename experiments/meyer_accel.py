#!/usr/bin/env python3
"""Acceleration of the Meyer decomposition: the reduced-composite theory.

THE IDENTIFICATION.  The Gilles/A2BC alternation

    u_{n+1} = ROF(f - v_n, lam),   v_n = P_{G_mu}(f - u_n)

is exactly proximal gradient descent (ISTA) at the maximal step 1/L on the
single-variable convex composite obtained by eliminating v in closed form:

    min_u  J(u) + S(u),   S(u) = (lam/2) dist^2(f - u, G_mu)
    grad S(u) = -lam * w(u),   w(u) = ROF(f - u, 1/mu)      (L = lam)

Proof of the step identity: f - v_n = u_n + (f-u_n) - P_{G_mu}(f-u_n)
= u_n + ROF(f-u_n, 1/mu) = u_n + w(u_n), so the alternation's u-update is
u_{n+1} = ROF(u_n + w(u_n), lam) = prox_{J/lam}(u_n - (1/lam) grad S(u_n)).

CONSEQUENCES.
- The measured slow geometric transfer is ISTA's behavior on an
  ill-conditioned composite; the flat modes are the low-frequency content
  observed as the swirl difference field.
- The measured primal heavy-ball failure was momentum on the TWO-BLOCK
  form, whose subproblem objectives move every pass (transport frame
  changes).  The reduced objective is STATIC: FISTA on u is sound, costs
  the same two ROF solves per iteration, and carries no stale transport
  (w is re-solved at the extrapolated point).
- "Blocking in the transport": Meyer's ball is exactly parametrized by a
  flux, v = div q with |q| <= mu pointwise.  Substituting makes the
  transport a standing primal variable in a STATIC box:
      min_{u, |q|<=mu} J(u) + (lam/2)||f - u - div q||^2
  solvable by a single-loop primal-dual method (Condat-Vu; p dual to
  grad u) with no inner solves: every step is a gradient, a clip, or a
  div/grad.  Nothing in the geometry moves between iterations.

Experiments:
  E1  numerical confirmation of the alternation == ISTA identity;
  E2  exact-prox race: ISTA vs FISTA vs FISTA+restart, err to reference;
  E3  inexact race at production budgets: 1-sweep warm SB inner solves,
      incumbent alternation vs reduced FISTA, sweeps-to-error;
  E4  explicit-flux Condat-Vu, no inner solves, wall-time-to-error.

Usage: meyer_accel.py [e1|e2|e3|e4|all]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from meyer_bregman import (SWEEPS, FGP_ITERS, grad, div, rig_scene, rof_sb,
                           rof_fgp, gball_fgp, reference_split, psnr)  # noqa: E402


# ----------------------------------------------------------------------
# exact-prox building blocks (FGP inner: reliable dual convergence)
# ----------------------------------------------------------------------

def rof_exact(g, c, iters=800):
    u, _ = rof_fgp(g, c, iters)
    return u


def w_exact(f, u, mu, iters=800):
    """w(u) = ROF(f-u, 1/mu) = -(1/lam) grad S(u)."""
    return rof_exact(f - u, 1.0 / mu, iters)


# ----------------------------------------------------------------------
# E1: the identity
# ----------------------------------------------------------------------

def exp_e1():
    f, *_ = rig_scene(64)
    lam, mu = 0.05, 60.0
    inner = 600
    # two-block alternation, exact inner solves
    u_a = np.zeros_like(f)
    # reduced ISTA, exact inner solves
    u_i = np.zeros_like(f)
    print("[e1] alternation vs reduced ISTA, exact prox, 64^2")
    print(f"{'iter':>5s} {'||u_alt - u_ista|| / ||f||':>28s}")
    nf = np.linalg.norm(f)
    for k in range(12):
        # alternation: v = P_Gmu(f-u) = (f-u) - ROF(f-u,1/mu); u = ROF(f-v)
        v = (f - u_a) - rof_exact(f - u_a, 1.0 / mu, inner)
        u_a = rof_exact(f - v, lam, inner)
        # reduced: u = ROF(u + w(u), lam)
        u_i = rof_exact(u_i + rof_exact(f - u_i, 1.0 / mu, inner), lam, inner)
        print(f"{k + 1:5d} {np.linalg.norm(u_a - u_i) / nf:28.2e}")


# ----------------------------------------------------------------------
# E2: exact-prox acceleration race
# ----------------------------------------------------------------------

def run_reduced(f, lam, mu, iters, mode="ista", inner=600, u_ref=None,
                log_every=None):
    """mode: ista | fista | fista_restart.  Returns error trace."""
    u = np.zeros_like(f)
    u_prev = u.copy()
    t = 1.0
    nf = np.linalg.norm(f)
    trace = []
    for k in range(iters):
        if mode == "ista":
            y = u
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            y = u + beta * (u - u_prev)
            t = t_new
        w = rof_exact(f - y, 1.0 / mu, inner)
        u_next = rof_exact(y + w, lam, inner)
        if mode == "fista_restart":
            # gradient restart: momentum fighting descent direction
            if float(np.sum((y - u_next) * (u_next - u))) > 0.0:
                t = 1.0
        u_prev, u = u, u_next
        if u_ref is not None and (log_every is None or (k + 1) % log_every == 0
                                  or k == iters - 1):
            trace.append((k + 1, np.linalg.norm(u - u_ref) / nf))
    return u, trace


def exp_e2():
    f, u_true, vc, vf, *_ = rig_scene(128)
    lam, mu = 0.05, 60.0
    print("[e2] exact-prox race, 128^2 (reference: cached exact split)")
    u_ref, v_ref = reference_split(f, lam, mu)
    for mode in ("ista", "fista", "fista_restart"):
        t0 = time.perf_counter()
        u, tr = run_reduced(f, lam, mu, 120, mode=mode, inner=500,
                            u_ref=u_ref, log_every=10)
        dt = time.perf_counter() - t0
        pts = "  ".join(f"{k}:{e:.1e}" for k, e in tr)
        print(f"  {mode:>14s} ({dt:5.1f}s): {pts}")


# ----------------------------------------------------------------------
# E3: inexact production race (warm SB, few sweeps per operator)
# ----------------------------------------------------------------------

def incumbent(f, lam, mu, passes):
    st_u, st_v = None, None
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    ck = []
    for p in range(passes):
        u, st_u = rof_sb(f - v, lam, sweeps=1, state=st_u)
        g = f - u
        w, st_v = rof_sb(g, 1.0 / mu, eta=10.0 / mu, sweeps=1, state=st_v)
        v = g - w
        ck.append((SWEEPS["n"], u.copy()))
    return u, ck


def reduced_fista_inexact(f, lam, mu, iters, k_w=1, k_u=1, restart=True):
    """FISTA on the reduced composite with warm k-sweep SB inner solves.

    The w-solver state is warm across iterations (its input f-y moves
    slowly); the u-prox state likewise.  Momentum lives on u only; the
    transports inside both ROF solves are re-derived at the extrapolated
    point every iteration -- nothing stale is carried.
    """
    st_w, st_u = None, None
    u = np.zeros_like(f)
    u_prev = u.copy()
    t = 1.0
    ck = []
    for k in range(iters):
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        y = u + beta * (u - u_prev)
        t = t_new
        w, st_w = rof_sb(f - y, 1.0 / mu, eta=10.0 / mu, sweeps=k_w,
                         state=st_w)
        u_next, st_u = rof_sb(y + w, lam, sweeps=k_u, state=st_u)
        if restart and float(np.sum((y - u_next) * (u_next - u))) > 0.0:
            t = 1.0
        u_prev, u = u, u_next
        ck.append((SWEEPS["n"], u.copy()))
    return u, ck


def exp_e3():
    f, *_ = rig_scene(256)
    lam, mu = 0.05, 60.0
    u_ref, v_ref = reference_split(f, lam, mu)
    nf = np.linalg.norm(f)
    print("[e3] inexact production race, 256^2, err vs reference by sweeps")
    budgets = (100, 200, 400, 800, 1600)

    def report(name, ck):
        errs = []
        for b in budgets:
            best = None
            for sw, uu in ck:
                if sw <= b:
                    best = uu
            errs.append(np.linalg.norm(best - u_ref) / nf
                        if best is not None else float("nan"))
        print(f"  {name:>34s}: " +
              "  ".join(f"{b}:{e:.2e}" for b, e in zip(budgets, errs)))

    SWEEPS["n"] = 0
    _, ck = incumbent(f, lam, mu, 800)
    report("incumbent alternation (ISTA)", ck)
    for k_w, k_u, rs in ((1, 1, True), (1, 1, False), (2, 2, True),
                         (4, 4, True)):
        SWEEPS["n"] = 0
        _, ck = reduced_fista_inexact(f, lam, mu, 800 // (k_w + k_u) * 1,
                                      k_w=k_w, k_u=k_u, restart=rs)
        report(f"reduced FISTA k={k_w}/{k_u} restart={rs}", ck)


# ----------------------------------------------------------------------
# E4: explicit-flux Condat-Vu (no inner solves, static geometry)
# ----------------------------------------------------------------------

def condat_vu(f, lam, mu, iters, tau=None, u_ref=None, log_every=200):
    """min_{u, |q|<=mu} J(u) + (lam/2)||f - u - div q||^2.

    Condat-Vu: p dual to grad u (|p|<=1 pointwise, isotropic);
    u, q primal with the smooth coupling handled by gradient steps;
    q clipped to the mu-ball (isotropic).  All steps explicit.
    """
    H, W = f.shape
    u = np.zeros_like(f)
    qx = np.zeros_like(f)
    qy = np.zeros_like(f)
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    # step rule: 1/tau - sigma*||grad||^2 >= L_H/2, ||grad||^2 <= 8,
    # L_H = lam*||[I, div]||^2 <= 9*lam
    if tau is None:
        tau = 0.99 / np.sqrt(8.0)
    sigma = (1.0 / tau - 4.5 * lam) / 8.0
    nf = np.linalg.norm(f)
    trace = []
    u_bar = u.copy()
    for k in range(iters):
        # dual ascent on p at the extrapolated primal
        gx, gy = grad(u_bar)
        px = px + sigma * gx
        py = py + sigma * gy
        mag = np.maximum(1.0, np.sqrt(px * px + py * py))
        px /= mag
        py /= mag
        # primal descent
        r = f - u - div(qx, qy)
        u_new = u + tau * (div(px, py) + lam * r)
        rgx, rgy = grad(r)
        qx_new = qx - tau * lam * rgx
        qy_new = qy - tau * lam * rgy
        qmag = np.sqrt(qx_new * qx_new + qy_new * qy_new)
        scale = mu / np.maximum(mu, qmag)
        qx_new *= scale
        qy_new *= scale
        u_bar = 2.0 * u_new - u
        u, qx, qy = u_new, qx_new, qy_new
        if u_ref is not None and ((k + 1) % log_every == 0 or k == iters - 1):
            trace.append((k + 1, np.linalg.norm(u - u_ref) / nf))
    return u, (qx, qy), trace


def exp_e4():
    f, *_ = rig_scene(256)
    lam, mu = 0.05, 60.0
    u_ref, v_ref = reference_split(f, lam, mu)
    print("[e4] explicit-flux Condat-Vu, 256^2 (no inner solves)")
    # sanity: gradient signs via objective decrease
    t0 = time.perf_counter()
    u, q, tr = condat_vu(f, lam, mu, 4000, u_ref=u_ref, log_every=400)
    dt = time.perf_counter() - t0
    per = dt / 4000 * 1e3
    print(f"  {per:.2f} ms/iter; err trace: " +
          "  ".join(f"{k}:{e:.1e}" for k, e in tr))
    print(f"  total {dt:.1f}s; for scale: incumbent sweep ~0.9 ms at 256^2")


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    t0 = time.perf_counter()
    if which in ("e1", "all"):
        exp_e1()
    if which in ("e2", "all"):
        exp_e2()
    if which in ("e3", "all"):
        exp_e3()
    if which in ("e4", "all"):
        exp_e4()
    print(f"[total {time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main(sys.argv)
