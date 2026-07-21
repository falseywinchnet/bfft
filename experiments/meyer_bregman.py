#!/usr/bin/env python3
"""Gilles-Osher Bregman Meyer G-norm decomposition, fused with the
two-lattice program's levers: seeding, heavy-ball momentum, warm
interleaved inner sweeps, and the ratio-4 scale ladder.

Paper: Gilles & Osher, "Bregman implementation of Meyer's G-norm for
cartoon + textures decomposition" (arXiv 2410.22777).  Their Algorithm 3:

    u_{n+1} = P_ROF(f - v_n, lambda)
    v_{n+1} = f - u_{n+1} - (1/lambda) P_ROF(lambda (f - u_{n+1}), 1/(lambda mu))

with P_ROF(g, c) = argmin_w J(w) + (c/2)||g - w||^2 (second argument =
fidelity coefficient; convention fixed by their Prop-2 proof), solved by
Split Bregman (their Algorithm 2).  Substituting w = lambda z in the
texture step collapses it exactly:

    (1/lambda) P_ROF(lambda g, 1/(lambda mu)) = argmin_z J(z) + (1/(2 mu))||g - z||^2
                                              = ROF(g, 1/mu)
    =>  v_{n+1} = (f - u_{n+1}) - ROF(f - u_{n+1}, 1/mu) = P_{G_mu}(f - u_{n+1})

(the Chambolle identity: the ROF residual at fidelity c is the projection
onto the G-ball of radius 1/c).  So Algorithm 3 is a two-projector
alternation in disguise:

    u <- ROF(f - v, lambda)         # cartoon step, ball radius 1/lambda leaves u
    v <- P_{G_mu}(f - u)            # texture step, G-ball radius mu

Per outer pass the u-step can move at most a G-norm sliver of size 1/lambda
from u to v, so for a clean split (1/lambda << texture G-norm) the transfer
is a slow geometric contraction -- exactly the regime where our measured
levers apply (two_lattice exp_a: cold start stagnates, near-seed converges
in ~1 iteration; momentum beta=0.88 helps in both regimes).

Levers transplanted here:
  seed      -- loose-sweep first split (few Bregman sweeps), buys the basin;
  momentum  -- heavy ball on (u, v) trial points, beta=0.88;
  warm      -- persistent Bregman (d, b) fields across outer passes;
  interleave-- ONE inner sweep per subproblem per outer pass (flat loop);
  ladder    -- texture bands as ROF scale-space differences along a
               geometric mu-ladder, ratio 4 (task-10: r* in (4,8]).

Cost currency: total inner sweeps (1 sweep = 1 FFT u-solve + shrink + b
update on the full image); the honest cross-variant metric is the
fixed-point residual max(||u - ROF(f-v,lam)||, ||v - P_Gmu(f-u)||)/||f||
computed by high-accuracy offline probes.

Images in [0, 255] float64.  Periodic boundaries (FFT-diagonal u-solve).

Usage: meyer_bregman.py [engine|transition|iters|ladder|barbara|barbara_ladder|all]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SCRATCH = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
               "63c039dd-f490-43f0-aba0-e10153ab8f90/scratchpad")

# ----------------------------------------------------------------------
# engine: Split Bregman ROF (Gilles Alg. 2), periodic BC, FFT u-solve
# ----------------------------------------------------------------------

SWEEPS = {"n": 0}          # global inner-sweep counter (the cost currency)

try:                        # threaded FFT backend (2.2x at 512^2)
    import scipy.fft as _sfft

    def _rfft2(a):
        return _sfft.rfft2(a, workers=-1)

    def _irfft2(a, s):
        return _sfft.irfft2(a, s=s, workers=-1)
except ImportError:
    def _rfft2(a):
        return np.fft.rfft2(a)

    def _irfft2(a, s):
        return np.fft.irfft2(a, s=s)


def grad(u):
    """Forward differences, periodic."""
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy


def div(px, py):
    """Adjoint: div = -grad^T."""
    return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))


_LAP = {}


def lap_hat(shape):
    """FFT symbol of the periodic 5-point Laplacian (<= 0)."""
    if shape not in _LAP:
        n0, n1 = shape
        wy = 2.0 * np.cos(2 * np.pi * np.arange(n0) / n0) - 2.0
        wx = 2.0 * np.cos(2 * np.pi * np.arange(n1) / n1) - 2.0
        _LAP[shape] = wy[:, None] + wx[None, :n1 // 2 + 1]   # rfft2 layout
    return _LAP[shape]


class RofState:
    """Warm-startable Split Bregman state (d, b fields)."""

    def __init__(self, shape):
        z = np.zeros(shape)
        self.dx, self.dy = z.copy(), z.copy()
        self.bx, self.by = z.copy(), z.copy()
        self.u = None


def rof_sb(g, c, eta=None, state=None, sweeps=None, tol=1e-4, max_sweeps=400):
    """min_u J(u) + (c/2)||g-u||^2 by Split Bregman (Gilles Alg. 2).

    One sweep = FFT u-solve + anisotropic-norm... no: isotropic shrink + b update.
    Returns (u, state).  If `sweeps` is given, runs exactly that many;
    otherwise runs to relative change < tol (nested-baseline mode).
    """
    eta = 2.0 * c if eta is None else eta
    st = state if state is not None else RofState(g.shape)
    L = lap_hat(g.shape)
    denom = c - eta * L
    g_hat = _rfft2(g)
    u = st.u if st.u is not None else g.copy()
    n = sweeps if sweeps is not None else max_sweeps
    for it in range(n):
        rhs_hat = c * g_hat - eta * _rfft2(div(st.dx - st.bx,
                                               st.dy - st.by))
        u_new = _irfft2(rhs_hat / denom, s=g.shape)
        ux, uy = grad(u_new)
        tx, ty = ux + st.bx, uy + st.by
        s = np.sqrt(tx * tx + ty * ty)
        coef = np.maximum(s - 1.0 / eta, 0.0) / np.maximum(s, 1e-12)
        st.dx, st.dy = coef * tx, coef * ty
        st.bx += ux - st.dx
        st.by += uy - st.dy
        SWEEPS["n"] += 1
        done = (sweeps is None and
                np.linalg.norm(u_new - u) <= tol * np.linalg.norm(u_new))
        u = u_new
        if done:
            break
    st.u = u
    return u, st


def gball(g, mu, **kw):
    """P_{G_mu}(g) = g - ROF(g, 1/mu): projection onto the G-ball radius mu."""
    u, st = rof_sb(g, 1.0 / mu, **kw)
    return g - u, st


def chambolle_proj(g, mu, tau=0.124, iters=None, tol=1e-4, max_iters=4000):
    """P_{G_mu}(g) by the paper's Prop. 1 fixed point (context baseline).

    p^{n+1} = (p + tau q) / (1 + tau |q|),  q = grad(div p - g/mu);
    P_{G_mu}(g) = mu div p.  Counts its own iterations (returned).
    """
    px = np.zeros_like(g)
    py = np.zeros_like(g)
    gm = g / mu
    v_prev = None
    n = iters if iters is not None else max_iters
    for it in range(n):
        qx, qy = grad(div(px, py) - gm)
        mag = np.sqrt(qx * qx + qy * qy)
        px = (px + tau * qx) / (1.0 + tau * mag)
        py = (py + tau * qy) / (1.0 + tau * mag)
        if iters is None and it % 10 == 9:
            v = mu * div(px, py)
            if v_prev is not None and \
               np.linalg.norm(v - v_prev) <= tol * (np.linalg.norm(v) + 1e-12):
                return v, it + 1
            v_prev = v
    return mu * div(px, py), n


# ----------------------------------------------------------------------
# dual-flux engine: FGP (FISTA on the transport variable)
# ----------------------------------------------------------------------
#
# Meyer's G-norm is a transport norm: v = div p, ||p||_inf <= mu -- the
# flux field p is the transport object that carries texture mass.  ROF's
# dual is a constrained problem in that flux:
#
#     ROF(g, c):  u = g + div(p)/c,   p* = argmin_{|p|<=1} ||g + div p / c||^2
#
# Projected gradient on p with step c/8 (||grad div|| <= 8) is Chambolle;
# FISTA extrapolation ON THE FLUX is FGP (Beck-Teboulle) -- momentum on
# the dual/transport variable, which is convex-sound, in contrast to the
# measured failure of primal heavy-ball (beta >= 0.8 diverges).  States
# persist across outer A2BC passes: flux memory.

FGP_ITERS = {"n": 0}


class FluxState:
    def __init__(self, shape):
        self.px = np.zeros(shape)
        self.py = np.zeros(shape)
        self.px_prev = self.px.copy()
        self.py_prev = self.py.copy()
        self.t = 1.0


def rof_fgp(g, c, iters, state=None, restart=False):
    """ROF(g, c) by fast gradient projection on the dual flux."""
    st = state if state is not None else FluxState(g.shape)
    if restart:
        st.t = 1.0
        st.px_prev[:] = st.px
        st.py_prev[:] = st.py
    step = c / 8.0
    for _ in range(iters):
        # FISTA extrapolated point (momentum on the flux)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * st.t * st.t))
        a = (st.t - 1.0) / t_new
        yx = st.px + a * (st.px - st.px_prev)
        yy = st.py + a * (st.py - st.py_prev)
        u = g + div(yx, yy) / c
        gx, gy = grad(u)
        qx = yx + step * gx
        qy = yy + step * gy
        nrm = np.sqrt(qx * qx + qy * qy)
        scale = 1.0 / np.maximum(1.0, nrm)
        st.px_prev[:] = st.px
        st.py_prev[:] = st.py
        st.px = qx * scale
        st.py = qy * scale
        st.t = t_new
        FGP_ITERS["n"] += 1
    u = g + div(st.px, st.py) / c
    return u, st


def gball_fgp(g, mu, iters, state=None, **kw):
    """P_{G_mu}(g) = -div(p)/c with c = 1/mu: pure flux readout."""
    u, st = rof_fgp(g, 1.0 / mu, iters, state=state, **kw)
    return g - u, st


def a2bc_flux(f, lam, mu, iters_per_step=2, seed_iters=20, max_outer=400,
              outer_tol=1e-6, restart_each=False, checkpoints=None):
    """A2BC with both projections living on persistent flux fields.

    The entire decomposition state is two transport fields (p_u for the
    cartoon step's ball radius 1/lam, p_v for the texture ball mu); the
    images u, v are readouts.  Momentum is FISTA-on-flux, memory is the
    carried flux."""
    st_u, st_v = None, None
    u, st_u = rof_fgp(f, lam, seed_iters, state=st_u)
    v, st_v = gball_fgp(f - u, mu, seed_iters, state=st_v)
    ck = []
    for it in range(max_outer):
        u_new, st_u = rof_fgp(f - v, lam, iters_per_step, state=st_u,
                              restart=restart_each)
        v_new, st_v = gball_fgp(f - u_new, mu, iters_per_step, state=st_v,
                                restart=restart_each)
        delta = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u, v = u_new, v_new
        if checkpoints is not None:
            ck.append((FGP_ITERS["n"], u.copy(), v.copy()))
        if delta < outer_tol:
            break
    return u, v, it + 1, ck


# ----------------------------------------------------------------------
# A2BC variants
# ----------------------------------------------------------------------

def a2bc_cold(f, lam, mu, outer_tol=1e-4, inner_tol=1e-4, max_outer=200,
              checkpoints=None):
    """Gilles Algorithm 3, faithful: u0=v0=0, nested inner solves to tol."""
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    ck = []
    for it in range(max_outer):
        u_new, _ = rof_sb(f - v, lam, tol=inner_tol)
        v_new, _ = gball(f - u_new, mu, tol=inner_tol)
        delta = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u, v = u_new, v_new
        if checkpoints is not None:
            ck.append((SWEEPS["n"], u.copy(), v.copy()))
        if delta < outer_tol:
            break
    return u, v, it + 1, ck


def a2bc_nested_warm(f, lam, mu, inner_tol=1e-4, outer_tol=1e-4,
                     max_outer=200, eta_v_mult=10.0, max_sweeps=400):
    """Gilles Algorithm 3, optimized implementation.

    Identical nesting discipline (each inner ROF solved to relative
    tolerance, outer alternation to tolerance, cold start) with two
    implementation-level optimizations that do not alter the algorithm:
    warm-started inner Bregman states across outer passes, and the
    measured accuracy-per-sweep penalty eta = eta_v_mult/mu on the
    weak-fidelity texture step.  Threaded FFTs via the module backend.
    """
    st_u, st_v = None, None
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    for it in range(max_outer):
        u_new, st_u = rof_sb(f - v, lam, state=st_u, tol=inner_tol,
                             max_sweeps=max_sweeps)
        w, st_v = rof_sb(f - u_new, 1.0 / mu, eta=eta_v_mult / mu,
                         state=st_v, tol=inner_tol, max_sweeps=max_sweeps)
        v_new = (f - u_new) - w
        delta = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u, v = u_new, v_new
        if delta < outer_tol:
            break
    return u, v, it + 1


def a2bc_fused(f, lam, mu, beta=0.0, seed_sweeps=0, sweeps_per_step=1,
               max_outer=400, outer_tol=1e-5, checkpoints=None):
    """Seed + heavy ball + warm state + interleaved single sweeps."""
    st_u, st_v = None, None
    # --- seed: loose-sweep first split (cost = 2*seed_sweeps sweeps) ---
    u, st_u = rof_sb(f, lam, sweeps=seed_sweeps, state=st_u)
    v, st_v = gball(f - u, mu, sweeps=seed_sweeps, state=st_v)
    u_prev, v_prev = u.copy(), v.copy()
    ck = [(SWEEPS["n"], u.copy(), v.copy())] if checkpoints is not None else []
    for it in range(max_outer):
        # momentum trial points (inert on the first pass)
        v_trial = v + beta * (v - v_prev)
        u_new, st_u = rof_sb(f - v_trial, lam, sweeps=sweeps_per_step,
                             state=st_u)
        u_trial = u_new + beta * (u_new - u_prev)
        v_new, st_v = gball(f - u_trial, mu, sweeps=sweeps_per_step,
                            state=st_v)
        delta = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u_prev, v_prev = u, v
        u, v = u_new, v_new
        if checkpoints is not None:
            ck.append((SWEEPS["n"], u.copy(), v.copy()))
        if delta < outer_tol:
            break
    return u, v, it + 1, ck


def a2bc_budget(f, lam, mu, budget_s=0.8, eta_v_mult=10.0, omega=1.0,
                cont=1.0, cont_frac=0.5, seed_sweeps=0, est_passes=400):
    """Wall-clock-budgeted fused split with memoryless levers only.

    eta_v_mult -- SB penalty eta = mult/mu on the weak-fidelity v-step;
    omega      -- Krasnoselskii-Mann over-relaxation of the v transfer:
                  v <- v + omega (v_new - v)  (memoryless, no history);
    cont       -- lambda continuation: start at lam/cont, geometric anneal
                  to lam over the first cont_frac of the budget (coarse
                  fidelity = fat transfer slivers early);
    seed_sweeps-- optional SB seed on u then v before the loop.
    """
    t0 = time.perf_counter()
    st_u, st_v = None, None
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    if seed_sweeps:
        u, st_u = rof_sb(f, lam / cont, sweeps=seed_sweeps, state=st_u)
        w, st_v = rof_sb(f - u, 1.0 / mu, eta=eta_v_mult / mu,
                         sweeps=seed_sweeps, state=st_v)
        v = (f - u) - w
    npass = 0
    while True:
        el = (time.perf_counter() - t0) / budget_s
        if el >= 1.0:
            break
        lam_t = lam / cont ** max(0.0, 1.0 - el / max(cont_frac, 1e-9)) \
            if cont > 1.0 else lam
        u, st_u = rof_sb(f - v, lam_t, sweeps=1, state=st_u)
        g = f - u
        w, st_v = rof_sb(g, 1.0 / mu, eta=eta_v_mult / mu, sweeps=1,
                         state=st_v)
        v = v + omega * ((g - w) - v)
        npass += 1
    return u, v, npass


def fp_residual(f, u, v, lam, mu):
    """High-accuracy fixed-point residual (offline probe, not counted)."""
    n0 = SWEEPS["n"]
    u_star, _ = rof_sb(f - v, lam, tol=1e-7, max_sweeps=800)
    v_star, _ = gball(f - u, mu, tol=1e-7, max_sweeps=800)
    SWEEPS["n"] = n0
    nf = np.linalg.norm(f)
    return max(np.linalg.norm(u - u_star), np.linalg.norm(v - v_star)) / nf


# ----------------------------------------------------------------------
# transport experiment
# ----------------------------------------------------------------------

_REF = {}


def reference_split(f, lam, mu, outers=120, inner=2000):
    """High-accuracy fixed point (FGP inner), disk-cached: the truth stick."""
    key = f"ref_{f.shape[0]}x{f.shape[1]}_{lam}_{mu}_{outers}_{inner}_" \
          f"{float(np.sum(f)):.3f}"
    if key in _REF:
        return _REF[key]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (key.replace(".", "p") + ".npz")
    if path.exists():
        z = np.load(path)
        _REF[key] = (z["u"], z["v"])
        return _REF[key]
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    drift_probe = None
    for k in range(outers):
        u, _ = rof_fgp(f - v, lam, inner)
        v, _ = gball_fgp(f - u, mu, inner)
        if k == outers // 2:
            drift_probe = u.copy()
    drift = np.linalg.norm(u - drift_probe) / np.linalg.norm(f)
    print(f"  [reference] outer drift over last half: {drift:.2e} "
          f"(err floor of all comparisons)")
    np.savez_compressed(path, u=u, v=v)
    _REF[key] = (u, v)
    return _REF[key]


def exp_transport(make_fig=True):
    """Momentum/memory on the flux (dual) vs on the image (primal)."""
    f, u_true, vc, vf, m_c, m_f = rig_scene(256)
    lam, mu = 0.05, 60.0
    print(f"[transport] reference split (FGP 3000x60, cached)...")
    u_ref, v_ref = reference_split(f, lam, mu)
    nf = np.linalg.norm(f)
    print(f"  reference sanity: PSNR u* {psnr(u_ref, u_true):.2f}, "
          f"v* {psnr(v_ref, vc + vf):.2f}")

    # ---- T1: the weak-fidelity bottleneck, solver race ----
    g = f - u_ref                    # the actual v-step input at the answer
    v_star, _ = gball_fgp(g, mu, 12000)
    solvers = {
        "SB eta=2c": lambda k: gball(g, mu, sweeps=k)[0],
        "SB eta=10c": lambda k: gball(g, mu, eta=10.0 / mu, sweeps=k)[0],
        "Chambolle (no momentum)": lambda k: chambolle_proj(g, mu,
                                                            iters=k)[0],
        "FGP (momentum on flux)": lambda k: gball_fgp(g, mu, k)[0],
    }
    print(f"  T1: v-step error vs iterations (weak fidelity c=1/{mu:.0f})")
    ks = (25, 50, 100, 200, 400, 800)
    print(f"  {'solver':>26s} " + " ".join(f"{k:>8d}" for k in ks) +
          f" {'ms/it':>6s}")
    t1 = {}
    for name, fn in solvers.items():
        errs = []
        t0 = time.perf_counter()
        for k in ks:
            v = fn(k)
            errs.append(np.linalg.norm(v - v_star) / np.linalg.norm(v_star))
        dt = (time.perf_counter() - t0) / sum(ks) * 1e3
        t1[name] = (ks, errs, dt)
        print(f"  {name:>26s} " + " ".join(f"{e:8.1e}" for e in errs) +
              f" {dt:6.2f}")

    # ---- T2: full pipeline race, error-to-reference vs wall time ----
    print("  T2: pipeline race (error to reference vs wall time)")
    runs = {}

    def race(name, fn, counter):
        counter["n"] = 0
        t0 = time.perf_counter()
        u, v, nout, ck = fn()
        dt = time.perf_counter() - t0
        err = np.linalg.norm(u - u_ref) / nf
        runs[name] = dict(u=u, v=v, ck=ck, time=dt, iters=counter["n"])
        print(f"  {name:>34s} outer {nout:4d} inner {counter['n']:6d} "
              f"{dt:6.2f}s err {err:.2e}")

    race("SB fused (Bregman-state memory)",
         lambda: a2bc_fused(f, lam, mu, beta=0.0, seed_sweeps=0,
                            max_outer=400, checkpoints=True), SWEEPS)
    race("flux k=1 (flux memory)",
         lambda: a2bc_flux(f, lam, mu, iters_per_step=1, seed_iters=0,
                           checkpoints=True), FGP_ITERS)
    race("flux k=2",
         lambda: a2bc_flux(f, lam, mu, iters_per_step=2, seed_iters=0,
                           checkpoints=True), FGP_ITERS)
    race("flux k=4",
         lambda: a2bc_flux(f, lam, mu, iters_per_step=4, seed_iters=0,
                           max_outer=200, checkpoints=True), FGP_ITERS)
    race("flux k=2, restart each pass",
         lambda: a2bc_flux(f, lam, mu, iters_per_step=2, seed_iters=0,
                           restart_each=True, checkpoints=True), FGP_ITERS)

    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for name, (ks_, errs, _) in t1.items():
            ax.loglog(ks_, errs, "-o", ms=3, label=name)
        ax.set_xlabel("inner iterations")
        ax.set_ylabel("relative error vs converged v-step")
        ax.set_title("T1: weak-fidelity texture projection\n"
                     "(momentum on flux vs primal-space solvers)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        for name, r in runs.items():
            xs, ys = [], []
            per = r["time"] / max(r["iters"], 1)
            for it_n, u, v in r["ck"]:
                xs.append(it_n * per)
                ys.append(np.linalg.norm(u - u_ref) / nf)
            ax.loglog(xs, ys, "-", label=name)
        ax.set_xlabel("wall time (s, est. from per-iteration cost)")
        ax.set_ylabel("||u - u*|| / ||f||")
        ax.set_title("T2: full split, error to reference vs time")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "meyer_transport.png", dpi=130)
        print(OUT / "meyer_transport.png")
    return t1, runs


def exp_budget():
    """Sub-second gate: best split achievable in <= 0.8 s at 256^2."""
    f, u_true, vc, vf, m_c, m_f = rig_scene(256)
    lam, mu = 0.05, 60.0
    u_ref, v_ref = reference_split(f, lam, mu)
    nf = np.linalg.norm(f)
    print(f"[budget] all variants capped at 0.8 s wall; err vs reference")
    print(f"  {'variant':>42s} {'passes':>6s} {'time':>6s} {'err_u':>9s} "
          f"{'PSNR u':>7s} {'PSNR v':>7s}")

    def go(name, **kw):
        t0 = time.perf_counter()
        u, v, npass = a2bc_budget(f, lam, mu, budget_s=0.8, **kw)
        dt = time.perf_counter() - t0
        print(f"  {name:>42s} {npass:6d} {dt:5.2f}s "
              f"{np.linalg.norm(u - u_ref) / nf:9.2e} "
              f"{psnr(u, u_true):7.2f} {psnr(v, vc + vf):7.2f}")
        return u, v

    go("incumbent (eta10)")
    go("eta2 (old default)", eta_v_mult=2.0)
    for om in (1.3, 1.6, 1.9):
        go(f"over-relax omega={om}", omega=om)
    for c in (4.0, 8.0, 16.0):
        go(f"lam-continuation /{c:.0f}", cont=c)
    go("continuation /8 + omega 1.6", cont=8.0, omega=1.6)
    go("seed 40 + continuation /8", cont=8.0, seed_sweeps=40)
    go("seed 40 + cont /8 + omega 1.6", cont=8.0, omega=1.6, seed_sweeps=40)
    return None


# ----------------------------------------------------------------------
# the Meyer scale ladder
# ----------------------------------------------------------------------

def texture_ladder(w, mus, tol=1e-5):
    """Split w into bands along a mu-ladder (mus descending, coarse->fine).

    ROF scale space s_k = ROF(w, 1/mu_k): larger mu = weaker fidelity =
    smoother s_k.  Bands tile exactly:
        w = P_{G_mu0}-complement ... concretely
        s_0 = ROF(w, 1/mu_0)            # coarsest survivor (-> cartoon-ward)
        band_k = s_{k+1} - s_k          # scale band between rungs k, k+1
        band_last = w - s_K             # finest content, inside ball mu_K
    Each rung solves independently: Bregman (d, b) states are eta- and
    c-scaled and must NOT be carried across rungs (measured: the carried
    state pins the next rung at the previous fixed point and the
    relative-change stop fires immediately -> empty bands).
    Returns (s0, bands[list, coarse->fine]) with w = s0 + sum(bands).
    """
    s = []
    for mu in mus:
        u, _ = rof_sb(w, 1.0 / mu, eta=10.0 / mu, tol=tol, max_sweeps=600)
        s.append(u)
    bands = [s[k + 1] - s[k] for k in range(len(s) - 1)]
    bands.append(w - s[-1])
    return s[0], bands


# ----------------------------------------------------------------------
# rig A: synthetic ground truth
# ----------------------------------------------------------------------

def rig_scene(n=256):
    """Cartoon* + coarse texture* (T=12) + fine texture* (T=3), overlapping.

    Returns f, u*, v_coarse*, v_fine*, and the two texture region masks.
    Amplitudes equal (20), so G-norms differ by the frequency ratio 4:
    G_coarse ~ a/w = 20/(2pi/12) ~ 38, G_fine ~ 20/(2pi/3) ~ 9.5.
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    # cartoon: shaded background + disk + rectangle with step edges
    u = 90.0 + 40.0 * x / n + 20.0 * y / n
    disk = ((x - 0.32 * n) ** 2 + (y - 0.34 * n) ** 2) < (0.18 * n) ** 2
    u[disk] = 170.0 - 30.0 * (y[disk] / n)
    rect = (x > 0.55 * n) & (x < 0.9 * n) & (y > 0.55 * n) & (y < 0.85 * n)
    u[rect] = 60.0
    # smooth-edged region masks (windowing keeps texture G-content local)
    def smooth_mask(cx, cy, r):
        d = np.sqrt((x - cx * n) ** 2 + (y - cy * n) ** 2) / (r * n)
        return np.clip(1.0 - d, 0.0, 1.0) ** 0.5 * (d < 1.0)
    m_c = smooth_mask(0.38, 0.62, 0.30)          # coarse region
    m_f = smooth_mask(0.62, 0.40, 0.30)          # fine region (overlaps)
    a = 20.0
    vc = a * m_c * np.cos(2 * np.pi * (x * np.cos(0.4) + y * np.sin(0.4)) / 12.0)
    vf = a * m_f * np.cos(2 * np.pi * (x * np.cos(1.2) + y * np.sin(1.2)) / 3.0)
    f = u + vc + vf
    return f, u, vc, vf, m_c, m_f


def psnr(a, ref, rng=255.0):
    return 10.0 * np.log10(rng ** 2 / np.mean((a - ref) ** 2))


def edge_leak(v, u_star):
    """corr(|grad u*|, |v|): how much texture layer hugs cartoon edges."""
    gx, gy = grad(u_star)
    e = np.sqrt(gx * gx + gy * gy).ravel()
    a = np.abs(v).ravel()
    e = e - e.mean()
    a = a - a.mean()
    return float(e @ a / (np.linalg.norm(e) * np.linalg.norm(a) + 1e-12))


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------

def exp_engine():
    """Smoke: ROF fixed point, G-ball identity, Chambolle agreement."""
    rng = np.random.default_rng(0)
    f, u_true, vc, vf, _, _ = rig_scene(128)
    # 1. ROF optimality: subgradient check via duality gap proxy --
    #    compare two solvers (Bregman vs Chambolle) on the same problem.
    mu = 30.0
    vb, _ = gball(f, mu, tol=1e-7, max_sweeps=2000)
    vch, nit = chambolle_proj(f, mu, tol=1e-7)
    rel = np.linalg.norm(vb - vch) / np.linalg.norm(vch)
    print(f"[engine] P_Gmu Bregman vs Chambolle rel diff {rel:.2e} "
          f"(chambolle iters {nit})")
    # 2. pure-oscillation capture: ||P_Gmu|| should be ~all for mu >> a/w
    n = 128
    yx = np.mgrid[0:n, 0:n][1].astype(np.float64)
    a, T = 20.0, 8.0
    g = a * np.cos(2 * np.pi * yx / T)
    gn = a / (2 * np.pi / T)
    for m in (0.25 * gn, gn, 4 * gn):
        v, _ = gball(g, m, tol=1e-6, max_sweeps=1500)
        print(f"[engine] cosine T={T} Gnorm~{gn:.1f} mu={m:6.1f} "
              f"captured {np.linalg.norm(v)/np.linalg.norm(g):.3f}")
    # 3. warm state = same answer
    v1, st = gball(f, mu, sweeps=40)
    v2, _ = gball(f, mu, state=st, sweeps=200)
    v3, _ = gball(f, mu, tol=1e-7, max_sweeps=2000)
    print(f"[engine] warm-continued vs cold-converged rel "
          f"{np.linalg.norm(v2-v3)/np.linalg.norm(v3):.2e}")


def _variants(f, lam, mu):
    """Run all iteration variants with checkpoint capture."""
    runs = {}

    def go(name, fn):
        SWEEPS["n"] = 0
        t0 = time.perf_counter()
        u, v, nout, ck = fn()
        dt = time.perf_counter() - t0
        runs[name] = dict(u=u, v=v, outer=nout, sweeps=SWEEPS["n"],
                          time=dt, ck=ck)
        print(f"  {name:28s} outer {nout:4d}  sweeps {SWEEPS['n']:5d}  "
              f"{dt:6.2f}s")

    go("cold nested (Gilles Alg.3)",
       lambda: a2bc_cold(f, lam, mu, checkpoints=True))
    go("fused warm 1-sweep (ours)",
       lambda: a2bc_fused(f, lam, mu, checkpoints=True))
    go("fused + beta 0.88 (harmful)",
       lambda: a2bc_fused(f, lam, mu, beta=0.88, checkpoints=True))
    return runs


def exp_iters(make_fig=True):
    """The minimal-iterations study on rig A."""
    f, u_true, vc, vf, m_c, m_f = rig_scene(256)
    v_true = vc + vf
    lam, mu = 0.05, 60.0
    print(f"[iters] rig 256^2, lam={lam} (1/lam={1/lam:.0f}), mu={mu}")
    runs = _variants(f, lam, mu)

    # fixed-point residual at checkpoints (offline probes)
    curves = {}
    for name, r in runs.items():
        pts = []
        ckpts = r["ck"]
        idx = np.unique(np.geomspace(1, len(ckpts), min(12, len(ckpts))
                                     ).astype(int) - 1)
        for i in idx:
            sw, u, v = ckpts[i]
            pts.append((sw, fp_residual(f, u, v, lam, mu)))
        curves[name] = pts

    ref = runs["cold nested (Gilles Alg.3)"]
    print(f"{'variant':>30s} {'sweeps':>7s} {'fp-resid':>9s} "
          f"{'PSNR u':>7s} {'PSNR v':>7s} {'edge-leak':>9s} {'|d(u)| vs cold':>14s}")
    for name, r in runs.items():
        fp = fp_residual(f, r["u"], r["v"], lam, mu)
        du = np.linalg.norm(r["u"] - ref["u"]) / np.linalg.norm(f)
        print(f"{name:>30s} {r['sweeps']:7d} {fp:9.2e} "
              f"{psnr(r['u'], u_true):7.2f} {psnr(r['v'], v_true):7.2f} "
              f"{edge_leak(r['v'], u_true):9.3f} {du:14.2e}")

    # context: Chambolle-projector inner (the paper's own baseline)
    SWEEPS["n"] = 0
    t0 = time.perf_counter()
    _, nit_v = chambolle_proj(f - ref["u"], mu, tol=1e-4)
    dt = time.perf_counter() - t0
    print(f"[iters] context: ONE Chambolle-projector v-step at tol 1e-4 = "
          f"{nit_v} p-iterations ({dt:.2f}s); Bregman does it in "
          f"~{ref['sweeps']//max(ref['outer'],1)//2} sweeps")

    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for name, pts in curves.items():
            xs, ys = zip(*pts)
            ax.loglog(xs, ys, "-o", ms=3, label=name)
        ax.set_xlabel("total inner Bregman sweeps")
        ax.set_ylabel("fixed-point residual")
        ax.set_title("A2BC Meyer split: cost to the fixed point "
                     f"(rig A, lam={lam}, mu={mu})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "meyer_iterations.png", dpi=130)
        print(OUT / "meyer_iterations.png")
    return runs


def exp_transition(make_fig=True):
    """The Meyer-rung coverage function: capture fraction vs mu*w/a."""
    n = 256
    x = np.mgrid[0:n, 0:n][1].astype(np.float64)
    a = 20.0
    print("[transition] capture fraction of P_Gmu on pure cosine")
    rows = {}
    for T in (4.0, 8.0, 16.0):
        w = 2 * np.pi / T
        gn = a / w
        g = a * np.cos(w * x)
        ratios = np.geomspace(1 / 16, 16, 17)
        cap = []
        for r in ratios:
            v, _ = gball(g, r * gn, tol=1e-6, max_sweeps=1200)
            cap.append(np.linalg.norm(v) / np.linalg.norm(g))
        rows[T] = (ratios, np.array(cap))
        c = rows[T][1]
        # 10-90% transition width in log-ratio units
        lo = np.interp(0.1, c, np.log(ratios))
        hi = np.interp(0.9, c, np.log(ratios))
        # capture at the worst-case rung placements of a ratio-4 ladder
        c_half = np.interp(np.log(0.5), np.log(ratios), c)
        c_two = np.interp(np.log(2.0), np.log(ratios), c)
        print(f"  T={T:5.1f}: 10-90% width = e^{hi-lo:.2f} = "
              f"{np.exp(hi-lo):.2f}x in mu; capture at mu=G/2: "
              f"{c_half:.2f}, at mu=2G: {c_two:.2f}")
    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for T, (r, c) in rows.items():
            ax.semilogx(r, c, "-o", ms=3, label=f"period {T:.0f} px")
        for rr in (0.5, 2.0):        # a ratio-4 ladder's worst placement
            ax.axvline(rr, color="k", ls=":", alpha=0.5)
        ax.set_xlabel("mu / ||g||_G")
        ax.set_ylabel("captured fraction ||P_Gmu g|| / ||g||")
        ax.set_title("Meyer rung coverage function "
                     "(dotted: worst-case placement, ratio-4 ladder)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "meyer_transition.png", dpi=130)
        print(OUT / "meyer_transition.png")
    return rows


def _band_route(bands, mus, mu_split):
    """Sum ladder bands into (coarse, fine) by rung scale.

    bands are coarse->fine between rungs; band k lives between mu_k and
    mu_{k+1}; assign to coarse if its geometric-mean rung > mu_split.
    Final band (inside ball mu_last) counts as fine.
    """
    coarse = np.zeros_like(bands[0])
    fine = np.zeros_like(bands[0])
    for k, b in enumerate(bands):
        hi = mus[k]
        lo = mus[k + 1] if k + 1 < len(mus) else mus[-1] / 4.0
        if np.sqrt(hi * lo) > mu_split:
            coarse += b
        else:
            fine += b
    return coarse, fine


def exp_ladder(make_fig=True):
    """Two-texture separation: single-mu split vs mu-ladders r=2,4,8."""
    f, u_true, vc, vf, m_c, m_f = rig_scene(256)
    lam, mu_max = 0.05, 60.0
    G_c, G_f = 20.0 / (2 * np.pi / 12), 20.0 / (2 * np.pi / 3)
    mu_split = np.sqrt(G_c * G_f)          # geometric midpoint ~19
    print(f"[ladder] texture G-norms: coarse ~{G_c:.1f}, fine ~{G_f:.1f}, "
          f"split at {mu_split:.1f}")

    # converge the full split once (plateau budget), then band the texture
    SWEEPS["n"] = 0
    u, v, _ = a2bc_budget(f, lam, mu_max, budget_s=2.0)
    base_sw = SWEEPS["n"]
    print(f"  base split: sweeps {base_sw}, "
          f"PSNR u {psnr(u, u_true):.2f}, PSNR v {psnr(v, vc + vf):.2f}")
    print(f"  single-mu split CANNOT separate the two textures: "
          f"one v layer holds both (by construction).")
    print(f"  NOTE 2-way routing telescopes: interior rungs cancel; only "
          f"the rung STRADDLING the pair matters.  Rung ratio r sets the "
          f"worst-case straddle margin sqrt(r).")

    span_hi = 60.0
    ladders = {
        "r=2": list(span_hi * 2.0 ** -np.arange(0, 6)),      # 60..1.9
        "r=4": list(span_hi * 4.0 ** -np.arange(0, 3)),      # 60,15,3.75
        "r=8": list(span_hi * 8.0 ** -np.arange(0, 2)),      # 60,7.5
    }
    results = {}
    print(f"{'ladder':>6s} {'rungs':>28s} {'straddle rung':>13s} "
          f"{'margins':>11s} {'PSNR coarse':>11s} {'PSNR fine':>10s} "
          f"{'x-leak c':>9s} {'x-leak f':>9s}")
    for name, mus in ladders.items():
        SWEEPS["n"] = 0
        _, bands = texture_ladder(v, mus)
        sw = SWEEPS["n"]
        vc_hat, vf_hat = _band_route(bands, mus, mu_split)
        # the rung nearest the pair's geometric midpoint, and its margins
        # to each texture's G-norm (both should exceed the transition
        # half-width for a clean split)
        straddle = mus[int(np.argmin([abs(np.log(m / mu_split))
                                      for m in mus]))]
        marg_c, marg_f = G_c / straddle, straddle / G_f
        # cross-leakage: energy of the WRONG texture's region in each layer
        zc = (m_c > 0.5) & (m_f < 0.1)
        zf = (m_f > 0.5) & (m_c < 0.1)
        xl_c = np.linalg.norm(vc_hat[zf]) / (np.linalg.norm(vf[zf]) + 1e-9)
        xl_f = np.linalg.norm(vf_hat[zc]) / (np.linalg.norm(vc[zc]) + 1e-9)
        results[name] = dict(mus=mus, bands=bands, vc=vc_hat, vf=vf_hat,
                             sweeps=sw)
        print(f"{name:>6s} {str([f'{m:.1f}' for m in mus]):>28s} "
              f"{straddle:13.1f} {marg_c:5.2f}/{marg_f:4.2f} "
              f"{psnr(vc_hat, vc):11.2f} {psnr(vf_hat, vf):10.2f} "
              f"{xl_c:9.3f} {xl_f:9.3f}")

    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        r4 = results["r=4"]
        rows = [("f (input)", f, None), ("u recovered", u, u_true),
                ("v total", v, vc + vf),
                ("coarse* ", vc, None), ("r=4 coarse layer", r4["vc"], vc),
                ("fine*", vf, None), ("r=4 fine layer", r4["vf"], vf),
                ("u*", u_true, None)]
        fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
        for ax, (name, im, ref) in zip(axes.ravel(), rows):
            lo, hi = (0, 255) if im.mean() > 40 else (-40, 40)
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            t = name if ref is None else f"{name}\n{psnr(im, ref):.2f} dB"
            ax.set_title(t, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Meyer scale ladder on rig A: one latent, per-scale "
                     "texture layers (single-mu Gilles split gives only "
                     "'v total')", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "meyer_ladder_rig.png", dpi=130)
        print(OUT / "meyer_ladder_rig.png")
    return results


# ----------------------------------------------------------------------
# Barbara
# ----------------------------------------------------------------------

def load_barbara():
    import matplotlib.image as mpimg
    img = mpimg.imread(str(SCRATCH / "set12_09.png")).astype(np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img * 255.0 if img.max() <= 1.5 else img


def downsample2(a):
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2]
                   + a[1::2, 1::2])


def upsample2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def a2bc_budget_pyramid(f, lam, mu, budget_s=0.9, boot_frac=0.3,
                        eta_v_mult=10.0):
    """Coarse-to-fine: spend boot_frac of the budget at half resolution
    (mu/2, 2*lam: G-norms halve under 2x downsampling), upsample (u, v),
    finish at full resolution.  Memoryless continuation, no history."""
    t0 = time.perf_counter()
    fh = downsample2(f)
    uh, vh, np_h = a2bc_budget(fh, 2.0 * lam, 0.5 * mu,
                               budget_s=boot_frac * budget_s,
                               eta_v_mult=eta_v_mult)
    u = upsample2(uh)
    v = upsample2(vh)
    st_u, st_v = None, None
    npass = 0
    while (time.perf_counter() - t0) < budget_s:
        u, st_u = rof_sb(f - v, lam, sweeps=1, state=st_u)
        g = f - u
        w, st_v = rof_sb(g, 1.0 / mu, eta=eta_v_mult / mu, sweeps=1,
                         state=st_v)
        v = g - w
        npass += 1
    return u, v, (np_h, npass)


def exp_barbara(make_fig=True):
    """Head-to-head on the paper's own image, sub-second gate at 512^2."""
    f = load_barbara()
    lam, mu = 0.05, 40.0
    nf = np.linalg.norm(f)
    print(f"[barbara] 512^2, lam={lam}, mu={mu}")
    # the paper's own answer: Algorithm 3, nested, run tight (the target)
    SWEEPS["n"] = 0
    t0 = time.perf_counter()
    u_ref, v_ref, nout, _ = a2bc_cold(f, lam, mu, outer_tol=1e-5,
                                      inner_tol=1e-5, max_outer=120)
    t_ref = time.perf_counter() - t0
    sw_ref = SWEEPS["n"]
    print(f"  Gilles Alg.3 nested tight: outer {nout}, sweeps {sw_ref}, "
          f"{t_ref:.1f}s")
    runs = {}
    for name, fn in [
        ("fused 0.9s", lambda: a2bc_budget(f, lam, mu, budget_s=0.9)),
        ("fused 0.9s + coarse boot",
         lambda: a2bc_budget_pyramid(f, lam, mu, budget_s=0.9)),
        ("fused 2.5s", lambda: a2bc_budget(f, lam, mu, budget_s=2.5)),
    ]:
        t0 = time.perf_counter()
        u, v, npass = fn()
        dt = time.perf_counter() - t0
        err = np.linalg.norm(u - u_ref) / nf
        runs[name] = dict(u=u, v=v)
        print(f"  {name:>26s}: passes {npass}, {dt:.2f}s, "
              f"||u-u_Gilles||/||f|| {err:.2e}  "
              f"(speedup {t_ref / dt:.0f}x)")
    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        fu = runs["fused 0.9s + coarse boot"]
        rows = [("f", f, 0, 255),
                ("cartoon u (Gilles, nested, "
                 f"{t_ref:.0f}s)", u_ref, 0, 255),
                ("texture v (Gilles)", v_ref, -40, 40),
                ("u (fused, 0.9 s)", fu["u"], 0, 255),
                ("v (fused, 0.9 s)", fu["v"], -40, 40),
                ("|u_fused - u_Gilles| x20",
                 20 * np.abs(fu["u"] - u_ref), 0, 255)]
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6))
        for ax, (name, im, lo, hi) in zip(axes.ravel(), rows):
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / "meyer_barbara.png", dpi=130)
        print(OUT / "meyer_barbara.png")
    return runs


def exp_barbara_ladder(make_fig=True):
    """The capability Gilles lacks: per-scale texture maps of Barbara."""
    f = load_barbara()
    lam, mu_max = 0.05, 40.0
    SWEEPS["n"] = 0
    u, v, _ = a2bc_budget(f, lam, mu_max, budget_s=4.0)
    mus = [40.0, 10.0, 2.5]      # ratio-4 rungs: GRADED bands (soft rungs)
    s0, bands = texture_ladder(v, mus)
    print(f"[barbara-ladder] split sweeps {SWEEPS['n']}, rungs {mus}, "
          f"band energies "
          f"{[f'{np.linalg.norm(b):.0f}' for b in bands]}, "
          f"residual-to-cartoon {np.linalg.norm(s0):.0f}")
    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        # u absorbs the coarsest survivor of v (it is cartoon-scale)
        rows = [("f", f, 0, 255), ("cartoon u + s0", u + s0, 0, 255),
                ("band mu 40-10 (coarse texture)", bands[0], -30, 30),
                ("band mu 10-2.5 (mid)", bands[1], -30, 30),
                ("band mu <2.5 (fine)", bands[2], -30, 30),
                ("v total (what single-mu gives)", v, -40, 40)]
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6))
        for ax, (name, im, lo, hi) in zip(axes.ravel(), rows):
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / "meyer_barbara_ladder.png", dpi=130)
        print(OUT / "meyer_barbara_ladder.png")
        # zooms: scarf (coarse stripes) vs tablecloth (fine check)
        zooms = {"scarf": (np.s_[30:180, 280:430]),
                 "tablecloth": (np.s_[300:450, 0:150]),
                 "face": (np.s_[40:190, 320:470])}
        fig, axes = plt.subplots(3, 5, figsize=(15, 9.6))
        cols = [("f", f, 0, 255), ("u + s0", u + s0, 0, 255),
                ("coarse band", bands[0], -30, 30),
                ("mid band", bands[1], -30, 30),
                ("fine band", bands[2], -30, 30)]
        for i, (zn, sl) in enumerate(zooms.items()):
            for j, (cn, im, lo, hi) in enumerate(cols):
                ax = axes[i, j]
                ax.imshow(im[sl], cmap="gray", vmin=lo, vmax=hi)
                if i == 0:
                    ax.set_title(cn, fontsize=9)
                if j == 0:
                    ax.set_ylabel(zn, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / "meyer_barbara_zooms.png", dpi=130)
        print(OUT / "meyer_barbara_zooms.png")
    return u, v, s0, bands


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    t0 = time.perf_counter()
    if which in ("engine", "all"):
        exp_engine()
    if which in ("transition", "all"):
        exp_transition()
    if which in ("iters", "all"):
        exp_iters()
    if which in ("transport", "all"):
        exp_transport()
    if which in ("budget", "all"):
        exp_budget()
    if which in ("ladder", "all"):
        exp_ladder()
    if which in ("barbara", "all"):
        exp_barbara()
    if which in ("barbara_ladder", "all"):
        exp_barbara_ladder()
    print(f"[total {time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main(sys.argv)
