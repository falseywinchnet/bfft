#!/usr/bin/env python3
"""
THLE discovery: the truncated half leading edge transform and its adaptive form.

Target objects (2026-07-09 design conversation):

  FIXED THLE-alpha:   y[k] = sum_{n=0}^{N-1-k} x[n] exp(-2*pi*i*(k+alpha)*n/N)
      the DFT square sliced down the anti-diagonal: bin 0 full support,
      bin N-1 one sample.  alpha = 0 (DFT edge) or 1/2 (ODFT edge).

  ADAPTIVE (the real goal): per bin k a marching leading-edge family
      C(k,tau) = sum_{n<tau} x[n] e^{-i w_k n}  (= B + iA, the cosine/sine
      correlation pair; observable phase arctan(A/B)).  Output C(k, tau_k)
      at the slice tau_k where the bin maximally correlates -- for all bins,
      at FFT-like cost, NOT as N sliding correlators (no DDC bank).

Established here, numerically:

  A.  THLE == valid overlap of a finite chirp correlation (a Bluestein-like
      walk whose destination is NOT the Fourier identity).  The diagonal
      truncation is never applied as a mask: it is the finite support of the
      chirp packet itself.  Exact at 1e-13, O(N log N).  Rank: |det| = 1
      exactly (unimodular anti-diagonal pivots), so THLE is algebraically a
      bijection -- but its condition number grows exponentially with N
      (measured), so the truncation GENUINELY erases late-sample information
      in floating point.  Inversion is practical only for small N (back-
      substitution) or from partial/regularized data.  This is a property
      of the object, not of an algorithm.
  B1. The ordinary radix-2 walk does NOT close on THLE: the child support
      law floor((N-1-k)/2) depends on the parent bin, so children cannot be
      shared.  The chirp correlation IS the walk for this target.
  B2. The quadratic packet JET closes: (value, d/dparam, d2/dparam2)
      transports exactly through every linear cell of the chirp walk
      (validated against exact analytic derivatives, machine precision).
      This is the mechanism that moves a paused walk along the tau manifold.
  C0. Contiguous-block spectra tile the leading-edge manifold EXACTLY on
      the dyadic lattice:  C(g, m*L) = sum_{b<m} blockDFT_L[b][g*L/N]
      for on-grid bins g (integer cycles per block kill every cross term;
      no Dirichlet leak ON the lattice).
  CL. The block tower closes on the (DFT, ODFT) channel pair:
        parent[2f]   = DFT_h(u)[f] + DFT_h(v)[f]
        parent[2f+1] = ODFT_h(u)[f] - ODFT_h(v)[f]
      i.e. the half-bin BODFT primitive is exactly the missing channel that
      makes contiguous-block towers walkable.
  C1. The adaptive transform: build the dyadic manifold (log N walk passes),
      per-bin coarse-to-fine level selection with a coherence stop (proxy
      rows die when the level's frequency grid outruns the component's
      bandwidth -- detected, not ignored), cluster-shared exact boundary
      refinement by endpoint recurrence, egress through a handful of shared
      truncated transforms + per-bin endpoint polish.  Validated against
      the brute-force manifold; cost accounted against the DDC bank.

Numpy only (no scipy).  np.fft stands in for the house complex FFT.
"""

import time
import numpy as np

rng = np.random.default_rng(7)


# =====================================================================
# Part A: fixed THLE-alpha, chirp valid-correlation identity + inverse
# =====================================================================

def thle_direct(x, alpha=0.0):
    """O(N^2) reference: y[k] = sum_{n=0}^{N-1-k} x[n] W^{(k+alpha)n}."""
    N = len(x)
    n = np.arange(N)
    y = np.empty(N, complex)
    for k in range(N):
        m = N - k
        y[k] = np.sum(x[:m] * np.exp(-2j * np.pi * (k + alpha) * n[:m] / N))
    return y


def _lin_conv(u, v):
    L = 1
    while L < len(u) + len(v) - 1:
        L <<= 1
    return np.fft.ifft(np.fft.fft(u, L) * np.fft.fft(v, L))[: len(u) + len(v) - 1]


def thle_chirp(x, alpha=0.0):
    """
    O(N log N) chirp valid-correlation:
      gamma = exp(-i*pi/N);  W^{(k+a)n} = g^{(n+k+a)^2} g^{-n^2} g^{-(k+a)^2}
      a[n] = x[n] g^{-n^2};  b[m] = g^{(m+a)^2}, m = 0..N-1  (FINITE packet)
      y[k] = g^{-(k+a)^2} * (reverse(a) linconv b)[N-1+k]
    The truncation n <= N-1-k is nowhere applied: it is the valid overlap
    of the finite chirp packet b.
    """
    N = len(x)
    n = np.arange(N, dtype=float)
    ph = np.pi / N
    a = x * np.exp(1j * ph * n * n)
    b = np.exp(-1j * ph * (n + alpha) ** 2)
    h = _lin_conv(a[::-1], b)
    k = np.arange(N, dtype=float)
    post = np.exp(1j * ph * (k + alpha) ** 2)
    return post * h[N - 1 : 2 * N - 1]


def thle_inverse_tri(y, alpha=0.0):
    """
    Exact inverse by anti-triangular back-substitution.  Row k has support
    n <= N-1-k and its anti-diagonal pivot W^{(k+alpha)(N-1-k)} is
    unimodular, so the system solves sequentially:
      k = N-1 determines x[0], k = N-2 determines x[1], ...   O(N^2).
    """
    N = len(y)
    x = np.zeros(N, complex)
    n = np.arange(N)
    for k in range(N - 1, -1, -1):
        m = N - k                      # support length of row k
        row = np.exp(-2j * np.pi * (k + alpha) * n[:m] / N)
        x[m - 1] = (y[k] - np.dot(row[: m - 1], x[: m - 1])) / row[m - 1]
    return x


def _series_reciprocal(e, N):
    inv = np.array([1.0 / e[0]], complex)
    m = 1
    while m < N:
        m = min(2 * m, N)
        t = _lin_conv(e[:m], inv)[:m]
        corr = -t
        corr[0] += 2.0
        inv = _lin_conv(inv, corr)[:m]
    return inv[:N]


def thle_inverse_series(y, alpha=0.0):
    """O(N log N) inverse (exact algebra; float-unstable, see part A)."""
    N = len(y)
    n = np.arange(N, dtype=float)
    ph = np.pi / N
    c = y * np.exp(-1j * ph * (n + alpha) ** 2)
    d = c[::-1]
    b = np.exp(-1j * ph * (n + alpha) ** 2)
    inv = _series_reciprocal(b[::-1], N)
    a = _lin_conv(d, inv)[:N]
    return a * np.exp(-1j * ph * n * n), np.max(np.abs(inv))


def part_A():
    print("=" * 72)
    print("A. fixed THLE-alpha: chirp valid-correlation identity + inverse")
    print("=" * 72)
    ok = True
    # forward identity: exact at every size
    for N in (256, 1024):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        for alpha in (0.0, 0.5):
            yd = thle_direct(x, alpha)
            yc = thle_chirp(x, alpha)
            fw = np.max(np.abs(yd - yc)) / np.max(np.abs(yd))
            ok &= fw < 1e-10
            print(f"  N={N:5d} alpha={alpha:3.1f}:  chirp-vs-direct {fw:.2e}"
                  f"   {'PASS' if fw < 1e-10 else 'FAIL'}")
    # invertibility: |det| = 1 exactly, yet cond grows exponentially
    print("  invertibility: |det(THLE)| = 1 (unimodular pivots), but")
    for N in (64, 128, 256):
        n = np.arange(N)
        T = np.exp(-2j * np.pi * np.outer(n.astype(float), n) / N)
        T *= (n[None, :] + n[:, None] < N)
        s = np.linalg.svd(T, compute_uv=False)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        xt = thle_inverse_tri(thle_chirp(x, 0.0), 0.0)
        bt = np.max(np.abs(xt - x)) / np.max(np.abs(x))
        print(f"    N={N:4d}: cond = {s[0]/s[-1]:9.2e}   "
              f"back-substitution roundtrip err {bt:.2e}")
        if N == 64:
            ok &= bt < 1e-8
    print("  => THLE is an exact chirp valid-correlation; algebraically a")
    print("     bijection (|det|=1) but exponentially conditioned: the")
    print("     leading-edge slice GENUINELY erases late-sample information")
    print("     in floats.  A property of the object, not the algorithm.")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# =====================================================================
# Part B1: the ordinary radix walk does not close on THLE
# =====================================================================

def part_B_nonclosure():
    print("=" * 72)
    print("B1. the ordinary radix-2 walk does NOT close on THLE")
    print("=" * 72)
    N = 256
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    M = N // 2
    E = thle_direct(x[0::2], 0.0)
    O = thle_direct(x[1::2], 0.0)
    k = np.arange(N)
    y_combine = E[k % M] + np.exp(-2j * np.pi * k / N) * O[k % M]
    y_true = thle_direct(x, 0.0)
    err = np.abs(y_combine - y_true) / np.max(np.abs(y_true))
    exact = np.flatnonzero(err < 1e-12)
    print(f"  standard combine of shared THLE children: max rel err "
          f"{err.max():.3f}, median {np.median(err):.3f}")
    print(f"  bins reproduced exactly: {exact.tolist()[:8]}  ({len(exact)}/{N})")
    print("  reason: the child support bound floor((N-1-k)/2) depends on the")
    print("  PARENT bin k, not k mod N/2 -- outputs k and k+N/2 need different")
    print("  children; no sharing, no radix walk.  The chirp correlation IS")
    print("  the walk for this target.")
    return len(exact) <= 2


# =====================================================================
# Part B2: the quadratic packet jet closes through the whole walk
# =====================================================================

def thle_chirp_jet(x, alpha):
    """Transport (value, d/dalpha, d2/dalpha2) through the chirp walk.
    alpha enters only through packet phases; the correlation cell is linear,
    so the jet closes channel-wise."""
    N = len(x)
    n = np.arange(N, dtype=float)
    ph = np.pi / N
    a = x * np.exp(1j * ph * n * n)
    u = n + alpha
    b0 = np.exp(-1j * ph * u * u)
    b1 = (-2j * ph * u) * b0
    b2 = (-2j * ph) * b0 + (-2j * ph * u) * b1
    r = a[::-1]
    h0 = _lin_conv(r, b0)[N - 1 : 2 * N - 1]
    h1 = _lin_conv(r, b1)[N - 1 : 2 * N - 1]
    h2 = _lin_conv(r, b2)[N - 1 : 2 * N - 1]
    kk = np.arange(N, dtype=float) + alpha
    p0 = np.exp(1j * ph * kk * kk)
    p1 = (2j * ph * kk) * p0
    p2 = (2j * ph) * p0 + (2j * ph * kk) * p1
    return (p0 * h0,
            p1 * h0 + p0 * h1,
            p2 * h0 + 2 * p1 * h1 + p0 * h2)


def part_B_jet():
    print("=" * 72)
    print("B2. the quadratic packet JET closes through the whole walk")
    print("=" * 72)
    N = 512
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    alpha = 0.3
    y0, y1, y2 = thle_chirp_jet(x, alpha)
    # exact analytic derivatives of the direct sum (not finite differences)
    n = np.arange(N)
    g0 = np.empty(N, complex); g1 = np.empty(N, complex); g2 = np.empty(N, complex)
    for k in range(N):
        m = N - k
        w = np.exp(-2j * np.pi * (k + alpha) * n[:m] / N)
        d = -2j * np.pi * n[:m] / N
        g0[k] = np.sum(x[:m] * w)
        g1[k] = np.sum(x[:m] * d * w)
        g2[k] = np.sum(x[:m] * d * d * w)
    e0 = np.max(np.abs(y0 - g0)) / np.max(np.abs(g0))
    e1 = np.max(np.abs(y1 - g1)) / np.max(np.abs(g1))
    e2 = np.max(np.abs(y2 - g2)) / np.max(np.abs(g2))
    ok = e0 < 1e-10 and e1 < 1e-10 and e2 < 1e-10
    print(f"  value {e0:.2e}   d/dparam {e1:.2e}   d2/dparam2 {e2:.2e}"
          f"   (vs exact analytic)   {'PASS' if ok else 'FAIL'}")
    print("  => a packet carrying (C, C_tau, C_tautau) transports exactly")
    print("     through the linear walk cells: the paused walk can Newton-step")
    print("     along the manifold without re-entering the input.")
    return ok


# =====================================================================
# Part C: the adaptive transform -- paused-walk manifold + selection
# =====================================================================

def leading_edge_bruteforce(x, alpha=0.0):
    """Ground truth: C[k, tau-1] = C(k,tau), tau = 1..N; score |C|^2/tau."""
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)
    U = x[None, :] * np.exp(-2j * np.pi * np.outer(k + alpha, n) / N)
    C = np.cumsum(U, axis=1)
    tau = np.arange(1, N + 1, dtype=float)
    S = np.abs(C) ** 2 / tau[None, :]
    return C, S, np.argmax(S, axis=1) + 1


def build_block_pyramid(x, t_min=3):
    """
    pyr[t]: (N/L, L) block DFTs, L = 2^t, for t = t_min..log2(N).
    Built as log N walk passes (N log^2 N cells total).  A production
    kernel halves this via the (DFT, ODFT) closure -- see part CL.
    """
    N = len(x)
    T = int(np.log2(N))
    return {t: np.fft.fft(x.reshape(N >> t, 1 << t), axis=1)
            for t in range(t_min, T + 1)}


def part_C0_and_CL(N=1024):
    print("=" * 72)
    print("C0. block spectra tile the leading-edge manifold exactly on the")
    print("    dyadic lattice;  CL. the tower closes on the (DFT,ODFT) pair")
    print("=" * 72)
    x, _ = make_signal(N, "events")
    pyr = build_block_pyramid(x)
    C_bf, _, _ = leading_edge_bruteforce(x)
    scale = np.max(np.abs(C_bf))
    worst = 0.0
    for t in (3, 5, 7, 9):
        L = 1 << t
        cum = np.cumsum(pyr[t], axis=0)
        for f in (1, L // 4, L // 2 + 1):
            f %= L
            g = f * (N // L)
            for m in (0, (N >> t) // 2, (N >> t) - 1):
                worst = max(worst,
                            abs(cum[m, f] - C_bf[g, (m + 1) * L - 1]) / scale)
    okc = worst < 1e-12
    print(f"  C0: C(g, mL) == running block-DFT sum, worst rel err {worst:.2e}"
          f"   {'PASS' if okc else 'FAIL'}")
    print("      (integer cycles per block kill every cross term: no")
    print("       Dirichlet leak ON the lattice; leak exists only off-grid)")
    # CL: block combine closes on (DFT, ODFT)
    h = 64
    u = rng.standard_normal(h) + 1j * rng.standard_normal(h)
    v = rng.standard_normal(h) + 1j * rng.standard_normal(h)
    P = np.fft.fft(np.concatenate([u, v]))
    Du, Dv = np.fft.fft(u), np.fft.fft(v)
    tw = np.exp(-1j * np.pi * np.arange(h) / h)          # half-bin pre-twist
    Ou, Ov = np.fft.fft(u * tw), np.fft.fft(v * tw)      # ODFT children
    even_err = np.max(np.abs(P[0::2] - (Du + Dv)))
    odd_err = np.max(np.abs(P[1::2] - (Ou - Ov)))
    okl = even_err < 1e-10 and odd_err < 1e-10
    print(f"  CL: parent[2f] = Du+Dv ({even_err:.2e}),  "
          f"parent[2f+1] = Ou-Ov ({odd_err:.2e})   {'PASS' if okl else 'FAIL'}")
    print("      => contiguous-block towers are NOT walkable by the standard")
    print("         butterfly alone; the half-bin BODFT channel is exactly")
    print("         the missing sibling that closes them.")
    return okc and okl


def make_signal(N, kind):
    n = np.arange(N)
    if kind == "events":
        x = np.zeros(N)
        x += 0.8 * np.cos(2 * np.pi * 100 * n / N + 0.7)              # full tone
        b1 = (n >= 200) & (n < 500)
        x[b1] += 1.0 * np.cos(2 * np.pi * 300.4 * n[b1] / N + 1.1)    # off-grid burst
        b2 = (n >= 600) & (n < 656)
        x[b2] += 1.2 * np.cos(2 * np.pi * 421 * n[b2] / N + 2.0)      # short burst
        x += 1e-3 * rng.standard_normal(N)
        return x, {100: None, 300: 500, 421: 656}
    if kind == "chirp":
        f0, f1 = 50, 400
        phase = 2 * np.pi * (f0 * n + (f1 - f0) * n ** 2 / (2 * N)) / N
        return np.cos(phase) + 1e-3 * rng.standard_normal(N), None
    raise ValueError(kind)


def adaptive_transform(x, t_min=4, rel=0.5, act=1.5):
    """
    The adaptive leading-edge transform (paused-walk form).

    1. MANIFOLD: block pyramid = exact C(g, mL) lattice (part C0).
    2. LANDSCAPES (row-shared): per level, cumulative block sums -> score
       |C|^2/tau per (row, boundary); per row keep best boundary + score.
       Cost O(N) per level.
    3. PER-BIN LEVEL CHOICE with coherence stop: each bin reads its proxy
       row at every level; a level is trusted only if its proxy score holds
       rel * (bin's best over levels) -- when the level's frequency grid
       outruns the component bandwidth the proxy dies and the descent stops
       (this replaces blind windowed descent).  Choose the finest trusted
       level: boundary known to +-L.
    4. ACTIVITY GATE: bins whose best score is < act * mean|x|^2 have no
       coherent leading edge; they default to tau = N (plain FFT readout).
    5. PER-BIN WINDOWED REFINEMENT (anchor-shared): each active bin gets an
       exact score landscape on [tau_c - w, tau_c + w], w = max(2L, N/8),
       by endpoint recurrence from a SHARED anchor: one truncated transform
       per distinct window start gives C(:, lo) for every member bin, then
       each bin walks its own window (O(w) endpoint ops).  The boundary is
       per-bin exact inside the window; only the window placement is shared.
    6. EGRESS: the refinement holds the exact slice value per active bin;
       inactive bins read the one plain full transform; real input mirrors
       its conjugate half.
    Returns tau_k, C_k, stats.
    """
    N = len(x)
    T = int(np.log2(N))
    k = np.arange(N)
    ops = 0                                   # exact endpoint ops (refine+polish)

    pyr = build_block_pyramid(x, t_min)
    levels = list(range(t_min, T + 1))

    # -- 2. row-shared landscapes --
    best_m = {}
    best_s = {}
    for t in levels:
        L, B = 1 << t, N >> t
        cum = np.cumsum(pyr[t], axis=0)
        tau = (np.arange(B, dtype=float) + 1) * L
        S = np.abs(cum) ** 2 / tau[:, None]
        best_m[t] = np.argmax(S, axis=0)                  # per row f
        best_s[t] = S[best_m[t], np.arange(L)]

    # -- 3. per-bin level choice --
    nl = len(levels)
    sc = np.zeros((nl, N))
    bd = np.zeros((nl, N), int)
    for i, t in enumerate(levels):
        L = 1 << t
        f = np.rint(k * L / N).astype(int) % L
        sc[i] = best_s[t][f]
        bd[i] = (best_m[t][f] + 1) * L
    smax = sc.max(axis=0)
    trusted = sc >= rel * smax[None, :]
    # finest trusted level = smallest i with trusted
    lev_idx = np.argmax(trusted, axis=0)                  # first True top-down
    tau_c = bd[lev_idx, k]
    win = np.array([1 << levels[i] for i in lev_idx])

    # -- 4. activity gate (real input: walk the independent half only) --
    active = smax > act * np.mean(np.abs(x) ** 2)
    if np.isrealobj(x):
        active &= (k >= 1) & (k <= N // 2)
    tau_c[~active] = N
    win[~active] = 0

    # -- 5. per-bin windowed refinement from shared anchors --
    tau_ref = tau_c.astype(int).copy()
    n_idx = np.arange(N)
    wide = np.maximum(2 * win, N // 8)
    lo_k = np.maximum(tau_c - wide, 1)
    hi_k = np.minimum(tau_c + wide, N)
    anchors = {}
    for kk in np.flatnonzero(active):
        anchors.setdefault(int(lo_k[kk]), []).append(kk)
    C_ref = np.zeros(N, complex)
    n_fft = 0
    for lo0, members in anchors.items():
        buf = np.zeros(N, complex)
        buf[:lo0] = x[:lo0]
        Ca = np.fft.fft(buf)                              # exact C(:, lo0)
        n_fft += 1
        for kk in members:
            w = 2 * np.pi * kk / N
            lo, hi = lo0, int(hi_k[kk])
            c_lo = Ca[kk]
            while True:
                seg = x[lo:hi] * np.exp(-1j * w * n_idx[lo:hi])
                c = c_lo + np.cumsum(seg)                 # C(kk, lo+1..hi)
                ops += hi - lo
                tt = np.arange(lo + 1, hi + 1, dtype=float)
                s = np.abs(c) ** 2 / tt
                j = int(np.argmax(s))
                span = hi - lo
                # extend the window while the max sits on an edge
                if j >= span - 3 and hi < N:
                    hi = min(N, hi + span)
                    continue
                if j <= 2 and lo > 1:
                    lo = max(1, lo - span)
                    c_lo = np.dot(x[:lo],
                                  np.exp(-1j * w * n_idx[:lo]))
                    ops += lo
                    continue
                s_lo = np.abs(c_lo) ** 2 / max(lo, 1)
                if s_lo > s[j]:
                    tau_ref[kk], C_ref[kk] = lo, c_lo
                else:
                    tau_ref[kk], C_ref[kk] = lo + 1 + j, c[j]
                break

    # -- 6. egress: the refinement already holds the exact slice value per
    #       active bin; inactive bins read the ONE plain full transform;
    #       real input mirrors its conjugate half --
    tau_out = tau_ref.copy()
    C_out = C_ref.copy()
    Cfull = np.fft.fft(x)
    n_fft += 1
    C_out[~active] = Cfull[~active]
    if np.isrealobj(x):
        for kk in np.flatnonzero(active):
            if 1 <= kk < N - kk:
                tau_out[N - kk] = tau_out[kk]
                C_out[N - kk] = np.conj(C_out[kk])
    stats = dict(anchors=len(anchors), ffts=n_fft,
                 endpoint_ops=ops, active=int(active.sum()))
    return tau_out, C_out, stats


def part_C1(N=1024):
    print("=" * 72)
    print("C1. adaptive transform vs brute-force manifold")
    print("=" * 72)
    ok = True
    for kind in ("events", "chirp"):
        x, truth = make_signal(N, kind)
        t0 = time.perf_counter()
        C_bf, S_bf, tstar = leading_edge_bruteforce(x)
        t_bf = time.perf_counter() - t0
        t0 = time.perf_counter()
        tau_k, C_k, st = adaptive_transform(x)
        t_fast = time.perf_counter() - t0

        kk = np.arange(N)
        smax = S_bf[kk, tstar - 1]
        coh = np.flatnonzero(smax > 8.0 * np.mean(np.abs(x) ** 2))
        coh = coh[(coh > 0) & (coh < N // 2)]
        dtau = np.abs(tau_k[coh] - tstar[coh])
        s_fast = np.abs(C_k[coh]) ** 2 / tau_k[coh]
        s_ratio = s_fast / smax[coh]
        C_star = C_bf[coh, tstar[coh] - 1]
        dphi = np.abs(np.angle(C_k[coh] * np.conj(C_star)))
        print(f"  [{kind:6s}] coherent bins {len(coh):4d}  active {st['active']:4d}"
              f"  anchors {st['anchors']:3d}  shared FFTs {st['ffts']:3d}"
              f"  endpoint ops {st['endpoint_ops']:7d} (DDC bank = {N*N})")
        print(f"     tau err: med {np.median(dtau):5.1f}  p90 "
              f"{np.percentile(dtau, 90):6.1f}   score found/max: med "
              f"{np.median(s_ratio):.4f}  p10 {np.percentile(s_ratio, 10):.4f}")
        print(f"     phase arctan(A/B) at slice: med err "
              f"{np.degrees(np.median(dphi)):5.2f} deg   time: brute "
              f"{t_bf*1e3:6.1f} ms  fast {t_fast*1e3:6.1f} ms")
        if truth:
            for kt in truth:
                print(f"       bin {kt:4d}: brute tau* {tstar[kt]:4d}   "
                      f"fast tau {tau_k[kt]:4d}")
            ok &= np.median(s_ratio) > 0.98
        else:
            ok &= np.median(s_ratio) > 0.90
    return ok


# =====================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()
    rA = part_A()
    rB1 = part_B_nonclosure()
    rB2 = part_B_jet()
    rC0 = part_C0_and_CL()
    rC1 = part_C1()
    print("=" * 72)
    print(f"A  fixed THLE identity + inverse       : {'PASS' if rA else 'FAIL'}")
    print(f"B1 radix non-closure                   : {'PASS' if rB1 else 'FAIL'}")
    print(f"B2 packet jet closure                  : {'PASS' if rB2 else 'FAIL'}")
    print(f"C0/CL manifold lattice + BODFT closure : {'PASS' if rC0 else 'FAIL'}")
    print(f"C1 adaptive paused-walk transform      : {'PASS' if rC1 else 'FAIL'}")
    print(f"total {time.perf_counter()-t0:.1f}s")
