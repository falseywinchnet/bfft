"""The aperture ladder: multi-scale spectrogram fusion on the Zak/DIP stack.

This is the principled replacement for (a) OT-barycenter fusion of
spectrograms (arXiv:2604.15055) and (b) the blunt-force paused-level solvers
(zak_suffix_paused_level_solver.py / zak_middle_stack_vectorized_solver.py).
Design doc: notes/aperture_ladder_design.md.  Everything below is built on
operators VALIDATED to machine precision (see selftest and the design doc):

1. LEVEL STATES ARE ZAK TRANSFORMS.  The DIP level-t state of an N-frame is
   the finite Zak transform on the lattice (e, q) = (2^t, N/2^t):
       Z[d, j] = sum_r x[j + r q] e^{-2 pi i d r / e}
   prefix = one e-point FFT per column; suffix = the remaining butterflies;
   the suffix never mixes level-t rows (nesting theorem).

2. THE WEYL GROUP ACTS BY TWISTED TRANSLATIONS (exact, circular): time
   shift s = a q + b -> j-roll by b with a delta-dependent phase; modulation
   -> delta-roll with a j-dependent phase.

3. FRAME CONSISTENCY IS EXACT IN COMB COORDINATES.  When q | hop, adjacent
   frames of the same record satisfy C_i[r + hop/q, j] == C_{i+1}[r, j]
   IDENTICALLY (a reshape, no transform, no circular error).  The circular
   twisted-shift consensus used by the paused solvers carries a ~45%%
   relative residual on real audio; overlap-add of dewindowed frames IS the
   exact consensus, so the canonical state of this solver is THE RECORD.

4. EQUAL-q APERTURE COUPLING IS COLUMN-LOCAL.  Pause every aperture at the
   level with the same column count q*: states are (e_a, q*) with nested
   rows.  For centered rect frames, Z_small = M Z_big per column with ONE
   j-independent e_s x e_b matrix M = F_{e_s} crop F_{e_b}^{-1} (a Dirichlet
   band).  Small apertures share the fine-time axis exactly (timewise);
   large apertures extend the delta axis (frequencywise).  Fusion of many
   scales is therefore a consistent linear system through one record --
   contradiction is impossible; transport is unnecessary.

5. PHASE IS A GAUGE FIELD DETERMINED BY MAGNITUDE.  For the Gaussian-window
   STFT, V = e^{-pi lam w^2} G(tau - i lam w) with G entire, so the
   Cauchy-Riemann equations give
       dphi/dtau = 2 pi w + (1/lam) ds/dw,      dphi/dw = -lam ds/dtau
   (s = log|V|): the phase field is INTEGRATED from the magnitudes (PGHI),
   not descended for.  Hann windows use the effective lam = 0.25645 n^2.

THE SOLVER: PGHI seed (closed form, O(data)) -> a few batched multi-aperture
projection sweeps with over-relaxation momentum (all operators fixed;
acceleration persists) -> single synthesis readout.  No L-BFGS, no per-frame
Python loops, no transport plans, no descent at the final representation.

Run:  .venv/bin/python experiments/aperture_ladder.py [--wav PATH]
      (defaults to the daveandsimon test wav; writes experiments/out/)
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =========================================================================== #
#  Exact operators (the validated algebra)                                    #
# =========================================================================== #

def zak_prefix(frames: np.ndarray, level: int) -> np.ndarray:
    """Frames (I, n) -> level-`level` Zak states (I, e, q).  One batched
    e-point FFT per column: this IS the DIP prefix."""
    I, n = frames.shape
    e = 1 << level
    return np.fft.fft(frames.reshape(I, e, n >> level), axis=1)


def zak_inverse_prefix(B: np.ndarray) -> np.ndarray:
    """Level states (I, e, q) -> frames (I, n).  Exact inverse prefix."""
    I, e, q = B.shape
    return np.fft.ifft(B, axis=1).reshape(I, e * q)


def time_shift_zak(B: np.ndarray, s: int) -> np.ndarray:
    """Exact CIRCULAR Weyl time shift x[n] -> x[n+s] on Zak states
    (I, e, q).  s = a q + b: j-roll by b, delta phase for wrapped columns."""
    I, e, q = B.shape
    a, b = s // q, s % q
    d = np.arange(e)
    out = np.roll(B, -b, axis=2)
    if b:
        # rolled-in columns (the last b) wrapped: extra unit of a.
        ph_hi = np.exp(2j * np.pi * d * (a + 1) / e)[None, :, None]
        ph_lo = np.exp(2j * np.pi * d * a / e)[None, :, None]
        out[:, :, :q - b] = out[:, :, :q - b] * ph_lo
        out[:, :, q - b:] = out[:, :, q - b:] * ph_hi
    else:
        out = out * np.exp(2j * np.pi * d * a / e)[None, :, None]
    return out


def aperture_coupling(e_big: int, e_small: int) -> np.ndarray:
    """The exact column-local map from a big-aperture Zak state to the
    centered small-aperture Zak state at equal q (rect windows):
    M = F_{e_s} . crop . F_{e_b}^{-1},  Z_small[:, j] = M @ Z_big[:, j]."""
    r0 = (e_big - e_small) // 2
    Feb_inv = np.conj(np.fft.fft(np.eye(e_big))).T / e_big
    Fes = np.fft.fft(np.eye(e_small))
    return Fes @ Feb_inv[r0:r0 + e_small]


# =========================================================================== #
#  Analysis families (batched, vectorized)                                    #
# =========================================================================== #

class Family:
    """One aperture: window size n, hop h, Hann analysis of a length-L
    record.  All per-frame operations are batched."""

    def __init__(self, L: int, n: int, hop: int):
        self.L, self.n, self.hop = L, n, hop
        self.offs = np.arange(0, L - n + 1, hop, dtype=np.int64)
        self.idx = self.offs[:, None] + np.arange(n, dtype=np.int64)[None, :]
        self.win = np.hanning(n)
        self.wsq = np.zeros(L)
        np.add.at(self.wsq, self.idx.ravel(),
                  np.broadcast_to(self.win ** 2, self.idx.shape).ravel())
        self.den = np.maximum(self.wsq, 1e-8)
        # samples with near-zero window coverage (the record's outer edges)
        # are unobservable through this family: synthesis must not divide
        # by their vanishing coverage.
        self.covered = self.wsq > 1e-2 * np.median(self.wsq)

    def analyze(self, u: np.ndarray) -> np.ndarray:
        """Record -> full complex spectra (I, n)."""
        return np.fft.fft(u[self.idx] * self.win, axis=1)

    def magnitudes(self, u: np.ndarray) -> np.ndarray:
        return np.abs(self.analyze(u))

    def synthesize(self, Y: np.ndarray) -> np.ndarray:
        """Spectra (I, n) -> record by least-squares (windowed) overlap-add.
        This IS the exact comb-coordinate consensus across frames.
        Uncovered edge samples are zeroed, not amplified."""
        f = np.real(np.fft.ifft(Y, axis=1))
        acc = np.zeros(self.L)
        np.add.at(acc, self.idx.ravel(), (f * self.win).ravel())
        out = acc / self.den
        out[~self.covered] = 0.0
        return out

    def project(self, u: np.ndarray, target: np.ndarray) -> np.ndarray:
        """One magnitude projection through this family (batched GL step)."""
        Y = self.analyze(u)
        Y *= target / (np.abs(Y) + 1e-12)
        return self.synthesize(Y)

    def loss(self, u: np.ndarray, target: np.ndarray) -> float:
        return float(np.sum((self.magnitudes(u) - target) ** 2))


# =========================================================================== #
#  PGHI: closed-form phase from magnitude (the seed)                          #
# =========================================================================== #

def pghi_phases(mag: np.ndarray, n: int, hop: int, offs: np.ndarray,
                lam_factor: float = 0.25645, rel_tol: float = 1e-9,
                seed: int = 0) -> np.ndarray:
    """Phase-gradient heap integration on the half spectrum.

    mag: (I, n) full magnitudes (Hermitian).  Returns full phases (I, n) in
    the frame-local convention (phase referenced at frame START, matching
    np.fft.fft of x[o:o+n]*win).

    Theory (design doc S5): Gaussian-window Cauchy-Riemann relations
        dphi/dtau = 2 pi w + (1/lam) ds/dw     [w cycles/sample]
        dphi/dw   = -lam ds/dtau
    in the window-CENTERED, global-time convention; lam = lam_factor * n^2
    is the standard Hann-equivalent time-frequency ratio.  We integrate on
    the (bin, frame) lattice by magnitude-ordered heap propagation and then
    convert to the frame-local convention.
    """
    I, nfft = mag.shape
    F = nfft // 2 + 1
    A = mag[:, :F].T.copy()                    # (F, I) half spectrum
    s = np.log(A + 1e-30)
    lam = lam_factor * n * n

    # centered-convention gradients on the lattice.
    # ds/dw: w step = 1/nfft cycles/sample -> d/dw = nfft * central diff.
    dsdw = np.zeros_like(s)
    dsdw[1:-1] = 0.5 * nfft * (s[2:] - s[:-2])
    # ds/dtau: tau step = hop samples.
    dsdt = np.zeros_like(s)
    dsdt[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * hop)
    w = (np.arange(F) / nfft)[:, None]
    phi_t = 2 * np.pi * w + dsdw / lam         # dphi/dtau  (rad/sample)
    phi_w = -lam * dsdt                        # dphi/dw    (rad per cyc/sm)

    # heap integration, highest magnitude first.
    phase = np.zeros((F, I))
    Amax = A.max() + 1e-30
    active = A > rel_tol * Amax
    done = ~active                             # low cells: random phase later
    order = np.argsort(A.ravel())[::-1]
    heap: list = []
    # seed the globally largest active cell with zero phase, then flood.
    for flat in order:
        m, i = np.unravel_index(flat, A.shape)
        if not active[m, i]:
            break
        if done[m, i]:
            continue
        done[m, i] = True
        heap.clear()
        heapq.heappush(heap, (-A[m, i], m, i))
        while heap:
            _, m0, i0 = heapq.heappop(heap)
            # push phase to unphased active neighbours.
            if i0 + 1 < I and not done[m0, i0 + 1]:
                phase[m0, i0 + 1] = phase[m0, i0] + 0.5 * hop * (
                    phi_t[m0, i0] + phi_t[m0, i0 + 1])
                done[m0, i0 + 1] = True
                heapq.heappush(heap, (-A[m0, i0 + 1], m0, i0 + 1))
            if i0 - 1 >= 0 and not done[m0, i0 - 1]:
                phase[m0, i0 - 1] = phase[m0, i0] - 0.5 * hop * (
                    phi_t[m0, i0] + phi_t[m0, i0 - 1])
                done[m0, i0 - 1] = True
                heapq.heappush(heap, (-A[m0, i0 - 1], m0, i0 - 1))
            if m0 + 1 < F and not done[m0 + 1, i0]:
                phase[m0 + 1, i0] = phase[m0, i0] + 0.5 / nfft * (
                    phi_w[m0, i0] + phi_w[m0 + 1, i0])
                done[m0 + 1, i0] = True
                heapq.heappush(heap, (-A[m0 + 1, i0], m0 + 1, i0))
            if m0 - 1 >= 0 and not done[m0 - 1, i0]:
                phase[m0 - 1, i0] = phase[m0, i0] - 0.5 / nfft * (
                    phi_w[m0, i0] + phi_w[m0 - 1, i0])
                done[m0 - 1, i0] = True
                heapq.heappush(heap, (-A[m0 - 1, i0], m0 - 1, i0))
    rng = np.random.default_rng(seed)
    low = ~active
    phase[low] = rng.uniform(-np.pi, np.pi, size=int(low.sum()))

    # centered convention -> frame-local START convention.  Exact identity
    # (pure-tone checked): S_local[m,i] = V(tau_i, m/nfft) e^{-2 pi i m c0/nfft}
    # -- the global-time advance is already inside the 2 pi w term of phi_t.
    c0 = (n - 1) / 2.0
    m_idx = np.arange(F)[:, None]
    phase_local = phase - 2 * np.pi * m_idx * c0 / nfft

    full = np.zeros((I, nfft))
    full[:, :F] = phase_local.T
    full[:, F:] = -phase_local.T[:, nfft - F:0:-1][:, :nfft - F]
    # Hermitian completion: phase[n-k] = -phase[k] for k=1..nfft/2-1.
    pos = np.arange(1, nfft // 2)
    full[:, nfft - pos] = -full[:, pos]
    return full


def pghi_seed(fam: Family, target: np.ndarray, seed: int = 0) -> np.ndarray:
    """Closed-form record seed from one family's magnitudes."""
    ph = pghi_phases(target, fam.n, fam.hop, fam.offs, seed=seed)
    return fam.synthesize(target * np.exp(1j * ph))


# --------------------------------------------------------------------------- #
#  Gauge-integrated seed: weighted Poisson solve of the phase field.          #
#                                                                             #
#  The Cauchy-Riemann laws give a gradient ESTIMATE on every lattice edge.    #
#  Greedy (heap) integration trusts one spanning tree and accumulates the     #
#  frequency-law discretization error into the mainlobe skirts.  The phase   #
#  field is integrable up to isolated vortices, so the right estimator is    #
#  the weighted least-squares potential:                                      #
#      min_phi  sum_t-edges w (dphi - g_t)^2 + kappa sum_f-edges w (dphi-g_w)^2
#  -- one sparse SPD solve on the (F, I) lattice; errors average along every #
#  cycle instead of accumulating along any path.                              #
# --------------------------------------------------------------------------- #

def gauge_phases(mag: np.ndarray, n: int, hop: int,
                 lam_factor: float = 0.25645, kappa: float = 0.15,
                 eps_rel: float = 1e-6) -> np.ndarray:
    """Magnitudes (I, n) -> full phases (I, n), frame-local convention."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    I, nfft = mag.shape
    F = nfft // 2 + 1
    A = mag[:, :F].T.copy()                        # (F, I)
    s = np.log(A + 1e-30)
    lam = lam_factor * n * n

    dsdw = np.zeros_like(s)
    dsdw[1:-1] = 0.5 * nfft * (s[2:] - s[:-2])
    dsdt = np.zeros_like(s)
    dsdt[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * hop)
    w_cyc = (np.arange(F) / nfft)[:, None]
    phi_t = 2 * np.pi * w_cyc + dsdw / lam         # dphi/dtau
    phi_w = -lam * dsdt                            # dphi/domega

    W = A ** 2
    Wmax = W.max() + 1e-30
    Wn = W / Wmax + eps_rel

    def node(m, i):
        return m * I + i

    rows, cols, vals, rhs_entries = [], [], [], []
    nN = F * I
    b = np.zeros(nN)
    diag = np.zeros(nN)

    # time edges (m, i) -> (m, i+1): g = hop * avg(phi_t)
    wt = np.minimum(Wn[:, :-1], Wn[:, 1:]).ravel()
    g = (0.5 * hop * (phi_t[:, :-1] + phi_t[:, 1:])).ravel()
    m_idx, i_idx = np.meshgrid(np.arange(F), np.arange(I - 1), indexing='ij')
    a_ = (m_idx * I + i_idx).ravel()
    b_ = a_ + 1
    for (u, v, ww, gg) in ((a_, b_, wt, g),):
        rows += [u, v, u, v]
        cols += [u, v, v, u]
        vals += [ww, ww, -ww, -ww]
        np.add.at(b, v, ww * gg)
        np.add.at(b, u, -ww * gg)

    # frequency edges (m, i) -> (m+1, i): g = (1/nfft) * avg(phi_w)
    wf = kappa * np.minimum(Wn[:-1, :], Wn[1:, :]).ravel()
    gf = (0.5 / nfft * (phi_w[:-1, :] + phi_w[1:, :])).ravel()
    m_idx, i_idx = np.meshgrid(np.arange(F - 1), np.arange(I), indexing='ij')
    a_ = (m_idx * I + i_idx).ravel()
    b_ = a_ + I
    rows += [a_, b_, a_, b_]
    cols += [a_, b_, b_, a_]
    vals += [wf, wf, -wf, -wf]
    np.add.at(b, b_, wf * gf)
    np.add.at(b, a_, -wf * gf)

    rows = np.concatenate([np.asarray(r) for r in rows])
    cols = np.concatenate([np.asarray(c) for c in cols])
    vals = np.concatenate([np.asarray(v) for v in vals])
    Lap = coo_matrix((vals, (rows, cols)), shape=(nN, nN)).tocsr()
    # pin the gauge (constant nullspace)
    Lap = Lap + coo_matrix(([1.0], ([0], [0])), shape=(nN, nN)).tocsr()
    phase = spsolve(Lap, b).reshape(F, I)

    # centered -> frame-local START convention.
    c0 = (n - 1) / 2.0
    phase_local = phase - 2 * np.pi * np.arange(F)[:, None] * c0 / nfft
    full = np.zeros((I, nfft))
    full[:, :F] = phase_local.T
    pos = np.arange(1, nfft // 2)
    full[:, nfft - pos] = -full[:, pos]
    return full


def gauge_seed(fam: Family, target: np.ndarray) -> np.ndarray:
    """Closed-form record seed by gauge-field integration of the phases."""
    ph = gauge_phases(target, fam.n, fam.hop)
    return fam.synthesize(target * np.exp(1j * ph))


# =========================================================================== #
#  The ladder solver                                                          #
#                                                                             #
#  Pipeline: closed-form PGHI phase seed (one aperture) -> joint quasi-       #
#  Newton descent on the amplitude loss of ALL apertures at once.  The loss   #
#  geometry is fixed (linear operators + |.|), so curvature estimates and     #
#  momentum persist across every iteration — nothing is rebuilt per step      #
#  (the anti-OT-plan property).  Multi-aperture consistency is automatic:     #
#  every aperture is a linear observation of the ONE record variable; the     #
#  equal-q coupling algebra (aperture_coupling) explains why no aperture can  #
#  contradict another, and supplies the column-local data layout for kernel-  #
#  level / streaming implementations.                                         #
#                                                                             #
#  Measured basin law (design doc S7): the SHORT-window PGHI seed (dense      #
#  frames, reliable time law) leads to globally phase-true recovery (high     #
#  SNR); the LONG-window seed converges to high readout-corr fastest but      #
#  stalls at a magnitude-consistent, phase-frozen solution.  Pick by budget.  #
# =========================================================================== #

def make_loss_grad(families: list, targets: list, weights: list | None = None):
    """Joint amplitude loss  sum_a w_a || |F_a u| - r_a ||^2  and its exact
    gradient, fully batched (one FFT pair per aperture per evaluation)."""
    if weights is None:
        n_ref = max(f.n for f in families)
        weights = [n_ref / f.n for f in families]
    L = families[0].L

    def loss_grad(u):
        g = np.zeros(L)
        loss = 0.0
        for fam, r, w in zip(families, targets, weights):
            Y = fam.analyze(u)
            aY = np.abs(Y)
            loss += w * np.sum((aY - r) ** 2)
            res = (1.0 - r / (aY + 1e-12)) * Y
            gf = 2.0 * w * fam.n * (np.real(np.fft.ifft(res, axis=1)) * fam.win)
            np.add.at(g, fam.idx.ravel(), gf.ravel())
        return loss, g

    return loss_grad


def solve_ladder(families: list, targets: list, iters: int = 300,
                 seed_family: int | None = None, seed: int = 0,
                 u0: np.ndarray | None = None) -> tuple:
    """PGHI seed + joint L-BFGS on all apertures.  Returns (u, t_seed,
    t_loop, final_loss).  seed_family defaults to the SMALLEST aperture
    (the phase-true basin)."""
    from scipy.optimize import minimize
    if seed_family is None:
        seed_family = int(np.argmin([f.n for f in families]))
    t0 = time.perf_counter()
    u = pghi_seed(families[seed_family], targets[seed_family], seed=seed) \
        if u0 is None else u0.copy()
    t_seed = time.perf_counter() - t0

    lg = make_loss_grad(families, targets)
    t0 = time.perf_counter()
    res = minimize(lg, u, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=iters, maxcor=8))
    t_loop = time.perf_counter() - t0
    return res.x, t_seed, t_loop, float(res.fun)


# =========================================================================== #
#  Evaluation protocol (identical to the codex/paper demo)                    #
# =========================================================================== #

def load_first_second(path: str, work_sr: int = 8192, seconds: float = 1.0):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr0 = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[:int(round(seconds * sr0))].astype(np.float64)
    x -= x.mean()
    if sr0 != work_sr:
        g = gcd(work_sr, sr0)
        x = resample_poly(x, work_sr // g, sr0 // g)
    L = int(round(seconds * work_sr))
    x = np.pad(x, (0, max(0, L - len(x))))[:L]
    return 0.95 * x / (np.max(np.abs(x)) + 1e-12), work_sr


def reassigned_spectrogram(x, sr, n, times):
    F = n // 2 + 1
    g = np.hanning(n)
    dg = np.gradient(g)
    tg = (np.arange(n) - (n - 1) / 2.0) * g
    pad = n
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    centers = np.round(times * sr).astype(int)
    P = np.zeros((F, len(times)))
    t0 = centers[0]
    hop_est = int(round(np.median(np.diff(centers)))) if len(centers) > 1 else 1
    for i, c in enumerate(centers):
        seg = xp[c + pad - n // 2: c + pad + n // 2]
        Y = np.fft.rfft(g * seg)
        Yt = np.fft.rfft(tg * seg)
        Yd = np.fft.rfft(dg * seg)
        E = np.abs(Y) ** 2
        if E.max() <= 0:
            continue
        good = E > 1e-8 * E.max()
        that = c + np.real(Yt[good] * np.conj(Y[good])) / (E[good] + 1e-30)
        khat = (np.arange(F)[good] - np.imag(Yd[good] * np.conj(Y[good]))
                / (E[good] + 1e-30) * n / (2 * np.pi))
        cols = np.clip(np.round((that - t0) / hop_est), 0,
                       len(times) - 1).astype(int)
        rows = np.clip(np.round(khat), 0, F - 1).astype(int)
        np.add.at(P, (rows, cols), E[good])
    return P


def corrmap(P, Q):
    a = np.sqrt(np.maximum(P, 0)).ravel()
    b = np.sqrt(np.maximum(Q, 0)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def snr_db(x, y, cut=256):
    xs, ys = x[cut:-cut], y[cut:-cut]
    e = min(np.sum((xs - ys) ** 2), np.sum((xs + ys) ** 2))
    return float(10 * np.log10(np.sum(xs ** 2) / max(e, 1e-30)))


def interp_time(P, t_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, P[k], left=0., right=0.)
                     for k in range(P.shape[0])])


def interp_freq(P, f_src, f_dst):
    return np.stack([np.interp(f_dst, f_src, P[:, i], left=0., right=0.)
                     for i in range(P.shape[1])], axis=1)


# =========================================================================== #
#  Self-test of the exact operators                                           #
# =========================================================================== #

def selftest():
    rng = np.random.default_rng(1)
    # twisted translation
    u = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    for s in (1, 32, 100, 400):
        Z = zak_prefix(u[None], 5)
        Zs = zak_prefix(np.roll(u, -s)[None], 5)
        err = np.max(np.abs(Zs - time_shift_zak(Z, s)))
        assert err < 1e-10, (s, err)
    # equal-q coupling
    x = rng.standard_normal(4096)
    for (Nb, Ns) in ((1024, 128), (4096, 256)):
        qstar = 32
        tb = int(np.log2(Nb // qstar))
        ts = int(np.log2(Ns // qstar))
        c = 2048
        Zb = zak_prefix(x[c - Nb // 2:c + Nb // 2][None], tb)[0]
        Zs = zak_prefix(x[c - Ns // 2:c + Ns // 2][None], ts)[0]
        M = aperture_coupling(Nb // qstar, Ns // qstar)
        err = np.max(np.abs(Zs - M @ Zb))
        assert err < 1e-10, (Nb, Ns, err)
    # prefix roundtrip
    fr = rng.standard_normal((7, 1024))
    err = np.max(np.abs(zak_inverse_prefix(zak_prefix(fr, 5)).real - fr))
    assert err < 1e-12
    print("selftest: twisted translation, equal-q coupling, prefix -- all exact")


# =========================================================================== #
#  Main experiment                                                            #
# =========================================================================== #

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', default='/Users/quentinkuttenkuler/Downloads/daveandsimon.wav')
    ap.add_argument('--sweeps', type=int, default=12)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(outdir, exist_ok=True)
    selftest()

    x, sr = load_first_second(args.wav)
    L = len(x)

    # the two codex apertures + a third wide one for the ladder experiment
    fam_long = Family(L, 1024, 128)
    fam_short = Family(L, 128, 32)
    fam_wide = Family(L, 4096, 512)
    targ_long = fam_long.magnitudes(x)
    targ_short = fam_short.magnitudes(x)
    targ_wide = fam_wide.magnitudes(x)

    times2 = (fam_short.offs + fam_short.n / 2) / sr
    times1 = (fam_long.offs + fam_long.n / 2) / sr
    freqs1 = np.arange(1024 // 2 + 1) * sr / 1024
    freqs2 = np.arange(128 // 2 + 1) * sr / 128
    freqsw = np.arange(4096 // 2 + 1) * sr / 4096
    oracle = reassigned_spectrogram(x, sr, 1024, times2)

    P1 = (targ_long.T[:1024 // 2 + 1]) ** 2
    P2 = (targ_short.T[:128 // 2 + 1]) ** 2
    Pw = (targ_wide.T[:4096 // 2 + 1]) ** 2
    long_map = interp_time(P1, times1, times2)
    short_map = interp_freq(P2, freqs2, freqs1)
    geo2 = np.sqrt(np.maximum(long_map, 0) * np.maximum(short_map, 0))
    print(f"baselines: long {corrmap(long_map, oracle):.4f}  "
          f"short {corrmap(short_map, oracle):.4f}  "
          f"geomean {corrmap(geo2, oracle):.4f}")

    # ---- blind 3-way fusion fails (the user's observation, reproduced)
    timesw = (fam_wide.offs + fam_wide.n / 2) / sr
    wide_map = interp_freq(interp_time(Pw, timesw, times2), freqsw, freqs1)
    geo3 = (np.maximum(long_map, 0) * np.maximum(short_map, 0)
            * np.maximum(wide_map, 0)) ** (1.0 / 3.0)
    print(f"blind 3-way geomean: {corrmap(geo3, oracle):.4f}  "
          f"(vs 2-way {corrmap(geo2, oracle):.4f} -- blind fusion of scales degrades)")

    # ---- 2-aperture, FAST operating point (codex's problem, his budget)
    u_fast, ts_f, tl_f, _ = solve_ladder([fam_long, fam_short],
                                         [targ_long, targ_short],
                                         iters=60, seed_family=0)
    P_fast = reassigned_spectrogram(u_fast, sr, 1024, times2)
    c_fast = corrmap(P_fast, oracle)
    print(f"\n2-aperture FAST  (long seed, 60 it): corr {c_fast:.4f}  "
          f"SNR {snr_db(x, u_fast):+.1f} dB  "
          f"{(ts_f + tl_f) * 1e3:.0f} ms  [codex: 0.9754 @ 209 ms]")

    # ---- 2-aperture, QUALITY operating point (phase-true basin)
    u2, ts2, tl2, l2 = solve_ladder([fam_long, fam_short],
                                    [targ_long, targ_short],
                                    iters=500)
    P2r = reassigned_spectrogram(u2, sr, 1024, times2)
    c2 = corrmap(P2r, oracle)
    print(f"2-aperture QUALITY (short seed, 500 it): corr {c2:.4f}  "
          f"SNR {snr_db(x, u2):+.1f} dB  {(ts2 + tl2) * 1e3:.0f} ms  "
          f"loss {l2:.2e}")

    # ---- 3-aperture ladder: domain extension instead of blur
    u3, ts3, tl3, l3 = solve_ladder([fam_long, fam_short, fam_wide],
                                    [targ_long, targ_short, targ_wide],
                                    iters=500)
    P3r = reassigned_spectrogram(u3, sr, 1024, times2)
    c3 = corrmap(P3r, oracle)
    print(f"3-aperture ladder  (short seed, 500 it): corr {c3:.4f}  "
          f"SNR {snr_db(x, u3):+.1f} dB  {(ts3 + tl3) * 1e3:.0f} ms  "
          f"loss {l3:.2e}")
    # the wide aperture's own oracle: does the ladder gain fine frequency?
    oracle_w = reassigned_spectrogram(x, sr, 4096, times2)
    print(f"   fine-frequency readout (n=4096 reassigned vs its oracle): "
          f"2-ap {corrmap(reassigned_spectrogram(u2, sr, 4096, times2), oracle_w):.4f}  "
          f"3-ap {corrmap(reassigned_spectrogram(u3, sr, 4096, times2), oracle_w):.4f}")

    # ---- figures
    def show(ax, P, title):
        Pm = np.sqrt(np.maximum(P, 0))
        vmax = np.percentile(Pm, 99.5) if np.any(Pm) else 1.0
        ax.imshow(Pm, origin='lower', aspect='auto',
                  extent=[times2[0], times2[-1], 0, sr / 2],
                  vmax=vmax, cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 4000)
        ax.set_xlabel('time (s)', fontsize=8)
        ax.set_ylabel('freq (Hz)', fontsize=8)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    show(axes[0, 0], long_map, f'long source {corrmap(long_map, oracle):.3f}')
    show(axes[0, 1], short_map, f'short source {corrmap(short_map, oracle):.3f}')
    show(axes[0, 2], geo2, f'geomean 2-way {corrmap(geo2, oracle):.3f}')
    show(axes[0, 3], geo3, f'geomean 3-way {corrmap(geo3, oracle):.3f} (blind blur)')
    show(axes[1, 0], P_fast, f'2-ap FAST {c_fast:.3f} '
                             f'@{(ts_f + tl_f) * 1e3:.0f}ms')
    show(axes[1, 1], P2r, f'2-ap QUALITY {c2:.3f}, SNR {snr_db(x, u2):+.0f}dB')
    show(axes[1, 2], P3r, f'3-ap ladder {c3:.3f}, SNR {snr_db(x, u3):+.0f}dB')
    show(axes[1, 3], oracle, 'oracle readout')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'aperture_ladder.png'), dpi=130)
    print(f"\nfigure -> {outdir}/aperture_ladder.png")


if __name__ == '__main__':
    main()
