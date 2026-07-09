"""Vertical identification: the superresolution problem stated and solved
INSIDE the DIP walk.  No scipy anywhere; every operation is a walk stage, a
banded row operator, an adjoint, or a radial projection.

This file is the executable record of the vertical-identification research
(notes/dip_vertical_identification.md).  It is the corrected answer to "the
system must live inside the DIP forward": the multi-aperture observations are
attached EXACTLY to one paused state per long frame, and the update blocks are
closed-form residual pullbacks through the walk's own stages.

THE THEOREMS (each validated to machine precision below):

T1  POLARIZATION LADDER.  One DIF butterfly p = a + w b, q = a - w b gives
    |p|^2 - |q|^2 = 4 Re(conj(a) w b): the magnitudes at level t+1 measure the
    RELATIVE PHASES of level-t pairs.  The magnitude profile down the tower is
    a hierarchy of relative-phase measurements at all dyadic lags; phase
    retrieval hardness lives in the LEVEL GAPS between observed rungs.
    (Energy-weighted sign census on the real record: ~18% of butterfly energy
    has |cos| < 0.3 — the per-butterfly sign bits that remain discrete.)

T2  WINDOWS ARE TAPS, NEVER DIVISIONS.  The PERIODIC Hann is an exact twisted
    3-tap row operator at EVERY level of the walk:
        Zak(h.u)[d,j] = 0.5 Z[d,j] - 0.25 e^{2pi i j/n} Z[d-1,j]
                                   - 0.25 e^{-2pi i j/n} Z[d+1,j]
    (plain circular roll in d; the twist is a constant per column, so the
    operator is a per-column CIRCULANT with symbol 0.5 - 0.5cos(2pi m/e -
    theta_j), exactly one zero = the window's edge null).  np.hanning
    (symmetric) is detuned by 1/(n-1) bin and NOT finitely banded — use
    periodic Hann in-system.

T3' APERTURES ATTACH IN THE STATE.  Every short frame inside a long frame at a
    q*-aligned offset (hop 32 = q*, so ALL of them) satisfies EXACTLY
        Zak_s(short frame) = M_delta @ Zak_b(long frame level-5 columns),
        M_delta = F_{e_s} crop_delta F_{e_b}^{-1}   (4 x 32 Dirichlet band)
    and the short Hann is T2's 3-tap in the short tower.  Validated 3e-16 for
    all 29 offsets.  NO time-domain round trip exists in the formulation.
    (NOTE a false sibling claim was tested and rejected: contiguous-BLOCK
    spectra do NOT decouple into record-bin residue classes — rect crops leak
    by Dirichlet tails.  Exact attachment is comb-side, per-frame, as above.)

T5  PER-ROW PAULI BANK.  The suffix from level 5 restricted to one row d is a
    UNITARY twisted 32-DFT W_d onto the endpoint comb {k == d mod 32} (suffix
    nesting).  So per long frame the vertical problem is a bank of 32
    dimension-32 Pauli problems (recover a row from |row| and |W_d row|),
    coupled across rows ONLY by the short-aperture banded crops (T3') and
    across frames by the comb overlap bus.
    Census: the Pauli pair alone pins ~33% of row energy (rest ambiguous) —
    quantifying exactly what the short band must contribute.

THE ENGINE (closed blocks, no step sizes — all scales are operator norms):
unknown = rect-frame level-5 state Z (32x32, Hermitian rows for real frames);
observations y_long = |suffix(tap3_b(Z))|, y_short[i] = |suffix(tap3_s(M_i Z))|;
sweep = sum of ADJOINT RESIDUAL PULLBACKS (radial residual at each rung pulled
back through pure adjoints — the same law as the paused-L5 teacher gradient,
expressed per-frame with the short apertures attached in-state).  Measured:
LOCALLY CONTRACTING around truth (0.10 -> 0.064 in 60 sweeps, monotone, all
frames tested).  Tikhonov-inverse pullbacks were tested and REJECTED: biased
at truth (fixed point != solution).  Cold start does not reach the basin —
seeding is the open problem, consistent with all other measurements today.

Run:  .venv/bin/python experiments/dip_vertical_identification.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

assert 'scipy' not in sys.modules, 'this file must stay scipy-free'

WAV = '/Users/quentinkuttenkuler/Downloads/daveandsimon.wav'
L = 8192


def load_record():
    import soundfile as sf
    x48, sr0 = sf.read(WAV, always_2d=False)
    x1 = x48[:sr0].astype(np.float64)
    x1 -= x1.mean()
    Xc = np.fft.rfft(x1)[:L // 2 + 1] * (L / sr0)   # FFT-domain resample
    x = np.fft.irfft(Xc, n=L)
    return 0.95 * x / np.max(np.abs(x))


def hann_p(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def zak(u, t):
    e = 1 << t
    return np.fft.fft(u.reshape(e, -1), axis=0)


def suffix(Z, e0, n):
    C = Z.copy()
    e = e0
    while e < n:
        q2 = C.shape[1] // 2
        w = np.exp(-1j * np.pi * np.arange(e) / e)[:, None]
        A_, D_ = C[:, :q2], C[:, q2:]
        C = np.vstack([A_ + w * D_, A_ - w * D_])
        e *= 2
    return C[:, 0]


def suffix_inv(Xv, e0, n):
    C = Xv.reshape(-1, 1).astype(complex)
    e = n
    while e > e0:
        e //= 2
        half = C.shape[0] // 2
        q = C.shape[1] * 2
        w = np.exp(-1j * np.pi * np.arange(e) / e)[:, None]
        P_, Q_ = C[:half], C[half:]
        B = np.empty((half, q), complex)
        B[:, :q // 2] = 0.5 * (P_ + Q_)
        B[:, q // 2:] = 0.5 * np.conj(w) * (P_ - Q_)
        C = B
    return C


def tap3(Z, ph):
    return 0.5 * Z - 0.25 * ph * np.roll(Z, 1, axis=0) \
        - 0.25 * np.conj(ph) * np.roll(Z, -1, axis=0)


def tap3_adj(Z, ph):
    return 0.5 * Z - 0.25 * np.roll(np.conj(ph) * Z, -1, axis=0) \
        - 0.25 * np.roll(ph * Z, 1, axis=0)


# ---- problem constants (benchmark apertures) ------------------------------
Nb, Ns, qstar = 1024, 128, 32
eb, es, tb, ts = 32, 4, 5, 2
phb = np.exp(2j * np.pi * np.arange(Nb >> tb) / Nb)[None, :]
phs = np.exp(2j * np.pi * np.arange(Ns >> ts) / Ns)[None, :]
deltas = np.arange(0, Nb - Ns + 1, 32)

_Feb_inv = np.conj(np.fft.fft(np.eye(eb))).T / eb
_Fes = np.fft.fft(np.eye(es))
Ms = []
for delta in deltas:
    r0 = delta // qstar
    crop = np.zeros((es, eb))
    crop[np.arange(es), r0 + np.arange(es)] = 1.0
    Ms.append(_Fes @ crop @ _Feb_inv)
Ms = np.array(Ms)


# =========================================================================== #
def theorems(x):
    print("== T1: polarization ladder ==")
    fr = (hann_p(Nb) * x[256:256 + Nb]).astype(np.complex128)
    Z5 = zak(fr, tb)
    q2 = Z5.shape[1] // 2
    a, b = Z5[:, :q2], Z5[:, q2:]
    w = np.exp(-1j * np.pi * np.arange(eb) / eb)[:, None]
    Z6 = np.vstack([a + w * b, a - w * b])
    lhs = np.abs(Z6[:eb]) ** 2 - np.abs(Z6[eb:]) ** 2
    rhs = 4 * np.real(np.conj(a) * w * b)
    print(f"   ||p|^2-|q|^2 - 4Re(a* w b)|_max = {np.max(np.abs(lhs-rhs)):.2e}")
    en = (np.abs(a) * np.abs(b)).ravel()
    cs = np.abs(lhs / (4 * np.abs(a) * np.abs(b) + 1e-30)).ravel()
    wts = en / en.sum()
    print(f"   sign census: |cos|>0.9 energy {wts[cs > .9].sum():.2f}, "
          f"|cos|<0.3 (sign-critical) {wts[cs < .3].sum():.2f}")

    print("\n== T2: periodic Hann = exact twisted 3-tap at every level ==")
    u = x[256:256 + Nb].astype(np.complex128)
    h = hann_p(Nb)
    for t in (0, 3, 5, 8):
        q = Nb >> t
        ph = np.exp(2j * np.pi * np.arange(q) / Nb)[None, :]
        err = np.max(np.abs(zak(h * u, t) - tap3(zak(u, t), ph)))
        print(f"   level {t}: {err:.2e}")

    print("\n== T3': all short frames attach in-state (banded row crops) ==")
    ub = x[1024:1024 + Nb].astype(np.complex128)
    Zb = zak(ub, tb)
    hs = hann_p(Ns)
    err_r = err_h = 0.0
    for i, delta in enumerate(deltas):
        us = x[1024 + delta:1024 + delta + Ns]
        Zs_pred = Ms[i] @ Zb
        err_r = max(err_r, np.max(np.abs(
            Zs_pred - zak(us.astype(np.complex128), ts))))
        err_h = max(err_h, np.max(np.abs(
            tap3(Zs_pred, phs) - zak((hs * us).astype(np.complex128), ts))))
    print(f"   {len(deltas)} offsets: rect {err_r:.2e}, hann(3-tap) {err_h:.2e}")

    print("\n== T5: per-row suffix = unitary twisted 32-DFT (Pauli bank) ==")
    d = 7
    W = np.zeros((eb, eb), complex)
    Zt = np.zeros((eb, eb), complex)
    for j in range(eb):
        Zt[:] = 0
        Zt[d, j] = 1.0
        W[:, j] = suffix(Zt, eb, Nb)[d::eb]
    print(f"   |W W^H - 32 I|_max = "
          f"{np.max(np.abs(W @ np.conj(W.T) - eb * np.eye(eb))):.2e}")
    return W


def pauli_census(x, rng_seed=0):
    """Per-row 32-dim Pauli bank: how much does (|row|, |W_d row|) pin?"""
    print("\n== census: per-row Pauli identifiability ==")
    rng = np.random.default_rng(rng_seed)
    # build all W_d once
    Wbank = np.zeros((eb, eb, eb), complex)
    Zt = np.zeros((eb, eb), complex)
    for d in range(eb):
        for j in range(eb):
            Zt[:] = 0
            Zt[d, j] = 1.0
            Wbank[d, :, j] = suffix(Zt, eb, Nb)[d::eb]
    WH = np.conj(np.transpose(Wbank, (0, 2, 1)))
    hb = hann_p(Nb)
    ok_e = tot_e = 0.0
    nrows = nfail = 0
    for O in range(0, L - Nb + 1, 512):
        Z5 = zak((hb * x[O:O + Nb]).astype(np.complex128), tb)
        for d in range(1, eb // 2):
            u_true = Z5[d]
            A = np.abs(u_true)
            C = np.abs(Wbank[d] @ u_true)
            en = float(A @ A)
            if en < 1e-8 * float(np.abs(Z5).max()) ** 2:
                continue
            best = None
            for r in range(4):
                u = A * np.exp(2j * np.pi * rng.random(eb))
                for _ in range(150):
                    v = Wbank[d] @ u
                    v = C * v / (np.abs(v) + 1e-30)
                    u = (WH[d] @ v) / eb
                    u = A * u / (np.abs(u) + 1e-30)
                z = np.vdot(u_true, u)
                err = np.linalg.norm(u * np.exp(-1j * np.angle(z)) - u_true) \
                    / (np.linalg.norm(u_true) + 1e-30)
                best = err if best is None or err < best else best
            nrows += 1
            tot_e += en
            if best < 0.05:
                ok_e += en
            else:
                nfail += 1
    print(f"   rows {nrows}: energy solved by the Pauli pair alone "
          f"{ok_e / tot_e * 100:.1f}%  ({nfail} ambiguous rows)")
    print("   -> the remainder is what the short band + bus must pin.")


# =========================================================================== #
def observations(x, O):
    fr = x[O:O + Nb]
    Z5 = zak(fr.astype(np.complex128), tb)
    y_long = np.abs(suffix(tap3(Z5, phb), eb, Nb))
    y_short = np.array([np.abs(suffix(tap3(Ms[i] @ Z5, phs), es, Ns))
                        for i in range(len(deltas))])
    return Z5, y_long, y_short


def sweep(Z, y_long, y_short, its):
    """Closed-block vertical sweeps: adjoint residual pullbacks (unbiased at
    truth), Hermitian projection.  Scales = operator norms, nothing tuned."""
    nfam = 1 + len(deltas)
    for _ in range(its):
        G = np.zeros_like(Z)
        Yv = suffix(tap3(Z, phb), eb, Nb)
        R = y_long * Yv / (np.abs(Yv) + 1e-30) - Yv
        G += tap3_adj(suffix_inv(R, eb, Nb), phb)
        for i in range(len(deltas)):
            S = suffix(tap3(Ms[i] @ Z, phs), es, Ns)
            Rs = y_short[i] * S / (np.abs(S) + 1e-30) - S
            G += np.conj(Ms[i].T) @ tap3_adj(suffix_inv(Rs, es, Ns), phs)
        Z = Z + G / nfam
        Z = 0.5 * (Z + np.conj(Z[(-np.arange(eb)) % eb, :]))
    return Z


def rel_err(Z, Z5):
    return min(np.linalg.norm(Z - Z5), np.linalg.norm(Z + Z5)) \
        / np.linalg.norm(Z5)


def engine(x):
    print("\n== engine: closed-block vertical sweeps on Z5 ==")
    print("   (a) local contraction around truth:")
    for O in (1024, 3072, 5120):
        Z5, yl, ys = observations(x, O)
        rng = np.random.default_rng(0)
        Z = Z5 + 0.1 * np.linalg.norm(Z5) / np.sqrt(Z5.size) * (
            rng.standard_normal(Z5.shape) + 1j * rng.standard_normal(Z5.shape))
        errs = [rel_err(Z, Z5)]
        for _ in range(6):
            Z = sweep(Z, yl, ys, 10)
            errs.append(rel_err(Z, Z5))
        print(f"      frame @{O}: 0.100 -> "
              + " -> ".join(f"{e:.4f}" for e in errs[1:]))
    print("   (b) cold start (open problem = seeding the basin):")
    for O in (1024, 3072, 5120):
        Z5, yl, ys = observations(x, O)
        best = None
        for s in range(3):
            rng = np.random.default_rng(s)
            Yv = yl * np.exp(2j * np.pi * rng.random(Nb))
            Z = tap3_adj(suffix_inv(Yv, eb, Nb), phb)
            Z = sweep(Z, yl, ys, 120)
            e = rel_err(Z, Z5)
            best = e if best is None or e < best else best
        print(f"      frame @{O}: best rel err {best:.3f}")


def main():
    t0 = time.perf_counter()
    x = load_record()
    theorems(x)
    pauli_census(x)
    engine(x)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s; scipy imported: "
          f"{'scipy' in sys.modules}")


if __name__ == '__main__':
    main()
