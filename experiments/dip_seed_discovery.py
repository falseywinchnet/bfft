"""Seeding, discovered.  No scipy anywhere; every operation is a walk stage,
a banded op, an adjoint, or (in the finisher) a hand-rolled two-loop.

This file resolves the open problem of notes/dip_vertical_identification.md
S4: what seed reaches the basin of the in-walk vertical system.  The findings
(all measured on the first second of daveandsimon.wav @ 8192 Hz, periodic
Hann, targets |STFT| of long 1024/128 and short 128/32):

1. THE BASIN HAS NO WALL, ONLY A SADDLE.  The closed-block vertical sweeps
   contract from every tested radius (even rel err 1.14 -> 0.45), but cold
   random starts sit near a slow saddle: what a seed must buy is
   CORRELATION STRUCTURE, not small error.

2. LIFT SEEDS ARE STRUCTURALLY OBSTRUCTED (a real negative result).  Every
   spectral-initializer variant on the joint measurement lift
   T = sum A^H diag(f(y^2) - c) A  (raw / sqrt / log / clipped weights, and
   the null-space version) yields correlation <= 0.28 and post-sweep frame
   error ~1.0.  Reason, stated as an obstruction: the observable lift is the
   ambiguity strip BLURRED TO HOP RESOLUTION IN POSITION, and the lifted
   diagonals s[n]s[n+tau] of broadband audio oscillate at twice the carrier
   frequency — the hop-scale blur annihilates exactly the content the
   eigenvector needs.  Spectral initializers work for position-broadband
   (random) masks; smooth-window STFT masks are position-lowpass.  Do not
   revisit without a mechanism that defeats this.

3. THE SEED CURRENCY IS RIDGE-LOCAL PHASE COHERENCE, NOT WAVEFORM
   CORRELATION.  PGHI seeds carry ~zero waveform correlation (SNR ~ -2 dB)
   yet convert under a curvature finisher; lift seeds with HIGHER waveform
   correlation do not convert.  Confirmed at mechanism level by the
   'timeonly' ablation: per-row time-law integration with RANDOM row
   constants (frequency law dropped entirely) can reach +31 dB — the
   time-law ridge phase-velocity is the payload; constants are the
   finisher's job.  (But see 5: at short scale the random constants make it
   a lottery.)

4. THE FINISHER MUST CARRY CURVATURE, AND THE GRADIENT SCALE IS LOAD-
   BEARING.  First-order closed-block sweeps cannot convert weak seeds.  A
   hand-rolled L-BFGS (two-loop + Armijo, the user's ComplexLBFGS pattern —
   no scipy) converts them — but ONLY with the exact adjoint scaling
   (F^H = n * ifft): an innocent 1/n slip reweights the families 64:1 and
   the identical pipeline stalls at loss ~2000 forever.  With the exact
   gradient the phase-frozen plateau (loss ~49, SNR -3) is ESCAPED around
   iteration ~600.

5. THE DISCOVERED CHAIN (deterministic, scipy-free, ~1.3 s total):
       pghi_long (full, heap) seed  ->  hand L-BFGS(800) on the joint
       amplitude loss with exact adjoints
   = SNR +46.6 dB, identical across all rng draws (seed randomness only
   touches inactive cells).  pghi_short_timeonly reaches +31..+42 dB but
   only on ~1/4 of draws (random row constants).  All previous bests:
   scipy ladder QUALITY +17.8 dB, codex ~0.975 corr with no reported SNR.

Run:  .venv/bin/python experiments/dip_seed_discovery.py
"""
from __future__ import annotations

import heapq
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
    Xc = np.fft.rfft(x1)[:L // 2 + 1] * (L / sr0)
    x = np.fft.irfft(Xc, n=L)
    return 0.95 * x / np.max(np.abs(x))


def hann_p(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


class Fam:
    """One aperture as a walk operator on the record (batched)."""

    def __init__(self, n, hop):
        self.n, self.hop = n, hop
        self.offs = np.arange(0, L - n + 1, hop)
        self.idx = self.offs[:, None] + np.arange(n)[None, :]
        self.win = hann_p(n)
        self.cov = np.zeros(L)
        np.add.at(self.cov, self.idx.ravel(),
                  np.broadcast_to(self.win ** 2, self.idx.shape).ravel())

    def analyze(self, u):
        return np.fft.fft(u[self.idx] * self.win, axis=1)

    def adjoint(self, Y):
        # exact adjoint of u -> fft(win * u[idx]):  F^H = n * ifft.  The n is
        # LOAD-BEARING (finding 4): dropping it silently reweights families.
        f = np.fft.ifft(Y, axis=1) * self.n
        out = np.zeros(L)
        np.add.at(out, self.idx.ravel(), np.real(f * self.win).ravel())
        return out


# --------------------------------------------------------------------------- #
#  The joint amplitude loss with exact adjoint gradient                        #
# --------------------------------------------------------------------------- #

def make_loss_grad(fams, ys, ws):
    def loss_grad(u):
        g = np.zeros(L)
        loss = 0.0
        for fam, y, w in zip(fams, ys, ws):
            Y = fam.analyze(u)
            aY = np.abs(Y)
            loss += w * float(np.sum((aY - y) ** 2))
            g += 2.0 * w * fam.adjoint((1.0 - y / (aY + 1e-12)) * Y)
        return loss, g
    return loss_grad


def lbfgs(loss_grad, u0, iters=800, maxcor=8):
    """Hand-rolled two-loop L-BFGS with Armijo backtracking (no scipy)."""
    u = u0.copy()
    loss, g = loss_grad(u)
    S, Yv, rho = [], [], []
    scale0 = 1.0 / (np.linalg.norm(g) + 1e-30)
    for _ in range(iters):
        q = g.copy()
        alpha = []
        for s_, y_, r_ in zip(reversed(S), reversed(Yv), reversed(rho)):
            a = r_ * np.dot(s_, q)
            alpha.append(a)
            q -= a * y_
        if S:
            q *= np.dot(S[-1], Yv[-1]) / max(np.dot(Yv[-1], Yv[-1]), 1e-30)
        else:
            q *= scale0
        for s_, y_, r_, a in zip(S, Yv, rho, reversed(alpha)):
            q += s_ * (a - r_ * np.dot(y_, q))
        p = -q
        gtp = np.dot(g, p)
        if gtp >= 0:
            p = -scale0 * g
            gtp = np.dot(g, p)
        step = 1.0
        for _ in range(20):
            u_t = u + step * p
            l_t, g_t = loss_grad(u_t)
            if l_t <= loss + 1e-4 * step * gtp:
                break
            step *= 0.5
        s_vec = u_t - u
        y_vec = g_t - g
        sy = np.dot(s_vec, y_vec)
        if sy > 1e-12 * np.linalg.norm(s_vec) * np.linalg.norm(y_vec):
            S.append(s_vec)
            Yv.append(y_vec)
            rho.append(1.0 / sy)
            if len(S) > maxcor:
                S.pop(0)
                Yv.pop(0)
                rho.pop(0)
        u, loss, g = u_t, l_t, g_t
    return u, loss


# --------------------------------------------------------------------------- #
#  Seeds                                                                       #
# --------------------------------------------------------------------------- #

def pghi_seed(fam, mag, mode='full', seed=0, lam_factor=0.25645):
    """PGHI phase construction; mode='timeonly' drops the frequency law and
    uses random per-row constants (the ridge-coherence ablation)."""
    I, nfft = mag.shape
    F = nfft // 2 + 1
    A = mag[:, :F].T.copy()
    s_ = np.log(A + 1e-30)
    lam = lam_factor * fam.n * fam.n
    dsdw = np.zeros_like(s_)
    dsdw[1:-1] = 0.5 * nfft * (s_[2:] - s_[:-2])
    dsdt = np.zeros_like(s_)
    dsdt[:, 1:-1] = (s_[:, 2:] - s_[:, :-2]) / (2 * fam.hop)
    wcyc = (np.arange(F) / nfft)[:, None]
    phi_t = 2 * np.pi * wcyc + dsdw / lam
    phi_w = -lam * dsdt
    phase = np.zeros((F, I))
    rng = np.random.default_rng(seed)
    if mode == 'timeonly':
        row0 = rng.uniform(-np.pi, np.pi, F)
        csum = np.cumsum(0.5 * fam.hop * (phi_t[:, :-1] + phi_t[:, 1:]), axis=1)
        phase[:, 0] = row0
        phase[:, 1:] = row0[:, None] + csum
    else:
        active = A > 1e-9 * A.max()
        done = ~active
        order = np.argsort(A.ravel())[::-1]
        heap = []
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
                nbrs = []
                if i0 + 1 < I:
                    nbrs.append((m0, i0 + 1, 0.5 * fam.hop * (phi_t[m0, i0] + phi_t[m0, i0 + 1])))
                if i0 - 1 >= 0:
                    nbrs.append((m0, i0 - 1, -0.5 * fam.hop * (phi_t[m0, i0] + phi_t[m0, i0 - 1])))
                if m0 + 1 < F:
                    nbrs.append((m0 + 1, i0, 0.5 / nfft * (phi_w[m0, i0] + phi_w[m0 + 1, i0])))
                if m0 - 1 >= 0:
                    nbrs.append((m0 - 1, i0, -0.5 / nfft * (phi_w[m0, i0] + phi_w[m0 - 1, i0])))
                for (m1, i1, dphi) in nbrs:
                    if not done[m1, i1]:
                        phase[m1, i1] = phase[m0, i0] + dphi
                        done[m1, i1] = True
                        heapq.heappush(heap, (-A[m1, i1], m1, i1))
        phase[~active] = rng.uniform(-np.pi, np.pi, int((~active).sum()))
    c0 = (fam.n - 1) / 2.0
    phase_local = phase - 2 * np.pi * np.arange(F)[:, None] * c0 / nfft
    full = np.zeros((I, nfft))
    full[:, :F] = phase_local.T
    pos = np.arange(1, nfft // 2)
    full[:, nfft - pos] = -full[:, pos]
    Y = mag * np.exp(1j * full)
    f = np.real(np.fft.ifft(Y, axis=1))
    acc = np.zeros(L)
    np.add.at(acc, fam.idx.ravel(), (f * fam.win).ravel())
    u = acc / np.maximum(fam.cov, 1e-8)
    u[fam.cov < 1e-2 * np.median(fam.cov)] = 0.0
    return u


def lift_seed(fams, y2s, ws, weight_fn, iters=60, seed=0):
    """Spectral initializer on the joint measurement lift (matvec = walk
    ops).  Kept as the executable form of the OBSTRUCTION result (finding 2):
    measure its correlation and watch it not convert."""
    cs = [float(weight_fn(y2).mean()) for y2 in y2s]

    def mv(v):
        out = np.zeros(L)
        for fam, y2, w, c in zip(fams, y2s, ws, cs):
            out += w * fam.adjoint((weight_fn(y2) - c) * fam.analyze(v))
        return out

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(L)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w_ = mv(mv(v))
        v = w_ / (np.linalg.norm(w_) + 1e-30)
    return v


# --------------------------------------------------------------------------- #
def main():
    x = load_record()
    fam_s, fam_l = Fam(128, 32), Fam(1024, 128)
    y_s = np.abs(fam_s.analyze(x))
    y_l = np.abs(fam_l.analyze(x))
    lg = make_loss_grad([fam_l, fam_s], [y_l, y_s], [1.0, 1024 / 128])

    def snr(u):
        xs, us = x[256:-256], u[256:-256]
        e = min(np.sum((xs - us) ** 2), np.sum((xs + us) ** 2))
        return 10 * np.log10(np.sum(xs ** 2) / max(e, 1e-30))

    print("== finding 2: lift seeds are obstructed (correlation ceiling) ==")
    for name, fn in (('raw y^2', lambda y: y), ('sqrt', np.sqrt),
                     ('log', lambda y: np.log1p(y / y.mean()))):
        v = lift_seed([fam_s, fam_l], [y_s ** 2, y_l ** 2],
                      [1 / 128 ** 2, 1 / 1024 ** 2], fn)
        c = abs(np.dot(v, x)) / (np.linalg.norm(v) * np.linalg.norm(x))
        print(f"   lift[{name:8s}]: corr {c:.3f}   (position-lowpass ceiling)")

    print("\n== findings 3-5: seed x curvature-finisher matrix ==")
    rng = np.random.default_rng(0)
    seeds = {
        'random': 0.01 * rng.standard_normal(L),
        'pghi_long_full': pghi_seed(fam_l, y_l, 'full'),
        'pghi_short_full': pghi_seed(fam_s, y_s, 'full'),
        'pghi_long_timeonly': pghi_seed(fam_l, y_l, 'timeonly'),
        'pghi_short_timeonly': pghi_seed(fam_s, y_s, 'timeonly'),
    }
    for name, u0 in seeds.items():
        t0 = time.perf_counter()
        u, fl = lbfgs(lg, u0, iters=800)
        print(f"   {name:20s}: seed SNR {snr(u0):+5.1f} -> final "
              f"{snr(u):+6.1f} dB  loss {fl:.2e}  "
              f"[{time.perf_counter() - t0:.1f}s]")

    print("\n== stability of the discovered chain (pghi_long + LBFGS 800) ==")
    for s in range(4):
        u0 = pghi_seed(fam_l, y_l, 'full', seed=s)
        u, fl = lbfgs(lg, u0, iters=800)
        print(f"   rng{s}: SNR {snr(u):+6.1f} dB  loss {fl:.2e}")

    print("\n== escape trajectory of the phase-frozen plateau ==")
    u0 = pghi_seed(fam_l, y_l, 'full')
    for iters in (150, 400, 600, 800):
        u, fl = lbfgs(lg, u0, iters=iters)
        print(f"   it={iters:4d}: SNR {snr(u):+6.1f} dB  loss {fl:.2e}")

    print("\nscipy imported:", 'scipy' in sys.modules)


if __name__ == '__main__':
    main()
