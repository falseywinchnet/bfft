"""Super-resolution spectrogram fusion by alignment in DIP intermediate space.

THE PROBLEM (Valdivia, Cazelles, Fevotte 2026, arXiv:2604.15055): given two
spectrograms of the same signal — M1 long-window (sharp frequency, blurred
time), M2 short-window (sharp time, blurred frequency), on DIFFERENT t-f
grids, magnitudes only — produce a super-resolution t-f product on the grid
F1 x T2.  Their solution is an unbalanced-OT barycenter computed by block
majorization-minimization: Bregman descent on two constrained sources, where
the transport plan changes every step (so acceleration information is
discarded every step) and a sub-second signal costs ~30 minutes.

THE BASIS (proved numerically in dip_numba.py): the DIP's level-t
intermediate state is the finite ZAK TRANSFORM of the frame on the dyadic
lattice (2^t, N/2^t) — the walk is a curve through the whole family of
fractional time-frequency representations, from the time axis (t=0) to the
frequency axis (t=m).  At a mid level:

    row delta = fine frequency (bin index mod 2^t — the LOW bits of k)
    col j     = fine time      (sample index mod N/2^t)
    coarse frequency = oscillation of a row along j (phase, not magnitude)
    coarse time      = phase pattern down a column across rows

Consequences exploited here:
  * two tones Dk bins apart occupy DIFFERENT rows once 2^t does not divide
    Dk: close-tone separation happens in the first levels of the walk and is
    never in tension with time localization;
  * a steady tone is ONE row x unit-modulus oscillation; an evolving atom
    (chirp/vibrato) is a compact ridge — signals are sparse there, and the
    sparsity is of MIXES (phase-bearing linear functionals of the signal),
    not of energy products;
  * every stage map is a scaled unitary (P_t P_t^H = 2^t I): the geometry of
    descent in the intermediate space is FIXED.  Momentum / acceleration
    persists across iterations — precisely what the OT plan updates destroy.

THE MEASURED ALIGNMENT LAW (the discovery this file demonstrates, printed
as a table below): give every level its own natural rigid-motion group — the
e x q lattice translations, exactly N motions at every level, equal degrees
of freedom — and measure how well two captures of the same content under
different settings (time offsets) align.  Then:
  * COMPLEX states decohere with depth (the 'dissimilar phase at the final
    representation point');
  * MAGNITUDES of the mixes align best at INTERIOR levels, and the advantage
    over the final products grows with how fast the content wanders —
    chirp at 156 ms offset: 0.85 aligned-cosine at level 2-3 vs 0.55 at the
    final FFT.  Stationary tones saturate (1.00) from mid levels on — final-
    product fusion is adequate exactly and only for content that needs no
    fusing.  A mid-level lattice translation has BOTH a time axis and a
    frequency axis, so it can absorb a diagonal (evolving) move that no pure
    frequency shift of a final spectrum can express.

THE METHOD: treat both spectrograms as magnitude observations of one latent
signal and align the MIXES rather than fuse the products:
  1. fast alternating dual-family projections (each family's magnitudes are
     enforced while the phases — the mixes — circulate between families),
     with over-relaxation momentum, valid because the operators never move;
  2. restarts ranked by the observable consistency loss (which tracks
     recovery quality — printed);
  3. a short amplitude-flow gradient polish with Nesterov momentum.
When the mixes agree, the final representation is a pure feed-forward DIP
finish — no descent in final FFT form.  The super-resolution product (here a
reassigned long-window spectrogram on the F1 x T2 grid) then falls out of
the recovered coherent signal; no barycenter of the magnitudes can reach it.

Also here, kept honest: soft-thresholding priors interleaved with the
projections — in the time, mid-level Zak, or final FFT domain, calibrated to
equal survival budgets — were tested and are NOT reliably helpful at these
observation densities (the prox machinery is kept for experimentation,
default off).  The working levers are redundancy, momentum, and restart
selection; the mid-level's proper role is the alignment law above, whose
operationalization (cross-frame consensus of mid-level magnitudes) is the
identified next step.

What magnitude-marginal fusion (OT included) cannot do even in principle:
distinguish INTERFERENCE from SILENCE.  The 100+120 Hz beating makes the
short-window spectrogram pulse at 20 Hz; any fusion of the two magnitude
distributions imprints those pulses on the product as false gaps.  The phase
of the aligned mixes knows the lumps are superposition, not modulation.

Run: .venv/bin/python experiments/dip_zak_fusion.py     (writes experiments/out/)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dip_numba import (dip_fft_stages, dip_fft_finish, dip_fft_backward,
                       dip_partial_forward)

# ----------------------------------------------------------------- problem --
FS = 2048
L = 2048                    # 1.0 s
N1, H1 = 1024, 128          # long window  : df = 2 Hz,  500 ms
N2, H2 = 64, 16             # short window : df = 32 Hz, 31 ms
TSTAR = 5                   # DIP alignment level: e = 32 rows, q = 32 cols
F1 = N1 // 2 + 1            # 513 target frequency bins (2 Hz)
PAUSE = (0.50, 0.55)        # 50 ms silence
CHIRP = (0.10, 0.90, 300.0, 380.0)   # linear chirp: wandering support


def make_signal():
    t = np.arange(L) / FS
    x = np.zeros(L)
    b_on = t < 0.50                               # 120 Hz: first half only
    x += np.sin(2 * np.pi * 100.0 * t)            # 100 Hz: whole second
    x += np.where(b_on, np.sin(2 * np.pi * 120.0 * t + 0.7), 0.0)
    t0, t1, f0, f1 = CHIRP
    on = (t >= t0) & (t < t1)
    rate = (f1 - f0) / (t1 - t0)
    ph = 2 * np.pi * (f0 * (t - t0) + 0.5 * rate * (t - t0) ** 2)
    x += 0.8 * np.where(on, np.sin(ph + 1.3), 0.0)
    x[(t >= PAUSE[0]) & (t < PAUSE[1])] = 0.0     # hard global pause
    return x


def frame_offsets(n, hop):
    return np.arange(0, L - n + 1, hop)


def stft_mag_full(x, n, hop):
    """Hann amplitude spectrogram on the FULL spectrum (n, nframes)."""
    offs = frame_offsets(n, hop)
    h = np.hanning(n)
    out = np.empty((n, len(offs)))
    for i, o in enumerate(offs):
        out[:, i] = np.abs(np.fft.fft(h * x[o:o + n]))
    return out


def ideal_map(times):
    """Ground-truth t-f energy map on the F1 x T2 grid (1-bin lines)."""
    P = np.zeros((F1, len(times)))
    for i, tt in enumerate(times):
        if PAUSE[0] <= tt < PAUSE[1]:
            continue
        P[50, i] += 1.0                                   # 100 Hz = bin 50
        if tt < 0.50:
            P[60, i] += 1.0                               # 120 Hz = bin 60
        t0, t1, f0, f1 = CHIRP
        if t0 <= tt < t1:
            fb = (f0 + (f1 - f0) * (tt - t0) / (t1 - t0)) / 2.0
            k = int(np.floor(fb))
            w = fb - k
            P[k, i] += 0.64 * (1 - w)
            P[k + 1, i] += 0.64 * w
    return P


# ------------------------------------------------------------------- proxes --
def soft(z, lam):
    az = np.abs(z)
    return np.where(az > lam, (1.0 - lam / (az + 1e-30)) * z, 0.0)


def prox_zak(u, off, lam):
    """Soft-threshold the DIP level-TSTAR state of one N1 tile (alignment in
    the intermediate space), exact backward to time domain."""
    tile = u[off:off + N1].astype(np.complex128)
    z = dip_partial_forward(tile, TSTAR, N1)
    z = soft(z, lam)
    u[off:off + N1] = np.real(dip_fft_backward(z, TSTAR, N1))
    return u


def prox_final(u, off, lam):
    """Same shrinkage, but in the FINAL spectrum domain (the energy-mixture
    endpoint) — the ablation contrast.  lam is a fraction of that domain's
    own max coefficient (fair calibration)."""
    tile = u[off:off + N1].astype(np.complex128)
    Z = np.fft.fft(tile)
    Z = soft(Z, lam)
    u[off:off + N1] = np.real(np.fft.ifft(Z))
    return u


def _apply_prox(u, it, lam_frac, prox):
    if prox == 'none' or lam_frac <= 0:
        return u
    fn = prox_zak if prox == 'zak' else prox_final
    # calibrate lam per tile against that domain's max coefficient
    for o in range((it % 2) * (N1 // 2), L - N1 + 1, N1):
        tile = u[o:o + N1].astype(np.complex128)
        ref = (np.abs(dip_partial_forward(tile, TSTAR, N1)).max()
               if prox == 'zak' else np.abs(np.fft.fft(tile)).max())
        u = fn(u, o, lam_frac * ref)
    return u


# ---------------------------------------------------------------- alignment --
def _gl_project(u, targf, h, n, offs):
    """One projection through one analysis family: enforce the observed
    magnitudes, keep the current phases, least-squares resynthesis."""
    acc = np.zeros(L)
    den = np.zeros(L)
    for i, o in enumerate(offs):
        Y = np.fft.fft(h * u[o:o + n])
        Y = targf[:, i] * Y / (np.abs(Y) + 1e-12)
        f = np.real(np.fft.ifft(Y))
        acc[o:o + n] += h * f
        den[o:o + n] += h * h
    return acc / np.maximum(den, 1e-6)


def consistency_loss(u, targ1, targ2, h1, h2, offs1, offs2):
    loss = 0.0
    for targ, h, n, offs in ((targ1, h1, N1, offs1), (targ2, h2, N2, offs2)):
        for i, o in enumerate(offs):
            loss += np.sum((np.abs(np.fft.fft(h * u[o:o + n]))
                            - targ[:, i]) ** 2)
    return loss


def gl_align(u, targ1, targ2, iters, beta=0.9, lam_frac0=0.003,
             prox='none', trace=None, hops=None):
    """Fast dual-family alignment: over-relaxed alternating magnitude
    projections + annealed shrinkage in the chosen domain."""
    hop1, hop2 = hops if hops else (H1, H2)
    h1, h2 = np.hanning(N1), np.hanning(N2)
    offs1, offs2 = frame_offsets(N1, hop1), frame_offsets(N2, hop2)
    up = u.copy()
    for it in range(iters):
        y = u + beta * (u - up)
        up = u
        v = _gl_project(y, targ1, h1, N1, offs1)
        u = _gl_project(v, targ2, h2, N2, offs2)
        lam = lam_frac0 * max(1.0 - it / (0.7 * iters), 0.0)
        u = _apply_prox(u, it, lam, prox)
        if trace is not None and it % 10 == 0:
            trace.append(consistency_loss(u, targ1, targ2, h1, h2,
                                          offs1, offs2))
    return u


def flow_polish(u, targ1, targ2, iters=200, momentum=0.85, hops=None):
    """Amplitude-flow gradient descent with Nesterov momentum (fixed
    operators: the acceleration state is never discarded)."""
    hop1, hop2 = hops if hops else (H1, H2)
    h1, h2 = np.hanning(N1), np.hanning(N2)
    offs1, offs2 = frame_offsets(N1, hop1), frame_offsets(N2, hop2)
    w2 = N1 / N2
    d1 = np.zeros(L)
    for o in offs1:
        d1[o:o + N1] += h1 ** 2
    d2 = np.zeros(L)
    for o in offs2:
        d2[o:o + N2] += h2 ** 2
    alpha = 0.5 / (2 * N1 * d1.max() + w2 * 2 * N2 * d2.max())
    up = u.copy()
    hist = []
    for it in range(iters):
        v = u + momentum * (u - up)
        g = np.zeros(L)
        loss = 0.0
        for targ, h, n, offs, w in ((targ1, h1, N1, offs1, 1.0),
                                    (targ2, h2, N2, offs2, w2)):
            for i, o in enumerate(offs):
                Y = np.fft.fft(h * v[o:o + n])
                aY = np.abs(Y)
                r = targ[:, i]
                loss += w * np.sum((aY - r) ** 2)
                res = (1.0 - r / (aY + 1e-12)) * Y
                g[o:o + n] += w * 2.0 * h * np.real(np.fft.ifft(res)) * n
        up = u
        u = v - alpha * g
        hist.append(loss)
    return u, np.array(hist)


def recover(targ1, targ2, prox='none', nseeds=16, iters1=400, iters2=1200,
            beta=0.9, verbose=False, hops=None):
    """Full pipeline: restart search -> refine top candidates -> polish.
    Restart ranking uses only the observable consistency loss."""
    hop1, hop2 = hops if hops else (H1, H2)
    h1, h2 = np.hanning(N1), np.hanning(N2)
    offs1, offs2 = frame_offsets(N1, hop1), frame_offsets(N2, hop2)

    cands = []
    for seed in range(nseeds):
        rng = np.random.default_rng(seed)
        u = 0.01 * rng.standard_normal(L)
        u = gl_align(u, targ1, targ2, iters1, beta=beta, prox=prox, hops=hops)
        cands.append((consistency_loss(u, targ1, targ2, h1, h2,
                                       offs1, offs2), seed, u))
    cands.sort(key=lambda c: c[0])
    if verbose:
        print("    restart losses:", " ".join(f"{c[0]:.1e}" for c in cands))

    best = None
    for lo, seed, u in cands[:3]:
        u = gl_align(u, targ1, targ2, iters2, beta=beta, prox=prox, hops=hops)
        u, _ = flow_polish(u, targ1, targ2, iters=200, hops=hops)
        lo = consistency_loss(u, targ1, targ2, h1, h2, offs1, offs2)
        if best is None or lo < best[0]:
            best = (lo, u)
    return best[1], best[0]


def snr_db(x, xh, cut=128):
    """Interior recovery SNR up to global sign (the first/last ~cut samples
    receive near-zero Hann weight in every frame: genuinely unobserved)."""
    xs, us = x[cut:-cut], xh[cut:-cut]
    e = min(np.sum((xs - us) ** 2), np.sum((xs + us) ** 2))
    return 10 * np.log10(np.sum(xs ** 2) / max(e, 1e-30))


# ------------------------------------------------------ final-form readouts --
def baseline_geomean(P1, P2, times1, times2, freqs1, freqs2):
    """Magnitude-marginal fusion baseline on F1 x T2: interpolate each
    observed POWER map to the target grid, geometric mean."""
    A = np.empty((F1, len(times2)))
    for k in range(F1):
        A[k] = np.interp(times2, times1, P1[k])
    B = np.empty((F1, len(times2)))
    for i in range(len(times2)):
        B[:, i] = np.interp(freqs1, freqs2, P2[:, i])
    return np.sqrt(np.maximum(A, 0) * np.maximum(B, 0))


def reassigned_spectrogram(x, times2):
    """Long-window reassigned spectrogram on the F1 x T2 grid — the 'no
    descent in final FFT form' product, computable only WITH the signal."""
    n = N1
    tt = np.arange(n) - (n - 1) / 2.0
    g = np.hanning(n)
    dg = np.gradient(g)
    tg = tt * g
    pad = n
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    P = np.zeros((F1, len(times2)))
    centers = (times2 * FS).astype(int)
    for i, c in enumerate(centers):
        a = c + pad - n // 2
        seg = xp[a:a + n]
        Y = np.fft.rfft(g * seg)
        Yt = np.fft.rfft(tg * seg)
        Yd = np.fft.rfft(dg * seg)
        E = np.abs(Y) ** 2
        good = E > 1e-6 * E.max()
        that = c + np.real(Yt[good] * np.conj(Y[good])) / E[good]
        khat = (np.arange(F1)[good]
                - np.imag(Yd[good] * np.conj(Y[good])) / E[good]
                * n / (2 * np.pi))
        cols = np.clip(np.round((that - centers[0]) / H2), 0,
                       len(times2) - 1).astype(int)
        rows = np.clip(np.round(khat), 0, F1 - 1).astype(int)
        np.add.at(P, (rows, cols), E[good])
    return P


def corrmap(P, Q):
    """Cosine similarity of sqrt-energy maps (dynamic-range compressed)."""
    a, b = np.sqrt(np.abs(P)).ravel(), np.sqrt(np.abs(Q)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------- the alignment law --
def best_lattice_cosine(A, B):
    """Max over ALL 2D circular lattice translations of |<A, roll(B)>| /
    (|A||B|).  Every level's lattice has e*q = N translations: equal degrees
    of freedom at every depth — the fair mutual-alignment game."""
    cc = np.abs(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
    return float(cc.max() / (np.linalg.norm(A) * np.linalg.norm(B)))


def measure_alignment_law(x, offset=320, m1=10):
    """Two captures of the same content under different settings (frames
    `offset` samples apart): per DIP level, the best lattice-aligned cosine
    of the complex states and of their magnitudes ('the mixes')."""
    t = np.arange(L) / FS
    h = np.hanning(N1)
    contents = {
        'paper mix': x,
        'two tones': np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 120 * t
                                                          + 0.7),
        'chirp': 0.8 * np.sin(2 * np.pi * (300 * t + 50 * t ** 2) + 1.3),
        'fast chirp': np.sin(2 * np.pi * (200 * t + 150 * t ** 2)),
    }
    law = {}
    for name, sig in contents.items():
        fa = (h * sig[256:256 + N1]).astype(np.complex128)
        fb = (h * sig[256 + offset:256 + offset + N1]).astype(np.complex128)
        pc, pm = [], []
        for lv in range(m1 + 1):
            e, q = 1 << lv, N1 >> lv
            za = dip_partial_forward(fa, lv, N1).reshape(e, q)
            zb = dip_partial_forward(fb, lv, N1).reshape(e, q)
            pc.append(best_lattice_cosine(za, zb))
            pm.append(best_lattice_cosine(np.abs(za), np.abs(zb)))
        law[name] = (pc, pm)
    return law


# --------------------------------------------------------------------- main --
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(outdir, exist_ok=True)

    x = make_signal()
    offs1, offs2 = frame_offsets(N1, H1), frame_offsets(N2, H2)
    times1 = (offs1 + N1 / 2) / FS
    times2 = (offs2 + N2 / 2) / FS
    freqs1 = np.arange(F1) * FS / N1
    freqs2 = np.arange(N2 // 2 + 1) * FS / N2

    # observations: the ONLY inputs to every method below
    targ1 = stft_mag_full(x, N1, H1)      # |STFT| long,  full spectrum
    targ2 = stft_mag_full(x, N2, H2)      # |STFT| short, full spectrum
    M1 = targ1[:F1] ** 2                  # power half-spectrograms, as in
    M2 = targ2[:N2 // 2 + 1] ** 2         # the paper
    ideal = ideal_map(times2)

    print("== DIP/Zak-space fusion vs magnitude-marginal fusion ==")
    print(f"  signal: {L / FS:.2f} s @ {FS} Hz | long {N1} (df=2Hz) "
          f"x{len(offs1)} frames | short {N2} (df=32Hz) x{len(offs2)} frames")
    print(f"  alignment level t*={TSTAR}: e={1 << TSTAR} rows (freq mod "
          f"{1 << TSTAR}), q={N1 >> TSTAR} cols")
    print(f"  100 Hz = bin 50 -> row {50 % (1 << TSTAR)}, 120 Hz = bin 60 -> "
          f"row {60 % (1 << TSTAR)}  (separated at level {TSTAR})")

    # ---- recovery: alignment of the mixes in the DIP mid-level state
    print("\n-- recovery (dual-family mix alignment, momentum, restarts) --")
    t0 = time.perf_counter()
    xh, loss = recover(targ1, targ2, verbose=True)
    t_rec = time.perf_counter() - t0
    print(f"  best consistency loss {loss:.3e} | interior SNR "
          f"{snr_db(x, xh):+.1f} dB | wall time {t_rec:.1f} s")

    # ---- the alignment law: where do captures under different settings
    # agree?  Equal DOF at every level: the e x q lattice translations.
    print("\n-- the alignment law: best lattice-aligned cosine per level --")
    law = measure_alignment_law(x)
    m1 = 10
    print("   level:           " + " ".join(f"{lv:5d}" for lv in range(m1 + 1)))
    for name, (pc, pm) in law.items():
        print(f"   {name:12s} cplx " + " ".join(f"{v:.3f}" for v in pc))
        print(f"   {'':12s} |mix|" + " ".join(f"{v:.3f}" for v in pm))
    print("   (|mix| = magnitude alignability; the interior advantage grows"
          " with how fast\n    the content wanders — final products decohere;"
          " stationary tones saturate)")

    # ---- momentum persistence (fixed geometry is what allows it)
    print("\n-- momentum persistence (fixed operators, unlike OT plans) --")
    rng = np.random.default_rng(1)
    u0 = 0.01 * rng.standard_normal(L)
    tr_m, tr_p = [], []
    gl_align(u0.copy(), targ1, targ2, 400, beta=0.9, prox='none', trace=tr_m)
    gl_align(u0.copy(), targ1, targ2, 400, beta=0.0, prox='none', trace=tr_p)
    print(f"  loss @400 aligned iters: momentum {tr_m[-1]:.3e} | "
          f"plain {tr_p[-1]:.3e}")

    # ---- final-form products (feed-forward only past this point)
    fr = (np.hanning(N1) * xh[offs1[1]:offs1[1] + N1]).astype(np.complex128)
    fin = dip_fft_finish(dip_partial_forward(fr, TSTAR, N1), TSTAR, N1)
    ref = dip_fft_stages(fr)[-1]
    print(f"\n  DIP finish from level {TSTAR} == final FFT (feed-forward): "
          f"max diff {np.max(np.abs(fin - ref)):.2e}")

    P_base = baseline_geomean(M1, M2, times1, times2, freqs1, freqs2)
    P_ours = reassigned_spectrogram(xh, times2)
    P_oracle = reassigned_spectrogram(x, times2)

    disp1 = np.empty((F1, len(times2)))
    for k in range(F1):
        disp1[k] = np.interp(times2, times1, M1[k])
    disp2 = np.empty((F1, len(times2)))
    for i in range(len(times2)):
        disp2[:, i] = np.interp(freqs1, freqs2, M2[:, i])

    c_base = corrmap(P_base, ideal)
    c_ours = corrmap(P_ours, ideal)
    print("\n-- product quality (cosine sim of sqrt-energy vs ideal map) --")
    print(f"  long-window alone      : {corrmap(disp1, ideal):.4f}")
    print(f"  short-window alone     : {corrmap(disp2, ideal):.4f}")
    print(f"  baseline geomean fusion: {c_base:.4f}")
    print(f"  DIP/Zak-space fusion   : {c_ours:.4f}")
    print(f"  oracle (true signal)   : {corrmap(P_oracle, ideal):.4f}")

    # ---- the mid-level exhibit: atoms + alignment
    e, q = 1 << TSTAR, N1 >> TSTAR
    frame_t = np.hanning(N1) * x[offs1[4]:offs1[4] + N1]
    frame_r = np.hanning(N1) * xh[offs1[4]:offs1[4] + N1]
    z_true = dip_partial_forward(frame_t.astype(np.complex128), TSTAR, N1)
    z_rec = dip_partial_forward(frame_r.astype(np.complex128), TSTAR, N1)

    # ---------------------------------------------------------------- plots
    def show(ax, P, title):
        Pm = np.sqrt(np.abs(P))
        ax.imshow(Pm, origin='lower', aspect='auto',
                  extent=[times2[0], times2[-1], 0, FS / 2],
                  cmap='magma', vmax=Pm.max())
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 450)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("freq (Hz)", fontsize=8)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    show(axes[0, 0], disp1, f"observed M1: long window "
                            f"{N1 / FS * 1e3:.0f} ms (freq sharp, time blur)")
    show(axes[0, 1], disp2, f"observed M2: short window "
                            f"{N2 / FS * 1e3:.0f} ms (time sharp, freq blur)")
    show(axes[0, 2], ideal, "ideal t-f map")
    show(axes[1, 0], P_base, f"baseline geomean fusion (corr {c_base:.3f})")
    show(axes[1, 1], P_ours, f"DIP/Zak-space fusion (corr {c_ours:.3f})")
    ax = axes[1, 2]
    ax.semilogy(10 * np.arange(len(tr_m)), tr_m, label='momentum 0.9')
    ax.semilogy(10 * np.arange(len(tr_p)), tr_p, label='no momentum',
                alpha=0.7)
    ax.set_title("alignment loss (fixed geometry -> momentum persists)",
                 fontsize=9)
    ax.set_xlabel("iteration", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'dip_zak_fusion.png'), dpi=130)

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, z, name in ((ax2[0], z_true, 'true frame'),
                        (ax2[1], z_rec, 'recovered from magnitudes only')):
        ax.imshow(np.abs(z.reshape(e, q)), origin='lower', aspect='auto',
                  cmap='magma')
        ax.set_title(f"|Zak state| level {TSTAR}, {name}", fontsize=9)
        ax.set_xlabel("j (fine time)", fontsize=8)
        ax.set_ylabel("delta (freq mod 32)", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, 'dip_zak_state.png'), dpi=130)

    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    lv = np.arange(m1 + 1)
    for name, (pc, pm) in law.items():
        ax3[0].plot(lv, pm, marker='o', ms=3, label=name)
        ax3[1].plot(lv, pc, marker='o', ms=3, label=name)
    ax3[0].set_title("|mix| alignability (magnitudes of the mixes)",
                     fontsize=9)
    ax3[1].set_title("complex-state alignability (phases decohere)",
                     fontsize=9)
    for ax in ax3:
        ax.set_xlabel("DIP level t  (0 = time ... 10 = final FFT)",
                      fontsize=8)
        ax.axvline(TSTAR, color='gray', ls=':', lw=0.8)
        ax.legend(fontsize=7)
    ax3[0].set_ylabel("best lattice-aligned cosine", fontsize=8)
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, 'dip_alignment_law.png'), dpi=130)
    print(f"\n  figures -> {outdir}/dip_zak_fusion.png, dip_zak_state.png, "
          f"dip_alignment_law.png")


if __name__ == "__main__":
    main()
