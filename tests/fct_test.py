#!/usr/bin/env python3
"""FCT (Fast Correlated Transform) validation.

Checks the native kernel against first principles and against the brute-force
leading-edge manifold, then exercises both STFT integrations:

  1. EXACTNESS INVARIANT: for every bin, the emitted value must equal the
     direct prefix correlation C(k, tau_k) = sum_{t<tau_k} x[t] e^{-2pi i k t/N}
     at the emitted slice -- independent of which slice the selector chose.
  2. DEFAULT INVARIANT: inactive bins carry tau = N and the plain rfft bin.
  3. SELECTION QUALITY: on the reference event signal, the score at the
     chosen slice must reach the brute-force maximum (coherent bins), and the
     planted event boundaries must be recovered.
  4. C STFT PATH: bfft.STFTPlan(transform="fct") -- shape, per-frame
     consistency with FctPlan, istft rejection.
  5. PYTHON/NUMBA STFT PATH: stft.STFT(transform="fct") matches the C STFT
     path bit-for-bit framing; istft raises.

Run from the repo root:  .venv/bin/python tests/fct_test.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402

rng = np.random.default_rng(11)
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def prefix_correlation(x, k, tau):
    n = len(x)
    t = np.arange(tau)
    return np.sum(x[:tau] * np.exp(-2j * np.pi * k * t / n))


def make_events(N):
    n = np.arange(N)
    x = 0.8 * np.cos(2 * np.pi * 100 * n / N + 0.7)
    b1 = (n >= 200) & (n < 500)
    x[b1] += 1.0 * np.cos(2 * np.pi * 300.4 * n[b1] / N + 1.1)
    b2 = (n >= 600) & (n < 656)
    x[b2] += 1.2 * np.cos(2 * np.pi * 421 * n[b2] / N + 2.0)
    x += 1e-3 * rng.standard_normal(N)
    return x


# ---------------------------------------------------------------- 1 + 2
print("1/2. exactness + default invariants")
for N in (64, 256, 1024):
    for trial in range(3):
        x = rng.standard_normal(N) if trial == 0 else (
            make_events(N) if N == 1024 else rng.standard_normal(N) +
            np.cos(2 * np.pi * (N // 8) * np.arange(N) / N))
        C, tau = bfft.fct(x)
        assert C.shape == (N // 2 + 1,) and tau.shape == (N // 2 + 1,)
        worst = 0.0
        for k in range(N // 2 + 1):
            ref = prefix_correlation(x, k, int(tau[k]))
            worst = max(worst, abs(C[k] - ref) / (abs(ref) + 1.0))
        check(f"exactness N={N} trial={trial}", worst < 1e-9,
              f"worst rel err {worst:.2e}")
        full = bfft.rfft(x)
        dflt = np.flatnonzero(tau == N)
        err_dflt = np.max(np.abs(C[dflt] - full[dflt])) if len(dflt) else 0.0
        check(f"defaults N={N} trial={trial}", err_dflt < 1e-9,
              f"{len(dflt)} default bins, err {err_dflt:.2e}")

# ---------------------------------------------------------------- 3
print("3. selection quality vs brute-force manifold (events, N=1024)")
N = 1024
x = make_events(N)
C, tau = bfft.fct(x)
n_idx = np.arange(N)
kk = np.arange(N // 2 + 1)
U = x[None, :] * np.exp(-2j * np.pi * np.outer(kk, n_idx) / N)
Cbf = np.cumsum(U, axis=1)
tt = np.arange(1, N + 1, dtype=float)
S = np.abs(Cbf) ** 2 / tt[None, :]
tstar = np.argmax(S, axis=1) + 1
smax = S[kk, tstar - 1]
coh = np.flatnonzero(smax > 8.0 * np.mean(x ** 2))
coh = coh[(coh > 0) & (coh < N // 2)]
s_found = np.abs(C[coh]) ** 2 / tau[coh]
ratio = s_found / smax[coh]
check("coherent bins found the max slice",
      np.median(ratio) > 0.999 and np.percentile(ratio, 10) > 0.95,
      f"{len(coh)} bins, ratio med {np.median(ratio):.4f} "
      f"p10 {np.percentile(ratio, 10):.4f}")
check("full tone bin 100 -> tau = N", tau[100] == N, f"tau {tau[100]}")
check("burst bin 300 boundary ~502",
      abs(int(tau[300]) - int(tstar[300])) <= 2,
      f"tau {tau[300]} vs brute {tstar[300]}")
check("short burst bin 421 boundary ~661",
      abs(int(tau[421]) - int(tstar[421])) <= 2,
      f"tau {tau[421]} vs brute {tstar[421]}")
phase = np.abs(np.angle(C[coh] * np.conj(Cbf[coh, tau[coh] - 1])))
check("emitted phase = arctan(A/B) at slice", np.max(phase) < 1e-6,
      f"max {np.degrees(np.max(phase)):.2e} deg")

# Complex-IQ selection must be shared by I and Q.  Verify all wrapped bins,
# including negative frequencies, directly against the emitted tau.
print("3b. complex-IQ exactness at shared selected slices")
N = 256
t = np.arange(N)
xc = (0.8 * np.exp(2j * np.pi * 37.25 * t / N) +
      0.6 * (t < 143) * np.exp(-2j * np.pi * 61 * t / N + 0.4j) +
      1e-3 * (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
Cc, tauc = bfft.FctPlan(N).fct_complex(xc)
worst = 0.0
for k in range(N):
    ref = prefix_correlation(xc, k, int(tauc[k]))
    worst = max(worst, abs(Cc[k] - ref) / (abs(ref) + 1.0))
check("complex full-spectrum exactness", worst < 1e-9,
      f"worst rel err {worst:.2e}")
check("negative-frequency bins can adapt", bool(np.any(tauc[N // 2 + 1:] < N)),
      f"adaptive negative bins {(tauc[N // 2 + 1:] < N).sum()}")

Cm, taum, Mm = bfft.FctPlan(N).fct_complex_moment(xc)
moment_err = 0.0
for k in range(N):
    tt_m = np.arange(int(taum[k]))
    ref_m = np.sum(tt_m * xc[:len(tt_m)] *
                   np.exp(-2j * np.pi * k * tt_m / N))
    moment_err = max(moment_err,
                     abs(Mm[k] - ref_m) / (abs(ref_m) + 1.0))
check("selected-support phase moment exact", moment_err < 1e-9,
      f"worst rel err {moment_err:.2e}")
check("moment call preserves selection",
      np.array_equal(taum, tauc) and np.max(np.abs(Cm - Cc)) < 1e-9)

# The public explicit-plan path must carry t_min into the certified walk.
min_tau = 37
Cc_min, tauc_min = bfft.FctPlan(N, t_min=min_tau).fct_complex(xc)
S_min = np.empty((N, N - min_tau + 1))
for k in range(N):
    prefixes = np.cumsum(
        xc * np.exp(-2j * np.pi * k * np.arange(N) / N))
    S_min[k] = np.abs(prefixes[min_tau - 1:]) ** 2 / np.arange(min_tau, N + 1)
tstar_min = np.argmax(S_min, axis=1) + min_tau
check("explicit t_min domain is exact", np.array_equal(tauc_min, tstar_min),
      f"minimum emitted tau {tauc_min.min()}")

# ---------------------------------------------------------------- 4
print("4. C STFT path (bfft.STFTPlan transform='fct')")
n_sig, n_fft, hop = 4096, 512, 128
sig = rng.standard_normal(n_sig)
sig += np.cos(2 * np.pi * 60 * np.arange(n_sig) / n_fft)
sp = bfft.STFTPlan(n=n_sig, n_fft=n_fft, hop_length=hop, transform="fct")
Z = sp.stft(sig)
check("stft shape", Z.shape == (n_fft // 2 + 1, n_sig // hop),
      f"{Z.shape}")
check("stft finite", bool(np.all(np.isfinite(Z))))
try:
    sp.istft(Z)
    check("istft rejected (forward-only)", False)
except ValueError:
    check("istft rejected (forward-only)", True)

# per-frame consistency: frame 8 of the C path == FctPlan of the same
# windowed natural-order segment (reflect pad + window replicated here)
win = np.hanning(n_fft)
half = n_fft // 2
xp = np.pad(sig, (half, half - 1), mode="reflect")
seg = xp[8 * hop: 8 * hop + n_fft] * win
Cf, _ = bfft.FctPlan(n_fft).fct(seg)
check("frame == FctPlan(windowed segment)",
      np.max(np.abs(Z[:, 8] - Cf)) < 1e-9,
      f"max err {np.max(np.abs(Z[:, 8] - Cf)):.2e}")

# ---------------------------------------------------------------- 5
print("5. Python/numba STFT path (stft.STFT transform='fct')")
from stft import STFT  # noqa: E402

tf = STFT(n=n_sig, n_fft=n_fft, hop_length=hop, transform="fct")
Zp = tf.stft(sig)
check("numba path matches C path",
      np.max(np.abs(Zp - Z)) < 1e-9,
      f"max err {np.max(np.abs(Zp - Z)):.2e}")
try:
    tf.istft(Zp)
    check("python istft rejected (forward-only)", False)
except ValueError:
    check("python istft rejected (forward-only)", True)

# rfft path still intact after the wiring (skip the documented opening
# transient; steady-state reconstruction is exact)
tf2 = STFT(n=n_sig, n_fft=n_fft, hop_length=hop, transform="rfft")
xr, _ = tf2.istft(tf2.stft(sig))
lat = n_fft // 2
err = np.max(np.abs(xr[lat:] - sig[: len(xr) - lat])[2 * n_fft:])
check("rfft STFT roundtrip unaffected (steady state)", err < 1e-10,
      f"err {err:.2e}")

print("=" * 60)
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL FCT TESTS PASS")
