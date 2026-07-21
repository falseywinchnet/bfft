#!/usr/bin/env python3
"""Does the irfft-frame object denoise better than the ordinary STFT?

THE OBJECT.  np.fft.irfft applied to a REAL vector is a DCT-I: it treats
the vector as a half spectrum with zero imaginary part and returns an
even-symmetric signal of length 2(M-1).  Applied to a windowed time frame
of length N it gives 2(N-1) real samples, in which the phase geometry of
the frame appears as spatial structure rather than as a separate array.
The map is exact and its inverse is the forward transform:
``rfft(y).real`` recovers the frame (taking the real part is exactly the
projection onto the even component, so any modification of y is handled
without a separate symmetrization).

THE CLAIM UNDER TEST.  Laying phase out as geometry should let a
scale-selective 2-D decomposition -- which separates things with a shape
from things approaching noise -- do better than the same decomposition
applied to an ordinary spectrogram.

THE TEST.  Add white noise to speech at a known SNR, build a real 2-D
image in each domain, run the SAME decomposition with the SAME parameters,
drop the same number of fine layers, invert, and measure.  Domains:

  A   irfft   the object above, (T, 2(N-1))
  B1  logmag  log|X|, noisy phase reused -- the standard spectrogram
              denoising setup, and the strongest ordinary-domain
              comparison
  B2  reim    Re X and Im X decomposed independently, (T, N/2+1) each --
              the domain-B analogue that also carries phase

and a baseline that involves no decomposition at all:

  W   wiener  a Wiener gain from a noise floor estimated on the quietest
              frames, which is what anyone would reach for first

Run:  .venv/bin/python experiments/irfft_denoise_ab.py [--seconds 2.0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402

WAV = Path.home() / "Desktop" / "daveandsimon.wav"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

N = 512
HOP = 128


# --- STFT plumbing -------------------------------------------------------

def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def frames_of(x, n=N, hop=HOP):
    """Windowed analysis frames, (T, n)."""
    w = hann(n)
    t = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(t)[:, None]
    return x[idx] * w[None, :]


def overlap_add(fr, length, n=N, hop=HOP):
    """Weighted overlap-add with the same window; Hann squared at hop n/4
    satisfies COLA, so the normalization is exact."""
    w = hann(n)
    out = np.zeros(length)
    norm = np.zeros(length)
    for t in range(fr.shape[0]):
        s = t * hop
        out[s:s + n] += fr[t] * w
        norm[s:s + n] += w * w
    return out / np.maximum(norm, 1e-9)


def valid_span(length, t, n=N, hop=HOP):
    """The sample range fully covered by the analysis frames."""
    return slice(n, (t - 1) * hop)


# --- the two representations --------------------------------------------

def to_irfft(fr):
    """Frames -> the irfft object, (T, 2(N-1))."""
    return np.fft.irfft(fr, axis=1)


def from_irfft(obj):
    """The object -> frames.  Taking the real part of the forward
    transform is the projection onto the even component, which is the only
    part the object can carry."""
    return np.fft.rfft(obj, axis=1).real


def to_spec(fr):
    return np.fft.rfft(fr, axis=1)


def from_spec(X):
    return np.fft.irfft(X, n=N, axis=1)


# --- the shared decomposition -------------------------------------------

def layers(img, mu, passes=64, rung_sweeps=300):
    """Decompose a real image, coarse to fine.

    Returns a list [cartoon, band_coarse, band_mid, band_fine, residual]
    that sums to the input exactly, after affinely rescaling the image into
    the [0, 255] range the decomposition's parameters are set for."""
    lo = float(img.min())
    span = float(img.max()) - lo
    k = 255.0 / span if span > 1e-12 else 1.0
    y = (img - lo) * k
    cart, tex, b0, b1, b2 = bfft.meyer(y, mu=mu, passes=passes,
                                       rung_sweeps=rung_sweeps)
    resid = y - cart - b0 - b1 - b2
    return [cart, b0, b1, b2, resid], (lo, k)


def keep(ls, scale, drop):
    """Reassemble after dropping the `drop` finest layers."""
    lo, k = scale
    y = sum(ls[:len(ls) - drop]) if drop else sum(ls)
    return lo + y / k


# --- the four methods ----------------------------------------------------

#
# Each returns a closure over ONE decomposition, so a sweep over how many
# layers to drop costs nothing extra.

def run_irfft(noisy, mu, **kw):
    fr = frames_of(noisy)
    ls, sc = layers(to_irfft(fr), mu, **kw)
    return lambda drop: overlap_add(from_irfft(keep(ls, sc, drop)),
                                    len(noisy))


def run_logmag(noisy, mu, **kw):
    fr = frames_of(noisy)
    X = to_spec(fr)
    ls, sc = layers(np.log(np.abs(X) + 1e-6), mu, **kw)
    ph = np.exp(1j * np.angle(X))
    return lambda drop: overlap_add(
        from_spec(np.exp(keep(ls, sc, drop)) * ph), len(noisy))


def run_reim(noisy, mu, **kw):
    fr = frames_of(noisy)
    X = to_spec(fr)
    lr, sr = layers(X.real, mu, **kw)
    li, si = layers(X.imag, mu, **kw)
    return lambda drop: overlap_add(
        from_spec(keep(lr, sr, drop) + 1j * keep(li, si, drop)), len(noisy))


def run_wiener(noisy, floor_pct=10.0, over=1.0, **kw):
    """Wiener gain against a noise PSD taken from the quietest frames."""
    fr = frames_of(noisy)
    X = to_spec(fr)
    p = np.abs(X) ** 2
    energy = p.sum(1)
    quiet = p[energy <= np.percentile(energy, floor_pct)]
    npsd = quiet.mean(0) if len(quiet) else p.mean(0) * 0.1
    snr = np.maximum(p / (over * npsd[None, :] + 1e-12) - 1.0, 0.0)
    g = snr / (snr + 1.0)
    return overlap_add(from_spec(X * g), len(noisy))


# --- metrics -------------------------------------------------------------

def snr_db(est, ref):
    return 10 * np.log10((ref ** 2).sum() / (((est - ref) ** 2).sum() + 1e-30))


def seg_snr_db(est, ref, n=512, floor=-10.0, ceil=35.0):
    """Segmental SNR, clamped -- the standard speech figure, which does not
    let a few loud frames carry the whole score."""
    t = len(ref) // n
    vals = []
    for i in range(t):
        s = slice(i * n, (i + 1) * n)
        e = (ref[s] ** 2).sum()
        if e < 1e-9:
            continue
        v = 10 * np.log10(e / (((est[s] - ref[s]) ** 2).sum() + 1e-30))
        vals.append(min(max(v, floor), ceil))
    return float(np.mean(vals)) if vals else float("nan")


def lsd_db(est, ref):
    """Log-spectral distance over the analysis frames."""
    a = np.log10(np.abs(to_spec(frames_of(est))) + 1e-6)
    b = np.log10(np.abs(to_spec(frames_of(ref))) + 1e-6)
    return float(np.sqrt(((20 * (a - b)) ** 2).mean()))


# --- data ----------------------------------------------------------------

def load(seconds):
    from scipy.io import wavfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, x = wavfile.read(str(WAV))
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = x[:int(seconds * sr)]
    return sr, x / (np.abs(x).max() + 1e-12)


def add_noise(x, snr, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x))
    n *= np.sqrt((x ** 2).mean() / (n ** 2).mean()) * 10 ** (-snr / 20.0)
    return x + n


def figure(seconds=2.0, snr=0.0, mu=40.0, drop=2):
    """Spectrograms and the object itself, plus wav files to listen to."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.io import wavfile

    sr, clean = load(seconds)
    noisy = add_noise(clean, snr)
    outs = {"clean": clean, "noisy": noisy,
            "wiener": run_wiener(noisy)[:len(clean)],
            "irfft": run_irfft(noisy, mu)(drop)[:len(clean)],
            "reim": run_reim(noisy, mu)(drop)[:len(clean)]}
    for name, y in outs.items():
        p = OUT / f"irfft_ab_{int(snr)}dB_{name}.wav"
        wavfile.write(str(p), sr,
                      (np.clip(y, -1, 1) * 32767).astype(np.int16))
    print("wrote wavs to", OUT)

    t = 1 + (len(clean) - N) // HOP
    sl = valid_span(len(clean), t)      # the OLA edges are not reconstructed

    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6))
    for c, (name, y) in enumerate(outs.items()):
        S = 20 * np.log10(np.abs(to_spec(frames_of(y))) + 1e-4)
        axes[0, c].imshow(S.T, origin="lower", aspect="auto", vmin=-60,
                          vmax=20, cmap="magma")
        lab = name if name == "clean" else \
            f"{name}  {snr_db(y[sl], clean[sl]):.2f} dB  " \
            f"LSD {lsd_db(y, clean):.1f}"
        axes[0, c].set_title(lab, fontsize=9)
        axes[0, c].set_xticks([])

    # Row 2: how the ladder actually splits the noisy object -- the claim
    # under test is that shape separates from noise by scale.
    ls, _ = layers(to_irfft(frames_of(noisy)), mu)
    names = ("cartoon", "band coarse", "band mid", "band fine", "residual")
    for c, (a, nm) in enumerate(zip(ls, names)):
        v = a[:, :64]
        s = 2.5 * v.std() + 1e-9
        axes[1, c].imshow(v.T, origin="lower", aspect="auto", cmap="gray",
                          vmin=v.mean() - s, vmax=v.mean() + s)
        axes[1, c].set_title(f"{nm}  (rms {v.std():.2f})", fontsize=8)
        axes[1, c].set_xticks([])
    axes[0, 0].set_ylabel("STFT log-magnitude", fontsize=9)
    axes[1, 0].set_ylabel("noisy object, by scale\n(first 64 bins)",
                          fontsize=9)
    fig.suptitle(f"Denoising at {snr:.0f} dB input: Wiener vs the "
                 f"decomposition in two domains (mu={mu:.0f}, drop={drop})",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / f"irfft_denoise_ab_{int(snr)}dB.png"
    fig.savefig(p, dpi=110)
    print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--fig-snr", type=float, default=0.0)
    ap.add_argument("--fig-mu", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--snrs", type=float, nargs="+", default=[10.0, 5.0, 0.0])
    ap.add_argument("--mus", type=float, nargs="+", default=[40.0])
    ap.add_argument("--drops", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args(argv)

    if args.figure:
        figure(args.seconds, args.fig_snr, args.fig_mu)
        return 0

    sr, clean = load(args.seconds)
    t = 1 + (len(clean) - N) // HOP
    sl = valid_span(len(clean), t)
    print(f"{WAV.name}: {sr} Hz, {len(clean)} samples "
          f"({len(clean) / sr:.2f} s), {t} frames, scoring on {sl}")

    methods = [("A  irfft ", run_irfft),
               ("B1 logmag", run_logmag),
               ("B2 reim  ", run_reim)]

    for snr in args.snrs:
        noisy = add_noise(clean, snr)
        base = (snr_db(noisy[sl], clean[sl]),
                seg_snr_db(noisy[sl], clean[sl]), lsd_db(noisy, clean))
        print(f"\n=== input SNR {snr:.0f} dB "
              f"(measured {base[0]:.2f} dB, seg {base[1]:.2f}, "
              f"LSD {base[2]:.2f}) ===")
        w = run_wiener(noisy)[:len(clean)]
        print(f"{'W  wiener':10s} {'-':>4s} {'-':>4s} "
              f"{snr_db(w[sl], clean[sl]):8.2f} "
              f"{snr_db(w[sl], clean[sl]) - base[0]:+7.2f} "
              f"{seg_snr_db(w[sl], clean[sl]):8.2f} "
              f"{lsd_db(w, clean):7.2f}")
        print(f"{'method':10s} {'mu':>4s} {'drp':>4s} {'SNR':>8s} "
              f"{'gain':>7s} {'segSNR':>8s} {'LSD':>7s} {'time':>7s}")
        for name, fn in methods:
            for mu in args.mus:
                t0 = time.perf_counter()
                try:
                    rec = fn(noisy, mu)
                except Exception as exc:
                    print(f"{name:10s} {mu:4.0f}  FAILED "
                          f"{type(exc).__name__}: {exc}")
                    continue
                dt = time.perf_counter() - t0
                for drop in args.drops:
                    y = rec(drop)[:len(clean)]
                    s = snr_db(y[sl], clean[sl])
                    print(f"{name:10s} {mu:4.0f} {drop:4d} {s:8.2f} "
                          f"{s - base[0]:+7.2f} "
                          f"{seg_snr_db(y[sl], clean[sl]):8.2f} "
                          f"{lsd_db(y, clean):7.2f} "
                          f"{dt if drop == args.drops[0] else 0.0:6.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
