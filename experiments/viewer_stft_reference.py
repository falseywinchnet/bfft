#!/usr/bin/env python3
"""Reproduce the STFT geometry behind paused_suffix_superresolution.png.

This is the compact, internal reference used to audit the IQ viewer.  It
recreates the first-second Dave-and-Simon source maps and the reassigned oracle:

* 48 kHz mono -> 8192 Hz, one second;
* long Hann N=1024/H=128 (fine frequency);
* short Hann N=128/H=32 (fine time);
* long-frequency x short-center display lattice;
* long-window reassignment for the phase-aware F1 x T2 readout.

The expected cosine fingerprint is 0.720767 / 0.492446 / 0.653397 for
long / short / blind geomean against the reassigned oracle.
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_first_second(path, work_sr=8192):
    x, sr0 = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.asarray(x[:sr0], np.float64)
    x -= np.mean(x)
    if sr0 != work_sr:
        g = gcd(int(work_sr), int(sr0))
        x = resample_poly(x, work_sr // g, sr0 // g)
    x = np.pad(x[:work_sr], (0, max(0, work_sr - len(x))))
    x *= 0.95 / (np.max(np.abs(x)) + 1e-12)
    return np.ascontiguousarray(x), int(work_sr), int(sr0)


def stft_power(x, n, hop):
    offsets = np.arange(0, len(x) - n + 1, hop, dtype=np.int64)
    frames = x[offsets[:, None] + np.arange(n)[None, :]]
    spectrum = np.fft.rfft(frames * np.hanning(n), axis=1)
    return np.abs(spectrum.T) ** 2, offsets


def interp_rows(values, source, target):
    return np.stack([np.interp(target, source, row, left=0.0, right=0.0)
                     for row in values])


def interp_frequency(values, source, target):
    return np.stack([np.interp(target, source, values[:, col],
                               left=0.0, right=0.0)
                     for col in range(values.shape[1])], axis=1)


def reassigned_power_real(x, n, centers, hop):
    bins = n // 2 + 1
    g = np.hanning(n)
    dg = np.gradient(g)
    tg = (np.arange(n) - (n - 1) / 2.0) * g
    xp = np.concatenate([np.zeros(n), x, np.zeros(n)])
    out = np.zeros((bins, len(centers)), np.float64)
    first = int(centers[0])
    for col, center in enumerate(centers):
        a = int(center) + n - n // 2
        segment = xp[a:a + n]
        y = np.fft.rfft(g * segment)
        yt = np.fft.rfft(tg * segment)
        yd = np.fft.rfft(dg * segment)
        energy = np.abs(y) ** 2
        good = energy > 1e-8 * (energy.max() + 1e-30)
        that = center + np.real(yt[good] * np.conj(y[good])) / (energy[good] + 1e-30)
        khat = (np.arange(bins)[good] -
                np.imag(yd[good] * np.conj(y[good])) /
                (energy[good] + 1e-30) * n / (2 * np.pi))
        cc = np.clip(np.rint((that - first) / hop), 0, len(centers) - 1).astype(int)
        rr = np.clip(np.rint(khat), 0, bins - 1).astype(int)
        np.add.at(out, (rr, cc), energy[good])
    return out


def cosine_power(a, b):
    aa = np.sqrt(np.maximum(a, 0.0)).astype(np.float64, copy=False).ravel()
    bb = np.sqrt(np.maximum(b, 0.0)).astype(np.float64, copy=False).ravel()
    return float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-30))


def show(ax, power, title, times, frequencies):
    amplitude = np.sqrt(np.maximum(power, 0.0))
    vmax = np.percentile(amplitude, 99.5)
    ax.imshow(amplitude, origin="lower", aspect="auto", cmap="viridis",
              extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
              vmax=vmax)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    ax.set_title(title)


def run(wav, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x, sr, sr0 = load_first_second(wav)
    long_power, long_offsets = stft_power(x, 1024, 128)
    short_power, short_offsets = stft_power(x, 128, 32)
    long_times = (long_offsets + 512) / sr
    short_times = (short_offsets + 64) / sr
    long_freqs = np.arange(513) * sr / 1024
    short_freqs = np.arange(65) * sr / 128
    long_map = interp_rows(long_power, long_times, short_times)
    short_map = interp_frequency(short_power, short_freqs, long_freqs)
    geomean = np.sqrt(np.maximum(long_map, 0.0) * np.maximum(short_map, 0.0))
    centers = short_offsets + 64
    oracle = reassigned_power_real(x, 1024, centers, 32)

    # Exercise the exact backend used by the IQ viewer on the same geometry.
    # A real signal is represented as complex IQ with Q=0; after undoing the
    # viewer's fftshift, its nonnegative half must equal the real oracle.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "viewer"))
    import iqwaterfall as iqw  # noqa: E402
    iq = np.empty(2 * len(x), np.float32)
    iq[0::2] = x
    iq[1::2] = 0.0
    native_engine = iqw.Reassign(1024)
    native_db = native_engine.render_mem(iq, int(centers[0]), 32, len(centers))
    native_engine.close()
    native = np.fft.ifftshift(10.0 ** (native_db / 10.0), axes=1)
    native = native[:, :513].T
    native_corr = cosine_power(native, oracle)
    native_peak_relative_error = float(
        np.max(np.abs(native - oracle)) / (np.max(oracle) + 1e-30))
    metrics = {
        "source_rate": sr0,
        "work_rate": sr,
        "samples": len(x),
        "shape": list(oracle.shape),
        "long_corr": cosine_power(long_map, oracle),
        "short_corr": cosine_power(short_map, oracle),
        "geomean_corr": cosine_power(geomean, oracle),
        "native_oracle_corr": native_corr,
        "native_peak_relative_error": native_peak_relative_error,
    }
    expected = np.array([0.7207673617639888, 0.49244556704981524,
                         0.6533974476591363])
    got = np.array([metrics["long_corr"], metrics["short_corr"],
                    metrics["geomean_corr"]])
    assert np.max(np.abs(got - expected)) < 2e-9, (got, expected)
    assert native_corr > 0.9999999, native_corr
    assert native_peak_relative_error < 1e-5, native_peak_relative_error

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    show(axes[0, 0], long_map, f"long source {got[0]:.3f}", short_times,
         long_freqs)
    show(axes[0, 1], short_map, f"short source {got[1]:.3f}", short_times,
         long_freqs)
    show(axes[1, 0], geomean, f"geomean {got[2]:.3f}", short_times,
         long_freqs)
    show(axes[1, 1], oracle, "phase-aware reassigned readout", short_times,
         long_freqs)
    fig.tight_layout()
    png = outdir / "viewer_stft_reference.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(outdir / "reference_maps.npz", long=long_map,
                        short=short_map, geomean=geomean, oracle=oracle,
                        native_oracle=native,
                        times=short_times, frequencies=long_freqs)
    print(json.dumps(metrics, indent=2))
    print(png)
    return metrics, png


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="/Users/quentinkuttenkuler/Desktop/daveandsimon.wav")
    parser.add_argument("--outdir", default="/tmp/bfft_viewer_stft_reference")
    args = parser.parse_args()
    run(args.wav, args.outdir)
