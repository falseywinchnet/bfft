#!/usr/bin/env python3
"""
Stress-test rfft-STFT vs odft-STFT reconstruction behavior.

Run from repo root after installing the package:

    python experiments/stft_odft_reconstruction_stress.py

It imports ./stft.py and compares:
    1. identity STFT -> ISTFT reconstruction
    2. soft spectral gate
    3. random complex mask
    4. lowpass-style bin truncation
    5. alternating-bin comb mask
    6. frame dropout
    7. phase-only reconstruction
    8. magnitude-only reconstruction

The useful columns are:
    steady_snr_db
    steady_rel_l2
    steady_max_abs
    full_snr_db
    full_rel_l2
    edge_snr_db

steady_* ignores the expected opening transient documented by stft.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

import importlib.util

# Load loose repo-root stft.py directly. It is not a package module.
HERE = os.path.abspath(os.path.dirname(__file__))
CANDIDATES = [
    os.path.join(HERE, "stft.py"),
    os.path.join(os.path.dirname(HERE), "stft.py"),
]

STFT_PATH = next((x for x in CANDIDATES if os.path.exists(x)), None)
if STFT_PATH is None:
    raise FileNotFoundError("Could not find stft.py next to sbench.py or one directory above it")

REPO_ROOT = os.path.dirname(STFT_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

spec = importlib.util.spec_from_file_location("repo_stft_file", STFT_PATH)
repo_stft = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(repo_stft)

STFT = repo_stft.STFT


def db20(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-300))


def snr_db(ref: np.ndarray, err: np.ndarray) -> float:
    return db20(np.linalg.norm(ref) / max(np.linalg.norm(err), 1e-300))


def rel_l2(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.linalg.norm(test - ref) / max(np.linalg.norm(ref), 1e-300))


def max_abs(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.max(np.abs(test - ref)))


def crest_db(x: np.ndarray) -> float:
    return db20(np.max(np.abs(x)) / max(np.sqrt(np.mean(x * x)), 1e-300))


def hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float64)


def sqrt_hann(n: int) -> np.ndarray:
    return np.sqrt(np.maximum(np.hanning(n), 0.0)).astype(np.float64)


def blackman_harris_4(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    p = 2.0 * np.pi * i / (n - 1)
    return (a0 - a1 * np.cos(p) + a2 * np.cos(2 * p) - a3 * np.cos(3 * p)).astype(np.float64)


def make_signal(kind: str, n: int, fs: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / fs

    if kind == "white":
        x = rng.standard_normal(n)

    elif kind == "colored":
        w = rng.standard_normal(n)
        X = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        tilt = 1.0 / np.sqrt(1.0 + f)
        x = np.fft.irfft(X * tilt, n)

    elif kind == "tones_random":
        x = np.zeros(n, dtype=np.float64)
        for _ in range(64):
            f = rng.uniform(10.0, 0.49 * fs)
            a = 10.0 ** rng.uniform(-2.0, 0.0)
            ph = rng.uniform(0.0, 2.0 * np.pi)
            x += a * np.cos(2.0 * np.pi * f * t + ph)

    elif kind == "tones_integer":
        x = np.zeros(n, dtype=np.float64)
        frame_n = 1024
        for k in range(1, frame_n // 2, 19):
            f = k * fs / frame_n
            x += (1.0 / math.sqrt(k)) * np.cos(2.0 * np.pi * f * t)

    elif kind == "tones_half":
        x = np.zeros(n, dtype=np.float64)
        frame_n = 1024
        for k in range(1, frame_n // 2, 19):
            f = (k + 0.5) * fs / frame_n
            x += (1.0 / math.sqrt(k)) * np.cos(2.0 * np.pi * f * t)

    elif kind == "chirps":
        x = np.zeros(n, dtype=np.float64)
        for f0, f1, amp in [(40, 0.45 * fs, 0.8), (0.43 * fs, 80, 0.5), (400, 3000, 0.3)]:
            rate = (f1 - f0) / max(t[-1], 1e-12)
            phase = 2.0 * np.pi * (f0 * t + 0.5 * rate * t * t)
            x += amp * np.cos(phase)

    elif kind == "impulses":
        x = 0.02 * rng.standard_normal(n)
        count = max(1, n // 2048)
        idx = rng.integers(0, n, size=count)
        x[idx] += rng.choice([-1.0, 1.0], size=count) * rng.uniform(5.0, 20.0, size=count)

    elif kind == "drift":
        x = 0.4 * np.sin(2.0 * np.pi * 0.7 * t)
        x += 0.2 * np.sin(2.0 * np.pi * 3.1 * t)
        x += 0.02 * rng.standard_normal(n)

    else:
        raise ValueError(kind)

    x = x.astype(np.float64)
    x -= np.mean(x)
    rms = np.sqrt(np.mean(x * x))
    if rms > 0:
        x /= rms
    return x


def apply_edit(Z: np.ndarray, edit: str, rng: np.random.Generator) -> np.ndarray:
    Y = Z.copy()
    bins, frames = Y.shape

    if edit == "identity":
        return Y

    if edit == "soft_gate":
        mag = np.abs(Y)
        med = np.median(mag, axis=0, keepdims=True)
        gain = mag * mag / (mag * mag + (1.5 * med) ** 2 + 1e-300)
        return Y * gain

    if edit == "random_smooth_mask":
        raw = rng.uniform(0.0, 1.0, size=(bins, frames))
        for _ in range(4):
            raw[1:-1, :] = 0.25 * raw[:-2, :] + 0.5 * raw[1:-1, :] + 0.25 * raw[2:, :]
            raw[:, 1:-1] = 0.25 * raw[:, :-2] + 0.5 * raw[:, 1:-1] + 0.25 * raw[:, 2:]
        return Y * raw

    if edit == "random_complex_mask":
        mag = rng.uniform(0.0, 1.0, size=(bins, frames))
        ph = rng.uniform(-0.25 * np.pi, 0.25 * np.pi, size=(bins, frames))
        return Y * mag * np.exp(1j * ph)

    if edit == "lowpass_25pct":
        cut = max(1, bins // 4)
        Y[cut:, :] = 0.0
        return Y

    if edit == "highpass_25pct":
        cut = max(1, bins // 4)
        Y[:cut, :] = 0.0
        return Y

    if edit == "comb_even_bins":
        Y[::2, :] = 0.0
        return Y

    if edit == "comb_odd_bins":
        Y[1::2, :] = 0.0
        return Y

    if edit == "frame_dropout":
        Y[:, ::5] = 0.0
        return Y

    if edit == "phase_only":
        return np.exp(1j * np.angle(Y))

    if edit == "magnitude_only":
        return np.abs(Y).astype(np.complex128)

    raise ValueError(edit)


@dataclass
class Metrics:
    n: int
    n_fft: int
    hop: int
    window: str
    transform: str
    signal: str
    edit: str
    bins: int
    frames: int
    stft_ms: float
    istft_ms: float
    full_snr_db: float
    full_rel_l2: float
    full_max_abs: float
    steady_snr_db: float
    steady_rel_l2: float
    steady_max_abs: float
    edge_snr_db: float
    output_rms: float
    output_crest_db: float


def run_case(n: int, n_fft: int, hop: int, window_name: str, transform: str, signal: str, edit: str, fs: float, seed: int) -> Metrics:
    if window_name == "hann":
        win = hann(n_fft)
    elif window_name == "sqrt_hann":
        win = sqrt_hann(n_fft)
    elif window_name == "blackman_harris_4":
        win = blackman_harris_4(n_fft)
    else:
        raise ValueError(window_name)

    x = make_signal(signal, n, fs, seed)
    tf = STFT(n=n, n_fft=n_fft, hop_length=hop, window=win, transform=transform)

    t0 = time.perf_counter()
    Z = tf.stft(x)
    t1 = time.perf_counter()

    rng = np.random.default_rng(seed + 999)
    Y = apply_edit(Z, edit, rng)

    t2 = time.perf_counter()
    y, _ = tf.istft(Y)
    t3 = time.perf_counter()

    m = min(len(x), len(y))
    half = n_fft // 2

    # stft.py is a streaming centered STFT. ISTFT output is delayed by n_fft//2.
    # Compare y[half:] against x[:len(y)-half], then ignore an additional guard
    # region for steady-state metrics.
    if m <= half + 8:
        raise RuntimeError("signal too short for latency-compensated scoring")

    test_full = y[half:m]
    ref_full = x[:test_full.shape[0]]

    guard = min(n_fft, max(0, test_full.shape[0] // 4))
    start_i = guard
    stop_i = test_full.shape[0] - guard
    if stop_i <= start_i:
        start_i = 0
        stop_i = test_full.shape[0]

    ref_steady = ref_full[start_i:stop_i]
    y_steady = test_full[start_i:stop_i]

    err_full = test_full - ref_full
    err_steady = y_steady - ref_steady

    edge_len = min(n_fft, test_full.shape[0] // 2)
    edge_ref = np.concatenate([ref_full[:edge_len], ref_full[-edge_len:]])
    edge_y = np.concatenate([test_full[:edge_len], test_full[-edge_len:]])
    edge_err = edge_y - edge_ref

    return Metrics(
        n=n,
        n_fft=n_fft,
        hop=hop,
        window=window_name,
        transform=transform,
        signal=signal,
        edit=edit,
        bins=Z.shape[0],
        frames=Z.shape[1],
        stft_ms=(t1 - t0) * 1000.0,
        istft_ms=(t3 - t2) * 1000.0,
        full_snr_db=snr_db(ref_full, err_full),
        full_rel_l2=rel_l2(ref_full, test_full),
        full_max_abs=max_abs(ref_full, test_full),
        steady_snr_db=snr_db(ref_steady, err_steady),
        steady_rel_l2=rel_l2(ref_steady, y_steady),
        steady_max_abs=max_abs(ref_steady, y_steady),
        edge_snr_db=snr_db(edge_ref, edge_err),
        output_rms=float(np.sqrt(np.mean(test_full * test_full))),
        output_crest_db=crest_db(test_full),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stft_odft_reconstruction_stress.csv")
    ap.add_argument("--n", type=int, default=24576)
    ap.add_argument("--fs", type=float, default=48000.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-ffts", default="256,512,1024,2048")
    ap.add_argument("--hops", default="quarter,eighth,half")
    args = ap.parse_args()

    signals = [
        "white",
        "colored",
        "tones_random",
        "tones_integer",
        "tones_half",
        "chirps",
        "impulses",
        "drift",
    ]

    edits = [
        "identity",
        "soft_gate",
        "random_smooth_mask",
        "random_complex_mask",
        "lowpass_25pct",
        "highpass_25pct",
        "comb_even_bins",
        "comb_odd_bins",
        "frame_dropout",
        "phase_only",
        "magnitude_only",
    ]

    windows = ["hann", "sqrt_hann", "blackman_harris_4"]
    transforms = ["rfft", "odft"]
    n_ffts = [int(x) for x in args.n_ffts.split(",")]

    rows = []
    for n_fft in n_ffts:
        hop_values = []
        for h in args.hops.split(","):
            h = h.strip()
            if h == "quarter":
                hop_values.append(n_fft // 4)
            elif h == "eighth":
                hop_values.append(n_fft // 8)
            elif h == "half":
                hop_values.append(n_fft // 2)
            else:
                hop_values.append(int(h))

        for hop in hop_values:
            if n_fft % hop != 0:
                continue
            for window in windows:
                for signal in signals:
                    for edit in edits:
                        pair = []
                        for transform in transforms:
                            try:
                                r = run_case(args.n, n_fft, hop, window, transform, signal, edit, args.fs, args.seed)
                                rows.append(r.__dict__)
                                pair.append(r)
                                print(
                                    f"n_fft={n_fft:5d} hop={hop:5d} win={window:17s} "
                                    f"sig={signal:13s} edit={edit:18s} xf={transform:4s} "
                                    f"steady_snr={r.steady_snr_db:9.2f} dB "
                                    f"full_snr={r.full_snr_db:9.2f} dB "
                                    f"edge_snr={r.edge_snr_db:9.2f} dB",
                                    flush=True,
                                )
                            except Exception as e:
                                print(f"FAILED n_fft={n_fft} hop={hop} window={window} signal={signal} edit={edit} transform={transform}: {e}", flush=True)

                        if len(pair) == 2:
                            a, b = pair
                            delta = b.steady_snr_db - a.steady_snr_db
                            print(
                                f"    odft_minus_rfft steady_snr_delta={delta:+.2f} dB",
                                flush=True,
                            )

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
