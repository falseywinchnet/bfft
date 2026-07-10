#!/usr/bin/env python3
"""Synthesize a super-resolution IQ fixture with known ground truth.

The point of two-aperture fusion is joint resolution: content that only a
long window can separate in frequency coexisting with content that only a
short window can localize in time.  This fixture encodes both:

- two complex tones 1.5 kHz apart (needs the long aperture: at Fs=456 kHz a
  128-sample window has ~3.6 kHz resolution, a 1024-sample window ~450 Hz);
- a wideband click train with 60-sample clicks every 4096 samples (needs the
  short aperture: a 1024-sample frame smears a click across 2.2 ms);
- one linear chirp crossing the tones;
- a small noise floor.

Written as a stereo int16 WAV so the viewer's IQ reader parses it directly.
"""
from __future__ import annotations

import struct
import sys

import numpy as np

FS = 456_000


def make_iq(seconds=2.0, seed=5):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    z = np.zeros(n, np.complex128)
    # tone pair separated by 1.5 kHz (long-window job)
    z += 0.4 * np.exp(2j * np.pi * 50_000 * t)
    z += 0.4 * np.exp(2j * np.pi * 51_500 * t + 1.1j)
    # wideband click train (short-window job)
    click = np.hanning(60)
    for c in range(2048, n - 64, 4096):
        ph = rng.uniform(0, 2 * np.pi)
        z[c:c + 60] += 1.6 * click * np.exp(1j * ph)
    # a chirp crossing the tones: -150 kHz -> +150 kHz over the record
    z += 0.25 * np.exp(2j * np.pi * (-150_000 * t + 150_000 * t * t / seconds))
    z += 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return z


def write_wav_iq(path, z, fs=FS):
    peak = np.max(np.abs(np.concatenate([z.real, z.imag])))
    scale = 0.9 * 32767 / peak
    data = np.empty(2 * z.size, np.int16)
    data[0::2] = np.round(z.real * scale).astype(np.int16)
    data[1::2] = np.round(z.imag * scale).astype(np.int16)
    payload = data.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(payload)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 2, fs, fs * 4, 4, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(payload)))
        f.write(payload)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sr_fixture_iq.wav"
    write_wav_iq(out, make_iq())
    print(out)
