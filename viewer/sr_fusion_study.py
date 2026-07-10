#!/usr/bin/env python3
"""Quantify two-aperture fusion quality on the known-truth SR fixture.

The fixture (make_sr_fixture.py) contains a 1.5 kHz tone pair (long-aperture
job), a wideband click train (short-aperture job), a chirp, and a noise floor.
Measured per configuration:

- click_width_ms: participation-ratio width of the wideband time profile
  around one click (sharp time = small);
- halo_db: wideband power 5..16 rows away from the click (outside the short
  window, inside the long window) relative to the between-click background --
  the long-window leakage this study targets;
- tone_dip_db: spectral dip between the 50.0 and 51.5 kHz peaks on steady
  rows (resolved frequency = deep dip; the short window alone cannot do it);
- noise_cv: coefficient of variation of quiet-band row power (gate noise).

Baselines: long STFT (N=1024), short STFT (N=128), reassigned STFT.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import dip_stream
import iqwaterfall as iqw

FS = 456_000.0
HS = 32


def fusion_linear(src, start, rows, timeout=120.0, **kw):
    fs = dip_stream.FusionStream(src, **kw)
    try:
        frame_lo = max(0, start // fs.frame_hop - fs.lo)
        t0 = time.perf_counter()
        fs.request(frame_lo, rows)
        while True:
            mags, cov = fs.assemble(frame_lo, rows)
            if cov >= 0.999 or time.perf_counter() - t0 > timeout:
                break
            time.sleep(0.02)
        dt = time.perf_counter() - t0
        power = np.fft.fftshift((mags.astype(np.float64)) ** 2, axes=1)
        return power, cov, dt
    finally:
        fs.close()


def stft_linear(src, start, rows, n_fft):
    wf = iqw.Waterfall(n_fft, iqw.WINDOWS["Hann"])
    try:
        db = wf.render(src, start - n_fft // 2, HS, rows)
        return np.power(10.0, db.astype(np.float64) / 10.0)
    finally:
        wf.close()


def reassigned_linear(src, start, rows, n_fft=1024):
    ra = iqw.Reassign(n_fft)
    try:
        need = (rows - 1) * HS + n_fft
        z = src.read(start - n_fft // 2, need)
        iq = np.empty(2 * need, np.float32)
        iq[0::2] = z.real
        iq[1::2] = z.imag
        db = ra.render_mem(iq, n_fft // 2, HS, rows)
        return np.power(10.0, db.astype(np.float64) / 10.0)
    finally:
        ra.close()


def cols_for(nb, f_lo, f_hi):
    f = (np.arange(nb) - nb / 2) / nb * FS
    return np.where((f >= f_lo) & (f <= f_hi))[0]


def metrics(power, start, rows, nb):
    """power: [rows, nb] linear, fftshifted, row r at sample start + r*HS."""
    out = {}
    # The fixture click is a 60-sample Hann pulse: lowpass. Measure it (and
    # its long-window blob) where it lives, away from the tones and chirp.
    band = cols_for(nb, -40_000, 40_000)
    p = power[:, band].mean(axis=1)                # click-band time profile

    # clicks live at samples 2048 + 4096*k; distance is measured to EVERY
    # click touching the window (edge clicks included), interior clicks are
    # the ones analyzed.
    first = 2048 + 4096 * ((start - 2048 - 4096) // 4096 + 1)
    all_rows = [(c - start) / HS for c in range(first,
                                                start + rows * HS + 4096,
                                                4096)]
    interior = [int(r) for r in all_rows if 40 <= r < rows - 40]
    assert interior, "no interior click in the analysis window"
    dist = np.min(np.abs(np.arange(rows)[None, :] -
                         np.asarray(all_rows)[:, None]), axis=0)

    widths, halos, blobs = [], [], []
    for r in interior:
        rr = int(r + np.argmax(p[r:r + 8]))        # snap to actual peak
        idx = np.arange(max(0, rr - 32), min(rows, rr + 33))
        q = np.clip(p[idx] - np.median(p), 0.0, None)
        widths.append(float(np.sum(q > 0.5 * q.max())) * HS / FS * 1e3)
        d = np.abs(np.arange(rows) - rr)
        halos.append(p[(d >= 6) & (d <= 20)].mean())
        blobs.append(p[(d >= 8) & (d <= 45)].mean())
    bg = p[dist > 48].mean()
    out["click_width_ms"] = float(np.mean(widths))
    out["halo_db"] = float(10 * np.log10(np.mean(halos) / (bg + 1e-30)))
    out["blob_db"] = float(10 * np.log10(np.mean(blobs) / (bg + 1e-30)))

    # tone pair on steady rows (no click within the long window)
    steady = dist > 48
    spec = power[steady].mean(axis=0)
    f = (np.arange(nb) - nb / 2) / nb * FS
    tol = max(400.0, FS / nb * 0.75)
    pk1 = spec[np.abs(f - 50_000) < tol].max()
    pk2 = spec[np.abs(f - 51_500) < tol].max()
    mid = (f > 50_400) & (f < 51_100)
    if np.any(mid):
        valley = spec[mid].min()
        out["tone_dip_db"] = float(10 * np.log10(valley /
                                                 (0.5 * (pk1 + pk2) + 1e-30)))
    else:
        out["tone_dip_db"] = 0.0   # grid too coarse to even see between them

    quiet = cols_for(nb, -220_000, -160_000)
    qp = power[steady][:, quiet].mean(axis=1)
    out["noise_cv"] = float(np.std(qp) / (np.mean(qp) + 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--start", type=int, default=200_000)
    ap.add_argument("--rows", type=int, default=320)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    src = iqw.IQSource(args.path)
    start, rows = args.start, args.rows

    results = {}

    for name, fn in [
        ("stft_long_1024", lambda: stft_linear(src, start, rows, 1024)),
        ("stft_short_128", lambda: stft_linear(src, start, rows, 128)),
        ("reassigned_1024", lambda: reassigned_linear(src, start, rows)),
    ]:
        power = fn()
        results[name] = metrics(power, start, rows, power.shape[1])

    fusion_cfgs = [
        ("fusion_frame_b1_gf.05", dict(norm="frame", beta=1.0, gate_floor=.05)),
        ("fusion_frame_b2_gf.05", dict(norm="frame", beta=2.0, gate_floor=.05)),
        ("fusion_frame_seed_only",
         dict(norm="frame", beta=1.0, gate_floor=.05, n_steps=0)),
        ("fusion_row_seed_only",
         dict(norm="row", beta=1.0, gate_floor=.05, n_steps=0)),
        ("fusion_row_b1_gf0", dict(norm="row", beta=1.0, gate_floor=0.0)),
        ("fusion_row_b1_gf.05", dict(norm="row", beta=1.0, gate_floor=.05)),
        ("fusion_row_b1_gf.15", dict(norm="row", beta=1.0, gate_floor=.15)),
        ("fusion_row_b2_gf.05", dict(norm="row", beta=2.0, gate_floor=.05)),
        ("fusion_row_b1_gf.05_nb2048",
         dict(norm="row", beta=1.0, gate_floor=.05, nb=2048)),
        ("fusion_row_b1_gf.05_ns64",
         dict(norm="row", beta=1.0, gate_floor=.05, ns=64)),
        ("fusion_claim_b1_gf.05", dict(norm="claim", beta=1.0, gate_floor=.05)),
        ("fusion_claim_b1_gf.05_nb2048",
         dict(norm="claim", beta=1.0, gate_floor=.05, nb=2048)),
        ("fusion_claim_b2_gf.05_nb2048",
         dict(norm="claim", beta=2.0, gate_floor=.05, nb=2048)),
    ]
    for name, kw in fusion_cfgs:
        kw.setdefault("nb", 1024)
        kw.setdefault("ns", 128)
        power, cov, dt = fusion_linear(src, start, rows, **kw)
        m = metrics(power, start, rows, kw["nb"])
        m["coverage"] = cov
        m["render_s"] = round(dt, 4)
        results[name] = m

    hdr = f"{'config':30s} {'click_ms':>9s} {'halo_db':>8s} {'blob_db':>8s} " \
          f"{'tone_dip':>9s} {'noise_cv':>9s}"
    print(hdr)
    for name, m in results.items():
        print(f"{name:30s} {m['click_width_ms']:9.3f} {m['halo_db']:8.2f} "
              f"{m['blob_db']:8.2f} {m['tone_dip_db']:9.2f} "
              f"{m['noise_cv']:9.3f}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
