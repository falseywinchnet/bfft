#!/usr/bin/env python3
"""Render the three viewer transforms on one IQ fixture without the GUI.

Modes: streaming STFT, two-aperture DIP-Zak fusion super-resolution, and the
phase-reassigned STFT.  The fusion render goes through the production
``dip_stream.FusionStream`` path (polled to full coverage), and the fusion
energy-conservation invariant ``sum_e |F[e,k]|^2 = |L[k]|^2`` (beta=1) is
checked directly on a solved tile.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dip_stream
import iqwaterfall as iqw


def fusion_energy_check(src, nb, ns, gate_floor, n_steps, alpha):
    """sum over owned fine rows of |F|^2 vs |L|^2, per (frame, bin), tile 0.

    This is the frame-norm invariant; claim mode intentionally leaves
    off-band energy unclaimed, so the check pins norm="frame".
    """
    fs = dip_stream.FusionStream(src, nb=nb, ns=ns, workers=1,
                                 gate_floor=gate_floor, beta=1.0,
                                 n_steps=n_steps, alpha=alpha, norm="frame")
    try:
        tile = fs._solve_tile(0)                     # [Ib*owned, nb]
        shaped = tile.reshape(fs.Ib, fs.owned, fs.nb).astype(np.float64)
        fine_power = np.sum(shaped * shaped, axis=1)  # [Ib, nb]
        z = src.read(0, dip_stream.TILE)
        long_mag, _ = iqw.dip_fusion(np.ascontiguousarray(z), fs.dsel,
                                     mode=fs.mode, alpha=fs.alpha,
                                     n_steps=fs.n_steps,
                                     direct_seed=fs.direct, nb=fs.nb,
                                     ns=fs.ns)
        long_power = long_mag * long_mag
        num = np.abs(fine_power - long_power)
        rel = num.max() / (long_power.max() + 1e-30)
        return float(rel)
    finally:
        fs.close()


def render_fusion(src, start, rows, nb, ns, gate_floor=0.05, beta=1.0,
                  n_steps=1, alpha=1.0, norm="row", timeout=120.0):
    """Render `rows` fusion rows on the native HS lattice via FusionStream."""
    fs = dip_stream.FusionStream(src, nb=nb, ns=ns, gate_floor=gate_floor,
                                 beta=beta, n_steps=n_steps, alpha=alpha,
                                 norm=norm)
    try:
        frame_lo = max(0, start // fs.frame_hop - fs.lo)
        t0 = time.perf_counter()
        fs.request(frame_lo, rows)
        deadline = t0 + timeout
        while True:
            mags, cov = fs.assemble(frame_lo, rows)
            if cov >= 0.999 or time.perf_counter() > deadline:
                break
            time.sleep(0.05)
        dt = time.perf_counter() - t0
        block = 20.0 * np.log10(mags + 1e-9)
        return np.fft.fftshift(block, axes=1).astype(np.float32), cov, dt
    finally:
        fs.close()


def render_modes(path, start=0, n_fft=1024, hop=None, rows=160,
                 nb=1024, ns=128, gate_floor=0.05, beta=1.0,
                 n_steps=1, alpha=1.0, norm="claim"):
    hop = int(hop or max(1, n_fft // 32))
    src = iqw.IQSource(path)
    wf = iqw.Waterfall(n_fft, iqw.WINDOWS["Hann"])
    ra = iqw.Reassign(n_fft)

    timings = {}
    t0 = time.perf_counter()
    streaming = wf.render(src, start - n_fft // 2, hop, rows)
    timings["streaming_s"] = time.perf_counter() - t0

    need = (rows - 1) * hop + n_fft
    z = src.read(start - n_fft // 2, need)
    iq = np.empty(2 * need, np.float32)
    iq[0::2] = z.real
    iq[1::2] = z.imag
    t0 = time.perf_counter()
    reassigned = ra.render_mem(iq, n_fft // 2, hop, rows)
    timings["reassigned_s"] = time.perf_counter() - t0

    # Fusion rows live on the HS=32 lattice; ask for the same time span.
    span = rows * hop
    n_native = max(1, span // 32)
    fusion_native, cov, dt = render_fusion(
        src, start, n_native, nb, ns, gate_floor, beta, n_steps, alpha, norm)
    timings["fusion_s"] = dt
    idx = np.linspace(0, n_native - 1, rows).astype(np.intp)
    fusion = fusion_native[idx]

    energy_rel = fusion_energy_check(src, nb, ns, gate_floor, n_steps, alpha)

    blocks = {"streaming": streaming, "fusion": fusion,
              "reassigned": reassigned}
    metrics = {
        "sample_rate": src.sample_rate,
        "header_samples": src.num_samples,
        "n_fft": n_fft,
        "hop": hop,
        "rows": rows,
        "fusion_params": {"nb": nb, "ns": ns, "gate_floor": gate_floor,
                          "beta": beta, "n_steps": n_steps, "alpha": alpha,
                          "norm": norm, "coverage": cov,
                          "energy_conservation_rel_err": energy_rel},
        "timings": timings,
    }
    for name, block in blocks.items():
        finite = block[block > -239]
        metrics[name] = {
            "median_db": float(np.median(finite)) if finite.size else -240.0,
            "p99_db": float(np.percentile(finite, 99)) if finite.size else -240.0,
            "peak_db": float(np.max(finite)) if finite.size else -240.0,
        }
    return blocks, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--nb", type=int, default=1024)
    parser.add_argument("--ns", type=int, default=128)
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--norm", choices=("claim", "row", "frame"), default="claim")
    parser.add_argument("--out", default="/tmp/bfft_viewer_modes.png")
    args = parser.parse_args()
    blocks, metrics = render_modes(
        args.path, args.start, args.n_fft, args.hop, args.rows,
        args.nb, args.ns, args.gate_floor, args.beta, args.steps, args.alpha,
        args.norm)
    all_values = np.concatenate([b[b > -239].ravel() for b in blocks.values()])
    ceiling = float(np.percentile(all_values, 99.8))
    floor = ceiling - 80.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    labels = ["Streaming STFT",
              f"Two-aperture fusion (NB={args.nb}, NS={args.ns})",
              "Reassigned STFT"]
    for ax, (name, block), label in zip(axes, blocks.items(), labels):
        ax.imshow(block, origin="lower", aspect="auto", cmap="viridis",
                  vmin=floor, vmax=ceiling,
                  extent=[-metrics["sample_rate"] / 2,
                          metrics["sample_rate"] / 2,
                          0, metrics["rows"] * metrics["hop"] /
                          metrics["sample_rate"]])
        ax.set_title(label)
        ax.set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("time from start (s)")
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(out)


if __name__ == "__main__":
    main()
