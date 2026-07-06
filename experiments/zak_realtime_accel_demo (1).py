"""Vectorized coherent Zak/DIP-style superresolution fusion demo.

This is the fast path after the first standalone prototype.  The mathematical
problem is unchanged: use only two magnitude observations of the same first
second of audio, a long-window Hann STFT and a short-window Hann STFT, recover
one coherent latent waveform by alternating magnitude projections, then render a
reassigned long-window superresolution readout on the long-frequency by
short-time grid.

The acceleration is implementation-level and descent-level:
  * every STFT family is compiled into a static frame index matrix;
  * each projection uses one batched FFT and one batched inverse FFT instead of
    hundreds of Python-level FFT calls;
  * overlap-add is a single indexed scatter-add;
  * the same fixed projectors allow aggressive heavy-ball continuation;
  * modes expose either a single streaming trajectory or a robust seedbank.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from math import gcd
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class Mode:
    name: str
    nseeds: int
    scout_iters: int
    refine_iters: int
    polish_iters: int
    beta: float
    topk: int
    seed0: int = 0


MODES: Dict[str, Mode] = {
    # For a streaming system this is the relevant core: one warm trajectory.
    # seed0=5 is used for reproducibility on the attached first-second demo.
    "realtime": Mode("realtime", nseeds=1, scout_iters=0, refine_iters=580,
                     polish_iters=0, beta=0.94, topk=1, seed0=5),
    # Robust first-block mode: scout a small bank, refine the best few.
    "robust": Mode("robust", nseeds=6, scout_iters=160, refine_iters=420,
                   polish_iters=0, beta=0.94, topk=3, seed0=0),
    # Same schedule as the earlier numpy prototype, but on the vectorized engine.
    "prototype_equiv": Mode("prototype_equiv", nseeds=6, scout_iters=120,
                            refine_iters=360, polish_iters=80, beta=0.88,
                            topk=3, seed0=0),
}


@dataclass
class Config:
    wav: str = "/mnt/data/daveandsimon(4).wav"
    outdir: str = "/mnt/data/zak_realtime_out"
    work_sr: int = 8192
    seconds: float = 1.0
    n1: int = 1024
    h1: int = 128
    n2: int = 128
    h2: int = 32
    mode: str = "realtime"
    max_plot_hz: float = 4000.0


def load_first_second(path: str, work_sr: int, seconds: float) -> Tuple[np.ndarray, int, int]:
    x, sr0 = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[:min(len(x), int(round(seconds * sr0)))].astype(np.float64)
    x -= np.mean(x)
    if sr0 != work_sr:
        g = gcd(work_sr, sr0)
        x = resample_poly(x, work_sr // g, sr0 // g)
        sr = work_sr
    else:
        sr = sr0
    L = int(round(seconds * sr))
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    else:
        x = x[:L]
    x = 0.95 * x / (np.max(np.abs(x)) + 1e-12)
    return np.ascontiguousarray(x), sr, sr0


def frame_offsets(L: int, n: int, hop: int) -> np.ndarray:
    return np.arange(0, L - n + 1, hop, dtype=np.int64)


class STFTFamily:
    """One fixed magnitude family with batched projection and gradient."""
    def __init__(self, x: np.ndarray, n: int, hop: int):
        self.L = len(x)
        self.n = int(n)
        self.hop = int(hop)
        self.offsets = frame_offsets(self.L, self.n, self.hop)
        self.window = np.hanning(self.n).astype(np.float64)
        self.idx = self.offsets[:, None] + np.arange(self.n, dtype=np.int64)[None, :]
        self.wh = self.window[None, :]
        frames = x[self.idx] * self.wh
        self.target = np.abs(np.fft.fft(frames, axis=1))
        self.den = np.zeros(self.L, dtype=np.float64)
        np.add.at(self.den, self.idx.ravel(), np.broadcast_to(self.window ** 2, self.idx.shape).ravel())
        self.denom = np.maximum(self.den, 1e-8)

    def project(self, u: np.ndarray) -> np.ndarray:
        frames = u[self.idx] * self.wh
        Y = np.fft.fft(frames, axis=1)
        Y *= self.target / (np.abs(Y) + 1e-12)
        rec = np.fft.ifft(Y, axis=1).real * self.wh
        acc = np.zeros(self.L, dtype=np.float64)
        np.add.at(acc, self.idx.ravel(), rec.ravel())
        return acc / self.denom

    def loss(self, u: np.ndarray) -> float:
        frames = u[self.idx] * self.wh
        A = np.abs(np.fft.fft(frames, axis=1))
        return float(np.sum((A - self.target) ** 2))

    def gradient(self, u: np.ndarray, weight: float = 1.0) -> Tuple[np.ndarray, float]:
        frames = u[self.idx] * self.wh
        Y = np.fft.fft(frames, axis=1)
        A = np.abs(Y)
        res = (1.0 - self.target / (A + 1e-12)) * Y
        gf = weight * 2.0 * self.n * np.fft.ifft(res, axis=1).real * self.wh
        g = np.zeros(self.L, dtype=np.float64)
        np.add.at(g, self.idx.ravel(), gf.ravel())
        return g, float(weight * np.sum((A - self.target) ** 2))

    def half_power(self) -> np.ndarray:
        return self.target[:, :self.n // 2 + 1].T ** 2


def total_loss(u: np.ndarray, fam1: STFTFamily, fam2: STFTFamily) -> float:
    return fam1.loss(u) + fam2.loss(u)


def align(u: np.ndarray, fam1: STFTFamily, fam2: STFTFamily, iters: int,
          beta: float, trace_stride: int = 20) -> Tuple[np.ndarray, List[float], int]:
    up = u.copy()
    trace: List[float] = []
    resets = 0
    for it in range(iters):
        y = u + beta * (u - up)
        # The projection map can produce rare huge extrapolates early; restart
        # the inertial leg, not the candidate. This is essential for the high-beta mode.
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 50.0:
            y = u.copy()
            resets += 1
        up = u
        u = fam2.project(fam1.project(y))
        if trace_stride and (it % trace_stride == 0 or it == iters - 1):
            trace.append(total_loss(u, fam1, fam2))
    return u, trace, resets


def polish(u: np.ndarray, fam1: STFTFamily, fam2: STFTFamily,
           iters: int, momentum: float = 0.82) -> Tuple[np.ndarray, List[float]]:
    if iters <= 0:
        return u, []
    w2 = fam1.n / fam2.n
    alpha = 0.45 / (2 * fam1.n * fam1.den.max() + w2 * 2 * fam2.n * fam2.den.max() + 1e-12)
    up = u.copy()
    hist: List[float] = []
    for _ in range(iters):
        v = u + momentum * (u - up)
        g1, l1 = fam1.gradient(v, 1.0)
        g2, l2 = fam2.gradient(v, w2)
        up = u
        u = v - alpha * (g1 + g2)
        hist.append(l1 + l2)
    return u, hist


def recover(L: int, fam1: STFTFamily, fam2: STFTFamily, mode: Mode) -> Tuple[np.ndarray, Dict[str, object]]:
    t0 = time.perf_counter()
    scouts = []
    for seed in range(mode.seed0, mode.seed0 + mode.nseeds):
        rng = np.random.default_rng(seed)
        u = 0.01 * rng.standard_normal(L)
        if mode.scout_iters > 0:
            u, tr, resets = align(u, fam1, fam2, mode.scout_iters, mode.beta)
        else:
            tr, resets = [], 0
        loss = total_loss(u, fam1, fam2)
        scouts.append((loss, seed, u, tr, resets))
    scouts.sort(key=lambda c: c[0])

    best = None
    best_trace: List[float] = []
    refine_records = []
    for scout_loss, seed, u, tr0, reset0 in scouts[:min(mode.topk, len(scouts))]:
        u, tr1, reset1 = align(u, fam1, fam2, mode.refine_iters, mode.beta)
        u, tr2 = polish(u, fam1, fam2, mode.polish_iters)
        loss = total_loss(u, fam1, fam2)
        refine_records.append({"seed": seed, "scout_loss": scout_loss,
                               "final_loss": loss, "resets": reset0 + reset1})
        if best is None or loss < best[0]:
            best = (loss, seed, u, reset0 + reset1)
            best_trace = tr0 + tr1 + tr2
    assert best is not None
    wall = time.perf_counter() - t0
    info = {
        "mode": asdict(mode),
        "wall_seconds": wall,
        "best_seed": int(best[1]),
        "loss": float(best[0]),
        "trace": [float(v) for v in best_trace],
        "scouts": [{"seed": int(s), "loss": float(l), "resets": int(r)} for l, s, _, _, r in scouts],
        "refined": refine_records,
    }
    return best[2], info


def interp_time(P: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((P.shape[0], len(t_dst)), dtype=np.float64)
    for k in range(P.shape[0]):
        out[k] = np.interp(t_dst, t_src, P[k], left=0.0, right=0.0)
    return out


def interp_freq(P: np.ndarray, f_src: np.ndarray, f_dst: np.ndarray) -> np.ndarray:
    out = np.empty((len(f_dst), P.shape[1]), dtype=np.float64)
    for i in range(P.shape[1]):
        out[:, i] = np.interp(f_dst, f_src, P[:, i], left=0.0, right=0.0)
    return out


def baseline_maps(fam1: STFTFamily, fam2: STFTFamily, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P1 = fam1.half_power()
    P2 = fam2.half_power()
    times1 = (fam1.offsets + fam1.n / 2) / sr
    times2 = (fam2.offsets + fam2.n / 2) / sr
    freqs1 = np.arange(fam1.n // 2 + 1) * sr / fam1.n
    freqs2 = np.arange(fam2.n // 2 + 1) * sr / fam2.n
    long_map = interp_time(P1, times1, times2)
    short_map = interp_freq(P2, freqs2, freqs1)
    geomean = np.sqrt(np.maximum(long_map, 0.0) * np.maximum(short_map, 0.0))
    return long_map, short_map, geomean, times2, freqs1


def reassigned_spectrogram(x: np.ndarray, sr: int, n: int, times: np.ndarray) -> np.ndarray:
    F = n // 2 + 1
    g = np.hanning(n)
    dg = np.gradient(g)
    tt = np.arange(n) - (n - 1) / 2.0
    tg = tt * g
    pad = n
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    centers = np.round(times * sr).astype(int)
    P = np.zeros((F, len(times)), dtype=np.float64)
    t0 = centers[0]
    hop_est = int(round(np.median(np.diff(centers)))) if len(centers) > 1 else 1
    for c in centers:
        a = c + pad - n // 2
        seg = xp[a:a + n]
        if len(seg) != n:
            continue
        Y = np.fft.rfft(g * seg)
        Yt = np.fft.rfft(tg * seg)
        Yd = np.fft.rfft(dg * seg)
        E = np.abs(Y) ** 2
        if E.max() <= 0:
            continue
        good = E > 1e-8 * E.max()
        that = c + np.real(Yt[good] * np.conj(Y[good])) / (E[good] + 1e-30)
        khat = (np.arange(F)[good]
                - np.imag(Yd[good] * np.conj(Y[good])) / (E[good] + 1e-30) * n / (2 * np.pi))
        cols = np.clip(np.round((that - t0) / hop_est), 0, len(times) - 1).astype(int)
        rows = np.clip(np.round(khat), 0, F - 1).astype(int)
        np.add.at(P, (rows, cols), E[good])
    return P


def corrmap(P: np.ndarray, Q: np.ndarray) -> float:
    a = np.sqrt(np.maximum(P, 0.0)).ravel()
    b = np.sqrt(np.maximum(Q, 0.0)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def snr_db(x: np.ndarray, y: np.ndarray, cut: int) -> float:
    xs, ys = x[cut:-cut], y[cut:-cut]
    if len(xs) == 0:
        xs, ys = x, y
    err = min(np.sum((xs - ys) ** 2), np.sum((xs + ys) ** 2))
    return float(10 * np.log10(np.sum(xs ** 2) / max(err, 1e-30)))


def save_wav(path: str, x: np.ndarray, sr: int) -> None:
    y = 0.95 * x / (np.max(np.abs(x)) + 1e-12)
    sf.write(path, y, sr)


def plot_maps(path: str, x: np.ndarray, sr: int, times: np.ndarray, freqs: np.ndarray,
              long_map: np.ndarray, short_map: np.ndarray, geomean: np.ndarray,
              fused: np.ndarray, oracle: np.ndarray, metrics: Dict[str, float],
              trace: List[float], max_hz: float) -> None:
    def show(ax, P, title):
        Pm = np.sqrt(np.maximum(P, 0.0))
        vmax = np.percentile(Pm[Pm > 0], 99.5) if np.any(Pm > 0) else 1.0
        ax.imshow(Pm, origin="lower", aspect="auto",
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  cmap="magma", vmax=vmax)
        ax.set_ylim(0, min(max_hz, freqs[-1]))
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("frequency (Hz)", fontsize=8)
        ax.set_title(title, fontsize=9)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    t = np.arange(len(x)) / sr
    axes[0, 0].plot(t, x, lw=0.8)
    axes[0, 0].set_title("input waveform, first second", fontsize=9)
    axes[0, 0].set_xlabel("time (s)", fontsize=8)
    axes[0, 0].set_ylabel("amplitude", fontsize=8)
    show(axes[0, 1], long_map, f"long source, corr {metrics['long']:.3f}")
    show(axes[0, 2], short_map, f"short source, corr {metrics['short']:.3f}")
    show(axes[1, 0], geomean, f"geomean, corr {metrics['geomean']:.3f}")
    show(axes[1, 1], fused, f"coherent fused, corr {metrics['fused']:.3f}")
    ax = axes[1, 2]
    if trace:
        ax.semilogy(np.arange(len(trace)), np.maximum(trace, 1e-30), lw=1.2)
    ax.set_title(f"loss trace, wall {metrics['wall_seconds']:.3f}s, SNR {metrics['snr_db']:+.2f} dB", fontsize=9)
    ax.set_xlabel("logged step", fontsize=8)
    ax.set_ylabel("observable loss", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> Dict[str, object]:
    os.makedirs(cfg.outdir, exist_ok=True)
    mode = MODES[cfg.mode]
    x, sr, original_sr = load_first_second(cfg.wav, cfg.work_sr, cfg.seconds)
    fam1 = STFTFamily(x, cfg.n1, cfg.h1)
    fam2 = STFTFamily(x, cfg.n2, cfg.h2)
    xh, rec_info = recover(len(x), fam1, fam2, mode)
    long_map, short_map, geomean, times2, freqs1 = baseline_maps(fam1, fam2, sr)
    fused = reassigned_spectrogram(xh, sr, cfg.n1, times2)
    oracle = reassigned_spectrogram(x, sr, cfg.n1, times2)
    metrics = {
        "mode": cfg.mode,
        "original_sr": original_sr,
        "work_sr": sr,
        "L": len(x),
        "long": corrmap(long_map, oracle),
        "short": corrmap(short_map, oracle),
        "geomean": corrmap(geomean, oracle),
        "fused": corrmap(fused, oracle),
        "oracle": corrmap(oracle, oracle),
        "snr_db": snr_db(x, xh, cut=cfg.n1 // 2),
        "wall_seconds": float(rec_info["wall_seconds"]),
        "loss": float(rec_info["loss"]),
        "best_seed": int(rec_info["best_seed"]),
    }
    base = os.path.join(cfg.outdir, cfg.mode)
    comparison_png = base + "_comparison.png"
    source_wav = base + "_source_first_second_8192hz.wav"
    recovered_wav = base + "_recovered_first_second_8192hz.wav"
    metrics_json = base + "_metrics.json"
    plot_maps(comparison_png, x, sr, times2, freqs1, long_map, short_map, geomean,
              fused, oracle, metrics, rec_info["trace"], cfg.max_plot_hz)
    save_wav(source_wav, x, sr)
    save_wav(recovered_wav, xh, sr)
    payload = {"config": asdict(cfg), "metrics": metrics, "recovery": rec_info}
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"metrics": metrics, "comparison_png": comparison_png,
            "source_wav": source_wav, "recovered_wav": recovered_wav,
            "metrics_json": metrics_json}


def benchmark_all(cfg: Config) -> Dict[str, object]:
    rows = []
    outputs = {}
    for name in ("realtime", "robust", "prototype_equiv"):
        c = Config(**{**asdict(cfg), "mode": name})
        res = run(c)
        rows.append(res["metrics"])
        outputs[name] = res
    path_csv = os.path.join(cfg.outdir, "benchmark.csv")
    with open(path_csv, "w", encoding="utf-8") as f:
        keys = ["mode", "wall_seconds", "loss", "best_seed", "long", "short", "geomean", "fused", "snr_db"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    names = [r["mode"] for r in rows]
    axes[0].bar(names, [r["wall_seconds"] for r in rows])
    axes[0].set_ylabel("wall seconds")
    axes[0].set_title("recovery time")
    axes[1].bar(names, [r["fused"] for r in rows])
    axes[1].axhline(rows[0]["long"], linestyle=":", linewidth=1, label="long source")
    axes[1].axhline(rows[0]["geomean"], linestyle="--", linewidth=1, label="geomean")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("cosine to oracle")
    axes[1].set_title("superresolution quality")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path_png = os.path.join(cfg.outdir, "benchmark.png")
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return {"rows": rows, "outputs": outputs, "benchmark_csv": path_csv, "benchmark_png": path_png}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", default=Config.wav)
    p.add_argument("--outdir", default=Config.outdir)
    p.add_argument("--mode", choices=list(MODES.keys()) + ["all"], default="realtime")
    p.add_argument("--work-sr", type=int, default=Config.work_sr)
    args = p.parse_args()
    cfg = Config(wav=args.wav, outdir=args.outdir, work_sr=args.work_sr,
                 mode="realtime" if args.mode == "all" else args.mode)
    if args.mode == "all":
        res = benchmark_all(cfg)
        print("== vectorized Zak/DIP coherent fusion benchmark ==")
        for r in res["rows"]:
            print(f"{r['mode']:16s} wall {r['wall_seconds']:.3f}s | fused {r['fused']:.4f} | loss {r['loss']:.3e} | seed {r['best_seed']}")
        print(f"benchmark_csv: {res['benchmark_csv']}")
        print(f"benchmark_png: {res['benchmark_png']}")
    else:
        res = run(cfg)
        m = res["metrics"]
        print("== vectorized Zak/DIP coherent fusion ==")
        print(f"mode {m['mode']} | wall {m['wall_seconds']:.3f}s | fused corr {m['fused']:.4f} | loss {m['loss']:.3e} | seed {m['best_seed']}")
        print(f"source corrs: long {m['long']:.4f}, short {m['short']:.4f}, geomean {m['geomean']:.4f}")
        for k, v in res.items():
            if k != "metrics":
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
