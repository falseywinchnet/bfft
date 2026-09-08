"""ITD baseline knots reconstructed by cubic spline versus irregular CONV."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.convstar import ordered_sign_ledger, project_signed_fibres  # noqa: E402


OUT = ROOT / "experiments" / "out" / "conv_itd_comparison"


def extrema(signal: np.ndarray) -> np.ndarray:
    delta = np.diff(signal)
    sign = np.sign(delta)
    for i in range(1, sign.size):
        if sign[i] == 0:
            sign[i] = sign[i-1]
    for i in range(sign.size-2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i+1]
    turning = np.flatnonzero(sign[:-1] != sign[1:]) + 1
    return np.unique(np.r_[0, turning, signal.size-1]).astype(np.int64)


def itd_knots(signal: np.ndarray, index: np.ndarray) -> np.ndarray:
    value = np.empty(index.size, dtype=np.float64)
    # Match itd_baseline_extract_fast: the boundary knots are the signal at
    # the first and last supplied extrema, not free-boundary pair averages.
    value[0] = signal[index[0]]
    value[-1] = signal[index[-1]]
    for k in range(1, index.size-1):
        left, centre, right = index[k-1:k+2]
        chord = signal[left] + (centre-left)/(right-left)*(signal[right]-signal[left])
        value[k] = 0.5*(chord + signal[centre])
    return value


def derivative_weights(nodes: np.ndarray, centre: float, order: int) -> np.ndarray:
    dx = nodes-centre
    matrix = np.stack([dx**k for k in range(nodes.size)], axis=0)
    target = np.zeros(nodes.size)
    target[order] = float(math.factorial(order))
    return np.linalg.solve(matrix, target)


def irregular_jets(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    first = np.empty(n)
    second = np.empty(n)
    for i in range(n):
        start = min(max(i-2, 0), n-5)
        take = np.arange(start, start+5)
        first[i] = derivative_weights(x[take], x[i], 1) @ y[take]
        second[i] = derivative_weights(x[take], x[i], 2) @ y[take]
    return first, second


def conv_irregular(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    if x.size < 5:
        return np.interp(query, x, y)
    first, second = irregular_jets(x, y)
    h = np.diff(x)
    delta = np.diff(y)
    raw = np.empty((x.size-1, 5, 1), dtype=np.float64)
    raw[:, 0, 0] = h*first[:-1]/5
    raw[:, 1, 0] = h*first[:-1]/5 + h*h*second[:-1]/20
    raw[:, 2, 0] = delta - 2*h*(first[:-1]+first[1:])/5 + h*h*(second[1:]-second[:-1])/20
    raw[:, 3, 0] = h*first[1:]/5 - h*h*second[1:]/20
    raw[:, 4, 0] = h*first[1:]/5
    signs = ordered_sign_ledger(raw, delta[:, None])
    current = project_signed_fibres(raw, signs, delta[:, None])[..., 0]
    control = np.concatenate((y[:-1, None], y[:-1, None]+np.cumsum(current, axis=1)), axis=1)
    interval = np.clip(np.searchsorted(x, query, side="right")-1, 0, x.size-2)
    u = np.clip((query-x[interval])/(x[interval+1]-x[interval]), 0, 1)
    basis = np.stack([
        np.array((1, 5, 10, 10, 5, 1))[k]*u**k*(1-u)**(5-k)
        for k in range(6)
    ], axis=1)
    return np.sum(control[interval]*basis, axis=1)


def natural_cubic(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    n = x.size
    if n < 3:
        return np.interp(query, x, y)
    h = np.diff(x)
    matrix = np.zeros((n, n)); rhs = np.zeros(n)
    matrix[0, 0] = matrix[-1, -1] = 1
    for i in range(1, n-1):
        matrix[i, i-1:i+2] = (h[i-1], 2*(h[i-1]+h[i]), h[i])
        rhs[i] = 6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    moment = np.linalg.solve(matrix, rhs)
    interval = np.clip(np.searchsorted(x, query, side="right")-1, 0, n-2)
    hi = h[interval]; a = (x[interval+1]-query)/hi; b = 1-a
    return (a*y[interval]+b*y[interval+1]
            + ((a**3-a)*moment[interval]+(b**3-b)*moment[interval+1])*hi**2/6)


def turning_count(values: np.ndarray) -> int:
    return max(0, extrema(values).size-2)


def interval_excursion(baseline: np.ndarray, knot_i: np.ndarray, knot_y: np.ndarray) -> float:
    excess = 0.0
    for k in range(knot_i.size-1):
        local = baseline[knot_i[k]:knot_i[k+1]+1]
        lo, hi = sorted((knot_y[k], knot_y[k+1]))
        excess = max(excess, float(np.max(np.maximum(lo-local, local-hi))))
    return max(excess, 0.0)


def cases(n: int = 512) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t = np.linspace(0, 1, n)
    trend1 = 0.8*(t-.5)**2 + 0.25*t
    trend2 = 0.25*np.sin(2*np.pi*1.2*t)+0.35*t
    step = np.where(t < .53, -.3, .35) + .2*t
    burst = .25*t + .18*np.sin(2*np.pi*.7*t)
    window = np.exp(-((t-.62)/.11)**2)
    rng = np.random.default_rng(91)
    return {
        "chirp + trend": (trend1 + .32*np.sin(2*np.pi*(7*t+20*t*t)), trend1),
        "riding waves": (trend2 + .22*np.sin(2*np.pi*24*t), trend2),
        "jump + oscillation": (step + .16*np.sin(2*np.pi*17*t), step),
        "localized burst": (burst + .42*window*np.sin(2*np.pi*31*t), burst),
        "noise + smooth": (trend1 + .16*rng.standard_normal(n), trend1),
    }


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, float|int]] = {}
    examples = cases()
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 2.5*len(examples)), constrained_layout=True)
    for row, (name, (signal, truth)) in enumerate(examples.items()):
        index = extrema(signal); knot = itd_knots(signal, index)
        query = np.arange(signal.size, dtype=np.float64)
        cubic = natural_cubic(index.astype(float), knot, query)
        conv = conv_irregular(index.astype(float), knot, query)
        records[name] = {
            "sample_extrema": int(index.size-2),
            "knot_extrema": int(turning_count(knot)),
            "cubic_baseline_extrema": int(turning_count(cubic)),
            "conv_baseline_extrema": int(turning_count(conv)),
            "cubic_interval_excursion": interval_excursion(cubic, index, knot),
            "conv_interval_excursion": interval_excursion(conv, index, knot),
            "cubic_truth_mse": float(np.mean((cubic-truth)**2)),
            "conv_truth_mse": float(np.mean((conv-truth)**2)),
            "cubic_rotation_range": float(np.ptp(signal-cubic)),
            "conv_rotation_range": float(np.ptp(signal-conv)),
        }
        ax = axes[row, 0]
        ax.plot(signal, color="0.75", lw=.7, label="signal")
        ax.plot(truth, "k--", lw=1, label="known trend")
        ax.plot(cubic, color="#d97706", lw=1.2, label="ITD natural cubic")
        ax.plot(conv, color="#0369a1", lw=1.2, label="CONV knots")
        ax.scatter(index, knot, s=5, color="black", zorder=5)
        ax.set_title(name); ax.set_xticks([])
        if row == 0: ax.legend(ncol=4, fontsize=7)
        ax = axes[row, 1]
        ax.plot(signal-cubic, color="#d97706", lw=.8, label="cubic rotation")
        ax.plot(signal-conv, color="#0369a1", lw=.8, label="CONV rotation")
        ax.axhline(0, color="0.5", lw=.5); ax.set_xticks([])
        if row == 0: ax.legend(fontsize=7)
    fig.savefig(OUT / "itd_conv_vs_cubic.png", dpi=180)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
