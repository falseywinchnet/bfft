"""Fully CONV-clocked ITD on a noisy multi-component signal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_itd_comparison import conv_irregular  # noqa: E402
from experiments.conv_itd_extrema_ablation import (  # noqa: E402
    conv_current_extrema,
    itd_knots_at,
)


OUT = ROOT / "experiments" / "out" / "conv_itd_multicomponent"


def make_signal(n: int = 2048) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    t = np.linspace(0.0, 1.0, n)
    trend = 0.42*(t-.48)**2 + 0.24*t - 0.08
    steady = 0.18*np.sin(2*np.pi*33*t + 0.35)
    chirp = 0.24*np.sin(2*np.pi*(6*t + 13*t*t))
    envelope = np.exp(-((t-.64)/.075)**2)
    burst = 0.31*envelope*np.sin(2*np.pi*91*t - .4)
    am = 0.12*(1 + .55*np.sin(2*np.pi*.8*t))*np.sin(2*np.pi*16*t + .8)
    rng = np.random.default_rng(20260830)
    noise = 0.055*rng.standard_normal(n)
    parts = {
        "steady 33-cycle": steady,
        "quadratic chirp": chirp,
        "localized 91-cycle burst": burst,
        "amplitude-modulated 16-cycle": am,
        "seeded white noise": noise,
    }
    return t, trend + sum(parts.values()), {"trend": trend, **parts}


def decompose(signal: np.ndarray, max_levels: int = 14) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, float|int]]]:
    query = np.arange(signal.size, dtype=np.float64)
    baseline = np.asarray(signal, dtype=np.float64).copy()
    rotations: list[np.ndarray] = []
    records: list[dict[str, float|int]] = []
    for level in range(max_levels):
        ext_x, ext_y = conv_current_extrema(baseline)
        extrema_count = ext_x.size-2
        if extrema_count < 2 or ext_x.size < 5:
            break
        knot = itd_knots_at(ext_x, ext_y)
        next_baseline = conv_irregular(ext_x, knot, query)
        rotation = baseline-next_baseline
        spectrum = np.abs(np.fft.rfft(rotation-np.mean(rotation)))
        dominant_bin = int(np.argmax(spectrum[1:])+1) if spectrum.size > 1 else 0
        rotations.append(rotation)
        records.append({
            "level": level+1,
            "input_extrema": int(extrema_count),
            "baseline_extrema": int(conv_current_extrema(next_baseline)[0].size-2),
            "rotation_rms": float(np.sqrt(np.mean(rotation*rotation))),
            "dominant_cycles_per_record": dominant_bin,
            "rotation_mean": float(np.mean(rotation)),
        })
        baseline = next_baseline
    return rotations, baseline, records


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    t, signal, truth = make_signal()
    rotations, final_baseline, records = decompose(signal)
    reconstruction = np.sum(rotations, axis=0)+final_baseline
    closure = float(np.max(np.abs(signal-reconstruction)))

    # Correlations are diagnostics only; they do not assign or fit modes.
    truth_names = [name for name in truth if name != "trend"]
    correlation = np.zeros((len(rotations), len(truth_names)))
    for i, rotation in enumerate(rotations):
        for j, name in enumerate(truth_names):
            correlation[i, j] = np.corrcoef(rotation, truth[name])[0, 1]
    summary = {
        "sample_count": signal.size,
        "rotation_count": len(rotations),
        "reconstruction_closure_linf": closure,
        "final_baseline_truth_mse": float(np.mean((final_baseline-truth["trend"])**2)),
        "final_baseline_extrema": int(conv_current_extrema(final_baseline)[0].size-2),
        "rotations": records,
        "truth_component_names": truth_names,
        "rotation_truth_correlations": correlation.tolist(),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(
        OUT / "decomposition.npz",
        t=t.astype(np.float32), signal=signal.astype(np.float32),
        rotations=np.stack(rotations).astype(np.float32),
        final_baseline=final_baseline.astype(np.float32),
        reconstruction=reconstruction.astype(np.float32),
        **{f"truth_{name.replace(' ', '_')}": value.astype(np.float32)
           for name, value in truth.items()},
    )

    rows = len(rotations)+2
    fig, axes = plt.subplots(rows, 1, figsize=(14, 1.55*rows), sharex=True, constrained_layout=True)
    axes[0].plot(t, signal, color="0.35", lw=.55, label="noisy composite")
    axes[0].plot(t, truth["trend"], "k--", lw=1.1, label="known trend")
    axes[0].legend(ncol=2, fontsize=8); axes[0].set_title("fully CONV-clocked ITD")
    for i, rotation in enumerate(rotations):
        axes[i+1].plot(t, rotation, color="#0369a1", lw=.65)
        record = records[i]
        axes[i+1].set_ylabel(f"R{i+1}\n{record['dominant_cycles_per_record']} cyc", rotation=0,
                             ha="right", va="center", fontsize=7)
    axes[-1].plot(t, final_baseline, color="#15803d", lw=1.1, label="final CONV baseline")
    axes[-1].plot(t, truth["trend"], "k--", lw=.9, label="known trend")
    axes[-1].legend(fontsize=8); axes[-1].set_xlabel("normalized time")
    fig.savefig(OUT / "conv_itd_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, max(3, .45*len(rotations)+1.5)), constrained_layout=True)
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(truth_names)), truth_names, rotation=25, ha="right")
    ax.set_yticks(range(len(rotations)), [f"R{i+1}" for i in range(len(rotations))])
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("diagnostic alignment with known generating components")
    fig.savefig(OUT / "component_correlations.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
