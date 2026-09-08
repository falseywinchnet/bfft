"""Representation comparison: canonical cubic ITD versus fully CONV ITD."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_itd_comparison import natural_cubic  # noqa: E402
from experiments.conv_itd_extrema_ablation import (  # noqa: E402
    conv_current_extrema,
    itd_knots_at,
)
from experiments.conv_itd_comparison import conv_irregular  # noqa: E402
from experiments.conv_itd_multicomponent import make_signal  # noqa: E402


OUT = ROOT / "experiments" / "out" / "conv_itd_vs_canonical"


def matlab_extrema(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PyITD/MATLAB one-sided plateau convention for maxima and minima."""

    x = np.asarray(signal, dtype=np.float64)
    dx = np.diff(x)
    maxima = np.flatnonzero(
        (np.r_[dx, 0.0] > 0.0)
        & (np.r_[0.0, dx] <= 0.0)
    )
    # Spell the minimum condition independently to retain the same one-sided
    # plateau convention as applying the selector to -x.
    ndx = -dx
    minima = np.flatnonzero(
        (np.r_[ndx, 0.0] > 0.0)
        & (np.r_[0.0, ndx] <= 0.0)
    )
    interior = np.unique(np.r_[maxima, minima])
    interior = interior[(interior > 0) & (interior < x.size-1)]
    index = np.r_[0, interior, x.size-1].astype(np.float64)
    return index, x[index.astype(np.int64)]


def decompose(
    signal: np.ndarray,
    kind: str,
    max_levels: int = 14,
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, float|int]]]:
    query = np.arange(signal.size, dtype=np.float64)
    state = np.asarray(signal, dtype=np.float64).copy()
    rotations: list[np.ndarray] = []
    records: list[dict[str, float|int]] = []
    for level in range(max_levels):
        ex, ey = (
            matlab_extrema(state) if kind == "canonical"
            else conv_current_extrema(state)
        )
        count = ex.size-2
        if count < 2 or ex.size < 5:
            break
        knot = itd_knots_at(ex, ey)
        next_state = (
            natural_cubic(ex, knot, query)
            if kind == "canonical"
            else conv_irregular(ex, knot, query)
        )
        rotation = state-next_state
        spectrum = np.abs(np.fft.rfft(rotation-np.mean(rotation)))
        rotations.append(rotation)
        records.append({
            "level": level+1,
            "input_extrema": int(count),
            "output_extrema": int((matlab_extrema(next_state)[0].size-2)
                                  if kind == "canonical"
                                  else (conv_current_extrema(next_state)[0].size-2)),
            "rms": float(np.sqrt(np.mean(rotation*rotation))),
            "dominant_cycles": int(np.argmax(spectrum[1:])+1),
        })
        state = next_state
    return rotations, state, records


def representation_metrics(
    rotations: list[np.ndarray],
    baseline: np.ndarray,
    truth: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    names = [name for name in truth if name != "trend"]
    corr = np.array([
        [np.corrcoef(rotation, truth[name])[0, 1] for name in names]
        for rotation in rotations
    ])
    absolute = np.abs(corr)
    best_mode = np.argmax(absolute, axis=0)
    best_corr = absolute[best_mode, np.arange(len(names))]
    # How much each generator leaks into non-best rotations, measured in
    # squared correlation coordinates rather than reconstruction energy.
    concentration = best_corr**2 / np.maximum(np.sum(absolute**2, axis=0), 1e-30)
    mode_purity = []
    for row in absolute:
        ordered = np.sort(row)
        mode_purity.append(float(ordered[-1]-ordered[-2]) if row.size > 1 else float(ordered[-1]))
    burst_name = "localized 91-cycle burst"
    burst_column = names.index(burst_name)
    burst_mode = int(best_mode[burst_column])
    t = np.linspace(0, 1, baseline.size)
    window = np.abs(t-.64) <= .15
    selected = rotations[burst_mode]
    burst_localization = float(
        np.sum(selected[window]**2) / max(np.sum(selected**2), 1e-30)
    )
    return {
        "component_names": names,
        "best_rotation_by_component": (best_mode+1).tolist(),
        "best_abs_correlation": best_corr.tolist(),
        "correlation_concentration": concentration.tolist(),
        "mean_best_abs_correlation": float(np.mean(best_corr)),
        "mean_correlation_concentration": float(np.mean(concentration)),
        "mean_mode_purity_margin": float(np.mean(mode_purity)),
        "burst_rotation": burst_mode+1,
        "burst_energy_localization": burst_localization,
        "trend_mse": float(np.mean((baseline-truth["trend"])**2)),
        "trend_correlation": float(np.corrcoef(baseline, truth["trend"])[0, 1]),
    }, corr


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    t, signal, truth = make_signal()
    outputs = {}
    for kind in ("canonical", "conv"):
        rotations, baseline, levels = decompose(signal, kind)
        metrics, corr = representation_metrics(rotations, baseline, truth)
        outputs[kind] = {
            "rotations": rotations, "baseline": baseline,
            "levels": levels, "metrics": metrics, "corr": corr,
        }

    summary = {
        kind: {"levels": data["levels"], "representation": data["metrics"]}
        for kind, data in outputs.items()
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(
        OUT / "decompositions.npz",
        t=t.astype(np.float32), signal=signal.astype(np.float32),
        canonical_rotations=np.stack(outputs["canonical"]["rotations"]).astype(np.float32),
        canonical_baseline=np.asarray(outputs["canonical"]["baseline"], dtype=np.float32),
        conv_rotations=np.stack(outputs["conv"]["rotations"]).astype(np.float32),
        conv_baseline=np.asarray(outputs["conv"]["baseline"], dtype=np.float32),
    )

    rows = max(len(outputs["canonical"]["rotations"]), len(outputs["conv"]["rotations"]))+2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 1.45*rows), sharex=True, constrained_layout=True)
    for column, kind in enumerate(("canonical", "conv")):
        label = "canonical MATLAB-extrema + cubic" if kind == "canonical" else "CONV-extrema + CONV"
        axes[0, column].plot(t, signal, color="0.4", lw=.55)
        axes[0, column].set_title(label)
        rotations = outputs[kind]["rotations"]
        levels = outputs[kind]["levels"]
        for row in range(1, rows-1):
            if row <= len(rotations):
                axes[row, column].plot(t, rotations[row-1], lw=.65,
                                       color="#d97706" if kind=="canonical" else "#0369a1")
                axes[row, column].set_ylabel(
                    f"R{row}\n{levels[row-1]['dominant_cycles']} cyc",
                    rotation=0, ha="right", va="center", fontsize=7)
            else:
                axes[row, column].axis("off")
        axes[-1, column].plot(t, outputs[kind]["baseline"], lw=1,
                              color="#d97706" if kind=="canonical" else "#15803d")
        axes[-1, column].plot(t, truth["trend"], "k--", lw=.8)
        axes[-1, column].set_xlabel("normalized time")
        axes[-1, column].set_ylabel("trend", rotation=0, ha="right")
    fig.savefig(OUT / "rotation_comparison.png", dpi=180)
    plt.close(fig)

    names = outputs["canonical"]["metrics"]["component_names"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, kind in zip(axes, ("canonical", "conv")):
        corr = outputs[kind]["corr"]
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(corr.shape[0]), [f"R{i+1}" for i in range(corr.shape[0])])
        ax.set_title("canonical ITD" if kind=="canonical" else "fully CONV ITD")
    fig.colorbar(image, ax=axes, label="Pearson correlation")
    fig.savefig(OUT / "representation_correlations.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
