"""Render the saved full zonotopic-enrichment comparison without recomputing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.conv_itd_multicomponent import make_signal


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "conv_itd_zonotopic_enrichment"


def main() -> None:
    import matplotlib.pyplot as plt

    data = np.load(OUT/"decompositions.npz")
    metrics = json.loads((OUT/"metrics.json").read_text(encoding="utf-8"))["methods"]
    t, signal, truth = make_signal(data["signal"].size)
    selected = (
        ("plain CONV-ITD", "plain_CONV-ITD", "#475569"),
        ("static a=0.05", "static_a_0_05", "#b45309"),
        ("transported a=0.05", "transported_a_0_05", "#0369a1"),
    )
    rows = 7
    fig, axes = plt.subplots(rows, len(selected), figsize=(18, 12), sharex=True, constrained_layout=True)
    for column, (name, key, color) in enumerate(selected):
        baseline = data[f"{key}_baseline"]
        rotations = data[f"{key}_rotations"]
        record = metrics[name]
        ax = axes[0, column]
        ax.plot(t, signal, color=".84", lw=.45, label="composite")
        ax.plot(t, truth["trend"], "k--", lw=.9, label="truth trend")
        ax.plot(t, baseline, color=color, lw=1.1, label="final baseline")
        ax.set_title(
            f"{name}\nmean best |r|={record['mean_best_abs_correlation']:.4f}, "
            f"trend MSE={record['trend_mse']:.4g}, {record['rotation_count']} rotations",
            fontsize=9,
        )
        for row in range(1, 6):
            ax = axes[row, column]
            if row <= rotations.shape[0]:
                ax.plot(t, rotations[row-1], color=color, lw=.65)
                ax.set_ylabel(f"R{row}", rotation=0, ha="right", va="center")
            else:
                ax.axis("off")
        ax = axes[6, column]
        ax.plot(t, baseline-truth["trend"], color=color, lw=.75)
        ax.axhline(0.0, color="k", lw=.45)
        ax.set_ylabel("trend\nerror", rotation=0, ha="right", va="center")
        ax.set_xlabel("normalized time")
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.savefig(OUT/"transported_measure_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for name, _, color in selected:
        value = metrics[name]
        level = np.arange(1, len(value["levels"])+1)
        if name != "plain CONV-ITD":
            axes[0].plot(
                level,
                [row["noise_rotation_rms"] for row in value["levels"]],
                marker="o", ms=3, color=color, label=name,
            )
        axes[1].plot(
            level,
            [row["signal_output_extrema"] for row in value["levels"]],
            marker="o", ms=3, color=color, label=name,
        )
    axes[0].set(title="transported auxiliary rotation energy", xlabel="level", ylabel="noise-rotation RMS")
    axes[1].set(title="signal extrema clock", xlabel="level", ylabel="output extrema")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(OUT/"measure_transport_diagnostic.png", dpi=200)
    plt.close(fig)

    count_audit = json.loads((OUT/"probe_count_audit.json").read_text(encoding="utf-8"))
    seed_audit = json.loads((OUT/"seed_audit.json").read_text(encoding="utf-8"))
    count = np.array([row["probe_count"] for row in count_audit["transported"]])
    trend = np.array([row["trend_mse"] for row in count_audit["transported"]])
    correlation = np.array([row["mean_best_abs_correlation"] for row in count_audit["transported"]])
    seeds = np.array([row["seed"] for row in seed_audit["transported"]])
    seed_trend = np.array([row["trend_mse"] for row in seed_audit["transported"]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(count, trend, "o-", color="#0369a1")
    axes[0].axhline(count_audit["plain"]["trend_mse"], color=".35", ls="--", label="plain")
    axes[0].set(xlabel="cubature generators", ylabel="trend MSE", title="angular integration convergence")
    axes[1].plot(count, correlation, "o-", color="#0369a1")
    axes[1].axhline(count_audit["plain"]["mean_best_abs_correlation"], color=".35", ls="--", label="plain")
    axes[1].set(xlabel="cubature generators", ylabel="mean best |r|", title="component alignment")
    axes[2].plot(seeds-seeds[0]+1, seed_trend, "o", color="#7c3aed")
    axes[2].axhline(seed_audit["plain"]["trend_mse"], color=".35", ls="--", label="plain")
    axes[2].set(xlabel="eight-generator bank", ylabel="trend MSE", title="under-resolved bank variation")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(OUT/"cubature_convergence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
