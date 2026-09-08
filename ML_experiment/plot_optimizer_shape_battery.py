"""Render the six-arm neural battery and operator-witness summary."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "adamw",
    "sgd",
    "anchor",
    "anchor_restrained_momentum",
    "muon",
    "lepton_transport",
)
LABELS = {
    "adamw": "AdamW",
    "sgd": "SGD",
    "anchor": "Anchor",
    "anchor_restrained_momentum": "R-M Anchor",
    "muon": "Muon",
    "lepton_transport": "T-Lepton",
}
COLORS = {
    "adamw": "#6c5ce7",
    "sgd": "#ff5a5f",
    "anchor": "#f39c12",
    "anchor_restrained_momentum": "#c46f00",
    "muon": "#087e8b",
    "lepton_transport": "#00a878",
}


def _aggregate_history(rows: list[dict], key: str):
    by_step = defaultdict(list)
    for row in rows:
        for point in row["history"]:
            value = float(point[key])
            if math.isfinite(value):
                by_step[int(point["step"])].append(value)
    steps = sorted(by_step)
    return steps, [float(np.mean(by_step[step])) for step in steps]


def _bar(axis, values, title, ylabel):
    x = np.arange(len(ORDER))
    axis.bar(x, values, color=[COLORS[name] for name in ORDER], width=0.72)
    axis.set_xticks(x, [LABELS[name] for name in ORDER], fontsize=8)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("neural", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    neural = json.loads(args.neural.read_text())
    operator = json.loads(args.operator.read_text())

    figure, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
    figure.suptitle(
        "Optimizer geometry battery: acquisition, generalization, and operator recovery",
        fontsize=16,
        fontweight="bold",
    )
    overall = {row["optimizer"]: row for row in summary["neural"]["overall"]}
    _bar(
        axes[0, 0],
        [overall[name]["learning_auc"] for name in ORDER],
        "A  Mean acquisition AUC (144 paired fits per arm)",
        "validation-score AUC",
    )
    _bar(
        axes[0, 1],
        [overall[name]["score"] for name in ORDER],
        "B  Mean held-out score at best validation checkpoint",
        "held-out score",
    )
    counts = summary["neural"]["robust_winner_counts"]
    _bar(
        axes[0, 2],
        [counts[name] for name in ORDER],
        "C  Robust task × architecture AUC winners",
        "winner pairs (of 48)",
    )

    pair_rows = summary["neural"]["pairs"]
    for variant, marker, label in (
        ("ordinary_mlp", "o", "ordinary MLP"),
        ("self_context", "^", "self-context"),
    ):
        selected = [row for row in pair_rows if row["variant"] == variant]
        deltas = [
            row["means"]["anchor_restrained_momentum"]["learning_auc"]
            - row["means"]["anchor"]["learning_auc"]
            for row in selected
        ]
        axes[1, 0].scatter(
            range(len(selected)), deltas, marker=marker, s=34, alpha=0.8,
            label=label,
        )
    axes[1, 0].axhline(0, color="#222", linewidth=1)
    axes[1, 0].set_title(
        "D  Restrained-memory Anchor minus Anchor", loc="left", fontweight="bold"
    )
    axes[1, 0].set_xlabel("task index within architecture")
    axes[1, 0].set_ylabel("mean acquisition-AUC delta")
    axes[1, 0].grid(True, axis="y", alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=8)

    native_block = [
        row for row in operator["runs"]
        if row["protocol"] == "standing"
        and row["scenario"] == "block_sweep"
    ]
    for optimizer in ORDER:
        rows = [row for row in native_block if row["optimizer"] == optimizer]
        steps, loss = _aggregate_history(rows, "relative_loss")
        _, error = _aggregate_history(rows, "operator_error")
        axes[1, 1].plot(
            steps, np.maximum(loss, 1e-30), color=COLORS[optimizer],
            linewidth=2, label=LABELS[optimizer].replace("\n", " "),
        )
        axes[1, 2].plot(
            steps, np.maximum(error, 1e-30), color=COLORS[optimizer],
            linewidth=2, label=LABELS[optimizer].replace("\n", " "),
        )
    for axis, title, ylabel in (
        (axes[1, 1], "E  Standing-LR block loss", "loss / initial loss"),
        (axes[1, 2], "F  Standing-LR full recovery", "relative operator error"),
    ):
        axis.set_yscale("log")
        axis.set_xlabel("optimizer updates")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.axvline(4, color="#222", linestyle="--", linewidth=1)
        axis.grid(True, which="both", alpha=0.18)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].scatter([4], [5e-28], marker="*", s=90, color=COLORS["muon"], zorder=6)
    axes[1, 2].scatter([4], [2e-13], marker="*", s=90, color=COLORS["muon"], zorder=6)
    axes[1, 1].text(8, 2e-25, "exact polar primitive", fontsize=7, color=COLORS["muon"])
    axes[1, 2].text(8, 8e-12, "exact polar primitive", fontsize=7, color=COLORS["muon"])

    figure.text(
        0.5, -0.01,
        "Neural battery: 24 tasks × 2 architectures × 3 seeds. "
        "Operator witness: d=32, κ=10⁴, four rank-8 blocks × 3 rotations.",
        ha="center", fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
