#!/usr/bin/env python3
"""Render retained complex-spiral fits with the user's Matplotlib 3-D setup."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CONFIGURATIONS = (
    ("ordinary_mlp", "Ordinary MLP"),
    ("self_context", "Self-context"),
    ("frame_reference", "Continuous frame flow (AdamW)"),
    ("frame_muon", "Continuous frame flow (Muon)"),
    ("frame_capacity", "Continuous frame flow (width 32)"),
    ("frame_fast", "Continuous frame flow (two-probe)"),
)


def pretty_style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fcfcfd",
        "axes.grid": True,
        "grid.alpha": .25,
        "axes.edgecolor": "#e0e0e0",
        "axes.linewidth": 1.0,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8,
        "font.size": 10,
        "savefig.bbox": "tight",
    })


def complex_spiral(t):
    """Exact analytic generator from the supplied experiment."""
    radius = (.12*t + .015*t**2 + .07*np.sin(3.1*t)
              + .015*np.sin(.7*t**2))
    angle = t + .25*np.sin(5*t) + .00035*t**3
    z = .055*t + .018*np.sin(2.3*t) + .00055*t**2 - .000002*t**3
    return np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))


def scatter_spiral(ax, t, points, title):
    t_normalized = (t - t.min()) / max(1e-8, t.max() - t.min())
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=plt.get_cmap("turbo")(t_normalized), s=8, depthshade=True,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=.25)
    ax.view_init(elev=22, azim=45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probes", type=Path,
        default=HERE / "results_frame_refinement/probes.json",
    )
    parser.add_argument(
        "--results", type=Path,
        default=HERE / "results_frame_refinement/results.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=HERE / "complex_spiral_3d.png",
    )
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    pretty_style()

    probes = json.loads(args.probes.read_text())["probes"]
    rows = {
        row["configuration"]: row for row in probes
        if row["task"] == "complex_spiral_3d"
    }
    result_rows = json.loads(args.results.read_text())["runs"]
    scores = {
        row["configuration"]: row["score"] for row in result_rows
        if row["task"] == "complex_spiral_3d" and row["seed"] == 0
    }

    total = 12 * math.pi
    t_truth = np.linspace(0, total, 2000, dtype=np.float64)
    truth = complex_spiral(t_truth)
    source = rows["ordinary_mlp"]
    t_models = (np.asarray(source["input"], dtype=np.float64) + 1) * total / 2

    fig = plt.figure(figsize=(18, 9.6), constrained_layout=True)
    truth_axis = fig.add_subplot(2, 4, 1, projection="3d")
    scatter_spiral(truth_axis, t_truth, truth, "Complex 3-D Spiral: Ground Truth")

    for index, (configuration, name) in enumerate(CONFIGURATIONS, start=2):
        axis = fig.add_subplot(2, 4, index, projection="3d")
        points = np.asarray(rows[configuration]["prediction"], dtype=np.float64)
        scatter_spiral(axis, t_models, points, f"{name}\nscore {scores[configuration]:.3f}")
    unused = fig.add_subplot(2, 4, 8)
    unused.axis("off")
    unused.text(
        .5, .57, "Color: t / T", ha="center", va="center", fontsize=12,
    )
    unused.text(
        .5, .43, "Observed: 0–0.5\nContinuation: 0.5–1",
        ha="center", va="center", fontsize=11,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
