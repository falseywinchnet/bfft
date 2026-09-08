"""Render the proof certificate and convergence curves for muon_primacy.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "exact_muon": "#087e8b",
    "sgd": "#ff5a5f",
    "adam": "#6c5ce7",
    "adam_sign_limit": "#6c5ce7",
}
LABELS = {
    "exact_muon": "exact Muon",
    "sgd": "SGD",
    "adam": "Adam",
    "adam_sign_limit": "Adam sign limit",
}


def _selected_runs(payload: dict, section: str, best_section: str) -> list[dict]:
    runs = payload[section]
    return [
        next(
            row for row in runs
            if row["method"] == method and row["lr"] == best["lr"]
        )
        for method, best in payload[best_section].items()
    ]


def _curves(axis, runs: list[dict], key: str, title: str) -> None:
    for run in runs:
        axis.plot(
            [point["step"] for point in run["history"]],
            [max(point[key], 1e-30) for point in run["history"]],
            color=COLORS[run["method"]],
            linewidth=2.2,
            label=f'{LABELS[run["method"]]}  (lr={run["lr"]:g})',
        )
    axis.set_yscale("log")
    axis.set_xlabel("optimizer updates")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(True, which="both", alpha=0.18)
    axis.legend(frameon=False, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    figure.suptitle(
        "Muon primacy witness: spectrum removal versus scalar/coordinate scaling",
        fontsize=16,
        fontweight="bold",
    )

    certificate = payload["one_step_certificate"]
    names = ["exact_muon", "sgd", "adam_sign_limit"]
    values = [
        max(certificate[name]["relative_loss_after_one_step"], 1e-30)
        for name in names
    ]
    axes[0, 0].bar(
        [LABELS[name] for name in names],
        values,
        color=[COLORS[name] for name in names],
        width=0.68,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("best one-step loss / initial loss")
    axes[0, 0].set_title(
        "A  Population one-step certificate", loc="left", fontweight="bold"
    )
    axes[0, 0].grid(True, axis="y", which="both", alpha=0.18)
    axes[0, 0].text(
        0,
        values[0] * 4,
        "exact (numerical floor)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    population = _selected_runs(payload, "population_runs", "population_best")
    _curves(axes[0, 1], population, "relative_loss", "B  Population objective")

    block = _selected_runs(payload, "block_sweep_runs", "block_sweep_best")
    _curves(axes[1, 0], block, "relative_loss", "C  Rank-deficient block sweep")
    axes[1, 0].axvline(4, color="#222222", linestyle="--", linewidth=1)
    axes[1, 0].text(5, 2e-18, "one complete sweep", fontsize=8)

    _curves(
        axes[1, 1],
        block,
        "operator_error",
        "D  Full operator recovery (same block sweep)",
    )
    axes[1, 1].axvline(4, color="#222222", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel(r"$\|W-Q\|_2 / \|Q\|_2$")

    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_ylabel("loss / initial loss")
    figure.text(
        0.5,
        -0.01,
        "d=32, covariance condition number 10⁴, four rank-8 orthogonal blocks; "
        "learning rates selected from declared grids by fastest 10⁻⁸ objective crossing.",
        ha="center",
        fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
