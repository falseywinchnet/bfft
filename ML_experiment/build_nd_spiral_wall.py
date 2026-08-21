#!/usr/bin/env python3
"""Render the M4-produced N-D spiral wall screen without requiring Torch."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_model(row, probe, out):
    u = np.asarray(probe["u"])
    coordinates = [np.asarray(value) for value in probe["coordinates"]]
    probability = [np.asarray(value) for value in probe["probabilities"]]
    figure = plt.figure(figsize=(13.6, 4.1), constrained_layout=True)
    truth = figure.add_subplot(131, projection="3d")
    predicted = figure.add_subplot(132, projection="3d")
    horizon = figure.add_subplot(133)
    for label, coordinate in enumerate(coordinates):
        color = "#2474b5" if label == 0 else "#ef7b2d"
        truth.plot(*coordinate.T, color=color, linewidth=1.5, alpha=0.9)
        predicted.scatter(*coordinate.T, c=probability[label], cmap="coolwarm",
                          vmin=0, vmax=1, s=4, alpha=0.9)
        horizon.plot(u, probability[label], color=color, linewidth=1.5,
                     label=f"true class {label}")
    for axis in (truth, predicted):
        axis.set_xlabel("PC1"); axis.set_ylabel("PC2"); axis.set_zlabel("PC3")
        axis.view_init(elev=23, azim=-61)
    truth.set_title("True two-branch manifold")
    predicted.set_title("Predicted P(class 1) on manifold")
    horizon.axvspan(0.015, 0.5, color="#d5ead8", alpha=0.45, label="observed radius")
    horizon.axvline(0.5, color="black", linestyle="--", linewidth=1)
    horizon.axhline(0.5, color="#777", linewidth=0.8)
    horizon.set(xlabel="latent path coordinate (viewer only)", ylabel="P(class 1)",
                ylim=(-0.03, 1.03), title="Observed fit and unseen continuation")
    horizon.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{row['label']}  |  val {row['validation_accuracy']:.3f}  "
        f"tail {row['tail_accuracy']:.3f}  |  {row['parameters']:,} params", fontsize=12)
    filename = f"{row['configuration']}.png"
    figure.savefig(out / filename, dpi=155)
    plt.close(figure)
    return filename


def plot_overviews(rows, out):
    columns = 4
    row_count = int(np.ceil(len(rows) / columns))
    figure, axes = plt.subplots(row_count, columns, figsize=(16, 3.25 * row_count),
                               sharex=True, sharey=True,
                               constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        bins = row["tail_bins"]
        axis.plot(np.linspace(0.525, 0.975, len(bins)), bins, marker="o", markersize=3)
        axis.axhline(0.5, color="#888", linewidth=0.7)
        axis.set_title(f"{row['label']}\nval {row['validation_accuracy']:.2f} / tail {row['tail_accuracy']:.2f}", fontsize=9)
        axis.set_ylim(0.35, 1.02)
    for axis in axes.flat[len(rows):]: axis.axis("off")
    figure.supxlabel("unseen radial bin center"); figure.supylabel("accuracy")
    figure.suptitle("N-D spiral continuation survival", fontsize=15)
    figure.savefig(out / "tail_survival_overview.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for row in rows:
        axis.plot([point["step"] for point in row["history"]],
                  [point["accuracy"] for point in row["history"]], label=row["label"], linewidth=1.4)
    axis.set(xlabel="training step", ylabel="observed-region validation accuracy",
             ylim=(0.45, 1.01), title="Acquisition speed")
    axis.legend(ncol=2, fontsize=8); axis.grid(alpha=0.25)
    figure.savefig(out / "learning_curves.png", dpi=160); plt.close(figure)


def write_html(rows, out):
    sorted_rows = sorted(rows, key=lambda row: row["tail_accuracy"], reverse=True)
    table_rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td><td>{row['validation_accuracy']:.3f}</td>"
        f"<td>{row['tail_accuracy']:.3f}</td><td>{row['tail_class_0']:.3f}</td>"
        f"<td>{row['tail_class_1']:.3f}</td><td>{row['antipodal_error']:.3f}</td>"
        f"<td>{row['parameters']:,}</td><td>{row['seconds']:.1f}s</td>"
        f"<td>{row['inference_ms_2000']:.1f}ms</td></tr>" for row in sorted_rows)
    cards = "".join(f"<article><h2>{html.escape(row['label'])}</h2><img src='{row['image']}' loading='lazy'></article>" for row in rows)
    document = f"""<!doctype html><meta charset='utf-8'><title>N-D spiral wall screen</title>
<style>body{{font:15px system-ui;margin:0;background:#f5f3ed;color:#202020}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{margin-bottom:4px}}.note{{max-width:1000px;color:#4b4b4b}}table{{border-collapse:collapse;background:white;width:100%;margin:24px 0}}th,td{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.overview{{width:100%;background:white;margin:12px 0}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px}}article{{background:white;padding:12px;box-shadow:0 1px 5px #0002}}article h2{{font-size:16px;margin:0 0 8px}}article img{{width:100%}}</style>
<main><h1>N-D spiral: throw ideas at the wall</h1><p class='note'>All models saw only individual 16-D observations from u≤0.5. The path coordinate, task rotation, harmonic planes, neighbors, and tail labels were unavailable to the models. The 3-D PCA and u-axis below are viewer-only diagnostics. Sorting by tail is presentation only; no model was selected or tuned from it.</p><img class='overview' src='tail_survival_overview.png'><img class='overview' src='learning_curves.png'><table><thead><tr><th>model</th><th>val</th><th>tail</th><th>class 0</th><th>class 1</th><th>antipodal err ↓</th><th>params</th><th>train</th><th>infer / 2k</th></tr></thead><tbody>{table_rows}</tbody></table><section class='gallery'>{cards}</section></main>"""
    (out / "index.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("directory", type=Path); args = parser.parse_args()
    rows = json.loads((args.directory / "results.json").read_text())["runs"]
    probes = {p["configuration"]: p for p in json.loads((args.directory / "probes.json").read_text())["probes"]}
    for row in rows: row["image"] = plot_model(row, probes[row["configuration"]], args.directory)
    plot_overviews(rows, args.directory); write_html(rows, args.directory)
    print(args.directory / "index.html")


if __name__ == "__main__": main()
