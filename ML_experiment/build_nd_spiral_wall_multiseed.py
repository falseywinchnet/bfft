#!/usr/bin/env python3
"""Aggregate independent N-D spiral rotations into one viewer."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DIRECTORIES = [HERE / f"results_nd_spiral_wall_v2_seed{seed}" for seed in range(3)]
OUTPUT = HERE / "nd_spiral_wall.html"
KNOWN_FIXED_SCALARS = {
    "self_context": 3456,
    "cff_fast": 3472,
    "cff_fast_muon": 3472,
    "fixed_random_bispectrum": 18432,
}


def main():
    grouped = {}
    for seed, directory in enumerate(DIRECTORIES):
        for row in json.loads((directory / "results.json").read_text())["runs"]:
            grouped.setdefault(row["configuration"], []).append({**row, "seed": seed})
    summaries = []
    for configuration, rows in grouped.items():
        tails = np.asarray([row["tail_accuracy"] for row in rows])
        vals = np.asarray([row["validation_accuracy"] for row in rows])
        summaries.append({
            "configuration": configuration, "label": rows[0]["label"],
            "tail_mean": float(tails.mean()), "tail_min": float(tails.min()),
            "tail_max": float(tails.max()), "val_mean": float(vals.mean()),
            "parameters": rows[0]["parameters"],
            "fixed_scalars": rows[0].get(
                "fixed_scalars", KNOWN_FIXED_SCALARS.get(configuration, 0)
            ),
            "seconds_mean": float(np.mean([row["seconds"] for row in rows])),
            "runs": rows,
        })
    summaries.sort(key=lambda row: row["tail_mean"], reverse=True)

    labels = [row["label"] for row in summaries][::-1]
    means = np.asarray([row["tail_mean"] for row in summaries][::-1])
    low = means - np.asarray([row["tail_min"] for row in summaries][::-1])
    high = np.asarray([row["tail_max"] for row in summaries][::-1]) - means
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    colors = ["#256d45" if value >= .99 else "#3979a8" if value >= .8 else "#bc6c35" for value in means]
    axis.barh(labels, means, xerr=np.stack((low, high)), color=colors, alpha=.9,
              error_kw={"capsize": 3, "elinewidth": 1})
    axis.axvline(.5, color="#555", linewidth=1); axis.set_xlim(0, 1.02)
    axis.set_xlabel("unseen-region accuracy (mean and range across 3 rotations)")
    axis.set_title("What survives a changed 16-D task rotation?")
    axis.grid(axis="x", alpha=.2)
    figure.savefig(HERE / "nd_spiral_wall_multiseed.png", dpi=170)
    plt.close(figure)

    rows_html = "".join(
        f"<tr><td>{html.escape(row['label'])}</td><td>{row['val_mean']:.3f}</td>"
        f"<td>{row['tail_mean']:.3f}</td><td>{row['tail_min']:.3f}</td>"
        f"<td>{row['parameters']:,}</td><td>{row['fixed_scalars']:,}</td>"
        f"<td>{row['seconds_mean']:.2f}s</td></tr>"
        for row in summaries)
    seed_links = " ".join(
        f"<a href='results_nd_spiral_wall_v2_seed{seed}/index.html'>rotation {seed}: all 18 manifold plots</a>"
        for seed in range(3))
    featured_names = ("fixed_random_bispectrum", "shallow_odd_cubic", "cff_fast",
                      "self_context", "even_quadratic_control", "midpoint_hessian")
    cards = "".join(
        f"<article><h2>{html.escape(grouped[name][0]['label'])}</h2>"
        f"<img src='results_nd_spiral_wall_v2_seed0/{name}.png'></article>"
        for name in featured_names)
    OUTPUT.write_text(f"""<!doctype html><meta charset='utf-8'><title>N-D spiral wall: three rotations</title>
<style>body{{font:15px system-ui;margin:0;background:#f4f1e9;color:#222}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin-bottom:6px}}.lede{{font-size:17px;max-width:1050px}}.warning{{padding:14px 18px;background:#fff3cd;border-left:5px solid #d49b17;max-width:1100px}}a{{color:#145b8c;margin-right:18px}}.hero{{width:100%;background:white;margin:20px 0}}table{{width:100%;border-collapse:collapse;background:white;margin:22px 0}}th,td{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px}}article{{background:white;padding:12px;box-shadow:0 1px 5px #0002}}article h2{{font-size:16px;margin:0 0 8px}}article img{{width:100%}}</style>
<main><h1>N-D spiral wall screen · three independent rotations</h1>
<p class='lede'>Eighteen mechanisms, 500 CPU steps, one 16-D eight-plane spiral task. Models saw only individual observed-region vectors. The task coordinate, known rotation, harmonic decomposition, neighbors, and future samples were never model inputs.</p>
<p class='warning'><b>The surprise is also a diagnosis of the benchmark.</b> The two branches are exact antipodes. Even-order features erase their identity; odd third-order random features expose a global separator. This is genuine extrapolation under the stated generator, but it is not evidence for universal manifold induction. “Parameters” means learned scalars; frozen atlas storage is shown separately.</p>
<p>{seed_links}</p><img class='hero' src='nd_spiral_wall_multiseed.png'>
<table><thead><tr><th>mechanism</th><th>observed val</th><th>unseen mean</th><th>unseen worst rotation</th><th>learned scalars</th><th>fixed atlas scalars</th><th>train mean</th></tr></thead><tbody>{rows_html}</tbody></table>
<section class='gallery'>{cards}</section></main>""")
    print(OUTPUT)


if __name__ == "__main__": main()
