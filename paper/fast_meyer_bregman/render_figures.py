#!/usr/bin/env python3
"""Render the finite-flow Meyer paper figures from serialized evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FLOW = RESULTS / "flow_natural_m4"


def lineage() -> None:
    events = [
        ("Jun 23", "fast STFT\nsuper-resolution"),
        ("Jul 10", "transport replaces\nscalar interpolation"),
        ("Jul 21", "Meyer--Bregman\nreduction"),
        ("Jul 27", "six-field reduced\nstate"),
        ("Jul 29", "fused spectral\ntriangle"),
        ("Jul 31", "FACR and fixed\nshape policy"),
        ("Aug 21", "finite-flow\nstate jump"),
    ]
    x = np.arange(len(events))
    fig, axis = plt.subplots(figsize=(10.4, 2.15))
    axis.plot(x, np.zeros_like(x), color="#26394c", lw=2)
    axis.scatter(x, np.zeros_like(x), s=62, color="#315c8c", zorder=3)
    for index, (date, label) in enumerate(events):
        level = 0.39 if index % 2 == 0 else -0.39
        axis.plot([index, index], [0, level * 0.68], color="#8091a2", lw=1)
        axis.text(index, level, f"{date}\n{label}", ha="center", va="center",
                  fontsize=8.0, linespacing=1.02)
    axis.set_xlim(-0.45, len(events) - 0.55)
    axis.set_ylim(-0.72, 0.72)
    axis.axis("off")
    fig.tight_layout(pad=0.15)
    fig.savefig(RESULTS / "method_lineage.png", dpi=260)
    plt.close(fig)


def copy_surgical_atlases() -> None:
    sources = {
        "hard_jump_defect_atlas.png": ROOT / "experiments" / "out"
        / "meyer_hard_jump_defect_proof" / "defect_atlas.png",
        "finite_flow_surgical_atlas.png": ROOT / "experiments" / "out"
        / "meyer_semismooth_state_jump_256_exact" / "atlas.png",
    }
    for name, source in sources.items():
        shutil.copyfile(source, RESULTS / name)


def natural_comparison() -> None:
    arrays = np.load(FLOW / "arrays.npz")
    scenes = (
        ("barbara_512", "Barbara 512"),
        ("camera_256", "Cameraman 256"),
        ("synthetic_256", "analytic crossing 256"),
    )
    methods = (
        ("source", "source"),
        ("gilles_nested", "cold nested\nGilles"),
        ("fused_64", "fused 64\nreference"),
        ("hard_jump", "old hard\ncontrol"),
        ("flow_fast", "finite flow\nfast"),
        ("flow_quality", "finite flow\nquality"),
    )
    fig, axes = plt.subplots(2 * len(scenes), len(methods),
                             figsize=(10.75, 9.25))
    for scene_index, (name, label) in enumerate(scenes):
        source = arrays[f"{name}_source"]
        textures = [
            arrays[f"{name}_{method}_texture"]
            for method, _ in methods if method != "source"
        ]
        limit = max(float(np.percentile(np.abs(value), 99.5))
                    for value in textures)
        for column, (method, title) in enumerate(methods):
            cartoon_axis = axes[2 * scene_index, column]
            texture_axis = axes[2 * scene_index + 1, column]
            if method == "source":
                cartoon = source
                texture = np.zeros_like(source)
            else:
                cartoon = arrays[f"{name}_{method}_cartoon"]
                texture = arrays[f"{name}_{method}_texture"]
            cartoon_axis.imshow(cartoon, cmap="gray", vmin=0.0, vmax=255.0,
                                interpolation="nearest")
            texture_axis.imshow(texture, cmap="coolwarm", vmin=-limit,
                                vmax=limit, interpolation="nearest")
            if scene_index == 0:
                cartoon_axis.set_title(title, fontsize=8.2)
            if column == 0:
                cartoon_axis.set_ylabel(label + "\ncartoon", fontsize=7.8)
                texture_axis.set_ylabel("texture", fontsize=7.8)
            for axis in (cartoon_axis, texture_axis):
                axis.set_xticks([])
                axis.set_yticks([])
    fig.tight_layout(pad=0.35, w_pad=0.15, h_pad=0.18)
    fig.savefig(RESULTS / "natural_flow_comparison.png", dpi=260)
    plt.close(fig)


def pareto() -> None:
    data = json.loads((FLOW / "benchmark.json").read_text())
    scenes = (
        ("barbara_512", "Barbara 512"),
        ("camera_256", "Cameraman 256"),
        ("synthetic_256", "analytic crossing 256"),
    )
    fused_methods = ("fused_19", "fused_29", "fused_64")
    flow_methods = ("flow_fast", "flow_quality")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), sharey=False)
    for axis, (scene, label) in zip(axes, scenes):
        rows = data["scenes"][scene]
        fx = [rows[m]["timing"]["median_ms"] for m in fused_methods]
        fy = [rows[m]["to_fused64"]["texture_relative_l2"]
              for m in fused_methods]
        qx = [rows[m]["timing"]["median_ms"] for m in flow_methods]
        qy = [rows[m]["to_fused64"]["texture_relative_l2"]
              for m in flow_methods]
        axis.plot(fx, fy, "o-", color="#8795a1", label="ordinary fused")
        axis.plot(qx, qy, "D-", color="#315c8c", label="finite flow")
        hard = rows["hard_jump"]
        axis.scatter(hard["timing"]["median_ms"],
                     hard["to_fused64"]["texture_relative_l2"],
                     marker="x", s=60, color="#b44d35", label="old hard")
        gilles = rows["gilles_nested"]
        axis.scatter(gilles["timing"]["elapsed_ms"],
                     gilles["to_fused64"]["texture_relative_l2"],
                     marker="*", s=72, color="#242424", label="cold Gilles")
        axis.set_xscale("log")
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("wall time (ms, log)")
        axis.grid(True, which="both", alpha=0.23)
    axes[0].set_ylabel(r"$\|v-v_{64}\|_2/\|v_{64}\|_2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.65)
    fig.savefig(RESULTS / "finite_flow_pareto.png", dpi=260)
    plt.close(fig)


def scaling() -> None:
    path = ROOT / "experiments" / "out" / \
        "meyer_flow_jump_benchmark_m4_final.json"
    data = json.loads(path.read_text())
    sizes = np.array(sorted(int(k) for k in data["sizes"]))
    methods = (
        ("hard_jump", "old hard", "#b44d35", "x"),
        ("flow_fast", "finite flow fast", "#315c8c", "D"),
        ("flow_quality", "finite flow quality", "#3b7e62", "s"),
        ("fused_19", "ordinary fused 19", "#9ba6af", "o"),
        ("fused_29", "ordinary fused 29", "#6f7d88", "o"),
        ("fused_64", "ordinary fused 64", "#242424", "o"),
    )
    fig, axis = plt.subplots(figsize=(6.7, 3.85))
    for method, label, color, marker in methods:
        values = [data["sizes"][str(size)][method]["median_ms"]
                  for size in sizes]
        axis.loglog(sizes * sizes, values, marker=marker, color=color,
                    label=label)
    axis.set_xlabel("pixels")
    axis.set_ylabel("median M4 wall time (ms)")
    axis.grid(True, which="both", alpha=0.23)
    axis.legend(frameon=False, fontsize=7.6, ncol=2)
    fig.tight_layout(pad=0.65)
    fig.savefig(RESULTS / "native_flow_scaling.png", dpi=260)
    plt.close(fig)


def main() -> None:
    lineage()
    copy_surgical_atlases()
    natural_comparison()
    pareto()
    scaling()


if __name__ == "__main__":
    main()
