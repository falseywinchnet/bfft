#!/usr/bin/env python3
"""Generate the reproducible scientific figures for the FlowCells paper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from skimage import data


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for directory in (ROOT, ROOT / "viewer", ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from port_needed import SegmentingConfig, build_segmenting_representation  # noqa: E402
from port_needed.soft_support_diffusion import diffuse_soft_support  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    mask = np.zeros(labels.shape, dtype=bool)
    mask[1:] |= labels[1:] != labels[:-1]
    mask[:-1] |= labels[:-1] != labels[1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


def site_ids(labels: np.ndarray) -> np.ndarray:
    count = int(np.max(labels)) + 1
    index = np.arange(count, dtype=np.uint32)
    value = index * np.uint32(747796405) + np.uint32(2891336453)
    value = ((value >> ((value >> 28) + 4)) ^ value) * np.uint32(277803737)
    value = (value >> 22) ^ value
    colours = np.column_stack(
        (value & 255, (value >> 8) & 255, (value >> 16) & 255)
    ).astype(np.float64) / 255.0
    return (0.12 + 0.86 * colours)[labels]


def robust_field(field: np.ndarray, cmap: str = "magma") -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    finite = field[np.isfinite(field)]
    if not finite.size:
        z = np.zeros_like(field)
    else:
        lo, hi = np.percentile(finite, (1.0, 99.5))
        z = np.clip((field - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
    return mpl.colormaps[cmap](z)[..., :3]


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def method_overview() -> None:
    rng = np.random.default_rng(7)
    n = 240
    yy, xx = np.mgrid[:n, :n] / n
    cartoon = np.zeros((n, n, 3), dtype=float)
    cartoon[...] = np.array([0.78, 0.87, 0.95])
    disk = (xx - 0.43) ** 2 + (yy - 0.48) ** 2 < 0.23**2
    cartoon[disk] = np.array([0.96, 0.65, 0.23])
    band = np.abs(yy - (0.18 + 0.63 * xx)) < 0.055
    cartoon[band] = np.array([0.18, 0.24, 0.32])
    texture = 0.055 * np.sin(2 * np.pi * (37 * xx + 11 * yy))
    texture *= np.exp(-((xx - 0.7) ** 2 + (yy - 0.62) ** 2) / 0.13)
    texture += 0.018 * rng.standard_normal((n, n))
    image = np.clip(cartoon + texture[..., None], 0, 1)

    gy, gx = np.gradient(np.mean(image, axis=2))
    energy = np.hypot(gx, gy)
    density = robust_field(energy**0.65, "magma")

    fig, axes = plt.subplots(1, 5, figsize=(12.8, 2.55))
    titles = (
        "Input",
        "single cartoon--texture split",
        "normalized support tensor",
        "measure quantization",
        "causal support + readout",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#b8b8b8")

    axes[0].imshow(image)
    texture_view = np.repeat(
        np.clip(0.5 + 4 * texture[:, n // 2 :, None], 0, 1),
        3,
        axis=2,
    )
    split = np.concatenate((cartoon[:, : n // 2], texture_view), axis=1)
    axes[1].imshow(split)
    axes[1].axvline(n / 2, color="white", lw=1.0)
    axes[1].text(0.23, 0.05, "cartoon", transform=axes[1].transAxes, color="white", ha="center")
    axes[1].text(0.76, 0.05, "texture", transform=axes[1].transAxes, color="white", ha="center")

    axes[2].imshow(image, alpha=0.32)
    step = 18
    for y in range(step // 2, n, step):
        for x in range(step // 2, n, step):
            vx, vy = gx[y, x], gy[y, x]
            angle = np.degrees(np.arctan2(vy, vx)) + 90
            coherence = min(1.0, np.hypot(vx, vy) / (np.percentile(energy, 90) + 1e-9))
            axes[2].add_patch(
                Ellipse(
                    (x, y),
                    width=5 + 14 * coherence,
                    height=5,
                    angle=angle,
                    fill=False,
                    edgecolor="#144d73",
                    linewidth=0.6,
                    alpha=0.9,
                )
            )

    axes[3].imshow(density)
    prob = energy**0.7 + 0.006
    prob /= prob.sum()
    sample = rng.choice(n * n, size=390, replace=False, p=prob.ravel())
    sy, sx = np.unravel_index(sample, (n, n))
    axes[3].scatter(sx, sy, s=2.0, c="white", alpha=0.8, linewidths=0)

    axes[4].imshow(image)
    for k in range(72):
        x, y = rng.uniform(0, n, size=2)
        vx = gx[int(np.clip(y, 0, n - 1)), int(np.clip(x, 0, n - 1))]
        vy = gy[int(np.clip(y, 0, n - 1)), int(np.clip(x, 0, n - 1))]
        angle = np.degrees(np.arctan2(vy, vx)) + 90
        axes[4].add_patch(
            Ellipse(
                (x, y),
                width=rng.uniform(9, 25),
                height=rng.uniform(4, 10),
                angle=angle,
                facecolor=mpl.colormaps["turbo"](k / 72),
                edgecolor="white",
                linewidth=0.25,
                alpha=0.62,
            )
        )

    fig.subplots_adjust(wspace=0.08)
    save_both(fig, "method_overview")


def curvature_schematic() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 2.7))
    x = np.linspace(-1, 1, 500)
    straight = np.zeros_like(x)
    curved = 0.52 * x**2
    for ax in axes:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.42, 0.82)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].plot(x, straight, color="#25364a", lw=3)
    axes[0].add_patch(Ellipse((0, 0), 1.65, 0.18, facecolor="#ef8354", alpha=0.55))
    axes[0].set_title("straight coherent support")
    axes[0].text(0, -0.28, "one long cell remains locally valid", ha="center")

    axes[1].plot(x, curved, color="#25364a", lw=3)
    axes[1].add_patch(Ellipse((0, 0), 1.65, 0.18, facecolor="#ef8354", alpha=0.55))
    axes[1].plot([-0.82, 0.82], [0, 0], color="#a23e48", ls="--", lw=1)
    axes[1].add_patch(
        FancyArrowPatch(
            (0.82, 0),
            (0.82, 0.35),
            arrowstyle="<->",
            mutation_scale=10,
            color="#a23e48",
            lw=1,
        )
    )
    axes[1].set_title("curvature breaks the tangent horizon")
    axes[1].text(0.87, 0.17, r"$\kappa a^2/(2b)$", color="#a23e48", va="center")

    axes[2].plot(x, curved, color="#25364a", lw=3)
    for cx in (-0.62, 0.0, 0.62):
        cy = 0.52 * cx**2
        angle = np.degrees(np.arctan(1.04 * cx))
        axes[2].add_patch(
            Ellipse(
                (cx, cy),
                0.62,
                0.16,
                angle=angle,
                facecolor="#47a8bd",
                edgecolor="white",
                lw=0.7,
                alpha=0.72,
            )
        )
    axes[2].set_title("curvature-limited population")
    axes[2].text(0, -0.28, "extra measure appears only where required", ha="center")
    save_both(fig, "curvature_horizon")


def ablation_plot() -> None:
    names = ["camera", "Chelsea", "coins", "astronaut", "Pikachu"]
    curvature = np.array([2.68, 0.43, 0.46, 1.53, 1.58])
    soft = np.array([0.077, 0.136, 0.0, 0.116, 0.080])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.0))
    axes[0].bar(x, curvature, color="#277da1", width=0.66)
    axes[0].axhline(0, color="#333333", lw=0.7)
    axes[0].set_xticks(x, names, rotation=24, ha="right")
    axes[0].set_ylabel("PSNR change (dB)")
    axes[0].set_title("curvature-limited population")
    axes[0].grid(axis="y", color="#dddddd", lw=0.6)
    axes[1].bar(x, soft, color="#f8961e", width=0.66)
    axes[1].axhline(0, color="#333333", lw=0.7)
    axes[1].set_xticks(x, names, rotation=24, ha="right")
    axes[1].set_ylabel("accepted PSNR change (dB)")
    axes[1].set_title("objective-gated soft support")
    axes[1].grid(axis="y", color="#dddddd", lw=0.6)
    fig.subplots_adjust(wspace=0.32, bottom=0.22)
    save_both(fig, "ablation_summary")


def rocket_diagnostics() -> None:
    rgb = data.rocket().astype(np.float64) / 255.0
    config = SegmentingConfig(
        allocation_method="causal_density",
        allocation_max_side=512,
        tgfd_sweeps=24,
        flow_sweeps=24,
        metric_strength=1.5,
        safety_cells=32768,
        curvature_limited_density=True,
        null_evidence_strength=0.5,
        boundary_jump_strength=24.0,
        interface_coverage_strength=0.4,
        soft_support_passes=16,
        soft_support_coupling=0.8,
        soft_support_colour_percentile=60.0,
        characteristic_passes=1,
        characteristic_trust_fraction=0.5,
        characteristic_core_radius=3.0,
        ridge_count=1,
        threads=4,
    )
    result = build_segmenting_representation(rgb, config)
    labels = result["labels"]
    ids = site_ids(labels)
    soft = result["soft_support"]
    if soft is not None:
        ids = np.clip(
            diffuse_soft_support(
                ids,
                soft["conductance"],
                passes=soft["passes"],
                coupling=soft["coupling"],
            ),
            0,
            1,
        )
    hard = (
        result["interface_coverage"]["hard_record"]["rgb"]
        if result["interface_coverage"] is not None
        else result["record"]["rgb"]
    )
    final = result["record"]["rgb"]
    outline = boundaries(labels)
    ids_with_edges = ids.copy()
    ids_with_edges[outline] = 0.08

    panels = (
        (rgb, "input"),
        (robust_field(result["geometry"]["measure"]), "support measure"),
        (ids_with_edges, f"site support ({len(result['centers']):,})"),
        (hard, "hard affine + ridge readout"),
        (final, "accepted interface/soft finish"),
        (robust_field(result["residual_energy"], "inferno"), "final residual energy"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.25))
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaaaaa")
            spine.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.035, hspace=0.11)
    save_both(fig, "rocket_diagnostics")

    metrics = {
        "cells": len(result["centers"]),
        "record": {
            key: float(value)
            for key, value in result["record"].items()
            if np.isscalar(value)
        },
        "timing": {key: float(value) for key, value in result["timing"].items()},
        "interface_accepted": bool(
            result["interface_coverage"] is not None
            and result["interface_coverage"]["accepted"]
        ),
        "soft_accepted": bool(soft is not None and soft["accepted"]),
    }
    (OUT / "rocket_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    method_overview()
    curvature_schematic()
    ablation_plot()
    rocket_diagnostics()
