#!/usr/bin/env python3
"""Infer cell density and bifurcation rate from the BFFT flow itself.

There is one population.  Cells are transported through pass-time and
bifurcate only when the local supportable area shrinks.  No carrier/detail
types, residual births, target cell count, ranking, or deletion appear.

If Q is the normalized event tensor, a locally supportable ellipse has area

    A* = pi / sqrt(det(Q + l_max^-2 I)).

Thus the required density is 1/A*.  With transport v, population density
obeys

    partial_p rho + div(rho v) = beta rho.

This script evolves that density and records the positive source needed to
reach the tensor-implied density.  It intentionally stops before converting
the density into discrete cells.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from transport_voronoi import _fit_rgb  # noqa: E402


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _advect_density(
    density: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    """Conservatively push density through one transport step.

    Every source pixel sends its population mass to the four surrounding
    destination pixels.  Clipping the destination at the image boundary
    makes this exactly mass preserving (up to floating-point summation).
    Unlike an inverse semi-Lagrangian sample with an approximate Jacobian,
    this cannot manufacture apparent bifurcations from interpolation error.
    """
    height, width = density.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    destination_x = np.clip(xx + vx, 0.0, width - 1.0)
    destination_y = np.clip(yy + vy, 0.0, height - 1.0)
    x0 = np.floor(destination_x).astype(np.intp)
    y0 = np.floor(destination_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = destination_x - x0
    fy = destination_y - y0
    transported = np.zeros_like(density, dtype=np.float64)
    for yi, xi, weight in (
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x1, fx * (1.0 - fy)),
        (y1, x0, (1.0 - fx) * fy),
        (y1, x1, fx * fy),
    ):
        np.add.at(transported, (yi.ravel(), xi.ravel()),
                  (density * weight).ravel())
    return transported


def infer_bifurcation_density(
    volume: dict,
    initial_cells: int,
    transport_strength: float = 1.0,
) -> dict:
    energy = np.asarray(volume["energy"], dtype=np.float64)
    high = np.asarray(volume["high_frequency"], dtype=np.float64)
    low = np.asarray(volume["low_frequency"], dtype=np.float64)
    tx = np.asarray(volume["transport_x"], dtype=np.float64)
    ty = np.asarray(volume["transport_y"], dtype=np.float64)
    confidence = np.asarray(
        volume["transport_confidence"], dtype=np.float64)
    persistence = np.asarray(
        volume["transport_persistence"], dtype=np.float64)
    stages, height, width = energy.shape
    pixels = height * width
    max_length = 0.18 * max(height, width)
    frequency_floor = 1.0 / max(max_length * max_length, 1.0)
    initial_density = float(initial_cells) / pixels

    # Reliability is the same soft numerical floor used in the normalized
    # tensor.  It suppresses undefined geometry where no event occurred.
    amplitude = np.sqrt(np.maximum(energy, 0.0))
    stage_scale = np.maximum(
        np.percentile(amplitude, 99.5, axis=(1, 2), keepdims=True),
        1e-30,
    )
    reliability = amplitude / (amplitude + 1e-5 * stage_scale)
    required = (
        reliability
        * np.sqrt(
            (high + frequency_floor) * (low + frequency_floor))
        / math.pi
    )
    required = np.maximum(required, initial_density)

    density = np.full((height, width), initial_density, dtype=np.float64)
    densities = []
    transported_fields = []
    source_fields = []
    beta_fields = []
    stage_stats = []
    cumulative_bifurcations = 0.0

    for stage in range(stages):
        gate = confidence[stage] * persistence
        vx = float(transport_strength) * gate * tx[stage]
        vy = float(transport_strength) * gate * ty[stage]
        transported = _advect_density(density, vx, vy)
        transported = np.maximum(transported, 1e-12)

        # Existing cells can be compressed by transport or expand their
        # supports, but only a positive density deficit causes bifurcation.
        source = np.maximum(required[stage] - transported, 0.0)
        density = transported + source
        beta = np.log1p(source / transported)
        bifurcations = float(np.sum(source))
        cumulative_bifurcations += bifurcations
        source_weight = max(bifurcations, 1e-30)
        ratio = np.sqrt(
            (high[stage] + frequency_floor)
            / np.maximum(low[stage] + frequency_floor, 1e-30))
        stage_stats.append({
            "stage": stage + 1,
            "instantaneous_required_cells": float(
                np.sum(required[stage])),
            "transported_cells_before_bifurcation": float(
                np.sum(transported)),
            "inferred_bifurcations": bifurcations,
            "population_after_bifurcation": float(np.sum(density)),
            "cumulative_bifurcations": cumulative_bifurcations,
            "bifurcation_weighted_aspect_ratio": float(
                np.sum(source * ratio) / source_weight),
            "bifurcation_weighted_minor_px": float(
                np.sum(
                    source / np.sqrt(high[stage] + frequency_floor))
                / source_weight),
            "transport_rms_px": float(np.sqrt(np.mean(vx * vx + vy * vy))),
        })
        transported_fields.append(transported.astype(np.float32))
        source_fields.append(source.astype(np.float32))
        beta_fields.append(beta.astype(np.float32))
        densities.append(density.astype(np.float32))

    envelope = np.maximum(
        initial_density, np.max(required, axis=0))
    return {
        "required_density": required.astype(np.float32),
        "transported_density": np.stack(transported_fields),
        "bifurcation_source": np.stack(source_fields),
        "bifurcation_log_rate": np.stack(beta_fields),
        "population_density": np.stack(densities),
        "no_transport_envelope": envelope.astype(np.float32),
        "stage_stats": stage_stats,
        "initial_cells": initial_cells,
        "envelope_cells": float(np.sum(envelope)),
        "final_cells": float(np.sum(density)),
        "cumulative_bifurcations": cumulative_bifurcations,
    }


def _normalize(field: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    scale = max(float(np.percentile(field, percentile)), 1e-30)
    return np.clip(field / scale, 0.0, 1.0)


def save_panel(rgb: np.ndarray, volume: dict, result: dict, path: Path) -> None:
    stages = np.asarray(result["population_density"]).shape[0]
    selected = sorted({
        min(max(value, 1), stages)
        for value in (1, 2, 4, 8, 12, 16, 24, stages)
    })
    fig, axes = plt.subplots(
        4, len(selected), figsize=(3.0 * len(selected), 10.5),
        squeeze=False)
    event = np.sqrt(np.asarray(volume["energy"]))
    required = np.asarray(result["required_density"])
    density = np.asarray(result["population_density"])
    source = np.asarray(result["bifurcation_source"])
    for column, stage in enumerate(selected):
        index = stage - 1
        for row, (field, title, cmap) in enumerate((
            (event[index], "flow event", "magma"),
            (required[index], "tensor-required density", "viridis"),
            (density[index], "transported population", "viridis"),
            (source[index], "inferred bifurcation", "inferno"),
        )):
            axes[row, column].imshow(_normalize(field), cmap=cmap)
            axes[row, column].set_title(f"p={stage} {title}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle(
        "One population: transport determines support; support determines "
        "bifurcation",
        fontsize=15,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)


def save_summary(rgb: np.ndarray, volume: dict, result: dict, path: Path) -> None:
    stats = result["stage_stats"]
    passes = [record["stage"] for record in stats]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("input")
    axes[0, 1].imshow(
        _normalize(result["no_transport_envelope"]), cmap="viridis")
    axes[0, 1].set_title(
        f"tensor density envelope\n{result['envelope_cells']:.0f} cells")
    axes[0, 2].imshow(
        _normalize(result["population_density"][-1]), cmap="viridis")
    axes[0, 2].set_title(
        f"transported final density\n{result['final_cells']:.0f} cells")

    axes[1, 0].plot(
        passes,
        [record["instantaneous_required_cells"] for record in stats],
        label="instantaneous tensor requirement",
    )
    axes[1, 0].plot(
        passes,
        [record["population_after_bifurcation"] for record in stats],
        label="persistent transported population",
    )
    axes[1, 0].set_title("population inferred without a cell budget")
    axes[1, 0].set_xlabel("BFFT pass")
    axes[1, 0].set_ylabel("equivalent cells")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        passes, [record["inferred_bifurcations"] for record in stats])
    axes[1, 1].set_title("bifurcations per pass")
    axes[1, 1].set_xlabel("BFFT pass")
    axes[1, 1].set_ylabel("equivalent cells")

    axes[1, 2].plot(
        passes,
        [record["bifurcation_weighted_aspect_ratio"] for record in stats],
    )
    axes[1, 2].set_title("anisotropy at bifurcation")
    axes[1, 2].set_xlabel("BFFT pass")
    axes[1, 2].set_ylabel("major / minor")
    for axis in axes[0]:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--transport-strength", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/flow_bifurcation_density.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments/out/flow_bifurcation_summary.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/flow_bifurcation_density.json",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    rgb = _fit_rgb(image, args.side)
    volume = build_flow_volume(rgb, passes=args.passes)
    result = infer_bifurcation_density(
        volume, args.initial_cells, args.transport_strength)
    save_panel(rgb, volume, result, args.output)
    save_summary(rgb, volume, result, args.summary)
    report = {
        "source": source,
        "shape": list(rgb.shape),
        "passes": args.passes,
        "initial_cells": args.initial_cells,
        "transport_strength": args.transport_strength,
        "tensor_envelope_cells": result["envelope_cells"],
        "transported_final_cells": result["final_cells"],
        "cumulative_bifurcations": result["cumulative_bifurcations"],
        "stages": result["stage_stats"],
        "outputs": {
            "panel": str(args.output.resolve()),
            "summary": str(args.summary.resolve()),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
