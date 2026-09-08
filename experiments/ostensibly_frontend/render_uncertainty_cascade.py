#!/usr/bin/env python3
"""Render registered maximum and texture-cartoon cascade diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _relative_db(field: np.ndarray) -> np.ndarray:
    value = np.abs(np.asarray(field, dtype=np.float64))
    peak = max(float(np.max(value)), 1e-30)
    return 20.0 * np.log10(np.maximum(value, peak * 1e-4) / peak)


def run(npz_path: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(npz_path)
    offsets = arrays["offsets"].tolist()
    zero = offsets.index(0)
    panels = (
        ("single zero-centred lattice (failed control)", arrays["observations"][zero]),
        ("Fourier-circle registered per-pixel maximum", arrays["fused"]),
        ("first split: rejected cartoon", arrays["first_cartoon"]),
        ("first split: retained signed texture", arrays["first_texture"]),
        ("second split: cartoon(signed texture)", arrays["texture_cartoon"]),
        ("audit: cartoon(abs(texture))", arrays["magnitude_texture_cartoon"]),
    )
    rows = min(128, panels[0][1].shape[0])
    extent = [0, panels[0][1].shape[1] - 1, rows - 1, 0]
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), constrained_layout=True)
    image = None
    for ax, (title, field) in zip(axes.ravel(), panels):
        image = ax.imshow(
            _relative_db(field[:rows]),
            origin="upper",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=-60.0,
            vmax=0.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("STFT frame")
        ax.set_ylabel("double-IRFFT row")
    fig.colorbar(image, ax=axes, label="relative magnitude (dB)", shrink=0.9)
    overview = out_dir / "uncertainty_cascade_overview.png"
    fig.savefig(overview, dpi=220)
    plt.close(fig)

    raw = arrays["observations"][zero, :rows]
    fused = arrays["fused"][:rows]
    signed_final = arrays["texture_cartoon"][:rows]
    magnitude_final = arrays["magnitude_texture_cartoon"][:rows]
    path = arrays["fused_ridge_path"]
    active_columns = np.arange(fused.shape[1])
    path_values = fused[path, active_columns]
    bright = int(np.argmax(path_values))
    x0 = max(0, bright - 18)
    x1 = min(fused.shape[1], bright + 19)
    center_row = int(np.median(path[x0:x1]))
    y0 = max(0, center_row - 18)
    y1 = min(rows, center_row + 19)
    zoom_panels = (
        ("single lattice", raw[y0:y1, x0:x1]),
        ("registered maximum", fused[y0:y1, x0:x1]),
        ("cartoon(signed texture)", signed_final[y0:y1, x0:x1]),
        ("cartoon(abs(texture))", magnitude_final[y0:y1, x0:x1]),
    )
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    image = None
    for ax, (title, field) in zip(axes, zoom_panels):
        image = ax.imshow(
            _relative_db(field),
            origin="upper",
            aspect="auto",
            cmap="magma",
            vmin=-60.0,
            vmax=0.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(f"frame {x0}…{x1 - 1}")
        ax.set_ylabel(f"row {y0}…{y1 - 1}")
    fig.colorbar(image, ax=axes, label="relative magnitude (dB)", shrink=0.9)
    zoom = out_dir / "uncertainty_cascade_ridge_zoom.png"
    fig.savefig(zoom, dpi=260)
    plt.close(fig)
    return overview, zoom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in run(args.npz, args.out):
        print(path)


if __name__ == "__main__":
    main()
