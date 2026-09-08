"""Extract Eikonal-described traces from the N=2048 Meyer cartoon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.ostensibly_frontend.trace_geometry import (
    TraceGeometryConfig,
    extract_trace_components,
)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    palette = plt.get_cmap("turbo")
    count = int(labels.max(initial=0))
    normalized = labels.astype(np.float64) / max(count, 1)
    rgba = palette(normalized)
    rgba[labels == 0, :3] = 0.0
    rgba[labels == 0, 3] = 1.0
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.arrays)
    cartoon = np.asarray(archive["cartoon"], dtype=np.float64)
    config = TraceGeometryConfig()
    score, labels, components = extract_trace_components(cartoon, config)

    rows, frames = cartoon.shape
    extent = [0, frames, 0, rows]
    cartoon_peak = max(float(np.max(np.abs(cartoon))), 1e-30)
    cartoon_db = 20.0 * np.log10(
        np.maximum(np.abs(cartoon), cartoon_peak * 1e-6) / cartoon_peak
    )
    score_peak = max(float(np.max(score)), 1e-30)
    score_db = 20.0 * np.log10(
        np.maximum(score, score_peak * 1e-6) / score_peak
    )
    panels = (
        (np.flipud(cartoon_db), "Meyer cartoon", "magma", -60.0, 0.0),
        (np.flipud(score_db), "vertical top-hat ridge evidence", "magma", -60.0, 0.0),
        (np.flipud(colorize_labels(labels)), f"{len(components)} retained trace components", None, None, None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_yticks([0, rows])
        ax.set_yticklabels([str(rows - 1), "0"])
        ax.set_ylabel("original row (low at top)")
    figure_path = args.out / "n2048_trace_components.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    records = [
        {
            "label": item.label,
            "row_bounds": [item.row0, item.row1],
            "frame_bounds": [item.frame0, item.frame1],
            "area": item.area,
            "centroid": [item.centroid_row, item.centroid_frame],
            "radial": item.radial.tolist(),
            "spatial": item.spatial.tolist(),
        }
        for item in components
    ]
    report = {
        "config": vars(config),
        "shape": [rows, frames],
        "components": len(components),
        "records": records,
        "figure": str(figure_path),
    }
    (args.out / "trace_components.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.out / "trace_components.npz",
        ridge_score=score,
        labels=labels,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
