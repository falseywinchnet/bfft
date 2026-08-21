#!/usr/bin/env python3
"""Render assembly fields and their positive middle-scale curvature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.v3_object_transport.audit_resolution_stability import (
    DEFAULT_128,
    DEFAULT_384,
)
from experiments.v3_object_transport.render_research_atlas import (
    ANCHOR,
    _fit_panel,
    _heat_image,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
)


def _field(path: Path, name: str, specification: dict) -> np.ndarray:
    labels = np.load(path / name / "v3_stages.npz")["compound_labels"]
    x = int(round(specification["xy"][0] * (labels.shape[1] - 1)))
    y = int(round(specification["xy"][1] * (labels.shape[0] - 1)))
    anchor = int(labels[y, x])
    kernel = np.load(
        path / "support_manifold_transport" / name / "full.npz"
    )["region_kernel"]
    return np.asarray(kernel[:, anchor][labels], dtype=np.float64)


def _resize_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(field, dtype=np.float32), mode="F")
    return np.asarray(image.resize(
        (shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-128", type=Path, default=DEFAULT_128)
    parser.add_argument("--results-256", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--results-384", type=Path, default=DEFAULT_384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results_256 / "scale_curvature_atlas.png")
    paths = (args.results_128, args.results_256, args.results_384)
    landmarks = json.loads((
        Path(__file__).resolve().parent / "assets" / "landmarks.json"
    ).read_text())["images"]
    panel_side = 250
    label_width = 130
    header_height = 35
    canvas = Image.new(
        "RGB",
        (label_width + 5 * panel_side, header_height + len(CONTROLS) * panel_side),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, title in enumerate((
        "source", "assembly 128", "assembly 256", "assembly 384",
        "positive log-scale curvature",
    )):
        draw.text(
            (label_width + column * panel_side + 8, 10),
            title, fill="black", font=font)
    log_scale = np.log(np.asarray((128.0, 256.0, 384.0)))
    right = float(
        (log_scale[1] - log_scale[0]) / (log_scale[2] - log_scale[0]))
    for row, name in enumerate(CONTROLS):
        specification = landmarks[name][ANCHOR[name]]
        fields = [_field(path, name, specification) for path in paths]
        target_shape = fields[1].shape
        aligned = [_resize_field(field, target_shape) for field in fields]
        curvature = np.maximum(
            aligned[1] - ((1.0 - right) * aligned[0] + right * aligned[2]),
            0.0,
        )
        maximum = float(np.max(curvature, initial=0.0))
        if maximum > 0.0:
            curvature = curvature / maximum
        source = Image.open(
            args.results_256 / name / "source.png").convert("RGB")
        panels = [_fit_panel(source, panel_side)] + [
            _fit_panel(_heat_image(value), panel_side)
            for value in (*aligned, curvature)
        ]
        top = header_height + row * panel_side
        draw.multiline_text(
            (8, top + 12),
            f"{name.replace('_', ' ')}\nanchor: {ANCHOR[name]}",
            fill="black", font=font, spacing=4)
        for column, panel in enumerate(panels):
            canvas.paste(panel, (label_width + column * panel_side, top))
    canvas.save(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
