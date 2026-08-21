#!/usr/bin/env python3
"""Render complementary role, contour, and enclosure participation fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
)


ANCHOR = {
    "pikachu_hard": "body",
    "coffee": "cup_wall",
    "astronaut": "flag_blue",
    "checker": "black_a",
    "coins": "coin_00",
}


def _cosine_to_anchor(embedding: np.ndarray, anchor: int) -> np.ndarray:
    value = np.asarray(embedding, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1)
    denominator = norm * norm[anchor]
    cosine = np.divide(
        value @ value[anchor],
        denominator,
        out=np.zeros(len(value), dtype=np.float64),
        where=denominator > 1e-30,
    )
    return np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)


def _heat_image(value: np.ndarray) -> Image.Image:
    field = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    stops = np.asarray([
        (0.00, 0, 0, 4),
        (0.20, 45, 17, 95),
        (0.40, 127, 30, 110),
        (0.60, 210, 62, 78),
        (0.80, 249, 142, 8),
        (1.00, 252, 253, 191),
    ], dtype=np.float64)
    position = field * (len(stops) - 1)
    lower = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = position - lower
    first = stops[lower, 1:]
    second = stops[lower + 1, 1:]
    rgb = first + fraction[..., None] * (second - first)
    return Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")


def _fit_panel(image: Image.Image, side: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((side, side), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (side, side), "white")
    panel.paste(
        fitted,
        ((side - fitted.width) // 2, (side - fitted.height) // 2),
    )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "research_participation_atlas.png")
    contour_report = json.loads(
        (args.results / "contour_transport" / "report.json").read_text())
    enclosure_report = json.loads(
        (args.results / "relative_enclosure" / "report.json").read_text())

    panel_side = 280
    label_width = 130
    header_height = 35
    canvas = Image.new(
        "RGB",
        (label_width + 5 * panel_side, header_height + len(CONTROLS) * panel_side),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = (
        "source",
        "relation-role field",
        "one-sided contour field",
        "bounded-enclosure field",
        "complete tensor algebra",
    )
    for column, title in enumerate(titles):
        draw.text(
            (label_width + column * panel_side + 8, 10),
            title,
            fill="black",
            font=font,
        )

    for row, name in enumerate(CONTROLS):
        image_dir = args.results / name
        source = Image.open(image_dir / "source.png").convert("RGB")
        stages = np.load(image_dir / "v3_stages.npz")
        labels = stages["compound_labels"]
        anchor_name = ANCHOR[name]
        anchor_region = int(
            contour_report["images"][name]["audit"]["points"][
                anchor_name]["region"])

        role = np.load(
            args.results / "connection_bloom" / name / "full.npz"
        )["region_embedding"]
        role_field = _cosine_to_anchor(role, anchor_region)[labels]
        contour = np.load(
            args.results / "contour_transport" / name / "contour_transport.npz"
        )["region_kernel"]
        contour_field = contour[:, anchor_region][labels]
        enclosure = np.load(
            args.results / "relative_enclosure" / name
            / "relative_enclosure.npz"
        )["region_kernel"]
        enclosure_field = enclosure[:, anchor_region][labels]
        complete = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz"
        )["complete"]
        complete_field = complete[:, anchor_region][labels]

        panels = [
            _fit_panel(source, panel_side),
            _fit_panel(_heat_image(role_field), panel_side),
            _fit_panel(_heat_image(contour_field), panel_side),
            _fit_panel(_heat_image(enclosure_field), panel_side),
            _fit_panel(_heat_image(complete_field), panel_side),
        ]
        y = header_height + row * panel_side
        draw.multiline_text(
            (8, y + 12),
            f"{name.replace('_', ' ')}\nanchor: {anchor_name}",
            fill="black",
            font=font,
            spacing=4,
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel, (label_width + column * panel_side, y))
    canvas.save(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
