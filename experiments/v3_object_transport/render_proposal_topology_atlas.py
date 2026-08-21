#!/usr/bin/env python3
"""Render explicit proposal heat, transported parts, and topology nulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.v3_object_transport.render_research_atlas import (
    ANCHOR,
    _fit_panel,
    _heat_image,
)
from experiments.v3_object_transport.run_connection_bloom import (
    CONTROLS,
    DEFAULT_RESULTS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (args.results / "proposal_topology_atlas.png")
    landmarks = json.loads((
        Path(__file__).resolve().parent / "assets/landmarks.json"
    ).read_text())["images"]
    titles = (
        "source", "canonical parts", "proposal heat",
        "parts transported by proposals", "shuffled-topology transport",
    )
    panel_side = 280
    label_width = 140
    header_height = 35
    canvas = Image.new(
        "RGB",
        (label_width + len(titles) * panel_side,
         header_height + len(CONTROLS) * panel_side),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, title in enumerate(titles):
        draw.text(
            (label_width + column * panel_side + 8, 10),
            title, fill="black", font=font)

    for row, name in enumerate(CONTROLS):
        image_dir = args.results / name
        source = Image.open(image_dir / "source.png").convert("RGB")
        labels = np.load(image_dir / "v3_stages.npz")["compound_labels"]
        anchor_name = ANCHOR[name]
        xy = landmarks[name][anchor_name]["xy"]
        x = int(round(float(xy[0]) * (labels.shape[1] - 1)))
        y = int(round(float(xy[1]) * (labels.shape[0] - 1)))
        anchor = int(labels[y, x])
        canonical = np.load(
            args.results / "participation_algebra" / name
            / "participation_algebra.npz")["complete"]
        full = np.load(
            args.results / "proposal_topology_transport" / name / "full.npz")
        shuffled = np.load(
            args.results / "proposal_topology_transport" / name
            / "shuffled_manifold_members.npz")
        fields = (
            canonical[:, anchor],
            full["heat_kernel"][:, anchor],
            full["transported_base_kernel"][:, anchor],
            shuffled["transported_base_kernel"][:, anchor],
        )
        panels = [_fit_panel(source, panel_side)] + [
            _fit_panel(_heat_image(field[labels]), panel_side)
            for field in fields
        ]
        top = header_height + row * panel_side
        draw.multiline_text(
            (8, top + 12),
            f"{name.replace('_', ' ')}\nanchor: {anchor_name}",
            fill="black", font=font, spacing=4)
        for column, panel in enumerate(panels):
            canvas.paste(panel, (label_width + column * panel_side, top))
    canvas.save(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
