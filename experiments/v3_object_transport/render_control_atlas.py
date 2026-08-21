#!/usr/bin/env python3
"""Render the frozen control audit as one compact visual atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


ROWS = ("pikachu_hard", "coffee", "astronaut", "checker", "coins")
COLUMNS = (
    ("source.png", "source"),
    ("reconstruction.png", "V3 reconstruction"),
    ("compound_regions.png", "compound regions"),
    ("historical_family_control.png", "old family control"),
)


def _panel(path: Path, side: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((side, side), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (side, side), "white")
    panel.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--side", type=int, default=256)
    args = parser.parse_args()
    header = 32
    row_label = 112
    gap = 4
    width = row_label + len(COLUMNS) * (args.side + gap)
    height = header + len(ROWS) * (args.side + gap)
    atlas = Image.new("RGB", (width, height), (242, 242, 240))
    draw = ImageDraw.Draw(atlas)
    for column, (_, label) in enumerate(COLUMNS):
        x = row_label + column * (args.side + gap)
        draw.text((x + 4, 9), label, fill=(20, 20, 20))
    for row, name in enumerate(ROWS):
        y = header + row * (args.side + gap)
        draw.text((8, y + 12), name.replace("_", " "), fill=(20, 20, 20))
        for column, (filename, _) in enumerate(COLUMNS):
            x = row_label + column * (args.side + gap)
            atlas.paste(_panel(args.results / name / filename, args.side), (x, y))
    output = args.out or (args.results / "control_atlas.png")
    atlas.save(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
