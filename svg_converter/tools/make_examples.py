#!/usr/bin/env python3
"""Create deterministic vectorization controls without downloading assets."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
INPUT = ROOT / "examples" / "input"


def geometric_badge() -> Image.Image:
    image = Image.new("RGBA", (384, 256), (247, 241, 224, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((26, 24, 358, 232), radius=38, fill=(21, 62, 79, 255))
    draw.ellipse((62, 52, 226, 216), fill=(218, 66, 63, 240))
    draw.polygon([(192, 42), (350, 128), (192, 218)], fill=(241, 176, 48, 235))
    draw.ellipse((112, 94, 178, 160), fill=(247, 241, 224, 255))
    draw.rectangle((252, 89, 326, 168), fill=(45, 143, 126, 255))
    return image


def translucent_ribbons() -> Image.Image:
    height, width = 256, 384
    yy, xx = np.indices((height, width))
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(30 + 0.50 * xx, 0, 255)
    rgba[..., 1] = np.clip(45 + 0.55 * yy, 0, 255)
    rgba[..., 2] = np.clip(125 + 35 * np.sin(xx / 28), 0, 255)
    rgba[..., 3] = 255
    image = Image.fromarray(rgba, "RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for offset, color in ((0, (252, 196, 78, 185)), (42, (229, 64, 98, 165)), (84, (58, 210, 190, 150))):
        points = []
        for x in range(-30, width + 31, 8):
            y = 52 + offset + 30 * np.sin((x + offset) / 42)
            points.append((x, y))
        draw.line(points, fill=color, width=31, joint="curve")
    return Image.alpha_composite(image, overlay)


def main() -> int:
    INPUT.mkdir(parents=True, exist_ok=True)
    geometric_badge().save(INPUT / "geometric_badge.png")
    translucent_ribbons().save(INPUT / "translucent_ribbons.png")
    demo = REPOSITORY / "segmenting_v3_demo.png"
    if demo.exists():
        with Image.open(demo) as image:
            # The upper-left panel is the unchanged cameraman source. Keeping
            # this crop supplies a natural-image control tied to the v3 study.
            crop = image.convert("RGBA").crop((12, 65, 548, 601))
            crop = crop.resize((384, 384), Image.Resampling.LANCZOS)
            crop.save(INPUT / "cameraman_source.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
