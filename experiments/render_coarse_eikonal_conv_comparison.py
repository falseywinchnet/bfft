"""Render fair coarse-derived Eikonal-chart CONV cycle comparisons."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_basin_average, conv_resize, lanczos3_resize  # noqa: E402
from experiments.conv_fermi_owner_demo import (  # noqa: E402
    eikonal_basin_conv_resize,
)


def _float_image(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    return np.clip(image[..., :3], 0.0, 1.0).astype(np.float32)


def _cycle(source: np.ndarray, method: str, factor: int = 4) -> np.ndarray:
    height, width = source.shape[:2]
    coarse = conv_basin_average(
        source, ((height - 1) // factor + 1, (width - 1) // factor + 1)
    )
    if method == "Basin CONV":
        return conv_resize(coarse, source.shape[:2])
    if method == "EB-CONV":
        return eikonal_basin_conv_resize(coarse, source.shape[:2])
    if method == "Lanczos-3 sinc":
        return lanczos3_resize(coarse, source.shape[:2])
    raise ValueError(method)


def render(output: Path) -> None:
    sources = {
        "text": _float_image(data.text()),
        "camera": _float_image(data.camera()),
        "logo": _float_image(data.logo()),
    }
    methods = ("EB-CONV", "Lanczos-3 sinc")
    panel_width = 512
    header = 34
    rows: list[Image.Image] = []
    font = ImageFont.load_default(size=18)
    for source_name, source in sources.items():
        panels: list[Image.Image] = []
        for method in methods:
            result = _cycle(source, method)
            # Display conversion is not modular arithmetic.  Signed-lobe
            # overshoot is recorded numerically elsewhere; the visible raster
            # uses the ordinary bounded display domain.
            rgb = Image.fromarray(
                np.rint(255.0 * np.clip(result, 0.0, 1.0)).astype(np.uint8),
                "RGB",
            )
            scale = panel_width / rgb.width
            panel = rgb.resize(
                (panel_width, max(1, int(round(rgb.height * scale)))),
                Image.Resampling.NEAREST,
            )
            framed = Image.new("RGB", (panel.width, panel.height + header), (28, 28, 28))
            framed.paste(panel, (0, header))
            ImageDraw.Draw(framed).text(
                (8, 7), f"{source_name}: {method}", fill=(245, 245, 245), font=font
            )
            panels.append(framed)
        height = max(panel.height for panel in panels)
        row = Image.new("RGB", (sum(panel.width for panel in panels), height), "black")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


if __name__ == "__main__":
    render(ROOT / "output/support_geometry/eikonal_path_oscillation/coarse_eikonal_conv_comparison.png")
