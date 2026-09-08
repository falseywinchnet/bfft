"""Native 4x astronaut cycles for owner and direct Eikonal frames."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

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
    direct_distance_eikonal_conv_resize,
    eikonal_basin_conv_resize,
)


def _panel(name: str, image: np.ndarray, metrics: dict, font) -> Image.Image:
    raster = Image.fromarray(
        np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8), "RGB"
    )
    header = 62
    panel = Image.new("RGB", (raster.width, raster.height + header), (24, 24, 24))
    panel.paste(raster, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), name, fill="white", font=font)
    if metrics:
        draw.text(
            (8, 31),
            f"MSE {metrics['mse']:.9f}   max |e| {metrics['max_abs']:.6f}",
            fill=(220, 220, 220), font=font,
        )
    return panel


def _crop_sheet(
    results: dict[str, np.ndarray], output: Path, font
) -> None:
    # Helmet, face, suit hardware, and the high-contrast shoulder boundary.
    boxes = {
        "face and helmet": (120, 25, 355, 255),
        "suit and shoulder": (0, 205, 310, 500),
    }
    scale = 2
    rows = []
    for crop_name, box in boxes.items():
        panels = []
        for name, image in results.items():
            raster = Image.fromarray(
                np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8), "RGB"
            ).crop(box)
            raster = raster.resize(
                (raster.width * scale, raster.height * scale),
                Image.Resampling.NEAREST,
            )
            panel = Image.new("RGB", (raster.width, raster.height + 32), (24, 24, 24))
            panel.paste(raster, (0, 32))
            ImageDraw.Draw(panel).text(
                (7, 6), f"{crop_name}: {name}", fill="white", font=font
            )
            panels.append(panel)
        row = Image.new(
            "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black"
        )
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    sheet = Image.new(
        "RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black"
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output)


def run(output: Path) -> None:
    source = np.asarray(data.astronaut(), dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    coarse_shape = ((height - 1) // 4 + 1, (width - 1) // 4 + 1)

    timings = {}
    started = time.perf_counter()
    conv_coarse = conv_basin_average(source, coarse_shape)
    timings["conv_analysis"] = time.perf_counter() - started
    started = time.perf_counter()
    owner = eikonal_basin_conv_resize(conv_coarse, source.shape[:2])
    timings["owner_synthesis"] = time.perf_counter() - started
    started = time.perf_counter()
    direct = direct_distance_eikonal_conv_resize(
        conv_coarse, source.shape[:2], response="inverse_power_2"
    )
    timings["direct_synthesis"] = time.perf_counter() - started
    started = time.perf_counter()
    cartesian = conv_resize(conv_coarse, source.shape[:2])
    timings["cartesian_synthesis"] = time.perf_counter() - started
    started = time.perf_counter()
    lanczos_coarse = lanczos3_resize(source, coarse_shape)
    lanczos = lanczos3_resize(lanczos_coarse, source.shape[:2])
    timings["lanczos_cycle"] = time.perf_counter() - started

    results = {
        "truth": source,
        "owner frames": owner,
        "direct action frames": direct,
        "Cartesian CONV": cartesian,
        "matched Lanczos-3": lanczos,
    }
    metrics = {}
    for name, result in results.items():
        if name == "truth":
            continue
        residual = np.asarray(result, dtype=np.float64) - source
        metrics[name] = {
            "mse": float(np.mean(residual * residual)),
            "max_abs": float(np.max(np.abs(residual))),
            "range_excess": float(max(
                -float(np.min(result)), float(np.max(result)) - 1.0, 0.0
            )),
        }

    font = ImageFont.load_default(size=16)
    panels = [
        _panel(name, image, metrics.get(name, {}), font)
        for name, image in results.items()
    ]
    columns = 3
    panel_width = 512
    panel_height = 574
    sheet = Image.new(
        "RGB",
        (columns * panel_width, ((len(panels) + columns - 1) // columns) * panel_height),
        "black",
    )
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * panel_width, (index // columns) * panel_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    _crop_sheet(results, output.with_name(output.stem + "_crops.png"), font)
    output.with_suffix(".json").write_text(
        json.dumps({
            "source_shape": list(source.shape[:2]),
            "coarse_shape": list(coarse_shape),
            "cycles": {
                "owner frames": "CONV basin analysis -> owner-frame synthesis",
                "direct action frames": "CONV basin analysis -> direct action-frame synthesis",
                "Cartesian CONV": "CONV basin analysis -> Cartesian CONV synthesis",
                "matched Lanczos-3": "Lanczos-3 analysis -> Lanczos-3 synthesis",
            },
            "metrics": metrics,
            "timings_seconds": timings,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/astronaut_4x_frame_cycle.png")
