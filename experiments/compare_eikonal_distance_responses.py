"""Compare fixed all-source Eikonal response laws on one controlled cycle."""

from __future__ import annotations

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
    DISTANCE_RESPONSES,
    build_direct_distance_atlas,
    build_distance_atlas,
    eikonal_basin_conv_resize,
    synthesize_distance_responses,
    synthesize_distance_charts,
)


def _camera() -> np.ndarray:
    return np.asarray(data.camera(), dtype=np.float32) / 255.0


def run(output: Path) -> None:
    truth = _camera()
    coarse = conv_basin_average(truth, (128, 128))
    started = time.perf_counter()
    atlas = build_distance_atlas(coarse)
    distance_results = synthesize_distance_responses(
        coarse, truth.shape, atlas, DISTANCE_RESPONSES
    )
    direct_atlas = build_direct_distance_atlas(
        coarse, response="inverse_power_2"
    )
    direct_result = synthesize_distance_charts(
        coarse,
        truth.shape,
        direct_atlas,
        response="inverse_power_2",
    )
    elapsed = time.perf_counter() - started
    results = {
        "owner partition": eikonal_basin_conv_resize(coarse, truth.shape),
        "direct-frame inverse-2": direct_result,
        **distance_results,
        "Cartesian CONV": conv_resize(coarse, truth.shape),
        "Lanczos-3 sinc": lanczos3_resize(coarse, truth.shape),
    }
    owner = results["owner partition"]
    font = ImageFont.load_default(size=16)
    panel_side = 512
    header = 58
    panels = []
    rows = []
    for name, result in results.items():
        mse_truth = float(np.mean((result - truth) ** 2))
        mse_owner = float(np.mean((result - owner) ** 2))
        rows.append((name, mse_truth, mse_owner))
        image = Image.fromarray(
            np.rint(255.0 * np.clip(result, 0.0, 1.0)).astype(np.uint8)
        ).resize((panel_side, panel_side), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (panel_side, panel_side + header), (25, 25, 25))
        panel.paste(image.convert("RGB"), (0, header))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6), name, fill="white", font=font)
        draw.text(
            (8, 29),
            f"truth MSE {mse_truth:.8f}   owner MSE {mse_owner:.8f}",
            fill=(220, 220, 220), font=font,
        )
        panels.append(panel)
    columns = 3
    sheet = Image.new(
        "RGB",
        (columns * panel_side, ((len(panels) + columns - 1) // columns) * (panel_side + header)),
        "black",
    )
    for index, panel in enumerate(panels):
        sheet.paste(
            panel,
            ((index % columns) * panel_side, (index // columns) * (panel_side + header)),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    crop_box = (190, 120, 420, 390)
    crop_scale = 3
    crop_width = (crop_box[2] - crop_box[0]) * crop_scale
    crop_height = (crop_box[3] - crop_box[1]) * crop_scale
    crop_panels = []
    crop_results = {"truth": truth, **results}
    for name, result in crop_results.items():
        crop = Image.fromarray(
            np.rint(255.0 * np.clip(result, 0.0, 1.0)).astype(np.uint8)
        ).crop(crop_box).resize(
            (crop_width, crop_height), Image.Resampling.NEAREST
        )
        panel = Image.new("RGB", (crop_width, crop_height + 34), (25, 25, 25))
        panel.paste(crop.convert("RGB"), (0, 34))
        ImageDraw.Draw(panel).text((8, 7), name, fill="white", font=font)
        crop_panels.append(panel)
    crop_columns = 3
    crop_sheet = Image.new(
        "RGB",
        (
            crop_columns * crop_width,
            ((len(crop_panels) + crop_columns - 1) // crop_columns)
            * (crop_height + 34),
        ),
        "black",
    )
    for index, panel in enumerate(crop_panels):
        crop_sheet.paste(
            panel,
            (
                (index % crop_columns) * crop_width,
                (index // crop_columns) * (crop_height + 34),
            ),
        )
    crop_sheet.save(output.with_name(output.stem + "_diagnostic_crop.png"))
    report = output.with_suffix(".txt")
    report.write_text(
        f"all-source elapsed_seconds {elapsed:.9f}\n"
        + "\n".join(
            f"{name}\ttruth_mse={truth_mse:.12g}\towner_mse={owner_mse:.12g}"
            for name, truth_mse, owner_mse in rows
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/eikonal_distance_responses_full.png")
