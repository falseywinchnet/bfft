"""Render the decisive successes and failures of the minimal 2-D lift."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .cross_predictive_transport_2d import denoise_cross_predictive_transport_2d
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def _tile(value: np.ndarray, size: int) -> Image.Image:
    pixels = np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0))
    return Image.fromarray(pixels, mode="L").resize(
        (size, size), Image.Resampling.NEAREST).convert("RGB")


def render(output: Path, size: int = 96, seed: int = 9100) -> None:
    selected = (
        "cameraman",
        "tapered hair",
        "woven chirps",
        "line drawing",
    )
    columns = (
        "clean truth",
        "mixed observation",
        "theoretical seed",
        "integrated FMMT",
        "median 3x3",
    )
    display_size = 240
    left_margin = 190
    top_margin = 82
    row_height = display_size + 70
    canvas = Image.new(
        "RGB",
        (left_margin + display_size * len(columns),
         top_margin + row_height * len(selected)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        "Minimal 2-D characteristic lift: mixed replacement + uniform (25%)",
        fill="#111827",
    )
    draw.text(
        (12, 31),
        "The same law preserves distributed relations but weakens sparse interfaces.",
        fill="#374151",
    )
    for column, name in enumerate(columns):
        draw.text(
            (left_margin + column * display_size + 7, 59), name,
            fill="#111827",
        )

    images = sources(size)
    for row, name in enumerate(selected):
        truth = images[name]
        observation = corrupt(
            truth,
            "mixed replacement + uniform",
            amount=0.10,
            density=0.25,
            seed=seed,
        )
        candidate = denoise_cross_predictive_transport_2d(observation)[0]
        empirical = denoise_fmmt(observation)[0]
        median = ndimage.median_filter(observation, 3, mode="reflect")
        values = (truth, observation, candidate, empirical, median)
        top = top_margin + row * row_height
        draw.text((8, top + 9), name, fill="#111827")
        for column, value in enumerate(values):
            left = left_margin + column * display_size
            canvas.paste(_tile(value, display_size), (left, top))
            draw.rectangle(
                (left, top, left + display_size - 1, top + display_size - 1),
                outline="#9ca3af",
            )
            if column:
                score = metrics(value, truth)
                draw.text(
                    (left + 6, top + display_size + 8),
                    f"MSE {score['mse']:.4f}  SSIM {score['ssim']:.3f}",
                    fill="#374151",
                )
                draw.text(
                    (left + 6, top + display_size + 27),
                    f"edge {score['edge_retention']:.3f}",
                    fill="#374151",
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=9100)
    args = parser.parse_args()
    render(args.out, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
