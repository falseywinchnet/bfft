#!/usr/bin/env python3
"""Render the literal bright-support to erosion-skeleton diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _trace_rgb(values: np.ndarray) -> np.ndarray:
    source = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = source[source > 0.0]
    gauge = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    display = np.log1p(8.0 * source / max(gauge, 1e-30)) / np.log(9.0)
    gray = np.uint8(np.clip(255.0 * (1.0 - display), 0.0, 255.0))
    return np.repeat(gray[..., None], 3, axis=2)


def _binary_rgb(mask: np.ndarray) -> np.ndarray:
    gray = np.where(np.asarray(mask, dtype=bool), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _overlay_rgb(trace: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    output = _trace_rgb(trace)
    output[np.asarray(skeleton, dtype=bool)] = (225, 0, 0)
    return output


def _panel(array: np.ndarray, width: int = 336, height: int = 512) -> Image.Image:
    return Image.fromarray(array, mode="RGB").resize(
        (width, height), Image.Resampling.NEAREST
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    saved = np.load(args.arrays)
    metadata = json.loads(args.metadata.read_text())

    margin = 28
    column_width = 336
    gap = 48
    title_height = 82
    panel_height = 512
    row_gap = 64
    canvas_width = 2 * column_width + gap + 2 * margin
    canvas_height = title_height + 4 * (panel_height + row_gap) + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    label_font = _font(16)
    small_font = _font(13)

    draw.text(
        (margin, 14),
        "real 3×3 Gabors → one bright mask → morphological erosion skeleton",
        fill="black",
        font=title_font,
    )
    columns = (
        ("source", "BDL timed /R/ witness"),
        ("target", "Dave/Simon first /R/ candidate"),
    )
    for column, (prefix, heading) in enumerate(columns):
        x = margin + column * (column_width + gap)
        trace = saved[f"{prefix}_trace"]
        bright = saved[f"{prefix}_bright"]
        skeleton = saved[f"{prefix}_skeleton"]
        draw.text((x, 48), heading, fill="black", font=label_font)
        rows = (
            (_trace_rgb(trace), "1  real positive step-3 trace"),
            (
                _binary_rgb(bright),
                f"2  bright fused-Gabor pixels: {np.count_nonzero(bright)}",
            ),
            (
                _binary_rgb(skeleton),
                f"3  erosion skeleton: {np.count_nonzero(skeleton)} pixels",
            ),
            (_overlay_rgb(trace, skeleton), "4  skeleton (red) over trace"),
        )
        for row, (array, label) in enumerate(rows):
            y = title_height + row * (panel_height + row_gap)
            canvas.paste(_panel(array, column_width, panel_height), (x, y))
            draw.rectangle(
                (x, y, x + column_width - 1, y + panel_height - 1),
                outline=(96, 96, 96),
            )
            draw.text((x, y + panel_height + 6), label, fill="black", font=label_font)
            draw.text(
                (x, y + panel_height + 28),
                f"native array: {trace.shape[0]} rows × {trace.shape[1]} frames",
                fill=(80, 80, 80),
                font=small_font,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(json.dumps({"output": str(args.out), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
