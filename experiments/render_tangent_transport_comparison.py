"""Render the fixed astronaut 8x cycle used to inspect tangent transport."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import data

from experiments.conv_distilled_core import (
    convstar_bounded_resize,
    convstar_characteristic_transport_resize,
    convstar_tangent_transport_resize,
    symmetric_conv_synthesis,
)
from standalone_conv_resize_demo.backend import lanczos3_resize


def _u8(value: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def _cycle(method, source: np.ndarray) -> np.ndarray:
    coarse = method(source, (64, 64))
    return method(coarse, source.shape[:2])


def _lanczos_cycle(source: np.ndarray) -> np.ndarray:
    coarse = lanczos3_resize(source, (64, 64))
    return lanczos3_resize(coarse, source.shape[:2])


def _symmetric_cycle(source: np.ndarray) -> np.ndarray:
    coarse = convstar_bounded_resize(source, (64, 64))
    return symmetric_conv_synthesis(coarse, source.shape[:2])


def _labelled_strip(
    images: list[np.ndarray], labels: list[str], crop: tuple[int, int, int, int]
) -> Image.Image:
    pieces: list[Image.Image] = []
    for image, label in zip(images, labels, strict=True):
        piece = Image.fromarray(_u8(image)).crop(crop)
        canvas = Image.new("RGB", (piece.width, piece.height + 24), "#202020")
        canvas.paste(piece, (0, 24))
        ImageDraw.Draw(canvas).text((5, 6), label, fill="white")
        pieces.append(canvas)
    strip = Image.new(
        "RGB", (sum(piece.width for piece in pieces), max(p.height for p in pieces)),
        "#202020",
    )
    x = 0
    for piece in pieces:
        strip.paste(piece, (x, 0))
        x += piece.width
    return strip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("output/support_geometry/tangent_transport")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source = np.asarray(data.astronaut(), dtype=np.float32) / 255.0
    current = _cycle(convstar_bounded_resize, source)
    symmetric = _symmetric_cycle(source)
    transported = _cycle(convstar_tangent_transport_resize, source)
    characteristic = _cycle(convstar_characteristic_transport_resize, source)
    lanczos = _lanczos_cycle(source)
    names = (
        "truth", "current CONV", "symmetric CONV", "transported tensor",
        "curved characteristic", "Lanczos-3",
    )
    values = (source, current, symmetric, transported, characteristic, lanczos)

    for name, value in zip(names, values, strict=True):
        Image.fromarray(_u8(value)).save(
            args.out / f"{name.lower().replace(' ', '_')}.png"
        )

    # Native output pixels: no interpolation is used in either comparison.
    _labelled_strip(list(values), list(names), (230, 190, 512, 460)).save(
        args.out / "helmet_native_strip.png"
    )
    _labelled_strip(list(values), list(names), (330, 285, 512, 455)).save(
        args.out / "helmet_upper_right_native_strip.png"
    )

    for name, value in zip(names[1:], values[1:], strict=True):
        error = np.asarray(value, dtype=np.float64) - source
        print(
            f"{name}: mse={np.mean(error * error):.12g} "
            f"max_abs={np.max(np.abs(error)):.12g}"
        )


if __name__ == "__main__":
    main()
