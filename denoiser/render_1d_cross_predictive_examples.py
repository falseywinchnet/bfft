"""Render matched mixed-corruption examples from the broad 1-D battery."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .affine_relation_transport import denoise_affine_relations
from .cross_predictive_transport import denoise_cross_predictive_transport
from .run_1d_cross_predictive_battery import PRESET_NAMES
from .sample_series import PRESETS, compose_series, corrupt
from .transport_support import TransportResolution, denoise_1d


def render(output: Path, size: int = 256, seed: int = 8101) -> None:
    resolution = TransportResolution(
        scale_samples=5, histogram_bins=32, maximum_steps=2048)
    columns = (
        "clean truth",
        "mixed observation",
        "cross-predictive",
        "fixed affine",
        "legacy support flow",
    )
    cell_width = 340
    cell_height = 150
    left_margin = 170
    top_margin = 54
    canvas = Image.new(
        "RGB",
        (left_margin + cell_width * len(columns),
         top_margin + cell_height * len(PRESET_NAMES)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        "Full-scale relation transport: mixed replacement + uniform; "
        "black = truth, teal = estimate",
        fill="#111827",
    )
    for column, name in enumerate(columns):
        draw.text(
            (left_margin + column * cell_width + 8, 32), name,
            fill="#111827")

    def polyline(value: np.ndarray, column: int, row: int):
        left = left_margin + column * cell_width
        top = top_margin + row * cell_height
        xx = np.linspace(left + 5, left + cell_width - 5, value.size)
        yy = top + cell_height - 8 - np.clip(value, 0.0, 1.0) * (cell_height - 16)
        return list(zip(xx.tolist(), yy.tolist()))

    for row, preset in enumerate(PRESET_NAMES):
        x, truth, _fields = compose_series(size, PRESETS[preset])
        observation = corrupt(
            truth,
            "mixed replacement + uniform",
            amount=0.15,
            density=0.25,
            seed=seed,
        )
        candidate = denoise_cross_predictive_transport(observation)[0]
        affine = denoise_affine_relations(observation)[0]
        legacy = denoise_1d(
            observation,
            resolution,
            provisional_sigma=2.0,
            action_budget_multiplier=8.0,
            continuation_rounds=4,
        )[0]
        values = (truth, observation, candidate, affine, legacy)
        draw.text(
            (6, top_margin + row * cell_height + 8), preset,
            fill="#111827")
        for column, value in enumerate(values):
            left = left_margin + column * cell_width
            top = top_margin + row * cell_height
            draw.rectangle(
                (left, top, left + cell_width - 1, top + cell_height - 1),
                outline="#d1d5db")
            draw.line(polyline(truth, column, row), fill="#111827", width=2)
            if column != 0:
                draw.line(polyline(value, column, row), fill="#0f766e", width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8101)
    args = parser.parse_args()
    render(args.out, args.size, args.seed)


if __name__ == "__main__":
    main()
