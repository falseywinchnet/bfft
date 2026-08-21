"""Render oracle truth/noise retention maps for the scale posterior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .causal_scale_transport_2d import (
    _isotropic_selling_spectrum,
    causal_scale_transport_observation_2d,
)
from .probe_scale_retention_audit_2d import _filtration_components
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def render(size: int, output: Path) -> None:
    selected = ("cameraman", "tapered hair", "woven chirps")
    catalogue = sources(size)
    columns = (
        "truth",
        "mixed observation",
        "phase posterior",
        "truth retained",
        "truth lost",
        "noise retained",
    )
    magnification = max(1, 192 // size)
    panel = size * magnification
    gap = 10
    left = 112
    top = 54
    footer = 28
    row_height = panel + footer + gap
    canvas = Image.new(
        "RGB",
        (left + len(columns) * (panel + gap),
         top + len(selected) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (left, 8),
        "Continuous-scale posterior: oracle retention and loss audit",
        fill="black",
        font=font,
    )
    for column, title in enumerate(columns):
        draw.text(
            (left + column * (panel + gap), 34),
            title,
            fill="black",
            font=font,
        )

    for row, source in enumerate(selected):
        truth = catalogue[source]
        observation = corrupt(
            truth,
            "mixed replacement + uniform",
            amount=0.10,
            density=0.25,
            seed=271828,
        )
        _readout, _residual, diagnostic = (
            causal_scale_transport_observation_2d(observation))
        coordinate = np.asarray(
            diagnostic["phase_susceptibility_coarse_to_fine"])
        spectrum = _isotropic_selling_spectrum(truth.shape)
        times = np.asarray(diagnostic["transport_times"])
        truth_coarse, truth_components = _filtration_components(
            truth, times, spectrum)
        noise_coarse, noise_components = _filtration_components(
            observation - truth, times, spectrum)
        retained_truth = truth_coarse + np.sum(
            coordinate * truth_components, axis=0)
        retained_noise = noise_coarse + np.sum(
            coordinate * noise_components, axis=0)
        estimate = retained_truth + retained_noise
        lost_truth = truth - retained_truth
        signed_limit = max(
            float(np.max(np.abs(lost_truth))),
            float(np.max(np.abs(retained_noise))),
            np.finfo(float).eps,
        )
        images = (
            (truth, False),
            (observation, False),
            (estimate, False),
            (retained_truth, False),
            (lost_truth, True),
            (retained_noise, True),
        )
        y = top + row * row_height
        draw.text((6, y + panel // 2), source, fill="black", font=font)
        for column, (image, signed) in enumerate(images):
            if signed:
                unit = np.clip(image / signed_limit, -1.0, 1.0)
                magnitude = np.abs(unit)
                colour = np.empty(image.shape + (3,), dtype=np.float64)
                colour[..., 0] = np.where(
                    unit >= 0.0, 1.0, 1.0 - magnitude)
                colour[..., 1] = 1.0 - magnitude
                colour[..., 2] = np.where(
                    unit <= 0.0, 1.0, 1.0 - magnitude)
                pixels = np.rint(255.0 * colour).astype(np.uint8)
                tile = Image.fromarray(pixels, mode="RGB")
            else:
                pixels = np.rint(
                    255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
                tile = Image.fromarray(pixels, mode="L").convert("RGB")
            tile = tile.resize((panel, panel), resample=Image.Resampling.NEAREST)
            canvas.paste(tile, (left + column * (panel + gap), y))
        draw.text(
            (left + 4 * (panel + gap), y + panel + 5),
            f"RMS {np.sqrt(np.mean(lost_truth ** 2)):.4f}",
            fill="black",
            font=font,
        )
        draw.text(
            (left + 5 * (panel + gap), y + panel + 5),
            f"RMS {np.sqrt(np.mean(retained_noise ** 2)):.4f}",
            fill="black",
            font=font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.size, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
