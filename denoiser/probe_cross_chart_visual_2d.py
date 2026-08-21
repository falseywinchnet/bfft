"""Render the relative-chart and source-coverage Cameraman theorem gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cross_chart_transport_closure_2d import (
    denoise_cross_chart_transport_closure_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_complete_moment_lineage_2d import displacement
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def _save(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.round(
        np.clip(value, 0.0, 1.0) * 255.0))).save(path)


def _render(
    output: Path,
    estimates: dict[str, np.ndarray],
    measured: dict[str, dict[str, float]],
) -> None:
    size = next(iter(estimates.values())).shape[0]
    scale = max(1, 320 // size)
    panel_size = size * scale
    title_height = 72
    canvas = Image.new(
        "RGB", (panel_size * len(estimates), panel_size + title_height),
        "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, (name, value) in enumerate(estimates.items()):
        panel = Image.fromarray(np.uint8(np.round(
            np.clip(value, 0.0, 1.0) * 255.0))).resize(
                (panel_size, panel_size), Image.Resampling.NEAREST).convert("RGB")
        left = index * panel_size
        canvas.paste(panel, (left, title_height))
        if name == "truth":
            title = name
        else:
            row = measured[name]
            title = (
                f"{name}\nMSE {row['mse']:.5f} | SSIM {row['ssim']:.4f}\n"
                f"edge {row['edge_retention']:.4f} | "
                f"obs RMS {row['observation_displacement_rms']:.4f}")
        bounds = draw.multiline_textbbox((0, 0), title, font=font, align="center")
        draw.multiline_text(
            (left + (panel_size - (bounds[2] - bounds[0])) / 2.0, 5),
            title, fill="black", font=font, align="center")
    canvas.save(output / "cross_chart_comparison.png")


def run(size: int, output: Path) -> dict:
    truth = sources(size)["cameraman"]
    observation = corrupt(
        truth, "mixed replacement + uniform", amount=0.10, density=0.25,
        seed=271828)
    relative, diagnostic = denoise_cross_chart_transport_closure_2d(
        observation,
        angular_count=4,
        quantile_count=16,
        phase_count=1,
        complete_residual_moment=True,
    )
    readouts = diagnostic["readouts"]
    source_coverage = readouts["source_coverage_closure_barycenter"]
    fmmt = denoise_fmmt(observation)[0]
    estimates = {
        "truth": truth,
        "observation": observation,
        "HJ consensus": readouts["transport_chart_consensus"],
        "relative closure": relative,
        "source coverage": source_coverage,
        "FMMT": fmmt,
        "coverage + FMMT": denoise_fmmt(source_coverage)[0],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in estimates.items():
        _save(output / f"{name.replace(' ', '_').lower()}.png", value)
    measured = {
        name: {**metrics(value, truth), **displacement(value, observation)}
        for name, value in estimates.items()
    }
    _render(output, estimates, measured)
    result = {
        "purpose": (
            "visual gate for deblurrer-inspired relative operator closure and "
            "absolute source coverage under mixed unknown noise"),
        "source": "cameraman",
        "condition": "mixed replacement + uniform 0.25",
        "size": size,
        "metrics": measured,
        "closure_diagnostic": diagnostic["closure"],
        "theory_status": "visual theorem probe; not promoted",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.out)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
