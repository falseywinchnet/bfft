#!/usr/bin/env python3
"""Tune the moment-neutral CONV reduction passband on a geographic witness.

The diagnostic uses one source crop and one fixed destination lattice.  It
compares canonical ordered-CONV basin analysis with a fourth-order,
moment-neutral passband correction.  Selection is constrained by reconstructed
PSNR, reconstructed edge PSNR, source-range overshoot, and maximum code-value
change; retained gradient and local high-pass energy are reported, not used in
isolation as a quality oracle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from backend import (  # noqa: E402
    conv_passband_basin_lines_f64,
    conv_resize,
)


LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


def geographic_crop(
    source: Image.Image, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    west, east, south, north = bounds
    width, height = source.size
    left = int(math.floor((west + 180.0) / 360.0 * width))
    right = int(math.ceil((east + 180.0) / 360.0 * width))
    top = int(math.floor((90.0 - north) / 180.0 * height))
    bottom = int(math.ceil((90.0 - south) / 180.0 * height))
    return np.asarray(source.crop((left, top, right, bottom)), dtype=np.float64)


def reduce_axis(
    values: np.ndarray, target: int, axis: int, compensation: float
) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    shape = moved.shape
    lines = np.ascontiguousarray(moved.reshape(shape[0], -1))
    reduced = conv_passband_basin_lines_f64(
        lines, target, compensation=compensation
    )
    return np.moveaxis(reduced.reshape((target,) + shape[1:]), 0, axis)


def reduce_image(
    source: np.ndarray, target: tuple[int, int], compensation: float
) -> np.ndarray:
    # Canonical analysis composes in the reverse order of synthesis.
    vertical = reduce_axis(source, target[0], 0, compensation)
    return reduce_axis(vertical, target[1], 1, compensation)


def gradient(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.diff(plane, axis=1, append=plane[:, -1:])
    gy = np.diff(plane, axis=0, append=plane[-1:, :])
    return gx, gy


def gradient_energy(plane: np.ndarray) -> float:
    gx, gy = gradient(plane)
    return float(np.mean(gx * gx + gy * gy))


def binomial_blur(plane: np.ndarray) -> np.ndarray:
    kernel = np.array((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float64) / 16.0
    horizontal = np.empty_like(plane)
    padded = np.pad(plane, ((0, 0), (2, 2)), mode="reflect")
    horizontal[:] = sum(kernel[k] * padded[:, k:k + plane.shape[1]] for k in range(5))
    padded = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
    return sum(kernel[k] * padded[k:k + plane.shape[0]] for k in range(5))


def highpass_energy(plane: np.ndarray) -> float:
    residual = plane - binomial_blur(plane)
    return float(np.mean(residual * residual))


def psnr(first: np.ndarray, second: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean(np.square(first - second)))
    return math.inf if mse == 0.0 else 10.0 * math.log10(peak * peak / mse)


def edge_psnr(first: np.ndarray, second: np.ndarray, peak: float = 255.0) -> float:
    first_x, first_y = gradient(first)
    second_x, second_y = gradient(second)
    mse = float(np.mean(np.square(first_x - second_x) + np.square(first_y - second_y)))
    return math.inf if mse == 0.0 else 10.0 * math.log10(2.0 * peak * peak / mse)


def reconstruct(candidate: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(conv_resize(candidate.astype(np.float32), shape), dtype=np.float64)


def synthetic_audit(compensation: float) -> dict[str, float]:
    x = np.linspace(-1.0, 1.0, 2049, dtype=np.float64)
    source = (
        0.30 * x
        + 0.28 * np.tanh(35.0 * (x + 0.29))
        - 0.22 * np.tanh(52.0 * (x - 0.16))
        + 0.018 * np.sin(73.0 * np.pi * x)
    )[:, None]
    result = conv_passband_basin_lines_f64(
        source, 257, compensation=compensation
    )[:, 0]
    lower = float(source.min())
    upper = float(source.max())
    overshoot = max(lower - float(result.min()), float(result.max()) - upper, 0.0)
    step = np.zeros((2049, 1), dtype=np.float64)
    step[1024:] = 1.0
    reduced_step = conv_passband_basin_lines_f64(
        step, 257, compensation=compensation
    )[:, 0]
    step_overshoot = max(
        -float(reduced_step.min()), float(reduced_step.max()) - 1.0, 0.0
    )
    return {
        "source_range_overshoot": overshoot,
        "gradient_energy": float(np.mean(np.square(np.diff(result)))),
        "unit_step_overshoot": step_overshoot,
    }


def candidate_metrics(
    source: np.ndarray,
    canonical: np.ndarray,
    candidate: np.ndarray,
    compensation: float,
) -> dict[str, float]:
    source_luma = source @ LUMA
    canonical_luma = canonical @ LUMA
    candidate_luma = candidate @ LUMA
    reconstruction = reconstruct(candidate, source.shape[:2])
    reconstruction_luma = reconstruction @ LUMA
    source_min = source.min(axis=(0, 1))
    source_max = source.max(axis=(0, 1))
    overshoot = max(
        float(np.max(source_min - candidate.min(axis=(0, 1)))),
        float(np.max(candidate.max(axis=(0, 1)) - source_max)),
        0.0,
    )
    canonical_gradient = gradient_energy(canonical_luma)
    canonical_highpass = highpass_energy(canonical_luma)
    synthetic = synthetic_audit(compensation)
    return {
        "compensation": compensation,
        "gradient_energy": gradient_energy(candidate_luma),
        "gradient_energy_ratio_to_canonical": (
            gradient_energy(candidate_luma) / max(canonical_gradient, 1.0e-30)
        ),
        "highpass_energy": highpass_energy(candidate_luma),
        "highpass_energy_ratio_to_canonical": (
            highpass_energy(candidate_luma) / max(canonical_highpass, 1.0e-30)
        ),
        "reconstruction_psnr_db": psnr(source_luma, reconstruction_luma),
        "reconstruction_edge_psnr_db": edge_psnr(source_luma, reconstruction_luma),
        "source_range_overshoot_codes": overshoot,
        "maximum_change_from_canonical_codes": float(
            np.max(np.abs(candidate - canonical))
        ),
        "rms_change_from_canonical_codes": float(
            np.sqrt(np.mean(np.square(candidate - canonical)))
        ),
        "synthetic_source_range_overshoot": synthetic["source_range_overshoot"],
        "synthetic_gradient_energy": synthetic["gradient_energy"],
        "synthetic_unit_step_overshoot": synthetic["unit_step_overshoot"],
    }


def render_contact_sheet(
    canonical: np.ndarray,
    winner: np.ndarray,
    compensation: float,
    path: Path,
) -> None:
    canonical_image = Image.fromarray(np.uint8(np.clip(np.rint(canonical), 0, 255)), "RGB")
    winner_image = Image.fromarray(np.uint8(np.clip(np.rint(winner), 0, 255)), "RGB")
    width, height = canonical_image.size
    scale = 2
    canonical_image = canonical_image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    winner_image = winner_image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    sheet = Image.new("RGB", (width * scale * 2, height * scale + 34), "white")
    sheet.paste(canonical_image, (0, 34))
    sheet.paste(winner_image, (width * scale, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "canonical ordered-CONV basin", fill="black")
    draw.text(
        (width * scale + 8, 10),
        f"moment-neutral passband {compensation:.3f}",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--bounds", default="90,110,30,42")
    parser.add_argument("--working-width", type=int, default=12289)
    parser.add_argument("--working-height", type=int, default=6145)
    parser.add_argument(
        "--compensations",
        default="0,.0125,.025,.0375,.05,.0625,.075,.1,.125,.15,.2,.25",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bounds = tuple(float(value) for value in args.bounds.split(","))
    if len(bounds) != 4:
        raise ValueError("bounds must be west,east,south,north")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.source) as image:
        rgb = image.convert("RGB")
        source = geographic_crop(rgb, bounds)
        source_width, source_height = rgb.size
    target = (
        max(5, round(source.shape[0] * args.working_height / source_height)),
        max(5, round(source.shape[1] * args.working_width / source_width)),
    )
    canonical = reduce_image(source, target, 0.0)
    records = []
    candidates: dict[float, np.ndarray] = {0.0: canonical}
    for compensation in (float(value) for value in args.compensations.split(",")):
        candidate = canonical if compensation == 0.0 else reduce_image(
            source, target, compensation
        )
        candidates[compensation] = candidate
        records.append(candidate_metrics(
            source, canonical, candidate, compensation
        ))

    baseline = next(row for row in records if row["compensation"] == 0.0)
    admitted = [
        row for row in records
        if row["reconstruction_psnr_db"] >= baseline["reconstruction_psnr_db"] - 0.10
        and row["reconstruction_edge_psnr_db"] >= baseline["reconstruction_edge_psnr_db"] - 0.10
        and row["source_range_overshoot_codes"]
            <= baseline["source_range_overshoot_codes"] + 0.50
        and row["synthetic_unit_step_overshoot"] <= 1.0 / 255.0
        and row["maximum_change_from_canonical_codes"] <= 5.0
    ]
    winner = max(
        admitted or [baseline],
        key=lambda row: (
            row["highpass_energy_ratio_to_canonical"],
            row["gradient_energy_ratio_to_canonical"],
        ),
    )
    result = {
        "source": str(args.source.resolve()),
        "bounds_degrees": list(bounds),
        "source_crop_shape": list(source.shape),
        "target_crop_shape": list(target) + [3],
        "selection_constraints": {
            "maximum_psnr_regression_db": 0.10,
            "maximum_edge_psnr_regression_db": 0.10,
            "maximum_incremental_source_range_overshoot_codes": 0.50,
            "maximum_unit_step_overshoot": "1/255",
            "maximum_change_from_canonical_codes": 5.0,
        },
        "winner": winner,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    render_contact_sheet(
        canonical,
        candidates[float(winner["compensation"])],
        float(winner["compensation"]),
        args.out.with_suffix(".png"),
    )
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
