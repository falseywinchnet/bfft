"""Controlled phase/geometry experiment for the v3 reconstruction.

The image is rendered analytically on a supersampled grid.  Its thin curves
therefore have known continuous positions rather than positions inherited
from a raster drawing primitive.  The report separates:

* normal displacement of each reconstructed line;
* contrast gain (coherent amplitude) along that line; and
* matched Lanczos-preview error over four sampling phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3


def _line_fields(x, width, height):
    """Named continuous center lines and dy/dx slopes."""
    cable = (
        0.18 * height
        + 0.00072 * (x - 0.42 * width) ** 2,
        0.00144 * (x - 0.42 * width),
    )
    diagonal = (
        0.70 * height - 0.43 * x,
        np.full_like(x, -0.43),
    )
    wire = (
        0.675 * height + 0.006 * width * np.sin(2.0 * np.pi * x / width),
        0.012 * np.pi * np.cos(2.0 * np.pi * x / width),
    )
    return {"cable": cable, "diagonal": diagonal, "wire": wire}


def _coverage(y, center, slope, width):
    distance = (y - center) / np.sqrt(1.0 + slope * slope)
    return np.clip(0.5 + width * 0.5 - np.abs(distance), 0.0, 1.0)


def phase_geometry_rig(height=512, width=640, supersample=4):
    """Return an RGB float image in [0, 255] with exact subpixel geometry."""
    scale = int(supersample)
    yy, xx = np.mgrid[:height * scale, :width * scale].astype(np.float64)
    x = (xx + 0.5) / scale
    y = (yy + 0.5) / scale

    base = 0.53 + 0.06 * x / width - 0.04 * y / height
    base += 0.012 * np.sin(2.0 * np.pi * x / (0.73 * width))
    image = base

    # Large cartoon regions make the rig exercise the same two-stage path.
    disc = (x - 0.30 * width) ** 2 + (y - 0.48 * height) ** 2
    image += 0.115 * (disc < (0.155 * height) ** 2)
    block = (
        (x > 0.67 * width) & (x < 0.90 * width)
        & (y > 0.53 * height) & (y < 0.82 * height)
    )
    image -= 0.13 * block

    fields = _line_fields(x, width, height)
    image -= 0.31 * _coverage(y, *fields["cable"], 1.10)
    image += 0.24 * _coverage(y, *fields["diagonal"], 0.90)
    image += 0.28 * _coverage(y, *fields["wire"], 0.78)

    # Fence pickets include crossings and endpoints, the difficult cases.
    for position in np.arange(0.53 * width, 0.94 * width, 17.25):
        picket = np.clip(0.5 + 0.72 - np.abs(x - position), 0.0, 1.0)
        extent = (y > 0.62 * height) & (y < 0.88 * height)
        image += 0.18 * picket * extent

    # A sharp L corner and a fine oscillation expose corner/phase folding.
    horizontal = (
        np.clip(0.5 + 0.82 - np.abs(y - 0.875 * height), 0.0, 1.0)
        * (x > 0.55 * width) * (x < 0.91 * width)
    )
    vertical = (
        np.clip(0.5 + 0.82 - np.abs(x - 0.91 * width), 0.0, 1.0)
        * (y > 0.70 * height) * (y < 0.876 * height)
    )
    image -= 0.22 * np.maximum(horizontal, vertical)
    texture_zone = (x > 0.05 * width) & (x < 0.46 * width) & (y > 0.76 * height)
    image += (
        0.035 * np.sin(2.0 * np.pi * (0.31 * x + 0.17 * y))
        * texture_zone
    )

    image = image.reshape(height, scale, width, scale).mean(axis=(1, 3))
    rgb = np.repeat(np.clip(image, 0.0, 1.0)[..., None], 3, axis=2)
    return rgb * 255.0


def _luminance(image):
    value = np.asarray(image, dtype=np.float64)[..., :3]
    return value @ np.array([0.2126, 0.7152, 0.0722])


def _bilinear(image, x, y):
    height, width = image.shape
    x = np.clip(x, 0.0, width - 1.000001)
    y = np.clip(y, 0.0, height - 1.000001)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    dx = x - x0
    dy = y - y0
    return (
        image[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + image[y0, x0 + 1] * dx * (1.0 - dy)
        + image[y0 + 1, x0] * (1.0 - dx) * dy
        + image[y0 + 1, x0 + 1] * dx * dy
    )


def _normal_profile(image, name, samples=480):
    grey = _luminance(image)
    height, width = grey.shape
    x = np.linspace(0.08 * width, 0.92 * width, samples)
    center, slope = _line_fields(x, width, height)[name]
    offsets = np.linspace(-5.0, 5.0, 161)
    normal = np.sqrt(1.0 + slope * slope)
    profile = np.empty(len(offsets), dtype=np.float64)
    for index, offset in enumerate(offsets):
        profile[index] = np.mean(_bilinear(
            grey,
            x - offset * slope / normal,
            center + offset / normal,
        ))
    baseline = 0.5 * (np.mean(profile[:16]) + np.mean(profile[-16:]))
    return offsets, profile - baseline


def _phase_and_gain(source, reconstruction, name):
    offsets, target = _normal_profile(source, name)
    _, fitted = _normal_profile(reconstruction, name)
    shifts = np.linspace(-2.0, 2.0, 161)
    best = None
    for shift in shifts:
        shifted = np.interp(offsets, offsets + shift, fitted)
        gain = float(np.dot(target, shifted) / max(np.dot(shifted, shifted), 1e-15))
        error = float(np.mean((target - gain * shifted) ** 2))
        candidate = (error, shift, gain)
        if best is None or candidate < best:
            best = candidate
    error, shift, correction_gain = best
    reconstructed_amplitude = 1.0 / max(abs(correction_gain), 1e-15)
    return {
        "normal_displacement_px": float(shift),
        "reconstruction_amplitude_ratio": float(reconstructed_amplitude),
        "profile_mse_after_gain": float(error),
    }


def _lanczos_preview(image, output_size, crop_phase):
    value = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
    phase_y, phase_x = crop_phase
    value = value[phase_y:, phase_x:]
    return np.asarray(
        Image.fromarray(value, mode="RGB").resize(
            output_size, resample=Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0


def analyze(source, reconstruction):
    source = np.asarray(source, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    height, width = source.shape[:2]
    lines = {
        name: _phase_and_gain(source, reconstruction, name)
        for name in ("cable", "diagonal", "wire")
    }
    preview = []
    output_size = (max(1, width // 2), max(1, height // 2))
    for phase_y, phase_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        target = _lanczos_preview(source, output_size, (phase_y, phase_x))
        fitted = _lanczos_preview(
            reconstruction, output_size, (phase_y, phase_x))
        preview.append({
            "crop_phase": [phase_y, phase_x],
            "rmse": float(np.sqrt(np.mean((target - fitted) ** 2))),
            "bias": float(np.mean(fitted - target)),
        })
    return {"line_profiles": lines, "lanczos_half_scale": preview}


def _save_montage(path, source, reconstruction):
    source = np.clip(np.rint(source * 255.0), 0, 255).astype(np.uint8)
    reconstruction = np.clip(
        np.rint(reconstruction * 255.0), 0, 255).astype(np.uint8)
    error = np.abs(source.astype(np.int16) - reconstruction.astype(np.int16))
    error = np.clip(error * 5, 0, 255).astype(np.uint8)
    panels = [source, reconstruction, error]
    canvas = Image.new("RGB", (source.shape[1] * 3, source.shape[0]))
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (index * source.shape[1], 0))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/out/v3_phase_alignment_rig"))
    args = parser.parse_args()

    source = phase_geometry_rig(args.height, args.width) / 255.0
    result = build_segmenting_v3(
        source,
        SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_flow_sweeps=1,
            structural_characteristic_passes=1,
            texture_safety_cells=max(32768, args.height * args.width),
            threads=4,
        ),
    )
    reconstruction = result["reconstruction_rgb"]
    report = analyze(source, reconstruction)
    report["psnr_db"] = float(result["record"]["psnr"])
    report["texture_cells"] = int(len(result["texture_centers"]))
    report["total_ms"] = float(result["timing"]["total_ms"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    png_path = args.output.with_suffix(".png")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _save_montage(png_path, source, reconstruction)
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
