#!/usr/bin/env python3
"""Texture-only harmonic-grid relaxation inside the Meyer G-ball.

The source is split once into Meyer cartoon, texture, and model residual.
Only texture enters a smooth, overlapping local Fourier frame.  Each local
harmonic is demodulated by its predicted phase advance across the support
grid.  A genuine carrier therefore becomes a ground (constant) field on that
grid at any image frequency, including upper Nyquist.  Phase-inconsistent or
unsupported energy remains non-ground and receives one graph-diffusion step.

The relaxed texture is optionally projected back through the same G-ball
identity used by Meyer.  Cartoon and model residual are copied unchanged:

    output = cartoon + relaxed_texture + (source - cartoon - texture).

There is no frequency cutoff and no iterative per-cell model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import bfft  # noqa: E402
from bfft._core import meyer_split_batch  # noqa: E402
from experiments.jpeg_dct_geometry_reassembly import (  # noqa: E402
    _image_metrics,
    _save_montage,
)


def _sqrt_periodic_hann(size: int) -> np.ndarray:
    position = np.arange(size, dtype=np.float64)
    periodic_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * position / size)
    return np.sqrt(np.maximum(periodic_hann, 0.0))


def _frame_layout(
    shape: tuple[int, int],
    size: int,
    hop: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    height, width = shape
    margin = size // 2
    base_height = height + 2 * margin
    base_width = width + 2 * margin
    extra_height = (hop - (base_height - size) % hop) % hop
    extra_width = (hop - (base_width - size) % hop) % hop
    padding = (
        (margin, margin + extra_height),
        (margin, margin + extra_width),
    )
    padded_shape = (
        height + sum(padding[0]),
        width + sum(padding[1]),
    )
    return padding, padded_shape


def _analysis(
    fields: np.ndarray,
    size: int,
    hop: int,
) -> tuple[np.ndarray, dict]:
    """Smooth-window rFFT analysis of CxHxW fields."""
    channels, height, width = fields.shape
    padding, padded_shape = _frame_layout((height, width), size, hop)
    padded = np.pad(
        fields,
        ((0, 0), padding[0], padding[1]),
        mode="reflect",
    )
    window_1d = _sqrt_periodic_hann(size)
    window = window_1d[:, None] * window_1d[None, :]
    patches = np.lib.stride_tricks.sliding_window_view(
        padded, (size, size), axis=(1, 2)
    )[:, ::hop, ::hop]
    spectrum = np.fft.rfft2(
        patches * window[None, None, None],
        axes=(-2, -1),
        norm="ortho",
    )
    layout = {
        "source_shape": (height, width),
        "padded_shape": padded_shape,
        "padding": padding,
        "window": window,
        "hop": hop,
        "size": size,
    }
    return spectrum, layout


def _synthesis(spectrum: np.ndarray, layout: dict) -> np.ndarray:
    """Overlap-add inverse of `_analysis`."""
    size = int(layout["size"])
    hop = int(layout["hop"])
    window = np.asarray(layout["window"], dtype=np.float64)
    patches = np.fft.irfft2(
        spectrum,
        s=(size, size),
        axes=(-2, -1),
        norm="ortho",
    ) * window[None, None, None]
    channels, grid_y, grid_x = patches.shape[:3]
    padded_height, padded_width = layout["padded_shape"]
    output = np.zeros(
        (channels, padded_height, padded_width), dtype=np.float64)
    normalization = np.zeros(
        (padded_height, padded_width), dtype=np.float64)
    window_square = window * window
    for offset_y in range(size):
        target_y = slice(offset_y, offset_y + grid_y * hop, hop)
        for offset_x in range(size):
            target_x = slice(offset_x, offset_x + grid_x * hop, hop)
            output[:, target_y, target_x] += (
                patches[..., offset_y, offset_x])
            normalization[target_y, target_x] += window_square[
                offset_y, offset_x]
    output /= np.maximum(normalization[None], 1e-15)
    (top, _bottom), (left, _right) = layout["padding"]
    height, width = layout["source_shape"]
    return output[:, top:top + height, left:left + width]


def _demodulation_phase(
    spectrum: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Estimate each bin's actual phase transport across the support grid.

    A smooth window spreads an off-bin carrier into neighboring spectral
    bins. Those sidebands propagate at the carrier phase, not their nominal
    bin-centre phase. Cross-grid complex correlation measures that transport
    directly and avoids classifying spectral leakage as inconsistency.
    """
    _channels, grid_y, grid_x, _frequency_y, _frequency_x = spectrum.shape
    horizontal_product = (
        spectrum[:, :, 1:] * np.conjugate(spectrum[:, :, :-1]))
    horizontal_cross = np.sum(horizontal_product, axis=(0, 1, 2))
    horizontal_mass = np.sum(
        np.abs(spectrum[:, :, 1:])
        * np.abs(spectrum[:, :, :-1]),
        axis=(0, 1, 2),
    )
    vertical_product = (
        spectrum[:, 1:] * np.conjugate(spectrum[:, :-1]))
    vertical_cross = np.sum(vertical_product, axis=(0, 1, 2))
    vertical_mass = np.sum(
        np.abs(spectrum[:, 1:])
        * np.abs(spectrum[:, :-1]),
        axis=(0, 1, 2),
    )
    horizontal_angle = np.angle(horizontal_cross)
    vertical_angle = np.angle(vertical_cross)
    phase_x = np.exp(
        -1j
        * np.arange(grid_x)[:, None, None]
        * horizontal_angle[None]
    )
    phase_y = np.exp(
        -1j
        * np.arange(grid_y)[:, None, None]
        * vertical_angle[None]
    )
    phase = (
        phase_y[:, None]
        * phase_x[None]
    )
    transport = {
        "horizontal_phase_coherence": float(
            np.sum(np.abs(horizontal_cross))
            / max(float(np.sum(horizontal_mass)), 1e-15)
        ),
        "vertical_phase_coherence": float(
            np.sum(np.abs(vertical_cross))
            / max(float(np.sum(vertical_mass)), 1e-15)
        ),
    }
    return phase, transport


def _neighbor_statistics(
    demodulated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint-channel aligned neighbor mean and phase consistency."""
    # C, Gy, Gx, Fy, Fx
    total = demodulated.copy()
    count = np.ones(demodulated.shape[1:], dtype=np.float64)
    magnitude = np.sqrt(
        np.sum(np.square(np.abs(demodulated)), axis=0))
    total_magnitude = magnitude.copy()
    neighbor_magnitude = np.zeros_like(magnitude)

    total[:, 1:] += demodulated[:, :-1]
    count[1:] += 1.0
    total_magnitude[1:] += magnitude[:-1]
    neighbor_magnitude[1:] += magnitude[:-1]

    total[:, :-1] += demodulated[:, 1:]
    count[:-1] += 1.0
    total_magnitude[:-1] += magnitude[1:]
    neighbor_magnitude[:-1] += magnitude[1:]

    total[:, :, 1:] += demodulated[:, :, :-1]
    count[:, 1:] += 1.0
    total_magnitude[:, 1:] += magnitude[:, :-1]
    neighbor_magnitude[:, 1:] += magnitude[:, :-1]

    total[:, :, :-1] += demodulated[:, :, 1:]
    count[:, :-1] += 1.0
    total_magnitude[:, :-1] += magnitude[:, 1:]
    neighbor_magnitude[:, :-1] += magnitude[:, 1:]

    mean = total / count[None]
    coherent_magnitude = np.sqrt(
        np.sum(np.square(np.abs(total)), axis=0))
    phase_consistency = coherent_magnitude / np.maximum(
        total_magnitude, 1e-15)
    neighbor_support = np.minimum(
        1.0,
        neighbor_magnitude / np.maximum(magnitude, 1e-15),
    )
    consistency = np.clip(
        phase_consistency * neighbor_support, 0.0, 1.0)
    return mean, consistency, magnitude


def harmonic_grid_relax(
    texture: np.ndarray,
    *,
    window_size: int = 16,
    hop: int = 8,
    diffusion: float = 1.0,
    consistency_power: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """One phase-aware diffusion step in the texture's harmonic support."""
    fields = np.ascontiguousarray(texture, dtype=np.float64)
    spectrum, layout = _analysis(fields, window_size, hop)
    grid_y, grid_x = spectrum.shape[1:3]
    phase, phase_transport = _demodulation_phase(spectrum)
    demodulated = spectrum * phase[None]
    neighbor_mean, consistency, magnitude = _neighbor_statistics(
        demodulated)
    # `neighbor_mean` contains the centre and as many as four neighbors.
    # Consequently diffusion=1 is only a 1/5 graph-Laplacian step.  Values
    # through 5 expose the remaining damping without another iteration or
    # another scan; 5 replaces an interior coefficient by its four-neighbor
    # mean.  This remains a convex, non-overshooting update.
    relaxation = (
        float(np.clip(diffusion, 0.0, 5.0))
        * np.power(
            1.0 - consistency,
            max(float(consistency_power), 1e-12),
        )
    )
    # Texture DC is not an oscillatory artifact. The graph diffusion's
    # ground is the demodulated support field, not image-frequency zero.
    relaxation[..., 0, 0] = 0.0
    relaxed_demodulated = (
        demodulated
        + relaxation[None] * (neighbor_mean - demodulated)
    )
    relaxed_spectrum = relaxed_demodulated * np.conjugate(phase)[None]
    relaxed = _synthesis(relaxed_spectrum, layout)
    removed = fields - relaxed
    input_energy = float(np.sum(np.square(fields)))
    removed_energy = float(np.sum(np.square(removed)))
    weighted_consistency = float(
        np.sum(consistency * magnitude)
        / max(float(np.sum(magnitude)), 1e-15)
    )
    record = {
        "window_size": int(window_size),
        "hop": int(hop),
        "support_grid": [int(grid_y), int(grid_x)],
        "diffusion": float(diffusion),
        "interior_neighbor_fraction": float(
            np.clip(diffusion, 0.0, 5.0) / 5.0),
        "consistency_power": float(consistency_power),
        "magnitude_weighted_consistency": weighted_consistency,
        "texture_energy": input_energy,
        "removed_energy": removed_energy,
        "removed_energy_fraction": removed_energy / max(input_energy, 1e-30),
        "consistency_p10": float(np.percentile(consistency, 10.0)),
        "consistency_p50": float(np.percentile(consistency, 50.0)),
        "consistency_p90": float(np.percentile(consistency, 90.0)),
        **phase_transport,
    }
    return relaxed, record


def _gball_project(
    texture: np.ndarray,
    *,
    mu: float,
    sweeps: int,
    threads: int,
) -> tuple[np.ndarray, dict]:
    if sweeps <= 0:
        return texture.copy(), {
            "enabled": False,
            "relative_projection_motion": 0.0,
        }
    output = np.empty_like(texture)
    c = 1.0 / float(mu)
    for channel in range(texture.shape[0]):
        survivor = bfft.rof(
            texture[channel],
            c=c,
            eta=10.0 * c,
            sweeps=int(sweeps),
            tol=0.0,
            threads=int(threads),
            solver=1,
        )
        output[channel] = texture[channel] - survivor
    motion = output - texture
    return output, {
        "enabled": True,
        "sweeps": int(sweeps),
        "relative_projection_motion": float(
            np.linalg.norm(motion)
            / max(float(np.linalg.norm(texture)), 1e-30)
        ),
    }


def process_rgb(
    rgb: np.ndarray,
    *,
    lam: float,
    mu: float,
    meyer_iters: int,
    window_size: int,
    hop: int,
    diffusion: float,
    consistency_power: float,
    gball_sweeps: int,
    threads: int,
) -> dict:
    source = np.moveaxis(
        np.asarray(rgb, dtype=np.float64), -1, 0) * 255.0
    phase = time.perf_counter()
    cartoon, texture = meyer_split_batch(
        source,
        lam=float(lam),
        mu=float(mu),
        passes=int(meyer_iters),
        threads=int(threads),
        solver=1,
    )
    meyer_ms = 1000.0 * (time.perf_counter() - phase)
    residual = source - cartoon - texture
    phase = time.perf_counter()
    relaxed, harmonic = harmonic_grid_relax(
        texture,
        window_size=int(window_size),
        hop=int(hop),
        diffusion=float(diffusion),
        consistency_power=float(consistency_power),
    )
    harmonic_ms = 1000.0 * (time.perf_counter() - phase)
    phase = time.perf_counter()
    projected, projection = _gball_project(
        relaxed,
        mu=float(mu),
        sweeps=int(gball_sweeps),
        threads=int(threads),
    )
    projection_ms = 1000.0 * (time.perf_counter() - phase)
    pre_projection = cartoon + relaxed + residual
    reconstruction = cartoon + projected + residual
    exact_source = cartoon + texture + residual
    return {
        "source": np.moveaxis(exact_source, 0, -1) / 255.0,
        "cartoon": np.moveaxis(cartoon, 0, -1) / 255.0,
        "texture": np.moveaxis(texture, 0, -1) / 255.0,
        "relaxed_texture": np.moveaxis(relaxed, 0, -1) / 255.0,
        "projected_texture": np.moveaxis(projected, 0, -1) / 255.0,
        "pre_projection": np.clip(
            np.moveaxis(pre_projection, 0, -1) / 255.0, 0.0, 1.0),
        "reconstruction": np.clip(
            np.moveaxis(reconstruction, 0, -1) / 255.0, 0.0, 1.0),
        "cartoon_conservation_max_error": float(
            np.max(np.abs(
                (pre_projection - relaxed - residual) - cartoon))),
        "residual_conservation_max_error": float(
            np.max(np.abs(
                (pre_projection - cartoon - relaxed) - residual))),
        "harmonic": harmonic,
        "gball_projection": projection,
        "timing": {
            "meyer_ms": meyer_ms,
            "harmonic_ms": harmonic_ms,
            "gball_projection_ms": projection_ms,
        },
    }


def _analytic_scene(shape=(256, 512)) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    background = 0.62 + 0.12 * x / width + 0.05 * y / height
    image = np.repeat(background[..., None], 3, axis=2)
    # A genuine off-bin harmonic that crosses the entire support grid.
    carrier = 0.055 * np.sin(
        2.0 * np.pi * (x / 7.3 + y / 19.0) + 0.4)
    image += carrier[..., None] * np.array([1.0, 0.75, 0.45])
    circle = (x - 150) ** 2 + (y - 122) ** 2 <= 58 ** 2
    image[circle] = np.array([0.84, 0.30, 0.18])
    rectangle = (x >= 305) & (x < 447) & (y >= 82) & (y < 205)
    image[rectangle] = np.array([0.16, 0.22, 0.30])
    image[96:108, 332:421] = 0.92
    image[126:138, 332:405] = 0.92
    image[156:168, 332:429] = 0.92
    return np.clip(image, 0.0, 1.0)


def _analytic_carrier_diagnostic(shape=(256, 512)) -> dict:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    phase = 2.0 * np.pi * (x / 7.3 + y / 19.0) + 0.4
    circle = (x - 150) ** 2 + (y - 122) ** 2 <= 66 ** 2
    rectangle = (x >= 297) & (x < 455) & (y >= 74) & (y < 213)
    mask = ~(circle | rectangle)
    mask[:16] = False
    mask[-16:] = False
    mask[:, :16] = False
    mask[:, -16:] = False
    return {"phase": phase, "mask": mask}


def _carrier_amplitude(rgb: np.ndarray, diagnostic: dict) -> float:
    value = np.asarray(rgb, dtype=np.float64)
    luminance = (
        0.299 * value[..., 0]
        + 0.587 * value[..., 1]
        + 0.114 * value[..., 2]
    )
    phase = np.asarray(diagnostic["phase"], dtype=np.float64)
    mask = np.asarray(diagnostic["mask"], dtype=bool)
    height, width = luminance.shape
    y, x = np.mgrid[:height, :width]
    design = np.stack((
        np.ones_like(luminance),
        x / max(width - 1, 1),
        y / max(height - 1, 1),
        np.sin(phase),
        np.cos(phase),
    ), axis=-1)[mask]
    coefficient, *_ = np.linalg.lstsq(
        design, luminance[mask], rcond=None)
    return float(math.hypot(coefficient[-2], coefficient[-1]))


def _jpeg_roundtrip(
    clean: np.ndarray,
    path: Path,
    quality: int,
) -> np.ndarray:
    pixels = np.clip(np.rint(clean * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, "RGB").save(
        path,
        "JPEG",
        quality=int(quality),
        subsampling=2,
        progressive=True,
        optimize=True,
    )
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _save_field(field: np.ndarray, path: Path, signed=False) -> None:
    value = np.asarray(field, dtype=np.float64)
    if signed:
        scale = max(float(np.percentile(np.abs(value), 99.5)), 1e-12)
        value = 0.5 + 0.48 * value / scale
    Image.fromarray(
        np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8), "RGB"
    ).save(path)


def _run_one(
    rgb: np.ndarray,
    output: Path,
    stem: str,
    args: argparse.Namespace,
    truth: np.ndarray | None = None,
    carrier_diagnostic: dict | None = None,
) -> dict:
    result = process_rgb(
        rgb,
        lam=args.lam,
        mu=args.mu,
        meyer_iters=args.meyer_iters,
        window_size=args.window_size,
        hop=args.hop,
        diffusion=args.diffusion,
        consistency_power=args.consistency_power,
        gball_sweeps=args.gball_sweeps,
        threads=args.threads,
    )
    reconstruction = result["reconstruction"]
    pre_projection = result["pre_projection"]
    _save_field(reconstruction, output / f"{stem}_relaxed.png")
    _save_field(
        result["texture"], output / f"{stem}_meyer_texture.png", signed=True)
    _save_field(
        result["texture"] - result["projected_texture"],
        output / f"{stem}_removed_texture.png",
        signed=True,
    )
    reference = rgb if truth is None else truth
    reference_u8 = np.clip(
        np.rint(reference * 255.0), 0, 255).astype(np.uint8)
    source_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    pre_u8 = np.clip(
        np.rint(pre_projection * 255.0), 0, 255).astype(np.uint8)
    output_u8 = np.clip(
        np.rint(reconstruction * 255.0), 0, 255).astype(np.uint8)
    record = {
        "harmonic": result["harmonic"],
        "gball_projection": result["gball_projection"],
        "timing": result["timing"],
        "cartoon_conservation_max_error": (
            result["cartoon_conservation_max_error"]),
        "residual_conservation_max_error": (
            result["residual_conservation_max_error"]),
        "input_metrics": _image_metrics(reference_u8, source_u8),
        "pre_projection_metrics": _image_metrics(reference_u8, pre_u8),
        "output_metrics": _image_metrics(reference_u8, output_u8),
    }
    if carrier_diagnostic is not None:
        truth_amplitude = _carrier_amplitude(reference, carrier_diagnostic)
        input_amplitude = _carrier_amplitude(rgb, carrier_diagnostic)
        output_amplitude = _carrier_amplitude(
            reconstruction, carrier_diagnostic)
        record["carrier_amplitude"] = {
            "truth": truth_amplitude,
            "jpeg_input": input_amplitude,
            "relaxed_output": output_amplitude,
            "input_relative_error": abs(
                input_amplitude - truth_amplitude
            ) / max(truth_amplitude, 1e-15),
            "output_relative_error": abs(
                output_amplitude - truth_amplitude
            ) / max(truth_amplitude, 1e-15),
        }
    images = [("input", source_u8)]
    if truth is not None:
        images.insert(0, ("clean truth", reference_u8))
    images.append(("harmonic relaxation", pre_u8))
    if result["gball_projection"]["enabled"]:
        images.append(("G-ball projected relaxation", output_u8))
    _save_montage(images, output / f"{stem}_comparison.png")
    return record


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source).expanduser().resolve()
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    supplied = _run_one(source, output, "supplied", args)

    clean = _analytic_scene()
    carrier_diagnostic = _analytic_carrier_diagnostic(clean.shape[:2])
    _save_field(clean, output / "synthetic_clean.png")
    compressed = _jpeg_roundtrip(
        clean, output / "synthetic_compressed.jpg", args.jpeg_quality)
    synthetic = _run_one(
        compressed,
        output,
        "synthetic",
        args,
        truth=clean,
        carrier_diagnostic=carrier_diagnostic,
    )
    report = {
        "source": str(source_path),
        "parameters": {
            "lam": args.lam,
            "mu": args.mu,
            "meyer_iters": args.meyer_iters,
            "window_size": args.window_size,
            "hop": args.hop,
            "diffusion": args.diffusion,
            "consistency_power": args.consistency_power,
            "gball_sweeps": args.gball_sweeps,
            "jpeg_quality": args.jpeg_quality,
        },
        "supplied": supplied,
        "synthetic": synthetic,
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default="/Users/quentinkuttenkuler/Downloads/1500x500.jpeg",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments/out/meyer_harmonic_gball"),
    )
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--meyer-iters", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--hop", type=int, default=8)
    parser.add_argument(
        "--diffusion",
        type=float,
        default=1.5,
        help=(
            "single-pass graph damping in [0, 5]; an interior harmonic "
            "moves diffusion/5 of the way to its four-neighbor mean"
        ),
    )
    parser.add_argument("--consistency-power", type=float, default=1.0)
    parser.add_argument("--gball-sweeps", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=28)
    arguments = parser.parse_args()
    if arguments.window_size < 4 or arguments.window_size % 2:
        parser.error("--window-size must be an even integer >= 4")
    if arguments.hop < 1 or arguments.window_size % arguments.hop:
        parser.error("--hop must evenly divide --window-size")
    if not 0.0 <= arguments.diffusion <= 5.0:
        parser.error("--diffusion must be in [0, 5]")
    return arguments


def main() -> int:
    print(json.dumps(run(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
