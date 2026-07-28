#!/usr/bin/env python3
"""One adversarially honest moonlight benchmark for continual fusion.

Unlike the small Cameraman falsification rig, this benchmark uses a
floating-point HDR radiance field, non-cyclic camera motion, a calibrated but
imperfect sensor, and explicit overlap support. The unregistered mean receives
the exact same circular shrinkage as the registered reconstructions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from budgeted_fullres_demo import (
    Budget,
    block_sum,
    bounded_registration,
    circular_cross_wiener,
)


@dataclass(frozen=True)
class MoonlightCamera:
    size: int = 512
    canvas_margin: int = 40
    frames: int = 1024
    fps: float = 120.0
    scene_illuminance_lux: float = 0.05
    photons_at_white: float = 0.86
    read_noise_electrons: float = 1.5
    row_noise_electrons: float = 0.20
    column_noise_electrons: float = 0.08
    dark_electrons: float = 0.008
    dsnu_electrons: float = 0.12
    prnu_fraction: float = 0.008
    hot_pixel_fraction: float = 0.0002
    hot_pixel_electrons: float = 2.5
    electrons_per_dn: float = 0.25
    adc_bits: int = 12
    black_level_dn: int = 512
    shift_radius: int = 24
    motion_step: int = 1
    registration_group: int = 32
    thumbnail: int = 128
    registration_upsample: int = 8
    seed: int = 73


def hdr_scene(config: MoonlightCamera) -> np.ndarray:
    """Construct a float HDR radiance field with sub-8-bit scene structure."""
    side = config.size + 2 * config.canvas_margin
    base = data.camera().astype(np.float64) / 255.0
    base = transform.resize(
        base, (side, side), anti_aliasing=True, preserve_range=True)
    yy, xx = np.mgrid[0:side, 0:side]
    x = xx / max(side - 1, 1)
    y = yy / max(side - 1, 1)

    # Linearized reflectance plus continuous microtexture that does not exist
    # in the 8-bit source. Illumination spans more than eight stops.
    reflectance = 0.015 + 0.82 * np.power(base, 2.2)
    microtexture = np.exp(
        0.10 * np.sin(2.0 * np.pi * (23.7 * x + 4.1 * y))
        + 0.055 * np.sin(2.0 * np.pi * (51.3 * x - 17.9 * y)))
    illumination = np.power(
        2.0,
        -1.2 - 8.5 * np.clip(0.65 * x + 0.35 * y, 0.0, 1.0),
    )
    scene = reflectance * microtexture * illumination

    # A practical night scene contains sparse high-radiance sources whose
    # halos coexist with deeply buried diffuse structure.
    rng = np.random.default_rng(config.seed + 101)
    points = np.zeros((side, side), dtype=np.float64)
    for _ in range(95):
        py = int(rng.integers(5, int(0.48 * side)))
        px = int(rng.integers(5, side - 5))
        points[py, px] += float(np.exp(rng.uniform(
            np.log(0.025), np.log(0.9))))
    scene += gaussian_filter(points, 0.65)
    lamp_y, lamp_x = int(0.28 * side), int(0.80 * side)
    radius2 = (yy - lamp_y) ** 2 + (xx - lamp_x) ** 2
    scene += 0.92 * np.exp(-radius2 / (2.0 * 1.1 ** 2))
    scene += 0.045 * np.exp(-radius2 / (2.0 * 18.0 ** 2))
    return np.clip(scene, 2.0 ** -16, 1.0).astype(np.float32)


def sensor_maps(config: MoonlightCamera) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed + 202)
    shape = (config.size, config.size)
    prnu = rng.normal(1.0, config.prnu_fraction, shape).astype(np.float32)
    dsnu = rng.normal(
        0.0, config.dsnu_electrons, shape).astype(np.float32)
    hot_rate = np.zeros(shape, dtype=np.float32)
    hot_count = int(round(config.hot_pixel_fraction * hot_rate.size))
    hot_indices = rng.choice(hot_rate.size, hot_count, replace=False)
    hot_rate.ravel()[hot_indices] = rng.exponential(
        config.hot_pixel_electrons, hot_count).astype(np.float32)
    return {"prnu": prnu, "dsnu": dsnu, "hot_rate": hot_rate}


def capture_stream(
    scene: np.ndarray,
    config: MoonlightCamera,
    maps: dict[str, np.ndarray],
):
    """Yield calibrated linear electron frames and true reference shifts."""
    rng = np.random.default_rng(config.seed)
    dy = 0
    dx = 0
    margin = config.canvas_margin
    maximum_dn = (1 << config.adc_bits) - 1
    for index in range(config.frames):
        if index:
            dy = int(np.clip(
                dy + rng.integers(-config.motion_step,
                                  config.motion_step + 1),
                -config.shift_radius, config.shift_radius))
            dx = int(np.clip(
                dx + rng.integers(-config.motion_step,
                                  config.motion_step + 1),
                -config.shift_radius, config.shift_radius))
        crop = scene[
            margin + dy:margin + dy + config.size,
            margin + dx:margin + dx + config.size,
        ]
        rate = (
            config.photons_at_white * crop * maps["prnu"]
            + config.dark_electrons
            + maps["hot_rate"]
        )
        electrons = rng.poisson(rate).astype(np.float32)
        electrons += maps["dsnu"]
        electrons += rng.normal(
            0.0, config.read_noise_electrons,
            crop.shape).astype(np.float32)
        electrons += rng.normal(
            0.0, config.row_noise_electrons,
            (config.size, 1)).astype(np.float32)
        electrons += rng.normal(
            0.0, config.column_noise_electrons,
            (1, config.size)).astype(np.float32)
        dn = np.rint(
            electrons / config.electrons_per_dn + config.black_level_dn)
        dn = np.clip(dn, 0, maximum_dn)
        calibrated = (
            dn - config.black_level_dn) * config.electrons_per_dn
        calibrated -= config.dark_electrons
        yield index, calibrated.astype(np.float32), (dy, dx)


def translate_with_mask(
    image: np.ndarray,
    shift: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Translate without cyclic wrap and return valid-output support."""
    dy, dx = shift
    height, width = image.shape
    output = np.zeros_like(image)
    mask = np.zeros_like(image, dtype=np.float32)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    if source_y1 > source_y0 and source_x1 > source_x0:
        output[target_y0:target_y1, target_x0:target_x1] = image[
            source_y0:source_y1, source_x0:source_x1]
        mask[target_y0:target_y1, target_x0:target_x1] = 1.0
    return output, mask


def normalized_pair(
    sums: list[np.ndarray],
    supports: list[np.ndarray],
    photons_at_white: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for total, support in zip(sums, supports):
        result.append(total / np.maximum(
            support * photons_at_white, 1e-8))
    return result[0], result[1]


def tone_map(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.log1p(120.0 * image) / np.log1p(120.0)


def score(
    estimate: np.ndarray,
    truth: np.ndarray,
    margin: int,
) -> dict[str, float]:
    crop = np.s_[margin:-margin, margin:-margin]
    estimate_linear = np.clip(estimate[crop], 0.0, 1.0)
    truth_linear = truth[crop]
    estimate_tone = tone_map(estimate_linear)
    truth_tone = tone_map(truth_linear)
    shadows = truth_linear < 0.025
    shadow_rmse = float(np.sqrt(np.mean(
        (estimate_linear[shadows] - truth_linear[shadows]) ** 2)))
    return {
        "linear_psnr_db": float(peak_signal_noise_ratio(
            truth_linear, estimate_linear, data_range=1.0)),
        "log_psnr_db": float(peak_signal_noise_ratio(
            truth_tone, estimate_tone, data_range=1.0)),
        "log_ssim": float(structural_similarity(
            truth_tone, estimate_tone, data_range=1.0)),
        "shadow_rmse": shadow_rmse,
    }


def run(
    config: MoonlightCamera,
) -> tuple[dict, dict[str, np.ndarray]]:
    scene = hdr_scene(config)
    margin = config.canvas_margin
    truth = scene[margin:margin + config.size, margin:margin + config.size]
    maps = sensor_maps(config)
    shape = truth.shape
    unregistered_sums = [
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    ]
    thumbnail_sum = np.zeros(
        (config.thumbnail, config.thumbnail), dtype=np.float64)
    counts = [0, 0]
    started = time.perf_counter()
    for index, frame, _ in capture_stream(scene, config, maps):
        parity = index % 2
        counts[parity] += 1
        unregistered_sums[parity] += frame
        thumbnail_sum += block_sum(frame, config.thumbnail)
    thumbnail_reference = (
        thumbnail_sum / config.frames).astype(np.float32)

    registered_sums = [
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    ]
    registered_supports = [
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    ]
    oracle_sums = [
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    ]
    oracle_supports = [
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    ]
    estimates = []
    truths = []
    group = []
    registration_config = Budget(
        size=config.size,
        frames=config.frames,
        photons_at_white=config.photons_at_white,
        shift_radius=config.shift_radius,
        registration_group=config.registration_group,
        thumbnail=config.thumbnail,
        registration_upsample=config.registration_upsample,
    )

    def consume(items):
        if not items:
            return
        pile = sum(
            (block_sum(frame, config.thumbnail)
             for _, frame, _ in items),
            np.zeros_like(thumbnail_reference),
        )
        estimated = bounded_registration(
            thumbnail_reference, pile, registration_config)
        for index, frame, truth_shift in items:
            parity = index % 2
            aligned, support = translate_with_mask(frame, estimated)
            registered_sums[parity] += aligned
            registered_supports[parity] += support
            oracle_aligned, oracle_support = translate_with_mask(
                frame, truth_shift)
            oracle_sums[parity] += oracle_aligned
            oracle_supports[parity] += oracle_support
            estimates.append(estimated)
            truths.append(truth_shift)

    for item in capture_stream(scene, config, maps):
        group.append(item)
        if len(group) == config.registration_group:
            consume(group)
            group.clear()
    consume(group)
    elapsed = time.perf_counter() - started

    unregistered = [
        unregistered_sums[p]
        / max(counts[p] * config.photons_at_white, 1e-8)
        for p in range(2)
    ]
    registered = normalized_pair(
        registered_sums, registered_supports, config.photons_at_white)
    oracle = normalized_pair(
        oracle_sums, oracle_supports, config.photons_at_white)
    raw_mean = 0.5 * (unregistered[0] + unregistered[1])
    raw_registered = 0.5 * (registered[0] + registered[1])
    unregistered_supported, _, _, unregistered_info = circular_cross_wiener(
        unregistered[0], unregistered[1], 2)
    registered_supported, _, _, registered_info = circular_cross_wiener(
        registered[0], registered[1], 2)
    oracle_supported, _, _, oracle_info = circular_cross_wiener(
        oracle[0], oracle[1], 2)

    estimates_array = np.asarray(estimates)
    truths_array = np.asarray(truths)
    difference = estimates_array - truths_array
    gauge = np.rint(np.median(difference, axis=0)).astype(np.int32)
    residual = difference - gauge
    errors = np.hypot(residual[:, 0], residual[:, 1])
    evaluation_margin = config.shift_radius + 12
    result = {
        "camera": asdict(config),
        "scene": {
            "radiance_min": float(np.min(truth)),
            "radiance_max": float(np.max(truth)),
            "radiance_dynamic_range_stops": float(np.log2(
                np.max(truth) / max(float(np.min(truth)), 1e-12))),
            "mean_signal_electrons_per_pixel_per_frame": float(
                config.photons_at_white * np.mean(truth)),
        },
        "elapsed_seconds": elapsed,
        "milliseconds_per_input_frame_two_pass": float(
            1000.0 * elapsed / config.frames),
        "registration_error_pixels": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p90": float(np.quantile(errors, 0.9)),
            "global_gauge_yx": [int(gauge[0]), int(gauge[1])],
        },
        "support": {
            "unregistered": unregistered_info,
            "registered": registered_info,
            "oracle": oracle_info,
        },
        "metrics": {
            "raw_mean": score(raw_mean, truth, evaluation_margin),
            "unregistered_supported": score(
                unregistered_supported, truth, evaluation_margin),
            "raw_registered": score(
                raw_registered, truth, evaluation_margin),
            "registered_supported": score(
                registered_supported, truth, evaluation_margin),
            "oracle_supported": score(
                oracle_supported, truth, evaluation_margin),
        },
    }
    images = {
        "truth": truth,
        "raw_mean": raw_mean,
        "unregistered_supported": unregistered_supported,
        "raw_registered": raw_registered,
        "registered_supported": registered_supported,
        "oracle_supported": oracle_supported,
    }
    return result, images


def render(
    images: dict[str, np.ndarray],
    result: dict,
    path: Path,
) -> None:
    panels = (
        ("truth", "Float HDR truth", None),
        ("raw_mean", "Raw unregistered mean", "raw_mean"),
        (
            "unregistered_supported",
            "Same filter, no registration",
            "unregistered_supported",
        ),
        ("raw_registered", "Raw bounded registration", "raw_registered"),
        (
            "registered_supported",
            "Bounded support + circular fusion",
            "registered_supported",
        ),
        ("oracle_supported", "Oracle registration ceiling", "oracle_supported"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.7))
    for axis, (key, title, metric_key) in zip(axes.ravel(), panels):
        axis.imshow(tone_map(images[key]), cmap="gray", vmin=0.0, vmax=1.0)
        if metric_key:
            metric = result["metrics"][metric_key]
            title += (
                f"\nlog {metric['log_psnr_db']:.2f} dB, "
                f"SSIM {metric['log_ssim']:.3f}")
        axis.set_title(title)
        axis.axis("off")
    camera = result["camera"]
    figure.suptitle(
        f"Physical moonlight bench: {camera['scene_illuminance_lux']:.2f} lux, "
        f"{camera['frames']} frames at {camera['fps']:.0f} fps, "
        f"{camera['read_noise_electrons']:.1f} e⁻ RMS read noise",
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.02, right=0.99, bottom=0.025, top=0.90,
        wspace=0.08, hspace=0.17)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("high_vision/out/realistic_moonlight_bench.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("high_vision/out/realistic_moonlight_bench.json"),
    )
    args = parser.parse_args()
    config = MoonlightCamera(frames=args.frames)
    result, images = run(config)
    render(images, result, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
