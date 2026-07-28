#!/usr/bin/env python3
"""Budgeted 512px low-light registration and fusion experiment.

The expensive orbit invariant is useful below registration, but it need not be
evaluated at every output frequency. This rig tests the complementary regime:
use a cheap, bounded thumbnail registration support, then spend the original
full-resolution photon counts only once in the final accumulation.

Frames are regenerated deterministically for two streaming passes, so memory is
O(image pixels), not O(frames * image pixels).
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
from skimage import data, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class Budget:
    size: int = 512
    frames: int = 512
    photons_at_white: float = 0.10
    dark_electrons: float = 0.002
    read_noise_electrons: float = 0.0
    shift_radius: int = 12
    motion_step: int = 1
    registration_group: int = 8
    registration_rounds: int = 2
    thumbnail: int = 128
    registration_upsample: int = 8
    wiener_ring_width: int = 2
    sampler: str = "auto"
    seed: int = 19


def source_image(size: int) -> np.ndarray:
    image = data.camera().astype(np.float32) / 255.0
    if image.shape != (size, size):
        image = transform.resize(
            image, (size, size), anti_aliasing=True,
            preserve_range=True).astype(np.float32)
    return np.clip(0.02 + 0.96 * image, 0.0, 1.0)


def block_sum(image: np.ndarray, thumbnail: int) -> np.ndarray:
    height, width = image.shape
    if height % thumbnail or width % thumbnail:
        return transform.resize(
            image, (thumbnail, thumbnail), anti_aliasing=True,
            preserve_range=True).astype(np.float32)
    fy = height // thumbnail
    fx = width // thumbnail
    return image.reshape(thumbnail, fy, thumbnail, fx).sum(axis=(1, 3))


def frame_stream(source: np.ndarray, config: Budget):
    rng = np.random.default_rng(config.seed)
    rate = config.photons_at_white * source
    use_events = (
        config.sampler == "events"
        or (config.sampler == "auto"
            and config.photons_at_white + config.dark_electrons < 0.25)
    )
    if config.sampler not in {"auto", "dense", "events"}:
        raise ValueError("sampler must be auto, dense, or events")
    source_mass = float(np.sum(source))
    source_cdf = np.cumsum(source.ravel(), dtype=np.float64)
    source_cdf /= source_cdf[-1]
    source_cdf[-1] = 1.0
    pixels = source.size
    width = source.shape[1]
    dy = 0
    dx = 0
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
        if use_events:
            photo_count = int(rng.poisson(
                config.photons_at_white * source_mass))
            base = np.searchsorted(
                source_cdf, rng.random(photo_count), side="right")
            event_y = (base // width + dy) % source.shape[0]
            event_x = (base % width + dx) % width
            shifted_events = event_y * width + event_x
            dark_count = int(rng.poisson(
                config.dark_electrons * pixels))
            dark_events = rng.integers(0, pixels, size=dark_count)
            frame = np.bincount(
                np.concatenate((shifted_events, dark_events)),
                minlength=pixels).reshape(source.shape).astype(np.float32)
        else:
            shifted = np.roll(rate, (dy, dx), axis=(0, 1))
            frame = rng.poisson(
                shifted + config.dark_electrons).astype(np.float32)
        if config.read_noise_electrons:
            frame += rng.normal(
                0.0, config.read_noise_electrons,
                frame.shape).astype(np.float32)
        frame -= config.dark_electrons
        yield index, frame, (dy, dx)


def scores(estimate: np.ndarray, source: np.ndarray) -> dict[str, float]:
    estimate = np.clip(estimate, 0.0, 1.0)
    return {
        "psnr_db": float(peak_signal_noise_ratio(
            source, estimate, data_range=1.0)),
        "ssim": float(structural_similarity(
            source, estimate, data_range=1.0)),
    }


def split_half_snr(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Estimate signal and noise from two independent reconstructions."""
    mean = 0.5 * (first + second)
    noise = 0.5 * (first - second)
    noise_variance = float(np.mean(noise ** 2))
    signal_variance = max(float(np.var(mean)) - noise_variance, 1e-15)
    return {
        "snr_db": float(10.0 * np.log10(
            signal_variance / max(noise_variance, 1e-15))),
        "noise_rms": float(np.sqrt(noise_variance)),
        "signal_rms": float(np.sqrt(signal_variance)),
    }


def circular_cross_wiener(
    first: np.ndarray,
    second: np.ndarray,
    ring_width: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Shrink unsupported Fourier circles using independent half-stacks.

    The half-stack cross-power estimates repeatable scene energy. Their
    half-difference estimates the noise in their mean. Pooling these estimates
    over projection circles gives enough samples for a stable, non-oracular
    Wiener gain without selecting an image direction.
    """
    if first.shape != second.shape:
        raise ValueError("split-half images must have the same shape")
    if ring_width < 1:
        raise ValueError("ring_width must be positive")
    height, width = first.shape
    first_spectrum = np.fft.fft2(first)
    second_spectrum = np.fft.fft2(second)
    mean_spectrum = 0.5 * (first_spectrum + second_spectrum)
    signal_sample = np.real(first_spectrum * np.conj(second_spectrum))
    noise_sample = 0.25 * np.abs(first_spectrum - second_spectrum) ** 2

    fy = np.fft.fftfreq(height) * height
    fx = np.fft.fftfreq(width) * width
    radius = np.hypot(fy[:, None], fx[None, :])
    rings = np.floor(radius / ring_width).astype(np.int32)
    count = np.bincount(rings.ravel())
    signal_sum = np.bincount(
        rings.ravel(), weights=signal_sample.ravel())
    noise_sum = np.bincount(
        rings.ravel(), weights=noise_sample.ravel())

    # A short symmetric dwell shares evidence between neighboring circles
    # while retaining a radial frequency response.
    dwell = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    signal_smooth = np.convolve(signal_sum, dwell, mode="same")
    noise_smooth = np.convolve(noise_sum, dwell, mode="same")
    count_smooth = np.convolve(count, dwell, mode="same")
    signal_power = np.maximum(
        signal_smooth / np.maximum(count_smooth, 1.0), 0.0)
    noise_power = np.maximum(
        noise_smooth / np.maximum(count_smooth, 1.0), 1e-15)
    gain = signal_power / (signal_power + noise_power)
    gain[0] = 1.0
    gain_map = gain[rings]

    filtered = np.fft.ifft2(mean_spectrum * gain_map).real
    filtered_first = np.fft.ifft2(first_spectrum * gain_map).real
    filtered_second = np.fft.ifft2(second_spectrum * gain_map).real
    supported = np.flatnonzero(gain >= 0.5)
    diagnostics = {
        "ring_width": int(ring_width),
        "rings": int(len(gain)),
        "median_gain": float(np.median(gain)),
        "half_gain_last_radius": float(
            ring_width * supported[-1] if len(supported) else 0.0),
    }
    return filtered, filtered_first, filtered_second, diagnostics


def bounded_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    config: Budget,
) -> tuple[int, int]:
    """Return a full-grid integer shift to apply to moving."""
    scale = config.size / config.thumbnail
    coarse_radius = config.shift_radius / scale + 1.5
    shift, _, _ = phase_cross_correlation(
        reference,
        moving,
        upsample_factor=config.registration_upsample,
        normalization=None,
    )
    shift = np.clip(shift, -coarse_radius, coarse_radius)
    return (
        int(np.rint(float(shift[0]) * scale)),
        int(np.rint(float(shift[1]) * scale)),
    )


def best_cyclic_alignment(
    estimate: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    correlation = np.fft.ifft2(
        np.fft.fft2(reference) * np.conj(np.fft.fft2(estimate))).real
    shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    return (
        np.roll(estimate, shift, axis=(0, 1)),
        (int(shift[0]), int(shift[1])),
    )


def run(config: Budget) -> tuple[dict, dict[str, np.ndarray]]:
    if config.size % config.thumbnail:
        raise ValueError("size must be divisible by thumbnail")
    source = source_image(config.size)
    unregistered_sum = np.zeros_like(source, dtype=np.float64)
    thumbnail_sum = np.zeros(
        (config.thumbnail, config.thumbnail), dtype=np.float64)
    first_frame = None

    started = time.perf_counter()
    for _, frame, _ in frame_stream(source, config):
        if first_frame is None:
            first_frame = frame.copy()
        unregistered_sum += frame
        thumbnail_sum += block_sum(frame, config.thumbnail)
    first_pass_seconds = time.perf_counter() - started

    # Because support is bounded, this blurred first-pass mean remains a valid
    # registration attractor. No frame is selected as a privileged reference.
    thumbnail_reference = (
        thumbnail_sum / config.frames).astype(np.float32)
    registration_started = time.perf_counter()
    registered_sum = None
    registered_even_sum = None
    registered_odd_sum = None
    registered_even_count = 0
    registered_odd_count = 0
    estimates = None
    truths = None
    round_summaries = []
    for round_index in range(config.registration_rounds):
        next_sum = np.zeros_like(source, dtype=np.float64)
        next_even_sum = np.zeros_like(source, dtype=np.float64)
        next_odd_sum = np.zeros_like(source, dtype=np.float64)
        next_even_count = 0
        next_odd_count = 0
        next_estimates = []
        next_truths = []
        group = []

        def consume_group(frames):
            nonlocal next_even_count, next_odd_count
            if not frames:
                return
            thumbnail = sum(
                (block_sum(frame, config.thumbnail)
                 for _, frame, _ in frames),
                np.zeros_like(thumbnail_reference))
            estimated = bounded_registration(
                thumbnail_reference, thumbnail, config)
            for index, frame, (dy, dx) in frames:
                truth = (-dy, -dx)
                aligned = np.roll(frame, estimated, axis=(0, 1))
                next_sum[:] += aligned
                if index % 2:
                    next_odd_sum[:] += aligned
                    next_odd_count += 1
                else:
                    next_even_sum[:] += aligned
                    next_even_count += 1
                next_estimates.append(estimated)
                next_truths.append(truth)

        for item in frame_stream(source, config):
            group.append(item)
            if len(group) == config.registration_group:
                consume_group(group)
                group.clear()
        consume_group(group)
        registered_sum = next_sum
        registered_even_sum = next_even_sum
        registered_odd_sum = next_odd_sum
        registered_even_count = next_even_count
        registered_odd_count = next_odd_count
        estimates = np.asarray(next_estimates)
        truths = np.asarray(next_truths)
        # The next round remarches only the small registration state. Its
        # attractor is the preceding full-resolution fusion projected once.
        thumbnail_reference = block_sum(
            (registered_sum / config.frames).astype(np.float32),
            config.thumbnail)
        difference = estimates - truths
        gauge = np.rint(np.median(difference, axis=0)).astype(np.int32)
        residual = difference - gauge
        round_errors = np.hypot(residual[:, 0], residual[:, 1])
        round_summaries.append({
            "round": round_index + 1,
            "gauge_yx": [int(gauge[0]), int(gauge[1])],
            "median_error": float(np.median(round_errors)),
            "p90_error": float(np.quantile(round_errors, 0.9)),
        })
    second_pass_seconds = time.perf_counter() - registration_started

    divisor = max(config.frames * config.photons_at_white, 1e-12)
    unregistered = np.clip(unregistered_sum / divisor, 0.0, 1.0)
    registered = np.clip(registered_sum / divisor, 0.0, 1.0)
    registered_even = registered_even_sum / max(
        registered_even_count * config.photons_at_white, 1e-12)
    registered_odd = registered_odd_sum / max(
        registered_odd_count * config.photons_at_white, 1e-12)
    split_before = split_half_snr(registered_even, registered_odd)
    (
        registered_wiener,
        registered_even_wiener,
        registered_odd_wiener,
        wiener_diagnostics,
    ) = circular_cross_wiener(
        registered_even,
        registered_odd,
        config.wiener_ring_width,
    )
    split_after = split_half_snr(
        registered_even_wiener, registered_odd_wiener)
    unregistered, unregistered_gauge = best_cyclic_alignment(
        unregistered, source)
    registered, registered_gauge = best_cyclic_alignment(registered, source)
    registered_wiener, wiener_gauge = best_cyclic_alignment(
        registered_wiener, source)
    registered_wiener = np.clip(registered_wiener, 0.0, 1.0)
    single = np.clip(first_frame / max(config.photons_at_white, 1e-12),
                     0.0, 1.0)
    difference = estimates - truths
    registration_gauge = np.rint(
        np.median(difference, axis=0)).astype(np.int32)
    residual = difference - registration_gauge
    errors = np.hypot(residual[:, 0], residual[:, 1])
    result = {
        "budget": asdict(config),
        "mean_detected_electrons_per_frame": float(
            np.sum(config.photons_at_white * source
                   + config.dark_electrons)),
        "nonzero_fraction_first_frame": float(np.mean(first_frame > 0.0)),
        "first_pass_seconds": first_pass_seconds,
        "registration_and_fusion_seconds": second_pass_seconds,
        "total_seconds": first_pass_seconds + second_pass_seconds,
        "milliseconds_per_input_frame_all_passes": float(
            1000.0 * (first_pass_seconds + second_pass_seconds)
            / config.frames),
        "streaming_passes": 1 + config.registration_rounds,
        "registration_rounds": round_summaries,
        "registration_error_pixels": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p90": float(np.quantile(errors, 0.9)),
            "exact_fraction": float(np.mean(errors == 0.0)),
            "global_gauge_yx": [
                int(registration_gauge[0]),
                int(registration_gauge[1]),
            ],
        },
        "evaluation_global_gauge": {
            "unregistered": list(unregistered_gauge),
            "registered": list(registered_gauge),
            "cross_wiener": list(wiener_gauge),
        },
        "split_half_support": {
            "before": split_before,
            "after": split_after,
            "noise_reduction_db": float(
                20.0 * np.log10(
                    max(split_before["noise_rms"], 1e-15)
                    / max(split_after["noise_rms"], 1e-15))),
            **wiener_diagnostics,
        },
        "metrics": {
            "single": scores(single, source),
            "unregistered": scores(unregistered, source),
            "bounded_registered": scores(registered, source),
            "circular_cross_wiener": scores(
                registered_wiener, source),
        },
    }
    images = {
        "source": source,
        "single": single,
        "unregistered": unregistered,
        "registered": registered,
        "registered_wiener": registered_wiener,
        "registered_even": registered_even,
        "registered_odd": registered_odd,
        "thumbnail_reference": thumbnail_reference,
        "errors": errors,
        "estimates": np.asarray(estimates),
        "estimates_gauge_corrected": (
            np.asarray(estimates) - registration_gauge),
        "truths": np.asarray(truths),
    }
    return result, images


def render(images: dict[str, np.ndarray], result: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.7))
    panels = (
        ("source", "Ground truth", None),
        ("single", "One photon frame", "single"),
        ("unregistered", "Unregistered mean", "unregistered"),
        ("registered", "Bounded registered fusion", "bounded_registered"),
        (
            "registered_wiener",
            "Cross-supported circular shrinkage",
            "circular_cross_wiener",
        ),
    )
    for axis, (key, title, metric) in zip(axes.ravel()[:5], panels):
        axis.imshow(images[key], cmap="gray", vmin=0.0,
                    vmax=(None if key == "thumbnail_reference" else 1.0))
        if metric:
            score = result["metrics"][metric]
            title += f"\n{score['psnr_db']:.2f} dB, SSIM {score['ssim']:.3f}"
        axis.set_title(title)
        axis.axis("off")
    axis = axes.ravel()[5]
    maximum = max(result["budget"]["shift_radius"], 1)
    axis.scatter(
        images["truths"][:, 1], images["truths"][:, 0],
        s=12, alpha=0.55, label="true alignment")
    axis.scatter(
        images["estimates_gauge_corrected"][:, 1],
        images["estimates_gauge_corrected"][:, 0],
        s=10, alpha=0.55, label="estimated")
    axis.set_xlim(-maximum - 1, maximum + 1)
    axis.set_ylim(maximum + 1, -maximum - 1)
    axis.set_aspect("equal")
    axis.set_title(
        "Registration support\n"
        f"median error {result['registration_error_pixels']['median']:.2f}px; "
        f"noise −{result['split_half_support']['noise_reduction_db']:.1f} dB")
    axis.set_xlabel("x shift")
    axis.set_ylabel("y shift")
    axis.legend(fontsize=8)
    budget = result["budget"]
    figure.suptitle(
        f"{budget['size']}², {budget['frames']} frames, "
        f"{budget['photons_at_white']:.3g} e⁻/white pixel, "
        f"bounded ±{budget['shift_radius']}px",
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.025, right=0.985, bottom=0.04, top=0.89,
        wspace=0.12, hspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--photons", type=float, default=0.10)
    parser.add_argument("--dark", type=float, default=0.002)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--shift-radius", type=int, default=12)
    parser.add_argument("--motion-step", type=int, default=1)
    parser.add_argument("--registration-group", type=int, default=8)
    parser.add_argument("--registration-rounds", type=int, default=2)
    parser.add_argument("--thumbnail", type=int, default=128)
    parser.add_argument("--upsample", type=int, default=8)
    parser.add_argument("--wiener-ring-width", type=int, default=2)
    parser.add_argument(
        "--sampler", choices=("auto", "dense", "events"), default="auto")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--output", type=Path,
        default=Path("high_vision/out/budgeted_fullres_demo.png"))
    parser.add_argument(
        "--json", type=Path,
        default=Path("high_vision/out/budgeted_fullres_demo.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = Budget(
        size=args.size,
        frames=args.frames,
        photons_at_white=args.photons,
        dark_electrons=args.dark,
        read_noise_electrons=args.read_noise,
        shift_radius=args.shift_radius,
        motion_step=args.motion_step,
        registration_group=args.registration_group,
        registration_rounds=args.registration_rounds,
        thumbnail=args.thumbnail,
        registration_upsample=args.upsample,
        wiener_ring_width=args.wiener_ring_width,
        sampler=args.sampler,
        seed=args.seed,
    )
    result, images = run(config)
    render(images, result, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
