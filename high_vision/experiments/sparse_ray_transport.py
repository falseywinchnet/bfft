#!/usr/bin/env python3
"""Sparse-registration recovery as tempered photon-ray transport.

Every detected photon at camera pixel ``x`` and frame ``i`` is compatible
with every latent location ``x-s`` in the admissible camera-motion orbit.
For a current radiance belief ``u``, the exact cyclic Poisson log likelihood
of shift ``s`` is

    L_i(s) = sum_x counts_i(x) log(photons * u(x-s) + background)

up to terms independent of ``s``.  A tempered posterior over the complete
shift orbit keeps every frame diffuse when it is individually unregistrable.
The M-step backprojects the actual photon counts through that distribution.

This experiment compares:

* a strictly one-pass continual ray integrator;
* replayed soft and nearly-hard Poisson EM controls;
* injected posterior noise;
* autocorrelation-only positive phase retrieval;
* persistent scene speckle versus independent photon background.

The last pair asks the precise "noise is our friend" question.  Stable
scene texture can act as a registration code.  Independent temporal noise
cannot; at best it is averaged away.  Posterior temperature is not sensor
noise: it is retained epistemic uncertainty over the motion manifold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from skimage import data, transform
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


@dataclass(frozen=True)
class RayBench:
    grid: int = 64
    frames: int = 1024
    validation_frames: int = 256
    photons_at_white: float = 0.08
    background_photons: float = 0.0
    shift_radius: int = 8
    batch: int = 32
    support_cap: int = 256
    temperature: float = 4.0
    replay_sweeps: int = 8
    seed: int = 149


def source_image(
    grid: int,
    *,
    speckle: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    image = transform.resize(
        data.camera().astype(np.float64) / 255.0,
        (grid, grid),
        anti_aliasing=True,
        preserve_range=True,
    )
    if speckle > 0.0:
        rng = np.random.default_rng(seed)
        texture = gaussian_filter(
            rng.standard_normal((grid, grid)), 0.55)
        texture /= max(float(np.std(texture)), 1e-12)
        image = np.clip(image + speckle * texture, 0.0, 1.0)
    return image


def shift_mask(grid: int, radius: int) -> np.ndarray:
    result = np.zeros((grid, grid), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                result[dy % grid, dx % grid] = True
    return result


def capture(
    source: np.ndarray,
    frames: int,
    photons_at_white: float,
    background_photons: float,
    radius: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    candidates = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]
    selected = rng.integers(0, len(candidates), size=frames)
    shifts = np.asarray(
        [candidates[index] for index in selected], dtype=np.int16)
    counts = np.empty(
        (frames,) + source.shape, dtype=np.uint16)
    for index, (dy, dx) in enumerate(shifts):
        rate = (
            photons_at_white
            * np.roll(source, (int(dy), int(dx)), axis=(0, 1))
            + background_photons
        )
        counts[index] = rng.poisson(rate)
    return counts, shifts


def calibrated_radiance(
    counts: np.ndarray,
    photons_at_white: float,
    background_photons: float,
) -> np.ndarray:
    return (
        np.asarray(counts, dtype=np.float64) - background_photons
    ) / max(photons_at_white, 1e-12)


def best_cyclic_alignment(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    correlation = np.fft.ifft2(
        np.fft.fft2(reference)
        * np.conj(np.fft.fft2(estimate))
    ).real
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    aligned = np.roll(estimate, peak, axis=(0, 1))
    height, width = estimate.shape
    shift = (
        int(peak[0] if peak[0] <= height // 2 else peak[0] - height),
        int(peak[1] if peak[1] <= width // 2 else peak[1] - width),
    )
    return aligned, shift


def metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    aligned, _ = best_cyclic_alignment(estimate, reference)
    aligned = np.clip(aligned, 0.0, 1.0)
    return {
        "psnr_db": float(peak_signal_noise_ratio(
            reference, aligned, data_range=1.0)),
        "ssim": float(structural_similarity(
            reference, aligned, data_range=1.0)),
    }


def shift_posterior(
    counts: np.ndarray,
    belief: np.ndarray,
    config: RayBench,
    admitted: np.ndarray,
    *,
    temperature: float | None = None,
    logit_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact cyclic Poisson shift posterior over every admitted ray."""
    rate = np.maximum(
        config.photons_at_white * belief
        + config.background_photons,
        1e-7,
    )
    log_rate_spectrum = np.fft.fft2(np.log(rate))
    count_spectrum = np.fft.fft2(
        np.asarray(counts, dtype=np.float64), axes=(-2, -1))
    score = np.fft.ifft2(
        count_spectrum * np.conj(log_rate_spectrum),
        axes=(-2, -1),
    ).real
    logits = score[:, admitted]
    if logit_noise > 0.0:
        if rng is None:
            raise ValueError("logit noise requires a random generator")
        logits = logits + logit_noise * rng.gumbel(size=logits.shape)
    used_temperature = (
        config.temperature if temperature is None else temperature)
    logits = (
        logits - np.max(logits, axis=1, keepdims=True)
    ) / max(float(used_temperature), 1e-8)
    probability = np.exp(np.clip(logits, -60.0, 0.0))
    probability /= np.maximum(
        np.sum(probability, axis=1, keepdims=True), 1e-30)
    posterior = np.zeros_like(score)
    posterior[:, admitted] = probability
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-30)),
        axis=1,
    )
    return posterior, {
        "mean_entropy_nats": float(np.mean(entropy)),
        "effective_shifts": float(np.exp(np.mean(entropy))),
        "mean_peak_probability": float(np.mean(
            np.max(probability, axis=1))),
    }


def backproject(
    counts: np.ndarray,
    posterior: np.ndarray,
    config: RayBench,
) -> np.ndarray:
    radiance = calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    )
    observation = np.fft.fft2(radiance, axes=(-2, -1))
    # A delta posterior at observed shift s has DFT exp(-ik.s);
    # conjugation applies the inverse shift to the observation.
    characteristic = np.conj(np.fft.fft2(
        posterior, axes=(-2, -1)))
    estimate = np.fft.ifft2(
        np.mean(observation * characteristic, axis=0)).real
    return np.maximum(estimate, 1e-7)


def continual_ray_transport(
    counts: np.ndarray,
    config: RayBench,
    *,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Consume the photon stream once; no frame or posterior is replayed."""
    admitted = shift_mask(config.grid, config.shift_radius)
    initial_count = min(config.batch, len(counts))
    belief = gaussian_filter(
        np.maximum(np.mean(calibrated_radiance(
            counts[:initial_count],
            config.photons_at_white,
            config.background_photons,
        ), axis=0), 1e-7),
        1.0,
    )
    support = initial_count
    trace = []
    for start in range(initial_count, len(counts), config.batch):
        batch = counts[start:start + config.batch]
        posterior, posterior_info = shift_posterior(
            batch, belief, config, admitted)
        estimate = backproject(batch, posterior, config)
        old_support = min(support, config.support_cap)
        belief = (
            old_support * belief + len(batch) * estimate
        ) / max(old_support + len(batch), 1)
        support += len(batch)
        record = {
            "frames": support,
            **posterior_info,
        }
        if reference is not None:
            record.update(metrics(belief, reference))
        trace.append(record)
    return belief, {
        "mode": "strictly_continual_tempered_ray_transport",
        "passes_over_frames": 1,
        "trace": trace,
    }


def replayed_ray_transport(
    counts: np.ndarray,
    config: RayBench,
    *,
    temperature: float,
    sweeps: int,
    logit_noise: float = 0.0,
    seed: int = 0,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    admitted = shift_mask(config.grid, config.shift_radius)
    belief = gaussian_filter(
        np.maximum(np.mean(calibrated_radiance(
            counts,
            config.photons_at_white,
            config.background_photons,
        ), axis=0), 1e-7),
        1.0,
    )
    rng = np.random.default_rng(seed)
    trace = []
    for sweep in range(sweeps):
        total = np.zeros_like(belief)
        total_frames = 0
        posterior_summary = []
        for start in range(0, len(counts), config.batch):
            batch = counts[start:start + config.batch]
            posterior, info = shift_posterior(
                batch,
                belief,
                config,
                admitted,
                temperature=temperature,
                logit_noise=logit_noise,
                rng=rng,
            )
            total += len(batch) * backproject(batch, posterior, config)
            total_frames += len(batch)
            posterior_summary.append(info)
        candidate = total / max(total_frames, 1)
        belief = np.maximum(
            0.30 * belief + 0.70 * candidate, 1e-7)
        record = {
            "sweep": sweep + 1,
            "mean_entropy_nats": float(np.mean([
                item["mean_entropy_nats"]
                for item in posterior_summary
            ])),
            "effective_shifts": float(np.mean([
                item["effective_shifts"]
                for item in posterior_summary
            ])),
        }
        if reference is not None:
            record.update(metrics(belief, reference))
        trace.append(record)
    return belief, {
        "mode": "replayed_tempered_ray_transport",
        "passes_over_frames": sweeps,
        "temperature": temperature,
        "logit_noise": logit_noise,
        "trace": trace,
    }


def marginal_poisson_score(
    counts: np.ndarray,
    belief: np.ndarray,
    config: RayBench,
) -> float:
    """Held-out log evidence, omitting only count-factorial constants."""
    admitted = shift_mask(config.grid, config.shift_radius)
    rate = np.maximum(
        config.photons_at_white * belief
        + config.background_photons,
        1e-7,
    )
    score = np.fft.ifft2(
        np.fft.fft2(
            np.asarray(counts, dtype=np.float64),
            axes=(-2, -1),
        )
        * np.conj(np.fft.fft2(np.log(rate))),
        axes=(-2, -1),
    ).real[:, admitted]
    evidence = (
        logsumexp(score, axis=1)
        - np.log(score.shape[1])
        - float(np.sum(rate))
    )
    return float(np.mean(evidence) / belief.size)


def shift_diagnostic(
    counts: np.ndarray,
    belief: np.ndarray,
    true_shifts: np.ndarray,
    config: RayBench,
) -> dict[str, float | list[int]]:
    """Truth-only audit of the final diffuse posterior's MAP gauge."""
    posterior, info = shift_posterior(
        counts,
        belief,
        config,
        shift_mask(config.grid, config.shift_radius),
    )
    peak = np.asarray(np.unravel_index(
        np.argmax(posterior.reshape(len(counts), -1), axis=1),
        (config.grid, config.grid),
    )).T
    peak = np.where(
        peak <= config.grid // 2, peak, peak - config.grid)
    difference = peak - np.asarray(true_shifts)
    gauge = np.rint(np.median(difference, axis=0)).astype(np.int32)
    residual = difference - gauge
    error = np.hypot(residual[:, 0], residual[:, 1])
    return {
        **info,
        "global_gauge_yx": [int(gauge[0]), int(gauge[1])],
        "median_shift_error": float(np.median(error)),
        "p90_shift_error": float(np.quantile(error, 0.9)),
        "mean_shift_error": float(np.mean(error)),
    }


def autocorrelation_phase_retrieval(
    counts: np.ndarray,
    config: RayBench,
    *,
    lanes: int = 4,
    sweeps: int = 200,
) -> tuple[np.ndarray, dict]:
    """Positive phase retrieval from the debiased second-order invariant."""
    radiance = calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    )
    spectrum = np.fft.fft2(radiance, axes=(-2, -1))
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    noise_floor = (
        config.grid ** 2
        * float(np.mean(counts))
        / max(config.photons_at_white ** 2, 1e-20)
    )
    magnitude = np.sqrt(np.maximum(power - noise_floor, 0.0))
    magnitude[0, 0] = abs(np.mean(spectrum[:, 0, 0]))
    rng = np.random.default_rng(config.seed + 700)
    reconstructions = []
    lane_metrics = []
    for _ in range(lanes):
        phase = rng.uniform(-np.pi, np.pi, magnitude.shape)
        candidate_spectrum = magnitude * np.exp(1j * phase)
        for _ in range(sweeps):
            candidate = np.maximum(
                np.fft.ifft2(candidate_spectrum).real, 0.0)
            current = np.fft.fft2(candidate)
            candidate_spectrum = (
                magnitude * current / np.maximum(np.abs(current), 1e-30))
        reconstructions.append(
            np.fft.ifft2(candidate_spectrum).real)

    anchor = reconstructions[0]
    aligned = [anchor]
    for candidate in reconstructions[1:]:
        ordinary, _ = best_cyclic_alignment(candidate, anchor)
        reflected, _ = best_cyclic_alignment(
            candidate[::-1, ::-1], anchor)
        ordinary_error = float(np.mean((ordinary - anchor) ** 2))
        reflected_error = float(np.mean((reflected - anchor) ** 2))
        aligned.append(
            ordinary if ordinary_error <= reflected_error else reflected)
        lane_metrics.append(min(ordinary_error, reflected_error))
    return np.mean(aligned, axis=0), {
        "mode": "autocorrelation_positive_phase_retrieval",
        "lanes": lanes,
        "sweeps_per_lane": sweeps,
        "mean_lane_disagreement": float(np.mean(lane_metrics)),
    }


def oracle_registered_mean(
    counts: np.ndarray,
    shifts: np.ndarray,
    config: RayBench,
) -> np.ndarray:
    radiance = calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    )
    return np.mean([
        np.roll(frame, (-int(dy), -int(dx)), axis=(0, 1))
        for frame, (dy, dx) in zip(radiance, shifts)
    ], axis=0)


def run(config: RayBench) -> tuple[dict, dict[str, np.ndarray]]:
    truth = source_image(config.grid)
    counts, shifts = capture(
        truth,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )
    validation, _ = capture(
        truth,
        config.validation_frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed + 1,
    )
    unregistered = np.mean(calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    ), axis=0)
    oracle = oracle_registered_mean(counts, shifts, config)
    continual, continual_info = continual_ray_transport(
        counts, config, reference=truth)
    soft, soft_info = replayed_ray_transport(
        counts,
        config,
        temperature=config.temperature,
        sweeps=config.replay_sweeps,
        seed=config.seed + 10,
        reference=truth,
    )
    hard, hard_info = replayed_ray_transport(
        counts,
        config,
        temperature=0.5,
        sweeps=config.replay_sweeps,
        seed=config.seed + 20,
        reference=truth,
    )
    stochastic_lanes = []
    stochastic_info = []
    for lane in range(4):
        estimate, info = replayed_ray_transport(
            counts,
            config,
            temperature=config.temperature,
            sweeps=3,
            logit_noise=2.0,
            seed=config.seed + 100 + lane,
        )
        estimate, _ = best_cyclic_alignment(estimate, soft)
        stochastic_lanes.append(estimate)
        stochastic_info.append(info)
    stochastic = np.mean(stochastic_lanes, axis=0)
    autocorrelation, autocorrelation_info = (
        autocorrelation_phase_retrieval(counts, config))

    speckled_truth = source_image(
        config.grid, speckle=0.005, seed=config.seed + 300)
    speckled_counts, speckled_shifts = capture(
        speckled_truth,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )
    speckled, speckled_info = continual_ray_transport(
        speckled_counts, config, reference=speckled_truth)

    noisy_config = RayBench(
        **{
            **asdict(config),
            "background_photons": (
                config.background_photons + config.photons_at_white),
        }
    )
    background_counts, background_shifts = capture(
        truth,
        config.frames,
        config.photons_at_white,
        noisy_config.background_photons,
        config.shift_radius,
        config.seed,
    )
    background, background_info = continual_ray_transport(
        background_counts, noisy_config, reference=truth)

    star_truth = truth.copy()
    star_rng = np.random.default_rng(config.seed + 500)
    star_indices = star_rng.choice(
        star_truth.size, size=8, replace=False)
    star_truth.ravel()[star_indices] = 10.0
    star_counts, star_shifts = capture(
        star_truth,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )
    star_recovery, star_info = continual_ray_transport(
        star_counts, config)

    estimates = {
        "unregistered_mean": unregistered,
        "oracle_registered_mean": oracle,
        "continual_soft_ray": continual,
        "replayed_soft_ray": soft,
        "nearly_hard_ray": hard,
        "noise_perturbed_ray_consensus": stochastic,
        "autocorrelation_phase_retrieval": autocorrelation,
    }
    result = {
        "config": asdict(config),
        "metrics": {
            name: {
                **metrics(image, truth),
                "heldout_log_evidence_per_pixel": marginal_poisson_score(
                    validation, image, config),
            }
            for name, image in estimates.items()
        },
        "continual": continual_info,
        "soft_replay": soft_info,
        "hard_replay": hard_info,
        "stochastic_lanes": stochastic_info,
        "autocorrelation": autocorrelation_info,
        "noise_ablation": {
            "persistent_scene_speckle": {
                **metrics(speckled, speckled_truth),
                **shift_diagnostic(
                    speckled_counts,
                    speckled,
                    speckled_shifts,
                    config,
                ),
            },
            "ordinary_scene": {
                **metrics(continual, truth),
                **shift_diagnostic(
                    counts, continual, shifts, config),
            },
            "added_independent_background": {
                **metrics(background, truth),
                **shift_diagnostic(
                    background_counts,
                    background,
                    background_shifts,
                    noisy_config,
                ),
            },
            "eight_hdr_guide_stars": {
                **shift_diagnostic(
                    star_counts,
                    star_recovery,
                    star_shifts,
                    config,
                ),
                "star_radiance_relative_to_white": 10.0,
                "continual_trace_final": star_info["trace"][-1],
            },
        },
    }
    images = {
        "truth": truth,
        "one_frame": calibrated_radiance(
            counts[0],
            config.photons_at_white,
            config.background_photons,
        ),
        **estimates,
        "speckled_truth": speckled_truth,
        "speckled_recovery": speckled,
        "background_recovery": background,
        "star_recovery": star_recovery,
    }
    return result, images


def render(
    result: dict,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.4))
    panels = (
        ("truth", "Oracle", None),
        ("one_frame", "One sparse photon frame", None),
        ("unregistered_mean", "Unregistered mean", "unregistered_mean"),
        (
            "oracle_registered_mean",
            "Oracle-registered mean",
            "oracle_registered_mean",
        ),
        (
            "continual_soft_ray",
            "One-pass soft ray transport",
            "continual_soft_ray",
        ),
        (
            "replayed_soft_ray",
            "Tempered replay control",
            "replayed_soft_ray",
        ),
        (
            "nearly_hard_ray",
            "Nearly-hard posterior",
            "nearly_hard_ray",
        ),
        (
            "noise_perturbed_ray_consensus",
            "Injected-noise consensus",
            "noise_perturbed_ray_consensus",
        ),
        (
            "autocorrelation_phase_retrieval",
            "Autocorrelation phase retrieval",
            "autocorrelation_phase_retrieval",
        ),
        ("speckled_recovery", "Persistent scene speckle", None),
    )
    for axis, (key, title, metric_key) in zip(axes.ravel(), panels):
        axis.imshow(
            np.clip(images[key], 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        if metric_key:
            value = result["metrics"][metric_key]
            title += (
                f"\n{value['psnr_db']:.2f} dB, "
                f"SSIM {value['ssim']:.3f}")
        axis.set_title(title)
        axis.axis("off")
    config = result["config"]
    figure.suptitle(
        "Every photon votes through its complete shift orbit: "
        f"{config['grid']}×{config['grid']}, {config['frames']} frames, "
        f"{config['photons_at_white']:.3f} photon/white pixel/frame"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--photons", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("high_vision/out/sparse_ray_transport.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("high_vision/out/sparse_ray_transport.json"),
    )
    args = parser.parse_args()
    config = RayBench(
        frames=args.frames,
        photons_at_white=args.photons,
        temperature=args.temperature,
        replay_sweeps=args.sweeps,
    )
    result, images = run(config)
    render(result, images, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
