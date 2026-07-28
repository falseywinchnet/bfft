#!/usr/bin/env python3
"""Finite-D Poisson-flow probes for sparse photon registration.

This experiment tests two distinct translations of PPFM/PFGM++ geometry into
the sparse-ray benchmark:

1. ``likelihood flow`` replaces the Gaussian/exponential softmax over shifts
   with the finite-D heavy-tailed kernel whose D -> infinity limit is the
   ordinary tempered posterior.
2. ``image charge flow`` treats posterior-sampled registered reconstructions
   as empirical charges and performs one local field fusion per incoming
   batch.

Neither method is allowed to select hyperparameters from the clean image.
The likelihood sweep is selected with a separately generated photon capture
and independently checked by bidirectional Poisson thinning.  The image-charge
sweep is selected by the complementary half of a thinned photon stream.
Oracle PSNR/SSIM are computed only after selection as a diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from sparse_ray_transport import (
    RayBench,
    backproject,
    calibrated_radiance,
    capture,
    continual_ray_transport,
    marginal_poisson_score,
    metrics,
    oracle_registered_mean,
    shift_mask,
    shift_posterior,
    source_image,
)


@dataclass(frozen=True)
class FlowBench:
    split_probability: float = 0.5
    likelihood_dimensions: tuple[float, ...] = (
        4.0, 16.0, 64.0, 128.0, float("inf"))
    likelihood_temperatures: tuple[float, ...] = (
        1.5, 2.0, 2.5, 3.0, 4.0)
    split_temperatures: tuple[float, ...] = (
        0.75, 1.0, 1.25, 1.5, 2.0)
    charge_dimensions: tuple[float, ...] = (
        16.0, 64.0, 128.0, float("inf"))
    charge_radii: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    charge_lanes: int = 16
    charge_patch: int = 8
    split_seed_offset: int = 77


def label_dimension(dimension: float) -> str:
    return "inf" if np.isinf(dimension) else str(int(dimension))


def poisson_thin(
    counts: np.ndarray,
    probability: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed events into complementary Poisson witnesses."""
    rng = np.random.default_rng(seed)
    first = rng.binomial(counts, probability).astype(counts.dtype)
    return first, counts - first


def finite_d_log_weights(
    loss: np.ndarray,
    dimension: float,
    temperature: float,
) -> np.ndarray:
    """Finite-D Student-like weights with an exponential D=infinity limit."""
    nonnegative = np.maximum(np.asarray(loss, dtype=np.float64), 0.0)
    scale = max(float(temperature), 1e-12)
    if np.isinf(dimension):
        return -nonnegative / scale
    d = max(float(dimension), 1e-12)
    return -0.5 * (d + 1.0) * np.log1p(
        2.0 * nonnegative / (d * scale))


def finite_d_shift_posterior(
    counts: np.ndarray,
    belief: np.ndarray,
    config: RayBench,
    admitted: np.ndarray,
    *,
    dimension: float,
    temperature: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map exact Poisson shift evidence through a finite-D flow kernel."""
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
    ).real
    admitted_score = score[:, admitted]
    loss = (
        np.max(admitted_score, axis=1, keepdims=True)
        - admitted_score
    )
    log_probability = finite_d_log_weights(
        loss, dimension, temperature)
    log_probability -= np.max(
        log_probability, axis=1, keepdims=True)
    probability = np.exp(np.clip(log_probability, -60.0, 0.0))
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


def initial_belief(
    counts: np.ndarray,
    config: RayBench,
) -> np.ndarray:
    initial_count = min(config.batch, len(counts))
    return gaussian_filter(
        np.maximum(
            np.mean(
                calibrated_radiance(
                    counts[:initial_count],
                    config.photons_at_white,
                    config.background_photons,
                ),
                axis=0,
            ),
            1e-7,
        ),
        1.0,
    )


def continual_likelihood_flow(
    counts: np.ndarray,
    config: RayBench,
    *,
    dimension: float,
    temperature: float,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """One pass with a finite-D posterior over the complete shift orbit."""
    admitted = shift_mask(config.grid, config.shift_radius)
    belief = initial_belief(counts, config)
    support = min(config.batch, len(counts))
    trace = []
    for start in range(support, len(counts), config.batch):
        batch = counts[start:start + config.batch]
        posterior, info = finite_d_shift_posterior(
            batch,
            belief,
            config,
            admitted,
            dimension=dimension,
            temperature=temperature,
        )
        estimate = backproject(batch, posterior, config)
        old_support = min(support, config.support_cap)
        belief = np.maximum(
            (
                old_support * belief
                + len(batch) * estimate
            )
            / max(old_support + len(batch), 1),
            1e-7,
        )
        support += len(batch)
        record = {"frames": support, **info}
        if reference is not None:
            record.update(metrics(belief, reference))
        trace.append(record)
    return belief, {
        "mode": "continual_finite_d_likelihood_flow",
        "passes_over_frames": 1,
        "dimension": label_dimension(dimension),
        "temperature": temperature,
        "trace": trace,
    }


def posterior_sampled_backprojections(
    counts: np.ndarray,
    posterior: np.ndarray,
    config: RayBench,
    *,
    lanes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw registered batch hypotheses from a diffuse shift posterior."""
    frame_count, height, width = counts.shape
    probability = posterior.reshape(frame_count, -1)
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.0
    radiance = calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    )
    result = np.zeros((lanes, height, width), dtype=np.float64)
    for lane in range(lanes):
        draw = np.asarray([
            np.searchsorted(cumulative[index], rng.random())
            for index in range(frame_count)
        ])
        y_index, x_index = np.unravel_index(
            draw, (height, width))
        y_shift = np.where(
            y_index <= height // 2, y_index, y_index - height)
        x_shift = np.where(
            x_index <= width // 2, x_index, x_index - width)
        result[lane] = np.mean([
            np.roll(
                radiance[index],
                (-int(y_shift[index]), -int(x_shift[index])),
                axis=(0, 1),
            )
            for index in range(frame_count)
        ], axis=0)
    return np.maximum(result, 1e-7)


def local_charge_field(
    belief: np.ndarray,
    charges: np.ndarray,
    *,
    exposure: float,
    dimension: float,
    radius: float,
    patch: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """One local finite-D electric-field barycenter.

    Distances use the square-root embedding of expected Poisson counts.  The
    D=infinity branch is the Gaussian-flow limit of the same kernel.
    """
    stabilized_belief = 2.0 * np.sqrt(
        np.maximum(exposure * belief, 0.0) + 0.375)
    stabilized_charges = 2.0 * np.sqrt(
        np.maximum(exposure * charges, 0.0) + 0.375)
    squared_distance = uniform_filter(
        (stabilized_charges - stabilized_belief) ** 2,
        size=(1, patch, patch),
        mode="wrap",
    ) * (patch * patch)
    scale = max(float(radius) ** 2, 1e-12)
    if np.isinf(dimension):
        log_weight = -squared_distance / (2.0 * scale)
    else:
        d = max(float(dimension), 1e-12)
        log_weight = (
            -0.5
            * (patch * patch + d)
            * np.log1p(squared_distance / (d * scale))
        )
    log_weight -= np.max(log_weight, axis=0, keepdims=True)
    weight = np.exp(np.clip(log_weight, -60.0, 0.0))
    weight /= np.maximum(
        np.sum(weight, axis=0, keepdims=True), 1e-30)
    effective = 1.0 / np.maximum(
        np.sum(weight * weight, axis=0), 1e-30)
    return np.sum(weight * charges, axis=0), {
        "mean_effective_charges": float(np.mean(effective)),
        "median_effective_charges": float(np.median(effective)),
    }


def continual_image_charge_flow(
    counts: np.ndarray,
    config: RayBench,
    *,
    dimension: float,
    radius: float,
    lanes: int,
    patch: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Maintain posterior-sampled lanes and fuse them once per new batch."""
    admitted = shift_mask(config.grid, config.shift_radius)
    belief = initial_belief(counts, config)
    lane_beliefs = np.repeat(belief[None, ...], lanes, axis=0)
    support = min(config.batch, len(counts))
    rng = np.random.default_rng(seed)
    trace = []
    for start in range(support, len(counts), config.batch):
        batch = counts[start:start + config.batch]
        posterior, posterior_info = shift_posterior(
            batch, belief, config, admitted)
        updates = posterior_sampled_backprojections(
            batch,
            posterior,
            config,
            lanes=lanes,
            rng=rng,
        )
        old_support = min(support, config.support_cap)
        lane_beliefs = (
            old_support * lane_beliefs
            + len(batch) * updates
        ) / max(old_support + len(batch), 1)
        support += len(batch)
        belief, field_info = local_charge_field(
            belief,
            lane_beliefs,
            exposure=(
                min(support, config.support_cap)
                * config.photons_at_white
            ),
            dimension=dimension,
            radius=radius,
            patch=patch,
        )
        trace.append({
            "frames": support,
            **posterior_info,
            **field_info,
        })
    return np.maximum(belief, 1e-7), {
        "mode": "continual_local_image_charge_flow",
        "passes_over_frames": 1,
        "dimension": label_dimension(dimension),
        "radius": radius,
        "lanes": lanes,
        "patch": patch,
        "trace": trace,
    }


def evaluate(
    estimate: np.ndarray,
    truth: np.ndarray,
    witness: np.ndarray,
    config: RayBench,
) -> dict[str, float]:
    return {
        "witness_log_evidence_per_pixel": marginal_poisson_score(
            witness, estimate, config),
        **metrics(estimate, truth),
    }


def best_by(
    records: list[dict],
    key: str,
) -> dict:
    return max(records, key=lambda item: float(item[key]))


def run(
    config: RayBench,
    flow: FlowBench,
) -> tuple[dict, dict[str, np.ndarray]]:
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
    first, second = poisson_thin(
        counts,
        flow.split_probability,
        config.seed + flow.split_seed_offset,
    )
    first_scale = flow.split_probability
    second_scale = 1.0 - flow.split_probability
    first_config = replace(
        config,
        photons_at_white=config.photons_at_white * first_scale,
        background_photons=config.background_photons * first_scale,
    )
    second_config = replace(
        config,
        photons_at_white=config.photons_at_white * second_scale,
        background_photons=config.background_photons * second_scale,
    )

    unregistered = np.mean(calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    ), axis=0)
    oracle = oracle_registered_mean(counts, shifts, config)
    standard, standard_info = continual_ray_transport(
        counts, config, reference=truth)

    likelihood_records = []
    likelihood_images: dict[tuple[str, float], np.ndarray] = {}
    for dimension in flow.likelihood_dimensions:
        for temperature in flow.likelihood_temperatures:
            estimate, info = continual_likelihood_flow(
                counts,
                config,
                dimension=dimension,
                temperature=temperature,
            )
            key = (label_dimension(dimension), temperature)
            likelihood_images[key] = estimate
            likelihood_records.append({
                "dimension": key[0],
                "temperature": temperature,
                **evaluate(estimate, truth, validation, config),
                "final_effective_shifts": (
                    info["trace"][-1]["effective_shifts"]
                ),
            })
    selected_likelihood = best_by(
        likelihood_records, "witness_log_evidence_per_pixel")
    likelihood_key = (
        selected_likelihood["dimension"],
        selected_likelihood["temperature"],
    )
    selected_likelihood_image = likelihood_images[likelihood_key]

    split_records = []
    split_images: dict[tuple[str, float], np.ndarray] = {}
    for dimension in flow.likelihood_dimensions:
        for temperature in flow.split_temperatures:
            first_estimate, _ = continual_likelihood_flow(
                first,
                first_config,
                dimension=dimension,
                temperature=temperature,
            )
            second_estimate, _ = continual_likelihood_flow(
                second,
                second_config,
                dimension=dimension,
                temperature=temperature,
            )
            first_to_second = marginal_poisson_score(
                second, first_estimate, second_config)
            second_to_first = marginal_poisson_score(
                first, second_estimate, first_config)
            crossfit = 0.5 * (first_estimate + second_estimate)
            key = (label_dimension(dimension), temperature)
            split_images[key] = crossfit
            split_records.append({
                "dimension": key[0],
                "temperature": temperature,
                "first_to_second_log_evidence": first_to_second,
                "second_to_first_log_evidence": second_to_first,
                "mean_cross_log_evidence": (
                    0.5 * (first_to_second + second_to_first)
                ),
                **metrics(crossfit, truth),
            })
    selected_split = best_by(
        split_records, "mean_cross_log_evidence")
    split_key = (
        selected_split["dimension"],
        selected_split["temperature"],
    )
    selected_split_image = split_images[split_key]

    charge_records = []
    charge_images: dict[tuple[str, float], np.ndarray] = {}
    for dimension in flow.charge_dimensions:
        for radius in flow.charge_radii:
            estimate, info = continual_image_charge_flow(
                first,
                first_config,
                dimension=dimension,
                radius=radius,
                lanes=flow.charge_lanes,
                patch=flow.charge_patch,
                seed=config.seed + 900,
            )
            key = (label_dimension(dimension), radius)
            charge_images[key] = estimate
            charge_records.append({
                "dimension": key[0],
                "radius": radius,
                **evaluate(
                    estimate, truth, second, second_config),
                "final_effective_charges": (
                    info["trace"][-1]["mean_effective_charges"]
                ),
            })
    selected_charge = best_by(
        charge_records, "witness_log_evidence_per_pixel")
    charge_key = (
        selected_charge["dimension"],
        selected_charge["radius"],
    )
    selected_charge_first = charge_images[charge_key]
    selected_charge_second, _ = continual_image_charge_flow(
        second,
        second_config,
        dimension=(
            float("inf")
            if selected_charge["dimension"] == "inf"
            else float(selected_charge["dimension"])
        ),
        radius=selected_charge["radius"],
        lanes=flow.charge_lanes,
        patch=flow.charge_patch,
        seed=config.seed + 901,
    )
    selected_charge_image = 0.5 * (
        selected_charge_first + selected_charge_second)
    selected_charge_crossfit_metrics = metrics(
        selected_charge_image, truth)

    images = {
        "truth": truth,
        "one_frame": calibrated_radiance(
            counts[0],
            config.photons_at_white,
            config.background_photons,
        ),
        "unregistered_mean": unregistered,
        "oracle_registered_mean": oracle,
        "standard_soft": standard,
        "selected_likelihood": selected_likelihood_image,
        "selected_split": selected_split_image,
        "selected_charge": selected_charge_image,
    }
    result = {
        "config": asdict(config),
        "flow": {
            **asdict(flow),
            "likelihood_dimensions": [
                label_dimension(value)
                for value in flow.likelihood_dimensions
            ],
            "charge_dimensions": [
                label_dimension(value)
                for value in flow.charge_dimensions
            ],
        },
        "baselines": {
            "unregistered_mean": metrics(unregistered, truth),
            "oracle_registered_mean": metrics(oracle, truth),
            "standard_soft": {
                **metrics(standard, truth),
                "heldout_log_evidence_per_pixel": (
                    marginal_poisson_score(
                        validation, standard, config)
                ),
                "final_effective_shifts": (
                    standard_info["trace"][-1]["effective_shifts"]
                ),
            },
        },
        "likelihood_flow": {
            "selection_rule": (
                "maximum marginal Poisson evidence on a separate capture"
            ),
            "selected": selected_likelihood,
            "sweep": likelihood_records,
        },
        "photon_thinned_crossfit": {
            "selection_rule": (
                "maximum mean bidirectional complementary-photon evidence"
            ),
            "selected": selected_split,
            "sweep": split_records,
        },
        "image_charge_flow": {
            "selection_rule": (
                "first photon stream proposes; complementary stream selects"
            ),
            "selected_first_direction": selected_charge,
            "selected_crossfit_metrics": (
                selected_charge_crossfit_metrics
            ),
            "sweep": charge_records,
        },
    }
    return result, images


def matrix(
    records: list[dict],
    row_values: list[str],
    column_values: list[float],
    *,
    column_key: str,
    value_key: str,
) -> np.ndarray:
    lookup = {
        (item["dimension"], float(item[column_key])): item[value_key]
        for item in records
    }
    return np.asarray([
        [
            lookup[(row, float(column))]
            for column in column_values
        ]
        for row in row_values
    ], dtype=np.float64)


def render(
    result: dict,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.8))
    panels = (
        ("truth", "Oracle", None),
        ("one_frame", "One sparse photon frame", None),
        ("unregistered_mean", "Unregistered mean", "unregistered_mean"),
        (
            "oracle_registered_mean",
            "Oracle-registered mean",
            "oracle_registered_mean",
        ),
        ("standard_soft", "Original soft transport", "standard_soft"),
        (
            "selected_likelihood",
            "Heldout-selected likelihood flow",
            "likelihood",
        ),
        (
            "selected_split",
            "Photon-crossfit likelihood flow",
            "split",
        ),
        (
            "selected_charge",
            "Posterior-sample image charges",
            "charge",
        ),
    )
    for axis, (key, title, metric_key) in zip(
        axes.ravel()[:8], panels
    ):
        axis.imshow(
            np.clip(images[key], 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        if metric_key == "likelihood":
            value = result["likelihood_flow"]["selected"]
        elif metric_key == "split":
            value = result["photon_thinned_crossfit"]["selected"]
        elif metric_key == "charge":
            value = result["image_charge_flow"][
                "selected_crossfit_metrics"]
        elif metric_key:
            value = result["baselines"][metric_key]
        else:
            value = None
        if value is not None:
            title += (
                f"\n{value['psnr_db']:.2f} dB, "
                f"SSIM {value['ssim']:.3f}"
            )
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    dimensions = result["flow"]["likelihood_dimensions"]
    temperatures = result["flow"]["likelihood_temperatures"]
    likelihood = matrix(
        result["likelihood_flow"]["sweep"],
        dimensions,
        temperatures,
        column_key="temperature",
        value_key="witness_log_evidence_per_pixel",
    )
    likelihood -= np.max(likelihood)
    heat = axes.ravel()[8].imshow(
        likelihood,
        cmap="magma",
        aspect="auto",
    )
    axes.ravel()[8].set_title(
        "Separate-capture evidence\n(relative; 0 is best)")
    axes.ravel()[8].set_xticks(
        range(len(temperatures)), temperatures)
    axes.ravel()[8].set_yticks(
        range(len(dimensions)), dimensions)
    axes.ravel()[8].set_xlabel("temperature")
    axes.ravel()[8].set_ylabel("finite D")
    figure.colorbar(heat, ax=axes.ravel()[8], fraction=0.046)

    charge_dimensions = result["flow"]["charge_dimensions"]
    radii = result["flow"]["charge_radii"]
    charge = matrix(
        result["image_charge_flow"]["sweep"],
        charge_dimensions,
        radii,
        column_key="radius",
        value_key="witness_log_evidence_per_pixel",
    )
    charge -= np.max(charge)
    heat = axes.ravel()[9].imshow(
        charge,
        cmap="magma",
        aspect="auto",
    )
    axes.ravel()[9].set_title(
        "Image-charge evidence\n(relative; 0 is best)")
    axes.ravel()[9].set_xticks(range(len(radii)), radii)
    axes.ravel()[9].set_yticks(
        range(len(charge_dimensions)), charge_dimensions)
    axes.ravel()[9].set_xlabel("field radius")
    axes.ravel()[9].set_ylabel("finite D")
    figure.colorbar(heat, ax=axes.ravel()[9], fraction=0.046)

    config = result["config"]
    figure.suptitle(
        "Finite-D Poisson-flow probes over sparse registration: "
        f"{config['grid']}×{config['grid']}, "
        f"{config['frames']} frames"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--photons", type=float, default=0.08)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "high_vision/out/finite_d_poisson_flow.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "high_vision/out/finite_d_poisson_flow.json"),
    )
    args = parser.parse_args()
    config = RayBench(
        frames=args.frames,
        photons_at_white=args.photons,
    )
    result, images = run(config, FlowBench())
    render(result, images, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "baselines": result["baselines"],
        "selected_likelihood": (
            result["likelihood_flow"]["selected"]
        ),
        "selected_split": (
            result["photon_thinned_crossfit"]["selected"]
        ),
        "selected_charge": (
            result["image_charge_flow"][
                "selected_first_direction"]
        ),
        "selected_charge_crossfit_metrics": (
            result["image_charge_flow"][
                "selected_crossfit_metrics"]
        ),
    }, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
