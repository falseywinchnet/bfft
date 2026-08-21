"""Uncertainty transport for positive exposure-path deblurring.

The state is a finite probability law over physically admissible blur paths,
not one prematurely selected kernel.  Each branch is scored by the registered
cross-observation closure, using a smooth robust loss.  Surviving branches are
transported through the inverse problem and combined with the law of total
variance.

This is deliberately weaker than a formal zonotopic image enclosure.  The
candidate catalog is finite and the consistency screen is calibrated from the
observations, so the returned intervals are empirical credible intervals.  The
code never labels them guaranteed bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .circles import circle_pool
from .kernels import TransportKernel, apply_circular
from .solver import DeblurResult, fuse_transport_observations


@dataclass(frozen=True)
class PairHypothesis:
    """One positive blur-pair branch and its evidence weight."""

    first: TransportKernel
    second: TransportKernel
    closure_score: float
    gauge_cost: float
    energy: float
    probability: float
    consistent: bool
    relative_transport_strength: float


@dataclass(frozen=True)
class PairPosterior:
    """Finite posterior-like law over relative blur hypotheses."""

    hypotheses: tuple[PairHypothesis, ...]
    temperature: float
    consistency_limit: float
    smoothing: float
    entropy: float
    effective_hypotheses: float
    common_blur_unidentifiable: bool

    @property
    def best(self) -> PairHypothesis:
        return self.hypotheses[0]


@dataclass(frozen=True)
class NoiseDiscrepancy:
    """Limited separation of fine sensor noise from structured model error."""

    read_sigma: float
    structured_rms: float
    total_rms: float
    outlier_fraction: float
    saturation_fraction: float


@dataclass(frozen=True)
class UncertainDeblurResult:
    """Mixture reconstruction and transported empirical uncertainty."""

    image: np.ndarray
    standard_deviation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    branch_images: tuple[np.ndarray, ...]
    branch_hypotheses: tuple[PairHypothesis, ...]
    branch_weights: np.ndarray
    retained_probability: float
    diagnostics: dict[str, object]


def pseudo_huber(value: np.ndarray, smoothing: float) -> np.ndarray:
    """Smooth absolute residual, quadratic near zero and linear in the tail."""
    delta = max(float(smoothing), np.finfo(float).eps)
    residual = np.asarray(value, dtype=np.float64)
    return np.sqrt(residual * residual + delta * delta) - delta


def _luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[2] >= 3:
        return (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    raise ValueError("observations must be HxW or RGB")


def _robust_closure_score(
    first_fft: np.ndarray,
    second_fft: np.ndarray,
    first_otf: np.ndarray,
    second_otf: np.ndarray,
    *,
    smoothing: float,
    ring_width: float,
    minimum_radius: float,
    maximum_radius_fraction: float,
) -> float:
    left = first_fft * second_otf
    right = second_fft * first_otf
    energy = np.abs(left) ** 2 + np.abs(right) ** 2
    floor = max(
        float(np.quantile(energy, 0.10)),
        float(np.max(energy)) * 1e-14,
        np.finfo(float).tiny,
    )
    normalized = np.abs(left - right) / np.sqrt(energy + floor)
    loss = pseudo_huber(normalized, smoothing)
    pooled, support = circle_pool(
        loss, weights=np.sqrt(energy + floor), width=ring_width)
    radius = (np.arange(len(pooled), dtype=np.float64) + 0.5) * ring_width
    positive_support = support[support > 0.0]
    support_floor = (
        float(np.quantile(positive_support, 0.10))
        if len(positive_support) else float("inf")
    )
    valid = (
        (radius >= minimum_radius)
        & (radius <= maximum_radius_fraction * 0.5 * min(first_fft.shape))
        & (support > support_floor)
    )
    selected = pooled[valid]
    if not len(selected):
        return float("inf")
    trimmed = np.sort(selected)[: max(1, int(math.ceil(0.8 * len(selected))))]
    return float(0.75 * np.median(selected) + 0.25 * np.mean(trimmed))


def estimate_pair_posterior(
    first: np.ndarray,
    second: np.ndarray,
    candidates: list[TransportKernel],
    *,
    noise_sigma: float = 0.0,
    smoothing: float = 0.02,
    temperature: float | None = None,
    consistency_multiplier: float = 8.0,
    common_blur_gauge_penalty: float = 0.04,
    priors: np.ndarray | None = None,
    ring_width: float = 1.5,
    minimum_radius: float = 2.0,
    maximum_radius_fraction: float = 0.92,
) -> PairPosterior:
    """Return a robust finite law over relative registered blur pairs.

    The posterior probabilities are evidence weights over the supplied finite
    catalog, not a claim that the catalog contains every real blur.  A common
    blur factor remains unidentifiable and is reported separately.
    """
    a = _luminance(first)
    b = _luminance(second)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("registered observations must share one 2-D shape")
    if not candidates:
        raise ValueError("a non-empty positive-kernel catalog is required")
    count = len(candidates)
    prior = (
        np.ones((count, count), dtype=np.float64)
        if priors is None else np.asarray(priors, dtype=np.float64)
    )
    if prior.shape != (count, count) or np.any(prior < 0.0):
        raise ValueError("priors must be a non-negative candidate-pair matrix")
    if float(np.sum(prior)) <= 0.0:
        raise ValueError("candidate-pair priors must carry positive mass")
    prior = prior / np.sum(prior)

    first_fft = np.fft.fft2(a - np.mean(a))
    second_fft = np.fft.fft2(b - np.mean(b))
    transfers = [candidate.otf(a.shape) for candidate in candidates]
    records: list[tuple[float, float, float, int, int, float]] = []
    for i, first_otf in enumerate(transfers):
        for j, second_otf in enumerate(transfers):
            closure = _robust_closure_score(
                first_fft,
                second_fft,
                first_otf,
                second_otf,
                smoothing=smoothing,
                ring_width=ring_width,
                minimum_radius=minimum_radius,
                maximum_radius_fraction=maximum_radius_fraction,
            )
            joint_transfer = 0.5 * (
                np.abs(first_otf) ** 2 + np.abs(second_otf) ** 2)
            gauge = float(np.mean(-np.log(np.maximum(joint_transfer, 1e-8))))
            energy = closure + max(float(common_blur_gauge_penalty), 0.0) * gauge
            relative = float(np.sqrt(np.mean(
                np.abs(first_otf - second_otf) ** 2)))
            records.append((energy, closure, gauge, i, j, relative))
    records.sort(key=lambda item: item[0])
    best_energy = float(records[0][0])
    evidence_temperature = (
        max(float(temperature), 1e-10)
        if temperature is not None
        else max(1e-7, 1.5 * max(float(noise_sigma), 0.0), 0.05 * best_energy)
    )
    limit = best_energy + max(float(consistency_multiplier), 1.0) * evidence_temperature
    log_weights = np.full(len(records), -np.inf, dtype=np.float64)
    consistent = np.asarray([record[0] <= limit for record in records])
    for index, record in enumerate(records):
        if consistent[index] and prior[record[3], record[4]] > 0.0:
            log_weights[index] = (
                -(record[0] - best_energy) / evidence_temperature
                + math.log(prior[record[3], record[4]])
            )
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        finite[0] = True
        log_weights[0] = 0.0
    log_weights[finite] -= np.max(log_weights[finite])
    weights = np.zeros(len(records), dtype=np.float64)
    weights[finite] = np.exp(log_weights[finite])
    weights /= np.sum(weights)

    hypothesis_list = [
        PairHypothesis(
            first=candidates[record[3]],
            second=candidates[record[4]],
            closure_score=float(record[1]),
            gauge_cost=float(record[2]),
            energy=float(record[0]),
            probability=float(weights[index]),
            consistent=bool(consistent[index]),
            relative_transport_strength=float(record[5]),
        )
        for index, record in enumerate(records)
    ]
    hypothesis_list.sort(key=lambda item: (-item.probability, item.energy))
    hypotheses = tuple(hypothesis_list)
    positive = weights[weights > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return PairPosterior(
        hypotheses=hypotheses,
        temperature=evidence_temperature,
        consistency_limit=float(limit),
        smoothing=float(smoothing),
        entropy=entropy,
        effective_hypotheses=float(np.exp(entropy)),
        common_blur_unidentifiable=bool(
            hypotheses[0].relative_transport_strength < 0.02),
    )


def _weighted_quantile(
    samples: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if values.ndim < 2 or weight.shape != (values.shape[0],):
        raise ValueError("one scalar weight is required per sample image")
    weight = weight / np.sum(weight)
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    expanded = np.broadcast_to(
        weight.reshape((len(weight),) + (1,) * (values.ndim - 1)), values.shape)
    sorted_weight = np.take_along_axis(expanded, order, axis=0)
    cumulative = np.cumsum(sorted_weight, axis=0)
    index = np.argmax(cumulative >= float(np.clip(quantile, 0.0, 1.0)), axis=0)
    return np.take_along_axis(sorted_values, index[None, ...], axis=0)[0]


def _linearized_noise_variance(
    shape: tuple[int, int],
    kernels: tuple[TransportKernel, TransportKernel],
    noise_sigma: float,
    flux_penalty: float,
) -> float:
    sigma = max(float(noise_sigma), 0.0)
    if sigma == 0.0:
        return 0.0
    transfers = [kernel.otf(shape) for kernel in kernels]
    coverage = sum(np.abs(transfer) ** 2 for transfer in transfers)
    height, width = shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    laplacian = (
        4.0 * np.sin(np.pi * fy[:, None]) ** 2
        + 4.0 * np.sin(np.pi * fx[None, :]) ** 2)
    denominator = coverage + max(float(flux_penalty), 0.0) * laplacian
    power = sigma * sigma * coverage / np.maximum(denominator * denominator, 1e-20)
    return float(np.mean(power))


def deblur_pair_posterior(
    first: np.ndarray,
    second: np.ndarray,
    posterior: PairPosterior,
    *,
    credibility: float = 0.95,
    maximum_branches: int = 8,
    noise_sigma: float = 0.0,
    tv_weight: float = 0.0012,
    flux_penalty: float = 0.035,
    passes: int = 20,
) -> UncertainDeblurResult:
    """Transport a finite kernel law through deblurring.

    Between-branch variation carries blur uncertainty.  A stationary
    linearization supplies a conservative within-branch read-noise floor.
    """
    selected: list[PairHypothesis] = []
    retained = 0.0
    target = float(np.clip(credibility, 0.0, 1.0))
    for hypothesis in posterior.hypotheses:
        if not hypothesis.consistent or hypothesis.probability <= 0.0:
            continue
        selected.append(hypothesis)
        retained += hypothesis.probability
        if retained >= target or len(selected) >= max(int(maximum_branches), 1):
            break
    if not selected:
        selected = [posterior.best]
        retained = posterior.best.probability
    weights = np.asarray([item.probability for item in selected], dtype=np.float64)
    weights /= np.sum(weights)
    branch_results: list[DeblurResult] = []
    for hypothesis in selected:
        branch_results.append(fuse_transport_observations(
            [first, second],
            [hypothesis.first, hypothesis.second],
            tv_weight=tv_weight,
            flux_penalty=flux_penalty,
            passes=passes,
        ))
    samples = np.stack([result.image for result in branch_results], axis=0)
    expanded = weights.reshape((len(weights),) + (1,) * (samples.ndim - 1))
    mean = np.sum(expanded * samples, axis=0)
    branch_variance = np.sum(expanded * (samples - mean) ** 2, axis=0)
    within = np.asarray([
        _linearized_noise_variance(
            mean.shape[:2], (item.first, item.second), noise_sigma, flux_penalty)
        for item in selected
    ])
    total_variance = branch_variance + float(np.sum(weights * within))
    alpha = 0.5 * (1.0 - target)
    lower = _weighted_quantile(samples, weights, alpha)
    upper = _weighted_quantile(samples, weights, 1.0 - alpha)
    return UncertainDeblurResult(
        image=np.clip(mean, 0.0, 1.0),
        standard_deviation=np.sqrt(np.maximum(total_variance, 0.0)),
        lower=np.clip(lower, 0.0, 1.0),
        upper=np.clip(upper, 0.0, 1.0),
        branch_images=tuple(result.image for result in branch_results),
        branch_hypotheses=tuple(selected),
        branch_weights=weights,
        retained_probability=float(retained),
        diagnostics={
            "posterior_entropy": posterior.entropy,
            "effective_hypotheses": posterior.effective_hypotheses,
            "selected_branches": len(selected),
            "retained_probability": float(retained),
            "mean_linearized_noise_variance": float(np.sum(weights * within)),
            "mean_blur_variance": float(np.mean(branch_variance)),
            "common_blur_unidentifiable": posterior.common_blur_unidentifiable,
            "branch_methods": [
                result.diagnostics["method"] for result in branch_results],
        },
    )


def estimate_noise_discrepancy(
    observation: np.ndarray,
    prediction: np.ndarray,
    *,
    saturation_epsilon: float = 1.0 / 255.0,
) -> NoiseDiscrepancy:
    """Separate fine residual noise from structured model discrepancy.

    This is intentionally a limited diagnostic.  A 3x3 local mean is treated
    as structured mismatch; the high-pass remainder gives a robust read-noise
    estimate.  It does not claim to separate arbitrary real blur and noise.
    """
    observed = _luminance(observation)
    predicted = _luminance(prediction)
    if observed.shape != predicted.shape:
        raise ValueError("observation and prediction must share a raster")
    residual = observed - predicted
    local = sum(
        np.roll(np.roll(residual, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ) / 9.0
    high = residual - local
    median = float(np.median(high))
    mad = float(np.median(np.abs(high - median)))
    # The delta-minus-3x3-mean filter has white-noise gain sqrt(8/9).
    sigma = mad / (0.6744897501960817 * math.sqrt(8.0 / 9.0) + 1e-15)
    threshold = max(4.0 * sigma, 1e-8)
    return NoiseDiscrepancy(
        read_sigma=float(sigma),
        structured_rms=float(np.sqrt(np.mean(local * local))),
        total_rms=float(np.sqrt(np.mean(residual * residual))),
        outlier_fraction=float(np.mean(np.abs(high - median) > threshold)),
        saturation_fraction=float(np.mean(
            (observed <= saturation_epsilon)
            | (observed >= 1.0 - saturation_epsilon))),
    )


def predict_observation(
    latent: np.ndarray,
    kernel: TransportKernel,
) -> np.ndarray:
    """Named forward prediction helper for uncertainty audits."""
    return apply_circular(latent, kernel)
