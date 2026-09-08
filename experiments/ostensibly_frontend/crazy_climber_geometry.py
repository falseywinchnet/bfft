"""Grayscale ridge saliency and Carmona-style stochastic occupation fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class HessianRidgeConfig:
    scales: tuple[float, ...] = (0.8, 1.2, 1.8, 2.6)
    line_to_blob_beta: float = 0.5
    structure_percentile: float = 99.0

    def __post_init__(self) -> None:
        if not self.scales or any(scale <= 0.0 for scale in self.scales):
            raise ValueError("ridge scales must be positive")
        if self.line_to_blob_beta <= 0.0:
            raise ValueError("line-to-blob beta must be positive")
        if not 0.0 < self.structure_percentile <= 100.0:
            raise ValueError("invalid structure percentile")


@dataclass(frozen=True)
class CrazyClimberConfig:
    climbers: int = 1000
    steps: int = 6000
    burn_in: int = 1000
    sample_stride: int = 4
    temperature_start: float = 0.08
    temperature_end: float = 0.006
    random_seed: int = 99173

    def __post_init__(self) -> None:
        if self.climbers < 1 or self.steps < 1:
            raise ValueError("climber count and steps must be positive")
        if not 0 <= self.burn_in < self.steps or self.sample_stride < 1:
            raise ValueError("invalid occupation schedule")
        if self.temperature_start <= 0.0 or self.temperature_end <= 0.0:
            raise ValueError("temperatures must be positive")


@dataclass(frozen=True)
class RidgeSurface:
    saliency: np.ndarray
    tangent: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class OccupationField:
    unweighted: np.ndarray
    weighted: np.ndarray
    visits: int
    accepted_vertical_fraction: float


def _hessian_scale_response(
    field: np.ndarray,
    sigma: float,
    beta: float,
    structure_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-line saliency and tangent from one Hessian scale."""

    scale_normalization = sigma * sigma
    hrr = scale_normalization * ndi.gaussian_filter(
        field, sigma=sigma, order=(2, 0), mode="nearest"
    )
    hrt = scale_normalization * ndi.gaussian_filter(
        field, sigma=sigma, order=(1, 1), mode="nearest"
    )
    htt = scale_normalization * ndi.gaussian_filter(
        field, sigma=sigma, order=(0, 2), mode="nearest"
    )
    hessian = np.empty(field.shape + (2, 2), dtype=np.float64)
    hessian[..., 0, 0] = hrr
    hessian[..., 0, 1] = hrt
    hessian[..., 1, 0] = hrt
    hessian[..., 1, 1] = htt
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    swap = np.abs(eigenvalues[..., 0]) > np.abs(eigenvalues[..., 1])
    lambda_tangent = np.where(swap, eigenvalues[..., 1], eigenvalues[..., 0])
    lambda_normal = np.where(swap, eigenvalues[..., 0], eigenvalues[..., 1])
    tangent0 = np.where(swap, eigenvectors[..., 0, 1], eigenvectors[..., 0, 0])
    tangent1 = np.where(swap, eigenvectors[..., 1, 1], eigenvectors[..., 1, 0])
    tangent = np.arctan2(tangent0, tangent1)

    line_ratio = np.abs(lambda_tangent) / np.maximum(np.abs(lambda_normal), 1e-30)
    structure = np.hypot(lambda_tangent, lambda_normal)
    positive_structure = structure[structure > 0.0]
    gauge = (
        float(np.percentile(positive_structure, structure_percentile))
        if positive_structure.size
        else 1.0
    )
    saliency = np.exp(-(line_ratio * line_ratio) / (2.0 * beta * beta))
    saliency *= 1.0 - np.exp(-(structure * structure) / (2.0 * gauge * gauge))
    saliency = np.where(lambda_normal < 0.0, saliency, 0.0)
    return saliency, tangent


def multiscale_hessian_ridge_surface(
    field: np.ndarray,
    config: HessianRidgeConfig = HessianRidgeConfig(),
) -> RidgeSurface:
    """Measure line-like relief without forming a binary support mask."""

    source = np.asarray(field, dtype=np.float64)
    if source.ndim != 2 or not source.size or not np.all(np.isfinite(source)):
        raise ValueError("ridge input must be one finite, nonempty 2-D field")
    source = np.maximum(source, 0.0)
    responses = []
    tangents = []
    for sigma in config.scales:
        response, tangent = _hessian_scale_response(
            source,
            sigma,
            config.line_to_blob_beta,
            config.structure_percentile,
        )
        responses.append(response)
        tangents.append(tangent)
    stack = np.stack(responses)
    winners = np.argmax(stack, axis=0)
    saliency = np.take_along_axis(stack, winners[None], axis=0)[0]
    tangent_stack = np.stack(tangents)
    tangent = np.take_along_axis(tangent_stack, winners[None], axis=0)[0]
    scale = np.asarray(config.scales, dtype=np.float64)[winners]
    return RidgeSurface(saliency=saliency, tangent=tangent, scale=scale)


def _reflect(indices: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(indices)
    return np.where(indices < 0, 1, np.where(indices >= size, size - 2, indices))


def crazy_climber_occupation(
    surface: np.ndarray,
    config: CrazyClimberConfig = CrazyClimberConfig(),
) -> OccupationField:
    """Accumulate unweighted and surface-weighted Markov occupation fields.

    Time performs an unbiased reflected random walk.  At every new time column,
    the row proposal is accepted whenever it climbs and otherwise according to
    a cooling Metropolis probability.  Walkers remain mobile after reaching a
    ridge, so recurrent mass—not terminal local maxima—is the result.
    """

    field = np.asarray(surface, dtype=np.float64)
    if field.ndim != 2 or not field.size or not np.all(np.isfinite(field)):
        raise ValueError("climber surface must be one finite, nonempty 2-D field")
    field = np.maximum(field, 0.0)
    positive = field[field > 0.0]
    if not positive.size:
        zeros = np.zeros(field.shape, dtype=np.float64)
        return OccupationField(zeros, zeros.copy(), 0, 0.0)
    gauge = max(float(np.percentile(positive, 99.5)), 1e-30)
    normalized = np.clip(field / gauge, 0.0, 1.0)
    row_count, frame_count = field.shape
    rng = np.random.default_rng(config.random_seed)
    rows = rng.integers(0, row_count, size=config.climbers, dtype=np.int64)
    frames = rng.integers(0, frame_count, size=config.climbers, dtype=np.int64)
    unweighted = np.zeros(field.size, dtype=np.float64)
    weighted = np.zeros(field.size, dtype=np.float64)
    samples = 0
    accepted = 0
    proposed = 0
    cooling_ratio = config.temperature_end / config.temperature_start

    for step in range(config.steps):
        frame_direction = 2 * rng.integers(0, 2, config.climbers) - 1
        frames = _reflect(frames + frame_direction, frame_count)
        row_direction = 2 * rng.integers(0, 2, config.climbers) - 1
        proposed_rows = _reflect(rows + row_direction, row_count)
        current_value = normalized[rows, frames]
        proposed_value = normalized[proposed_rows, frames]
        difference = proposed_value - current_value
        progress = step / max(config.steps - 1, 1)
        temperature = config.temperature_start * cooling_ratio**progress
        probability = np.exp(np.minimum(difference, 0.0) / temperature)
        move = (difference >= 0.0) | (rng.random(config.climbers) < probability)
        rows = np.where(move, proposed_rows, rows)
        accepted += int(np.count_nonzero(move))
        proposed += config.climbers

        if step >= config.burn_in and (step - config.burn_in) % config.sample_stride == 0:
            flat = rows * frame_count + frames
            unweighted += np.bincount(flat, minlength=field.size)
            weighted += np.bincount(
                flat,
                weights=normalized[rows, frames],
                minlength=field.size,
            )
            samples += config.climbers

    if samples:
        unweighted /= samples
        weighted /= samples
    return OccupationField(
        unweighted.reshape(field.shape),
        weighted.reshape(field.shape),
        samples,
        accepted / max(proposed, 1),
    )
