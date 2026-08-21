"""Exchange-symmetric non-Gaussian shape transport at fixed covariance.

For one covariance eigenaxis with variance ``lambda``, a symmetric positive
three-point measure has side mass ``w`` and extent

    a = sqrt(lambda / (2 w)).

Its fourth cumulant is ``lambda^2 * (1/(2w) - 3)``. Thus ``w=1/6`` is the
Gaussian-matched sigma rule, while every bounded ``0 < w < 1/2`` remains a
positive unit-mass exposure measure. Relative Fourier magnitudes estimate all
capture weights simultaneously; a common shape component remains a gauge and
is fixed by minimum departure from ``1/6`` rather than frame selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dense_estimation import _luminance


@dataclass(frozen=True)
class QuarticShapeTransport:
    side_weights: np.ndarray
    raw_side_weights: np.ndarray
    residual_displacements: tuple[np.ndarray, ...]
    residual_weights: tuple[np.ndarray, ...]
    authority: float
    diagnostics: dict[str, object]


def _pooled_relative_log_magnitudes(
    observations: Sequence[np.ndarray],
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    radial_bins: int,
    angular_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    images = tuple(_luminance(item) for item in observations)
    shape = images[0].shape
    if any(item.shape != shape for item in images[1:]):
        raise ValueError("quartic shape observations must share one raster")
    height, width = shape
    window = np.outer(np.hanning(height), np.hanning(width))
    magnitudes = []
    for image in images:
        centered = (image - np.mean(image)) * window
        magnitudes.append(np.abs(np.fft.fftshift(np.fft.fft2(centered))))
    magnitude_stack = np.stack(magnitudes, axis=0)
    floor = max(float(np.quantile(magnitude_stack, 0.20)), 1e-10)
    log_magnitude = np.log(magnitude_stack + floor)
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    grid_x = np.broadcast_to(fx, shape)
    grid_y = np.broadcast_to(fy, shape)
    radius = np.hypot(grid_x, grid_y)
    energy = np.exp(np.mean(np.log(magnitude_stack + floor), axis=0))
    band = (
        (radius >= max(float(minimum_frequency), 0.0))
        & (radius <= max(float(maximum_frequency), minimum_frequency))
    )
    threshold = float(np.quantile(energy[band], 0.30))
    supported = band & (energy >= threshold)
    radial_coordinate = np.clip(
        (radius - minimum_frequency)
        / max(maximum_frequency - minimum_frequency, 1e-12),
        0.0,
        1.0 - np.finfo(float).eps,
    )
    antipodal = (grid_y < 0.0) | ((grid_y == 0.0) & (grid_x < 0.0))
    direction_x = np.where(antipodal, -grid_x, grid_x)
    direction_y = np.where(antipodal, -grid_y, grid_y)
    angle = np.mod(np.arctan2(direction_y, direction_x), np.pi) / np.pi
    radial_index = np.floor(
        radial_coordinate * max(int(radial_bins), 4)).astype(np.int64)
    angular_index = np.floor(
        angle * max(int(angular_bins), 8)).astype(np.int64)
    angle_count = max(int(angular_bins), 8)
    bin_index = radial_index * angle_count + angular_index
    bins = max(int(radial_bins), 4) * angle_count
    index = bin_index[supported]
    spectral_mass = energy[supported]
    mass = np.bincount(index, weights=spectral_mass, minlength=bins)
    occupied = mass > 0.0

    def pool(value: np.ndarray) -> np.ndarray:
        total = np.bincount(
            index,
            weights=spectral_mass * value[supported],
            minlength=bins,
        )
        return total[occupied] / mass[occupied]

    pooled_x = pool(direction_x)
    pooled_y = pool(direction_y)
    pooled_log = np.stack([pool(value) for value in log_magnitude], axis=0)
    pooled_relative = pooled_log - np.mean(pooled_log, axis=0, keepdims=True)
    pooled_mass = mass[occupied]
    pooled_mass /= max(float(np.median(pooled_mass)), 1e-12)
    pooled_mass = np.minimum(pooled_mass, 20.0)
    occupied_index = np.flatnonzero(occupied)
    crossfit_fold = (
        occupied_index // angle_count + occupied_index % angle_count) % 2
    return pooled_x, pooled_y, pooled_relative, {
        "supported_fourier_coefficients": int(np.sum(supported)),
        "occupied_fourier_circle_direction_cells": int(np.sum(occupied)),
        "pooled_mass": pooled_mass,
        "crossfit_fold": crossfit_fold,
        "magnitude_floor": floor,
    }


def _measure_from_axes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    side_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (-1.0, 0.0, 1.0)
    axis_weights = tuple(
        (float(weight), 1.0 - 2.0 * float(weight), float(weight))
        for weight in side_weights)
    extents = np.sqrt(
        np.maximum(eigenvalues, 0.0) / np.maximum(2.0 * side_weights, 1e-12))
    axes = tuple(
        extents[index] * eigenvectors[:, index] for index in range(2))
    points = []
    weights = []
    for first_index, first in enumerate(coordinates):
        for second_index, second in enumerate(coordinates):
            points.append(first * axes[0] + second * axes[1])
            weights.append(
                axis_weights[0][first_index] * axis_weights[1][second_index])
    return np.asarray(points), np.asarray(weights)


def estimate_quartic_shape_transport(
    observations: Sequence[np.ndarray],
    covariances: np.ndarray,
    *,
    minimum_frequency: float = 0.02,
    maximum_frequency: float = 0.24,
    radial_bins: int = 20,
    angular_bins: int = 24,
    regularization: float = 0.03,
) -> QuarticShapeTransport:
    """Estimate continuous axis fourth-cumulant shape for every capture."""
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    covariance = np.asarray(covariances, dtype=np.float64)
    count = len(images)
    if count < 3 or covariance.shape != (count, 2, 2):
        raise ValueError("quartic shape transport needs N>=3 covariances")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pooled_x, pooled_y, observed, pool_record = (
        _pooled_relative_log_magnitudes(
            images,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            radial_bins=radial_bins,
            angular_bins=angular_bins,
        ))
    pooled_mass = np.asarray(pool_record.pop("pooled_mass"))
    crossfit_fold = np.asarray(pool_record.pop("crossfit_fold"), dtype=np.int8)
    root_mass = np.sqrt(pooled_mass / max(float(np.mean(pooled_mass)), 1e-12))
    baseline = np.full((count, 2), 1.0 / 6.0, dtype=np.float64)
    transfer_floor = 0.015

    def relative_transfer(parameters: np.ndarray) -> np.ndarray:
        side = parameters.reshape(count, 2)
        records = []
        for capture in range(count):
            log_transfer = np.zeros_like(pooled_x)
            for axis in range(2):
                weight = side[capture, axis]
                extent = np.sqrt(
                    max(eigenvalues[capture, axis], 0.0)
                    / max(2.0 * weight, 1e-12))
                projection = (
                    pooled_x * eigenvectors[capture, 0, axis]
                    + pooled_y * eigenvectors[capture, 1, axis])
                factor = (
                    1.0 - 2.0 * weight
                    + 2.0 * weight * np.cos(
                        2.0 * np.pi * extent * projection))
                log_transfer += 0.5 * np.log(
                    factor * factor + transfer_floor ** 2)
            records.append(log_transfer)
        stack = np.stack(records, axis=0)
        return stack - np.mean(stack, axis=0, keepdims=True)

    baseline_prediction = relative_transfer(baseline.ravel())
    baseline_residual = (
        (baseline_prediction - observed) * root_mass[None, :]).ravel()
    target = observed - baseline_prediction
    cumulant_coefficient = (2.0 * np.pi) ** 4 / 24.0
    basis = np.empty((count, 2, len(pooled_x)), dtype=np.float64)
    for capture in range(count):
        for axis in range(2):
            projection = (
                pooled_x * eigenvectors[capture, 0, axis]
                + pooled_y * eigenvectors[capture, 1, axis])
            basis[capture, axis] = (
                cumulant_coefficient
                * eigenvalues[capture, axis] ** 2
                * projection ** 4)
    design = np.empty((count * len(pooled_x), count * 2), dtype=np.float64)
    for capture in range(count):
        for axis in range(2):
            effect = np.zeros((count, len(pooled_x)), dtype=np.float64)
            effect[capture] = basis[capture, axis]
            effect -= np.mean(effect, axis=0, keepdims=True)
            design[:, 2 * capture + axis] = effect.ravel()
    row_weight = np.tile(root_mass, count)
    weighted_design = design * row_weight[:, None]
    weighted_target = target.ravel() * row_weight
    regularizer = np.sqrt(max(float(regularization), 0.0)) * np.eye(count * 2)
    fold_cumulants = []
    fold_ranks = []
    for fold in (0, 1):
        training_cells = crossfit_fold == fold
        training_rows = np.tile(training_cells, count)
        system = np.vstack((weighted_design[training_rows], regularizer))
        right = np.concatenate((
            weighted_target[training_rows], np.zeros(count * 2)))
        solution, _, rank, _ = np.linalg.lstsq(system, right, rcond=1e-8)
        cumulant = solution.reshape(count, 2)
        cumulant -= np.mean(cumulant, axis=0, keepdims=True)
        fold_cumulants.append(np.clip(cumulant, -1.95, 10.0))
        fold_ranks.append(int(rank))
    fold_cumulants = np.stack(fold_cumulants)
    raw_cumulant = np.mean(fold_cumulants, axis=0)
    crossfit_residual_parts = []
    for training_fold, cumulant in enumerate(fold_cumulants):
        held_out_cells = crossfit_fold != training_fold
        held_out_rows = np.tile(held_out_cells, count)
        predicted = design @ cumulant.ravel()
        crossfit_residual_parts.append(
            (predicted[held_out_rows] - target.ravel()[held_out_rows])
            * row_weight[held_out_rows])
    fitted_residual = np.concatenate(crossfit_residual_parts)
    baseline_rms = float(np.sqrt(np.mean(baseline_residual ** 2)))
    fitted_rms = float(np.sqrt(np.mean(fitted_residual ** 2)))
    crossfit_authority = float(np.clip(
        1.0 - fitted_rms / max(baseline_rms, 1e-12), 0.0, 1.0))
    fold_disagreement = 0.5 * np.abs(
        fold_cumulants[0] - fold_cumulants[1])
    parameter_coherence = (
        raw_cumulant ** 2
        / (raw_cumulant ** 2 + fold_disagreement ** 2 + 0.10 ** 2))
    parameter_authority = crossfit_authority * parameter_coherence
    normalized_kurtosis = np.clip(
        parameter_authority * raw_cumulant, -1.95, 10.0)
    side_weights = np.clip(
        1.0 / (2.0 * (normalized_kurtosis + 3.0)), 0.035, 0.48)
    raw = np.clip(
        1.0 / (2.0 * (raw_cumulant + 3.0)), 0.035, 0.48)
    authority = float(np.mean(parameter_authority))
    measures = tuple(
        _measure_from_axes(
            eigenvalues[index], eigenvectors[index], side_weights[index])
        for index in range(count)
    )
    return QuarticShapeTransport(
        side_weights=side_weights,
        raw_side_weights=raw,
        residual_displacements=tuple(item[0] for item in measures),
        residual_weights=tuple(item[1] for item in measures),
        authority=authority,
        diagnostics={
            "method": (
                "exchange_symmetric_fourier_circle_axis_quartic_transport"),
            "shape_gauge": (
                "exact_zero_mean_normalized_fourth_cumulant_common_shape_"
                "unidentifiable"),
            "side_weights": side_weights.tolist(),
            "raw_side_weights": raw.tolist(),
            "normalized_axis_fourth_cumulants": normalized_kurtosis.tolist(),
            "baseline_relative_log_magnitude_rms": baseline_rms,
            "fitted_relative_log_magnitude_rms": fitted_rms,
            "shape_authority": authority,
            "crossfit_predictive_authority": crossfit_authority,
            "raw_normalized_axis_fourth_cumulants": raw_cumulant.tolist(),
            "fold_cumulant_disagreement": fold_disagreement.tolist(),
            "parameter_coherence": parameter_coherence.tolist(),
            "parameter_authority": parameter_authority.tolist(),
            "authority_method": (
                "two_fold_held_out_fourier_circle_cross_prediction"),
            "linear_fold_ranks": fold_ranks,
            "cumulant_design_columns": int(count * 2),
            **pool_record,
            "capture_role": (
                "all_capture_axis_measures_remain_positive_no_family_selection"),
        },
    )
