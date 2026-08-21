"""Relative centered-mixing transport after deterministic center alignment.

For two captures of one latent scene, low-frequency Fourier magnitudes obey

    log |Y_2 / Y_1| = -2 pi^2 f^T (C_2 - C_1) f + O(|f|^4).

The latent spectrum and deterministic translation phase cancel.  Only the
covariance difference is identifiable: any common positive blur is a gauge.
We choose the minimum-trace positive gauge by assigning the positive and
negative eigenspaces of that difference to the two observations.  This gauge
is exactly exchange symmetric and does not select a preferred frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .dense_estimation import _luminance


@dataclass(frozen=True)
class RelativeMixingTransport:
    covariance_difference_second_minus_first: np.ndarray
    frame_covariances: tuple[np.ndarray, np.ndarray]
    residual_displacements: tuple[np.ndarray, np.ndarray]
    residual_weights: tuple[np.ndarray, np.ndarray]
    authority: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class MixingMagnitudeSpectrum:
    magnitude: np.ndarray
    shape: tuple[int, int]


def prepare_mixing_magnitude_spectrum(
    image: np.ndarray,
) -> MixingMagnitudeSpectrum:
    """Cache the observation FFT used by every covariance graph edge."""
    value = _luminance(image)
    if min(value.shape) < 16:
        raise ValueError("relative mixing estimation needs at least 16x16 pixels")
    height, width = value.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    centered = (value - np.mean(value)) * window
    return MixingMagnitudeSpectrum(
        np.abs(np.fft.fftshift(np.fft.fft2(centered))),
        (height, width),
    )


@lru_cache(maxsize=16)
def _mixing_frequency_geometry(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    return grid_x, grid_y, np.sqrt(grid_x * grid_x + grid_y * grid_y)


def _positive_sigma_measure(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered positive quadrature with exactly this covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(covariance, dtype=np.float64))
    axes: list[tuple[np.ndarray, np.ndarray]] = []
    for value, direction in zip(eigenvalues, eigenvectors.T):
        if value <= 1e-10:
            continue
        extent = np.sqrt(3.0 * float(value)) * direction
        axes.append((
            np.stack((-extent, np.zeros(2), extent), axis=0),
            np.asarray((1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)),
        ))
    if not axes:
        return np.zeros((1, 2), dtype=np.float64), np.ones(1, dtype=np.float64)
    points = np.zeros((1, 2), dtype=np.float64)
    weights = np.ones(1, dtype=np.float64)
    for axis_points, axis_weights in axes:
        points = (
            points[:, None, :] + axis_points[None, :, :]
        ).reshape(-1, 2)
        weights = (weights[:, None] * axis_weights[None, :]).reshape(-1)
    centroid = np.sum(points * weights[:, None], axis=0)
    return points - centroid[None, :], weights / np.sum(weights)


def estimate_relative_mixing_transport(
    first: np.ndarray,
    second: np.ndarray,
    *,
    minimum_frequency: float = 0.015,
    maximum_frequency: float = 0.16,
) -> RelativeMixingTransport:
    """Estimate the identifiable relative second moment of centered blur."""
    images = (_luminance(first), _luminance(second))
    if images[0].shape != images[1].shape:
        raise ValueError("relative mixing pair must share one raster")
    return estimate_relative_mixing_from_spectra(
        prepare_mixing_magnitude_spectrum(images[0]),
        prepare_mixing_magnitude_spectrum(images[1]),
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )


def estimate_relative_mixing_from_spectra(
    first: MixingMagnitudeSpectrum,
    second: MixingMagnitudeSpectrum,
    *,
    minimum_frequency: float = 0.015,
    maximum_frequency: float = 0.16,
) -> RelativeMixingTransport:
    """Estimate one covariance edge from observation-cached magnitudes."""
    if first.shape != second.shape:
        raise ValueError("relative mixing spectra must share one raster")
    magnitudes = (first.magnitude, second.magnitude)
    combined = np.concatenate([item.ravel() for item in magnitudes])
    magnitude_floor = max(float(np.quantile(combined, 0.25)), 1e-10)
    grid_x, grid_y, radius = _mixing_frequency_geometry(first.shape)
    energy = np.sqrt(magnitudes[0] * magnitudes[1])
    frequency_band = (
        (radius >= max(float(minimum_frequency), 0.0))
        & (radius <= max(float(maximum_frequency), minimum_frequency))
    )
    threshold = float(np.quantile(energy[frequency_band], 0.45))
    supported = frequency_band & (energy >= threshold)
    coordinates = -2.0 * np.pi ** 2 * np.stack((
        grid_x[supported] ** 2,
        2.0 * grid_x[supported] * grid_y[supported],
        grid_y[supported] ** 2,
    ), axis=1)
    log_ratio = (
        np.log(magnitudes[1][supported] + magnitude_floor)
        - np.log(magnitudes[0][supported] + magnitude_floor)
    )
    spectral_weight = energy[supported]
    spectral_weight /= max(float(np.median(spectral_weight)), 1e-12)
    spectral_weight = np.minimum(spectral_weight, 10.0)
    coefficient = np.zeros(3, dtype=np.float64)
    robust_weight = np.ones_like(log_ratio)
    for _ in range(6):
        weight = spectral_weight * robust_weight
        root = np.sqrt(weight)
        coefficient = np.linalg.lstsq(
            coordinates * root[:, None], log_ratio * root, rcond=None)[0]
        residual = log_ratio - coordinates @ coefficient
        scale = max(1.4826 * float(np.median(np.abs(residual))), 0.03)
        robust_weight = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
    covariance_difference = np.asarray((
        (coefficient[0], coefficient[1]),
        (coefficient[1], coefficient[2]),
    ))
    residual = log_ratio - coordinates @ coefficient
    residual_energy = float(np.average(residual * residual, weights=spectral_weight))
    measured_signal_energy = float(np.average(
        log_ratio * log_ratio, weights=spectral_weight))
    signal_energy = max(measured_signal_energy, 1e-12)
    fit_coherence = float(np.clip(1.0 - residual_energy / signal_energy, 0.0, 1.0))
    # Square-root tempering retains real but imperfect differential evidence;
    # zero-coherence evidence remains an exact null action.
    signal_rms = float(np.sqrt(measured_signal_energy))
    evidence_gate = float(1.0 - np.exp(-((signal_rms / 0.03) ** 4)))
    authority = float(np.sqrt(fit_coherence) * evidence_gate)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_difference)
    first_values = np.maximum(-eigenvalues, 0.0) * authority
    second_values = np.maximum(eigenvalues, 0.0) * authority
    frame_covariances = (
        (eigenvectors * first_values[None, :]) @ eigenvectors.T,
        (eigenvectors * second_values[None, :]) @ eigenvectors.T,
    )
    measures = tuple(
        _positive_sigma_measure(covariance) for covariance in frame_covariances)
    return RelativeMixingTransport(
        covariance_difference_second_minus_first=covariance_difference,
        frame_covariances=frame_covariances,
        residual_displacements=(measures[0][0], measures[1][0]),
        residual_weights=(measures[0][1], measures[1][1]),
        authority=authority,
        diagnostics={
            "relative_mixing_method": (
                "low_frequency_log_fourier_magnitude_covariance_transport"),
            "relative_mixing_gauge": (
                "minimum_trace_positive_parts_common_blur_unidentifiable"),
            "covariance_difference_second_minus_first": (
                covariance_difference.tolist()),
            "frame_covariances": [item.tolist() for item in frame_covariances],
            "raw_eigenvalues": eigenvalues.tolist(),
            "fit_coherence": fit_coherence,
            "relative_magnitude_signal_rms": signal_rms,
            "relative_magnitude_evidence_gate": evidence_gate,
            "relative_mixing_authority": authority,
            "fit_residual_rms": float(np.sqrt(residual_energy)),
            "supported_fourier_coefficients": int(np.sum(supported)),
            "residual_atom_counts": [len(item[0]) for item in measures],
            "identifiability_boundary": (
                "only_covariance_difference_is_observed_common_centered_blur_"
                "remains_gauge"),
            "role": (
                "exchange_symmetric_positive_measure_not_frame_or_family_"
                "selection"),
        },
    )


def estimate_adaptive_relative_mixing_from_spectra(
    first: MixingMagnitudeSpectrum,
    second: MixingMagnitudeSpectrum,
    *,
    minimum_frequency: float = 0.015,
    maximum_frequency: float = 0.16,
    cumulant_radius: float = 0.25,
) -> RelativeMixingTransport:
    """Refit inside a continuous low-frequency cumulant-validity radius.

    The preliminary covariance-difference scale sets ``f_max`` through
    ``f_max * sigma <= cumulant_radius``. This changes an analytical radius,
    never a blur family or reconstruction branch.
    """
    preliminary = estimate_relative_mixing_from_spectra(
        first,
        second,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    eigenvalues = np.linalg.eigvalsh(
        preliminary.covariance_difference_second_minus_first)
    differential_sigma = float(np.sqrt(max(
        float(np.max(np.abs(eigenvalues))), 0.0)))
    lower = min(max(
        2.0 * float(minimum_frequency),
        0.03,
        1.5 / min(first.shape),
    ), float(maximum_frequency))
    adaptive_maximum = float(np.clip(
        float(cumulant_radius) / max(differential_sigma, 1.0),
        lower,
        float(maximum_frequency),
    ))
    refined = estimate_relative_mixing_from_spectra(
        first,
        second,
        minimum_frequency=minimum_frequency,
        maximum_frequency=adaptive_maximum,
    )
    return RelativeMixingTransport(
        covariance_difference_second_minus_first=(
            refined.covariance_difference_second_minus_first),
        frame_covariances=refined.frame_covariances,
        residual_displacements=refined.residual_displacements,
        residual_weights=refined.residual_weights,
        authority=refined.authority,
        diagnostics={
            **refined.diagnostics,
            "frequency_radius_method": (
                "continuous_second_cumulant_validity_transport"),
            "preliminary_differential_sigma_pixels": differential_sigma,
            "adaptive_maximum_frequency_cycles_per_pixel": adaptive_maximum,
            "cumulant_radius": float(cumulant_radius),
        },
    )
