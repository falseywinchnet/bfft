"""Support-bearing multiscale fusion in the row--time Hodge geometry.

The short aperture observes temporal derivatives most sharply; the long
aperture observes frequency-row derivatives most sharply.  Combining those
two components produces a vector observation, not necessarily the gradient
of any scalar image.  The longitudinal Hodge component is the integrable
super-resolution field and the transverse component is contradictory
cross-aperture evidence.

This deliberately contrasts with a terminal per-pixel maximum.  Reassigned
power and cross-aperture agreement determine *where* an aperture may speak;
they never prescribe the correction direction.  One feed-forward
residualization then suppresses component observations contradicted by the
first integrable readout, mirroring the finite Meyer jump-measure route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from experiments.ostensibly_frontend.fourier_unrolled_densified import (
    FourierUnrolledDensified,
)


@dataclass(frozen=True)
class SupportedHodgeFusion:
    field: np.ndarray
    initial_field: np.ndarray
    support_confidence: np.ndarray
    refined_confidence: np.ndarray
    raw_time_derivative: np.ndarray
    raw_row_derivative: np.ndarray
    integrable_time_derivative: np.ndarray
    integrable_row_derivative: np.ndarray
    transverse_time_derivative: np.ndarray
    transverse_row_derivative: np.ndarray
    diagnostics: dict[str, object]


def _forward_gradient(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic forward derivative, returned as (time, row)."""
    source = np.asarray(value, dtype=np.float64)
    return (
        np.roll(source, -1, axis=1) - source,
        np.roll(source, -1, axis=0) - source,
    )


def _divergence(time_flux: np.ndarray, row_flux: np.ndarray) -> np.ndarray:
    return (
        time_flux - np.roll(time_flux, 1, axis=1)
        + row_flux - np.roll(row_flux, 1, axis=0)
    )


def _integrate_longitudinal(
    time_flux: np.ndarray,
    row_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a row--time vector observation onto exact scalar gradients."""
    tx = np.asarray(time_flux, dtype=np.float64)
    ry = np.asarray(row_flux, dtype=np.float64)
    if tx.shape != ry.shape or tx.ndim != 2:
        raise ValueError("flux components must be equal two-dimensional arrays")
    height, width = tx.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(width)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(height)[:, None]
    laplacian = (
        2.0 * np.cos(kx) - 2.0
        + 2.0 * np.cos(ky) - 2.0
    )
    safe = np.where(np.abs(laplacian) > 1e-15, laplacian, 1.0)
    potential_hat = np.fft.fft2(_divergence(tx, ry)) / safe
    potential_hat[0, 0] = 0.0
    potential = np.fft.ifft2(potential_hat).real
    gx, gy = _forward_gradient(potential)
    return potential, gx, gy


def _otsu_high_confidence(value: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Continuous high-population confidence with no fitted breakpoint."""
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    histogram, edges = np.histogram(source, bins=256, range=(0.0, 1.0))
    probability = histogram.astype(np.float64)
    probability /= max(float(np.sum(probability)), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(probability)
    moment = np.cumsum(probability * centers)
    total = moment[-1]
    between = (
        (total * weight - moment) ** 2
        / np.maximum(weight * (1.0 - weight), 1e-30)
    )
    split = int(np.argmax(between[:-1]))
    boundary = float(edges[split + 1])
    high_weight = max(1.0 - weight[split], 1e-30)
    high_mean = float(np.sum(
        probability[split + 1:] * centers[split + 1:]
    ) / high_weight)
    confidence = np.clip(
        (source - boundary) / max(high_mean - boundary, 1e-30),
        0.0,
        1.0,
    )
    return confidence, boundary, high_mean


def _normalize_support(stack: np.ndarray) -> np.ndarray:
    records = []
    for field in np.asarray(stack, dtype=np.float64):
        smooth = ndi.gaussian_filter(
            np.log1p(np.maximum(field, 0.0)),
            sigma=(1.0, 1.0),
            mode="reflect",
        )
        scale = max(float(np.quantile(smooth, 0.995)), 1e-30)
        records.append(np.clip(smooth / scale, 0.0, 1.0))
    return np.stack(records)


def _anchor_gauge(potential: np.ndarray, reference: np.ndarray,
                  confidence: np.ndarray) -> np.ndarray:
    active = confidence > 0.25
    if np.count_nonzero(active) < 16:
        active = np.ones(confidence.shape, dtype=bool)
    offset = float(np.median(
        np.asarray(reference, dtype=np.float64)[active] - potential[active]
    ))
    return np.maximum(potential + offset, 0.0)


def _low_residual_weight(residual: np.ndarray,
                         confidence: np.ndarray) -> np.ndarray:
    """Keep the low-error population after one integrable prediction."""
    magnitude = np.abs(np.asarray(residual, dtype=np.float64))
    active = confidence > 0.0
    if not np.any(active):
        return np.zeros_like(magnitude)
    scale = max(float(np.quantile(magnitude[active], 0.95)), 1e-30)
    normalized = np.clip(magnitude / scale, 0.0, 1.0)
    high_error, _, _ = _otsu_high_confidence(normalized)
    return 1.0 - high_error


def supported_hodge_fusion(
    result: FourierUnrolledDensified,
) -> SupportedHodgeFusion:
    """Fuse aperture derivatives once in a support-bearing Hodge coordinate."""
    apertures = result.apertures
    if len(apertures) < 3 or result.target_n_fft not in apertures:
        raise ValueError("a short, target, and long aperture are required")
    short_index = int(np.argmin(apertures))
    long_index = int(np.argmax(apertures))
    base_index = apertures.index(result.target_n_fft)
    fields = np.asarray(result.aligned_fields, dtype=np.float64)
    base = fields[base_index]
    short = fields[short_index]
    long = fields[long_index]

    supports = _normalize_support(result.aligned_support)
    # Independent evidence means the second strongest support, not the union.
    support_agreement = np.partition(supports, -2, axis=0)[-2]
    field_sorted = np.sort(np.maximum(fields, 0.0), axis=0)
    amplitude_agreement = field_sorted[-2] / np.maximum(field_sorted[-1], 1e-30)
    raw_confidence = np.sqrt(np.clip(
        support_agreement * amplitude_agreement, 0.0, 1.0))
    confidence, boundary, high_mean = _otsu_high_confidence(raw_confidence)

    base_time, base_row = _forward_gradient(base)
    short_time, _ = _forward_gradient(short)
    _, long_row = _forward_gradient(long)
    raw_time = base_time + confidence * (short_time - base_time)
    raw_row = base_row + confidence * (long_row - base_row)
    initial_potential, initial_time, initial_row = _integrate_longitudinal(
        raw_time, raw_row)
    initial_field = _anchor_gauge(initial_potential, np.median(fields, axis=0),
                                  confidence)

    # One feed-forward remeasurement.  Contradiction with the first
    # integrable state removes carrier/blur leakage from the accepted support.
    time_weight = _low_residual_weight(short_time - initial_time, confidence)
    row_weight = _low_residual_weight(long_row - initial_row, confidence)
    refined_time_confidence = confidence * time_weight
    refined_row_confidence = confidence * row_weight
    refined_confidence = np.sqrt(
        refined_time_confidence * refined_row_confidence)
    final_time_observation = (
        base_time
        + refined_time_confidence * (short_time - base_time)
    )
    final_row_observation = (
        base_row
        + refined_row_confidence * (long_row - base_row)
    )
    potential, integrable_time, integrable_row = _integrate_longitudinal(
        final_time_observation, final_row_observation)
    field = _anchor_gauge(
        potential, np.median(fields, axis=0), refined_confidence)
    transverse_time = final_time_observation - integrable_time
    transverse_row = final_row_observation - integrable_row
    observed_energy = max(float(np.sum(
        final_time_observation ** 2 + final_row_observation ** 2)), 1e-30)

    return SupportedHodgeFusion(
        field=field,
        initial_field=initial_field,
        support_confidence=confidence,
        refined_confidence=refined_confidence,
        raw_time_derivative=final_time_observation,
        raw_row_derivative=final_row_observation,
        integrable_time_derivative=integrable_time,
        integrable_row_derivative=integrable_row,
        transverse_time_derivative=transverse_time,
        transverse_row_derivative=transverse_row,
        diagnostics={
            "short_aperture": int(apertures[short_index]),
            "base_aperture": int(apertures[base_index]),
            "long_aperture": int(apertures[long_index]),
            "support_partition": "Otsu high population of independent support",
            "support_boundary": boundary,
            "support_high_mean": high_mean,
            "support_active_fraction": float(np.mean(confidence > 0.0)),
            "refined_support_active_fraction": float(
                np.mean(refined_confidence > 0.0)),
            "feed_forward_residualizations": 1,
            "transverse_energy_fraction": float(np.sum(
                transverse_time ** 2 + transverse_row ** 2
            ) / observed_energy),
            "terminal_semantics": (
                "longitudinal_Hodge_integration_of_support_gated_"
                "short_time_and_long_frequency_derivatives"
            ),
        },
    )

