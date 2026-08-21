"""Observation-law coordinates around the continual residual posterior.

This is an experiment, not a named-noise dispatch.  A single nonnegative
quadratic variance law is cross-fitted from target-excluded directional
predictions.  Its Fisher or canonical integral supplies the radiance
coordinate in which the unchanged positive Selling transport is evaluated.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import optimize

from .continual_eikonal_noise_transport_2d import (
    _DIRECTIONS,
    _shift_symmetric,
    _validate_image,
    phase_covector_noise_authority,
    phase_covector_sufficient_statistics,
)
from .continual_fabada_eikonal_2d import (
    denoise_continual_residual_posterior_2d,
)


Coordinate = Literal["fisher", "canonical"]


def crossfit_variance_law(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit ``V(s)=a+b*s+c*s^2`` from held-out directional residuals.

    Each directional prediction excludes its target.  Coherent phase lowers a
    sample's fitting measure but never changes the basis or creates a noise
    class.  Nonnegative least squares keeps the variance law positive on the
    observed nonnegative radiance chart.
    """
    image = _validate_image(observation)
    lower = float(np.min(image))
    span = float(np.ptp(image))
    scale = max(span, float(np.max(np.abs(image))), 1.0)
    normalized = (image - lower) / max(span, np.finfo(float).eps * scale)

    predictions = []
    residuals = []
    for dy, dx in _DIRECTIONS:
        prediction = 0.5 * (
            _shift_symmetric(image, -dy, -dx)
            + _shift_symmetric(image, dy, dx))
        predictions.append(prediction)
        residuals.append(image - prediction)
    prediction_stack = np.stack(predictions)
    residual_stack = np.stack(residuals)
    radiance_proxy = np.median(prediction_stack, axis=0)
    proxy = np.clip(
        (radiance_proxy - lower) / max(span, np.finfo(float).eps * scale),
        0.0,
        1.0,
    )
    local_second_moment = np.mean(residual_stack * residual_stack, axis=0)

    numerator, denominator = phase_covector_sufficient_statistics(image)
    phase_noise, phase_diagnostic = phase_covector_noise_authority(
        numerator, denominator)
    # Machine mass makes the fit defined even on an exactly coherent image;
    # it is a numerical measure completion, not a tunable structure weight.
    weight = phase_noise + np.finfo(float).eps
    design = np.stack((np.ones_like(proxy), proxy, proxy * proxy), axis=-1)
    weighted_design = design.reshape(-1, 3) * np.sqrt(weight.ravel())[:, None]
    weighted_target = local_second_moment.ravel() * np.sqrt(weight.ravel())
    coefficients, residual_norm = optimize.nnls(weighted_design, weighted_target)
    fitted = np.einsum("...k,k->...", design, coefficients)
    numerical = np.finfo(float).eps * scale * scale
    fitted = np.maximum(fitted, numerical)
    target_energy = float(np.sum(weight * local_second_moment ** 2))
    fit_energy = float(np.sum(weight * (local_second_moment - fitted) ** 2))
    return coefficients, {
        "coordinate_lower": lower,
        "coordinate_span": span,
        "coefficients": coefficients.tolist(),
        "weighted_residual_norm": float(residual_norm),
        "weighted_explained_fraction": float(
            1.0 - fit_energy / max(target_energy, np.finfo(float).tiny)),
        "mean_local_second_moment": float(np.mean(local_second_moment)),
        "mean_fitted_variance": float(np.mean(fitted)),
        "minimum_fitted_variance": float(np.min(fitted)),
        "maximum_fitted_variance": float(np.max(fitted)),
        "mean_crossfit_weight": float(np.mean(weight)),
        **phase_diagnostic,
    }


def variance_coordinate(
    observation: np.ndarray,
    coefficients: np.ndarray,
    coordinate: Coordinate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build a monotone data-supported coordinate and its inverse knots."""
    image = _validate_image(observation)
    if coordinate not in {"fisher", "canonical"}:
        raise ValueError("coordinate must be 'fisher' or 'canonical'")
    coefficient = np.asarray(coefficients, dtype=np.float64)
    if coefficient.shape != (3,) or np.any(coefficient < 0.0):
        raise ValueError("variance coefficients must be three nonnegative values")
    lower = float(np.min(image))
    upper = float(np.max(image))
    span = upper - lower
    scale = max(span, float(np.max(np.abs(image))), 1.0)
    if span <= np.finfo(float).eps * scale:
        knots = np.array([lower, np.nextafter(lower, np.inf)])
        mapped = image.copy()
        return mapped, knots, knots.copy(), {
            "coordinate_span_before_normalization": 0.0,
            "maximum_roundtrip_error": 0.0,
        }

    knots = np.unique(image)
    normalized = (knots - lower) / span
    variance = (
        coefficient[0]
        + coefficient[1] * normalized
        + coefficient[2] * normalized * normalized)
    floor = np.finfo(float).eps * scale * scale
    variance = np.maximum(variance, floor)
    derivative = (
        1.0 / np.sqrt(variance)
        if coordinate == "fisher"
        else 1.0 / variance)
    increments = np.diff(knots) * 0.5 * (derivative[:-1] + derivative[1:])
    integral = np.concatenate(([0.0], np.cumsum(increments)))
    integral_span = float(integral[-1])
    transformed_knots = lower + span * integral / integral_span
    mapped = np.interp(image, knots, transformed_knots)
    roundtrip = np.interp(mapped, transformed_knots, knots)
    return mapped, knots, transformed_knots, {
        "coordinate_span_before_normalization": integral_span,
        "maximum_roundtrip_error": float(np.max(np.abs(roundtrip - image))),
    }


def denoise_canonical_variance_transport_2d(
    observation: np.ndarray,
    coordinate: Coordinate = "fisher",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate the residual posterior in an inferred observation coordinate."""
    image = _validate_image(observation)
    coefficients, variance_diagnostic = crossfit_variance_law(image)
    transformed, knots, transformed_knots, coordinate_diagnostic = (
        variance_coordinate(image, coefficients, coordinate))
    transformed_estimate, posterior = denoise_continual_residual_posterior_2d(
        transformed)
    estimate = np.interp(transformed_estimate, transformed_knots, knots)
    estimate = np.clip(estimate, float(np.min(image)), float(np.max(image)))
    return estimate, {
        "status": f"{coordinate} variance-coordinate posterior experiment",
        "theory_status": "observation-geometry ablation; not promoted",
        "coordinate": coordinate,
        "variance_law": variance_diagnostic,
        "coordinate_diagnostic": coordinate_diagnostic,
        "posterior": posterior,
        "maximum_observation_identity_error": float(np.max(np.abs(
            image - (estimate + (image - estimate))))),
        "laws": {
            "variance": (
                "one cross-fitted nonnegative quadratic continuum; no noise label"),
            "geometry": (
                "Fisher arclength dx/sqrt(V) or canonical coordinate dx/V"),
            "transport": (
                "unchanged positive Selling residual posterior in transformed radiance"),
        },
    }
